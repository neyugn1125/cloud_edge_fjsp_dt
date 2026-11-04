"""
Gymnasium Discrete-Event Simulation Environment for Green Flexible Job Shop Scheduling (Green-FJSP).

Implements GreenFJSPEnv conforming to the Gymnasium API:
    - Speed-scaling cutting speed control (s in {1, 2, 3})
    - Tool wear deterioration and opportunistic preventive maintenance (PM)
    - Idle-gap energy-saving transitions based on Break-Even Time (T_BE)
    - Time-of-Use (TOU) electricity pricing cost integration
    - Shop-floor peak power ceiling (P_limit) penalty
    - Action masking for valid dispatching decisions
    - Multi-objective reward parameterized by Cloud-Edge sync weights (w_cmax, w_energy, w_tardiness)

Strict Unit Conventions:
    - Time: Minutes (min)
    - Power: Kilowatts (kW)
    - Energy: Kilowatt-hours (kWh)
    - Tool Wear VB: Micrometers (um)
"""

import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from sim_env.physical_machine import CNCPhysicalTwin, OperationExecutionResult
from utils.mqtt_schemas import MachineConfig, MachineState, SystemConfig, load_system_config


class GreenFJSPEnv(gym.Env):
    """
    Gymnasium environment for Energy-Aware Dynamic FJSP with Speed Scaling
    and Tool Deterioration.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        benchmark_path: Union[str, Path] = "data/benchmarks/Mk01_green.json",
        config_path: Union[str, Path] = "config/system_params.yaml",
        cloud_schedule_path: Optional[Union[str, Path]] = "data/benchmarks/cloud_baseline_schedule.json",
        cloud_guided_mask: bool = False,
        weights: Tuple[float, float, float] = (0.4, 0.4, 0.2),
        lambda_peak: float = 0.30,
        p_penalty_factor: Optional[float] = None,
        breakdown_penalty: float = 10.0,
        domain_randomization: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the Green FJSP environment.

        Args:
            benchmark_path: Path to Mk01_green.json.
            config_path: Path to system_params.yaml.
            cloud_schedule_path: Optional path to cloud_baseline_schedule.json for Cloud Guidance.
            cloud_guided_mask: If True, action mask guides machine selection to Cloud plan and locks critical wear.
            weights: Objective trade-off weights (w_cmax, w_energy, w_tardiness).
            lambda_peak: Peak demand surcharge per kWh of excess power ($0.30/kWh).
            p_penalty_factor: Backwards compatibility alias for lambda_peak.
            breakdown_penalty: Fixed penalty on tool breakdown event (default 10.0).
            domain_randomization: Whether to perturb initial wear and eta on reset.
            seed: Optional random seed.
        """
        super().__init__()

        self.cloud_guided_mask: bool = cloud_guided_mask

        # 1. Load System Configuration & Benchmark Dataset
        self.config_path = Path(config_path)
        self.benchmark_path = Path(benchmark_path)

        self.system_config: SystemConfig = load_system_config(self.config_path)

        with open(self.benchmark_path, "r", encoding="utf-8") as f:
            self.benchmark_data: Dict[str, Any] = json.load(f)

        if "global_parameters" in self.benchmark_data and "p_limit_kw" in self.benchmark_data["global_parameters"]:
            self.p_limit_kw: float = float(self.benchmark_data["global_parameters"]["p_limit_kw"])
        else:
            self.p_limit_kw: float = self.system_config.p_limit_kw
        self.tou_pricing = self.system_config.tou_pricing

        # Load Cloud Baseline Guidance if provided
        self.cloud_schedule_path = Path(cloud_schedule_path) if cloud_schedule_path else None
        self.cloud_op_machine: Dict[Tuple[int, int], str] = {}
        self.cloud_op_speed: Dict[Tuple[int, int], int] = {}
        self.cloud_op_pm: Dict[Tuple[int, int], bool] = {}
        self.cloud_op_order: Dict[Tuple[int, int], int] = {}
        self.cloud_step_order: List[int] = []
        self.cloud_sequence: List[Dict[str, Any]] = []

        if self.cloud_schedule_path and self.cloud_schedule_path.exists():
            with open(self.cloud_schedule_path, "r", encoding="utf-8") as f:
                cdata = json.load(f)
            self.cloud_sequence = cdata.get("operations_schedule", [])
            for idx, item in enumerate(self.cloud_sequence):
                j = int(item["job_index"])
                o = int(item["op_index"])
                self.cloud_op_machine[(j, o)] = item["machine_id"]
                self.cloud_op_speed[(j, o)] = int(item["speed_level"])
                self.cloud_op_pm[(j, o)] = bool(item["had_pm"])
                self.cloud_op_order[(j, o)] = idx
                self.cloud_step_order.append(j)

        # 2. Problem Dimensions
        self.jobs_data: List[Dict[str, Any]] = self.benchmark_data["jobs"]
        self.num_jobs: int = len(self.jobs_data)
        if "machines" in self.benchmark_data and self.benchmark_data["machines"]:
            self.machine_ids: List[str] = sorted(list(self.benchmark_data["machines"].keys()))
        else:
            self.machine_ids: List[str] = sorted(list(self.system_config.machines.keys()))
        self.num_machines: int = len(self.machine_ids)
        self.num_speed_levels: int = len(self.system_config.speed_scaling.levels)  # 3 (levels 1, 2, 3)
        self.num_pm_options: int = 2  # 0: no PM, 1: opportunistic PM

        self.total_operations: int = self.benchmark_data["global_parameters"]["total_operations"]

        # Objective weights: (w_cmax, w_energy, w_tardiness)
        self.set_objective_weights(weights)
        self.lambda_peak: float = float(p_penalty_factor) if p_penalty_factor is not None else float(lambda_peak)
        self.p_penalty_factor: float = self.lambda_peak
        self.breakdown_penalty: float = breakdown_penalty
        self.domain_randomization: bool = domain_randomization

        # Normalization reference baselines guided by Rule 2 (78.0m, $2.72, 345.0m)
        self.ref_baseline_cmax: float = 78.0       # min
        self.ref_baseline_cost: float = 2.72       # $
        self.ref_baseline_tardiness: float = 345.0 # min
        self.ref_max_makespan: float = 200.0  # min
        self.ref_max_energy: float = 150.0    # kWh
        self.ref_max_rate: float = 0.25       # $/kWh

        # 3. Action Space: Discrete(360)
        # Action encoding: job_idx * (M * S * PM) + machine_idx * (S * PM) + speed_idx * PM + pm_flag
        self.total_actions: int = self.num_jobs * self.num_machines * self.num_speed_levels * self.num_pm_options
        self.action_space = spaces.Discrete(self.total_actions)

        # 4. Observation Space: Box(low=0.0, high=1.0, shape=(87,), dtype=np.float32)
        # 10 Jobs * 5 features = 50
        # 6 Machines * 5 features = 30
        # 7 Global features = 7
        # Total = 87 features
        self.obs_dim: int = (self.num_jobs * 5) + (self.num_machines * 5) + 7
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # 5. Initialize Physical Machine Twins
        self.machines: Dict[str, CNCPhysicalTwin] = {}
        for m_id in self.machine_ids:
            m_cfg = self.system_config.get_machine(m_id)
            self.machines[m_id] = CNCPhysicalTwin(
                machine_id=m_id,
                config=m_cfg,
                alpha_p=self.system_config.speed_scaling.alpha_p,
                speed_levels={
                    k: v.speed_factor
                    for k, v in self.system_config.speed_scaling.levels.items()
                },
            )

        # Dynamic simulation tracking attributes
        self.current_op_idx: List[int] = [0] * self.num_jobs
        self.job_ready_time: List[float] = [0.0] * self.num_jobs
        self.job_completed: List[bool] = [False] * self.num_jobs
        self.cmax: float = 0.0
        self.total_cost: float = 0.0
        self.total_weighted_tardiness: float = 0.0
        self.completed_operations: int = 0

        # Cutting intervals for concurrent power tracking: (start, end, power_kw, machine_id)
        self.cutting_intervals: List[Tuple[float, float, float, str]] = []

        # Detailed audit schedule log of all executed events
        self.scheduled_events: List[Dict[str, Any]] = []

        if seed is not None:
            self.reset(seed=seed)

    def set_objective_weights(self, weights: Tuple[float, float, float]) -> None:
        """Update objective weights from Cloud layer (w_cmax, w_energy, w_tardiness)."""
        w_c, w_e, w_t = weights
        tot = w_c + w_e + w_t
        if tot <= 0:
            raise ValueError(f"Sum of weights must be positive, got {weights}")
        self.w_cmax: float = w_c / tot
        self.w_energy: float = w_e / tot
        self.w_tardiness: float = w_t / tot

    def encode_action(self, job_idx: int, machine_idx: int, speed_level: int, pm_flag: int) -> int:
        """Encode 4 decision components into a single integer action."""
        speed_idx = speed_level - 1
        return (
            job_idx * (self.num_machines * self.num_speed_levels * self.num_pm_options)
            + machine_idx * (self.num_speed_levels * self.num_pm_options)
            + speed_idx * self.num_pm_options
            + pm_flag
        )

    def decode_action(self, action: int) -> Tuple[int, int, int, int]:
        """Decode a single integer action into (job_idx, machine_idx, speed_level, pm_flag)."""
        job_stride = self.num_machines * self.num_speed_levels * self.num_pm_options
        machine_stride = self.num_speed_levels * self.num_pm_options

        job_idx, rem = divmod(action, job_stride)
        machine_idx, rem = divmod(rem, machine_stride)
        speed_idx, pm_flag = divmod(rem, self.num_pm_options)
        speed_level = speed_idx + 1

        return job_idx, machine_idx, speed_level, pm_flag

    def action_masks(self) -> np.ndarray:
        """
        Compute boolean mask of valid actions for the current simulation state.

        Conditions for validity:
            1. Job `job_idx` is not finished.
            2. Machine `M{machine_idx + 1}` is in the eligible set for the current operation.
            3. PM (pm_flag = 1) is strictly restricted to critical flank wear: wear >= 78% w_max.
            4. In cloud_guided_mask mode, prioritizes top ready jobs and aligns with Cloud machine allocation.
        """
        mask = np.zeros(self.total_actions, dtype=bool)

        ready_jobs = [
            j for j in range(self.num_jobs)
            if self.current_op_idx[j] < len(self.jobs_data[j]["operations"])
        ]

        if not ready_jobs:
            return mask

        # When cloud_guided_mask is True, prioritize the top ready jobs in Cloud's planned order
        if self.cloud_guided_mask and self.cloud_op_order:
            ready_jobs.sort(key=lambda j: self.cloud_op_order.get((j, self.current_op_idx[j]), 999))
            active_jobs = ready_jobs[:2]
        else:
            active_jobs = ready_jobs

        for job_idx in active_jobs:
            curr_op = self.current_op_idx[job_idx]
            job_ops = self.jobs_data[job_idx]["operations"]
            op_info = job_ops[curr_op]
            eligible_m_ids = {alt["machine_id"] for alt in op_info["eligible_machines"]}

            # Cloud guidance: guide machine selection to Cloud NSGA-III assignment
            cloud_target = self.cloud_op_machine.get((job_idx, curr_op))
            if self.cloud_guided_mask and cloud_target and cloud_target in eligible_m_ids:
                allowed_m_ids = {cloud_target}
            else:
                allowed_m_ids = eligible_m_ids

            for m_idx, m_id in enumerate(self.machine_ids):
                if m_id not in allowed_m_ids:
                    continue  # Machine not eligible or not guided for this operation

                twin = self.machines[m_id]
                wear = twin.current_wear_um
                w_max = twin.w_max_um

                alt = next(a_item for a_item in op_info["eligible_machines"] if a_item["machine_id"] == m_id)
                est_wear_inc = twin.base_eta * float(alt["t_base_min"])

                # PM is strictly permitted when current wear is in degradation: >= 78% w_max (or >= 75% in cloud-guided mode)
                # OR when the upcoming operation's cutting increment would push tool wear near failure (>= 95% w_max)
                pm_threshold = 0.75 if self.cloud_guided_mask else 0.78
                can_pm = (wear >= pm_threshold * w_max) or (wear + 1.35 * est_wear_inc >= 0.95 * w_max)

                # Critical wear protection (active in cloud-guided mode):
                # If wear >= 88% w_max, high cutting speed s=3 is locked (pm_flag=0)
                # If wear >= 92% w_max or the cut would cause breakdown, PM is mandatory (pm_flag=0 is locked)
                is_critical_wear = (wear >= 0.88 * w_max) and self.cloud_guided_mask
                is_extreme_wear = ((wear >= 0.92 * w_max) or (wear + 1.25 * est_wear_inc >= 0.98 * w_max)) and self.cloud_guided_mask

                for speed_level in range(1, self.num_speed_levels + 1):
                    # Normal cutting (pm_flag = 0)
                    if not is_extreme_wear:
                        if not (is_critical_wear and speed_level == 3):
                            act_no_pm = self.encode_action(job_idx, m_idx, speed_level, 0)
                            mask[act_no_pm] = True

                    # Opportunistic PM (pm_flag = 1)
                    if can_pm:
                        act_pm = self.encode_action(job_idx, m_idx, speed_level, 1)
                        mask[act_pm] = True

        return mask

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to start a new simulation episode."""
        super().reset(seed=seed)

        # Optional cloud synchronization parameter overrides
        if options and "weights" in options:
            self.set_objective_weights(options["weights"])

        # Reset all CNC Physical Twins with optional domain randomization
        dr = self.domain_randomization
        if options and "domain_randomization" in options:
            dr = bool(options["domain_randomization"])

        is_realistic = options.get("realistic_disturbance", False) if options else False

        for twin in self.machines.values():
            if is_realistic:
                # Realistic Shop-Floor Disturbance Scenario:
                # Non-uniform workpiece hardness causes wear rate eta to surge +15% to +35%
                # Initial tool wear varies by +8% to +18%
                eta_scale = float(self.np_random.uniform(1.15, 1.35))
                w_noise = float(self.np_random.uniform(0.08, 0.18))
                init_wear = max(0.0, twin.base_initial_wear_um * (1.0 + w_noise))
                eta_val = twin.base_eta * eta_scale
                twin.reset(initial_time=0.0, initial_wear_um=init_wear, eta=eta_val)
            elif dr:
                # Training Sim-to-Real Domain Randomization
                w_range = float(options.get("wear_noise_range", 0.10)) if options else 0.10
                eta_range = float(options.get("eta_noise_range", 0.05)) if options else 0.05
                w_noise = float(self.np_random.uniform(-w_range, w_range))
                eta_noise = float(self.np_random.uniform(-eta_range, eta_range))
                init_wear = max(0.0, twin.base_initial_wear_um * (1.0 + w_noise))
                eta_val = max(0.1, twin.base_eta * (1.0 + eta_noise))
                twin.reset(initial_time=0.0, initial_wear_um=init_wear, eta=eta_val)
            else:
                twin.reset(initial_time=0.0, initial_wear_um=twin.base_initial_wear_um, eta=twin.base_eta)

        # Reset Job tracking progress
        self.current_op_idx = [0] * self.num_jobs
        self.job_ready_time = [0.0] * self.num_jobs
        self.job_completed = [False] * self.num_jobs

        self.cmax = 0.0
        self.total_cost = 0.0
        self.total_weighted_tardiness = 0.0
        self.completed_operations = 0
        self.cutting_intervals.clear()
        self.scheduled_events.clear()

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one scheduling decision in the environment.

        Args:
            action: Discrete action index in [0, 359].

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict]: (obs, reward, terminated, truncated, info)
        """
        job_idx, machine_idx, speed_level, pm_flag = self.decode_action(action)
        m_id = self.machine_ids[machine_idx]
        machine = self.machines[m_id]

        # Defensive action validation
        curr_op = self.current_op_idx[job_idx]
        job_ops = self.jobs_data[job_idx]["operations"]

        if curr_op >= len(job_ops):
            # Invalid action on completed job
            return self._get_obs(), -50.0, False, False, {"error": "Job already completed"}

        op_info = job_ops[curr_op]
        alt = next((a for a in op_info["eligible_machines"] if a["machine_id"] == m_id), None)
        if alt is None:
            # Machine not eligible
            return self._get_obs(), -50.0, False, False, {"error": "Machine not eligible"}

        if pm_flag == 1 and machine.current_wear_um <= (0.5 * machine.w_max_um):
            # Premature PM disallowed
            return self._get_obs(), -50.0, False, False, {"error": "Tool wear <= 50% w_max"}

        t_base_min = float(alt["t_base_min"])

        # 1. Determine Earliest Start Time
        job_ready = self.job_ready_time[job_idx]
        mach_ready = machine.available_time
        start_time = max(job_ready, mach_ready)

        # 2. Idle Gap Management on Machine
        step_cost = 0.0
        action_taken = None
        if start_time > mach_ready:
            idle_gap = start_time - mach_ready
            action_taken, idle_energy = machine.apply_idle_gap(idle_gap)

            # Compute TOU cost during idle gap
            if action_taken == "IDLE":
                gap_power = machine.p_idle_kw
                gap_cost = self._integrate_tou_cost(gap_power, mach_ready, start_time)
            elif action_taken == "STANDBY":
                gap_power = machine.p_stdby_kw
                gap_cost = self._integrate_tou_cost(gap_power, mach_ready, start_time)
            else:  # TURN_OFF
                # Restart energy billed at the rate when machine warms up
                warmup_rate = self.system_config.get_tou_rate(start_time)
                gap_cost = machine.e_on_off_kwh_param * warmup_rate

            step_cost += gap_cost

        # 3. Machining & Tool Wear Execution
        prev_pm_reps = machine.num_pm_replacements
        prev_breakdowns = machine.num_breakdowns

        exec_res: OperationExecutionResult = machine.execute_operation(
            t_base_min=t_base_min,
            speed_level=speed_level,
            do_opportunistic_pm=bool(pm_flag),
            rng=random.Random(int(self.np_random.integers(0, 1_000_000))),
        )

        end_time = machine.available_time
        self.job_ready_time[job_idx] = end_time
        self.current_op_idx[job_idx] += 1
        self.completed_operations += 1

        # Calculate exact cutting time interval
        t_cut_start = start_time + exec_res.t_pm_min
        t_cut_end = t_cut_start + exec_res.t_proc_min

        # 4. TOU Electricity Cost Integration
        cost_proc = self._integrate_tou_cost(exec_res.p_actual_kw, t_cut_start, t_cut_end)
        cost_pm = (
            self._integrate_tou_cost(machine.p_stdby_kw, start_time, t_cut_start)
            if exec_res.had_pm
            else 0.0
        )
        cost_bd = (
            self._integrate_tou_cost(machine.p_stdby_kw, t_cut_end, end_time)
            if exec_res.had_breakdown
            else 0.0
        )

        op_cost = cost_proc + cost_pm + cost_bd
        step_cost += op_cost

        # 5. Shop-floor Peak Power Constraint Check
        excess_kwh = self._evaluate_peak_power_excess(t_cut_start, t_cut_end, exec_res.p_actual_kw)
        self.cutting_intervals.append((t_cut_start, t_cut_end, exec_res.p_actual_kw, m_id))
        peak_surcharge = self.lambda_peak * excess_kwh
        peak_penalty = peak_surcharge
        step_cost += peak_surcharge
        self.total_cost += step_cost

        # 6. Makespan and Tardiness Increments
        new_cmax = max(m.available_time for m in self.machines.values())
        delta_cmax = max(0.0, new_cmax - self.cmax)
        self.cmax = new_cmax

        delta_tardiness = 0.0
        if self.current_op_idx[job_idx] == len(job_ops):
            self.job_completed[job_idx] = True
            job_spec = self.jobs_data[job_idx]
            due_date = float(job_spec["due_date_min"])
            weight = float(job_spec["weight"])
            tardiness = max(0.0, end_time - due_date)
            weighted_tard = weight * tardiness
            delta_tardiness = weighted_tard
            self.total_weighted_tardiness += weighted_tard

        # 7. Multi-Objective Step Reward Guided by Baseline & Physical Domain Knowledge
        # Normalization denominators based on Rule 2 baseline (78.0m, $2.72, 345.0m)
        norm_delta_cmax = delta_cmax / self.ref_baseline_cmax
        norm_delta_cost = step_cost / self.ref_baseline_cost
        norm_delta_tardiness = delta_tardiness / self.ref_baseline_tardiness

        objective_penalty = (
            self.w_cmax * norm_delta_cmax
            + self.w_energy * norm_delta_cost
            + self.w_tardiness * norm_delta_tardiness
        )

        penalty_bd = self.breakdown_penalty if exec_res.had_breakdown else 0.0

        # Cloud Schedule Guidance Bonus:
        # Strongly guides the agent to follow Cloud NSGA-III's global machine allocation & sequence
        cloud_guidance_bonus = 0.0
        op_idx_before = self.current_op_idx[job_idx] - 1
        target_machine_id = self.cloud_op_machine.get((job_idx, op_idx_before))
        if target_machine_id is not None and m_id == target_machine_id:
            cloud_guidance_bonus += 0.15

        if self.cloud_step_order and (self.completed_operations - 1) < len(self.cloud_step_order):
            expected_job = self.cloud_step_order[self.completed_operations - 1]
            if job_idx == expected_job:
                cloud_guidance_bonus += 0.08

        # Adaptive speed shaping
        tou_rate = self.system_config.get_tou_rate(t_cut_start)
        is_off_peak = (tou_rate <= 0.06)
        is_on_peak = (tou_rate >= 0.20)
        adaptive_bonus = 0.0

        # High-speed cutting in Off-Peak with fresh tool (wear < 65 um)
        if is_off_peak and exec_res.wear_before_um < 65.0 and speed_level in (2, 3):
            adaptive_bonus += 0.08
        # Eco-speed in On-Peak window
        if is_on_peak and speed_level == 1:
            adaptive_bonus += 0.12
        elif is_on_peak and speed_level == 3:
            adaptive_bonus -= 0.10
        elif is_on_peak and speed_level == 2:
            adaptive_bonus -= 0.05

        # NOTE: Zero positive PM bonus! Tool replacement downtime is its own cost.

        step_reward = -(objective_penalty + penalty_bd) + cloud_guidance_bonus + adaptive_bonus

        # Record scheduled event for timeline audit & Gantt visualization
        event_record = {
            "step": self.completed_operations,
            "job_id": f"J{job_idx + 1}",
            "job_idx": job_idx,
            "op_id": f"J{job_idx + 1}_O{curr_op + 1}",
            "op_idx": curr_op,
            "machine_id": m_id,
            "machine_idx": machine_idx,
            "speed_level": speed_level,
            "speed_factor": exec_res.speed_factor,
            "p_actual_kw": exec_res.p_actual_kw,
            "mach_ready_before": mach_ready,
            "start_time": start_time,
            "t_pm_start": start_time if exec_res.had_pm else None,
            "t_pm_end": t_cut_start if exec_res.had_pm else None,
            "t_cut_start": t_cut_start,
            "t_cut_end": t_cut_end,
            "t_bd_start": t_cut_end if exec_res.had_breakdown else None,
            "t_bd_end": end_time if exec_res.had_breakdown else None,
            "end_time": end_time,
            "had_pm": exec_res.had_pm,
            "had_breakdown": exec_res.had_breakdown,
            "idle_gap_min": (start_time - mach_ready) if start_time > mach_ready else 0.0,
            "idle_action": action_taken if start_time > mach_ready else None,
        }
        self.scheduled_events.append(event_record)

        # 8. Termination Check
        terminated = (self.completed_operations >= self.total_operations)
        truncated = False

        return self._get_obs(), float(step_reward), terminated, truncated, self._get_info()

    def _integrate_tou_cost(self, power_kw: float, t_start: float, t_end: float) -> float:
        """
        Analytically integrate electricity cost ($) across piecewise constant TOU tariff tiers.
        """
        if t_end <= t_start or power_kw <= 0.0:
            return 0.0

        cycle = self.tou_pricing.cycle_min  # 1440 min
        start_day = int(t_start // cycle)
        end_day = int(t_end // cycle)
        cost = 0.0

        for day in range(start_day, end_day + 1):
            day_offset = day * cycle
            for tier in self.tou_pricing.tiers:
                tier_start = day_offset + tier.start_min
                tier_end = day_offset + tier.end_min

                overlap_start = max(t_start, tier_start)
                overlap_end = min(t_end, tier_end)

                if overlap_end > overlap_start:
                    duration_min = overlap_end - overlap_start
                    energy_kwh = power_kw * (duration_min / 60.0)
                    cost += energy_kwh * tier.rate_per_kwh

        return cost

    def _evaluate_peak_power_excess(self, new_start: float, new_end: float, new_power: float) -> float:
        """
        Evaluate marginal peak power threshold violation against concurrent cutting operations.
        Returns marginal excess energy (kWh) beyond p_limit_kw introduced by this operation.
        """
        overlapping = [
            (s, e, p)
            for (s, e, p, _) in self.cutting_intervals
            if not (e <= new_start or s >= new_end)
        ]

        if not overlapping:
            return max(0.0, new_power - self.p_limit_kw) * (new_end - new_start) / 60.0

        # Critical event transition points
        time_points = sorted(
            list(
                set(
                    [new_start, new_end]
                    + [max(new_start, s) for s, e, _ in overlapping]
                    + [min(new_end, e) for s, e, _ in overlapping]
                )
            )
        )

        excess_kwh = 0.0
        for i in range(len(time_points) - 1):
            t1, t2 = time_points[i], time_points[i + 1]
            dt_min = t2 - t1
            if dt_min <= 0:
                continue
            mid_t = 0.5 * (t1 + t2)
            prev_p = sum(p for s, e, p in overlapping if s <= mid_t < e)
            new_total_p = prev_p + new_power
            excess_now = max(0.0, new_total_p - self.p_limit_kw)
            excess_prev = max(0.0, prev_p - self.p_limit_kw)
            excess_kwh += (excess_now - excess_prev) * (dt_min / 60.0)

        return excess_kwh

    def _get_obs(self) -> np.ndarray:
        """
        Construct a normalized 87-dimensional observation vector in [0.0, 1.0].
        """
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        cursor = 0

        current_time = min(m.available_time for m in self.machines.values())

        # 1. Job features (10 jobs * 5 = 50)
        for j_idx, job in enumerate(self.jobs_data):
            num_ops = len(job["operations"])
            curr_op = self.current_op_idx[j_idx]
            due_date = float(job["due_date_min"])
            weight = float(job["weight"])

            # Progress
            progress = curr_op / num_ops
            # Work remaining
            rem_ops = job["operations"][curr_op:]
            rem_work = sum(op["mean_processing_time_min"] for op in rem_ops)
            tot_work = float(job["total_expected_work_min"])
            rem_work_ratio = rem_work / tot_work if tot_work > 0 else 0.0

            # Job ready time normalized
            ready_norm = np.clip(self.job_ready_time[j_idx] / self.ref_max_makespan, 0.0, 1.0)

            # Urgency / Slack ratio relative to Due Date
            slack = due_date - current_time
            urgency = np.clip(1.0 - (slack / self.ref_max_makespan), 0.0, 1.0)

            # Priority weight normalized (1, 2, or 3)
            weight_norm = weight / 3.0

            obs[cursor : cursor + 5] = [progress, rem_work_ratio, ready_norm, urgency, weight_norm]
            cursor += 5

        # 2. Machine features (6 machines * 5 = 30)
        for m_id in self.machine_ids:
            twin = self.machines[m_id]

            m_ready_norm = np.clip(twin.available_time / self.ref_max_makespan, 0.0, 1.0)
            wear_ratio = twin.tool_wear_ratio
            power_drift = np.clip(twin.gamma * wear_ratio, 0.0, 1.0)
            is_proc = 1.0 if twin.state == MachineState.PROCESSING else 0.0
            is_stby = 1.0 if twin.state in (MachineState.STANDBY, MachineState.OFF) else 0.0

            obs[cursor : cursor + 5] = [m_ready_norm, wear_ratio, power_drift, is_proc, is_stby]
            cursor += 5

        # 3. Global features (7)
        curr_rate = self.system_config.get_tou_rate(current_time)
        rate_norm = np.clip(curr_rate / self.ref_max_rate, 0.0, 1.0)

        tot_energy = sum(m.total_energy_kwh for m in self.machines.values())
        energy_norm = np.clip(tot_energy / self.ref_max_energy, 0.0, 1.0)

        time_norm = np.clip(current_time / self.ref_max_makespan, 0.0, 1.0)

        # Active power load ratio
        active_power = sum(
            m.p_cut_base_kw if m.state == MachineState.PROCESSING else m.p_idle_kw
            for m in self.machines.values()
        )
        active_power_ratio = np.clip(active_power / self.p_limit_kw, 0.0, 1.0)

        obs[cursor : cursor + 7] = [
            time_norm,
            rate_norm,
            energy_norm,
            active_power_ratio,
            self.w_cmax,
            self.w_energy,
            self.w_tardiness,
        ]

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Telemetry diagnostics dictionary."""
        tot_energy = sum(m.total_energy_kwh for m in self.machines.values())
        tot_breakdowns = sum(m.num_breakdowns for m in self.machines.values())
        tot_pms = sum(m.num_pm_replacements for m in self.machines.values())

        return {
            "cmax": round(self.cmax, 2),
            "total_energy_kwh": round(tot_energy, 3),
            "total_cost": round(self.total_cost, 3),
            "total_weighted_tardiness": round(self.total_weighted_tardiness, 2),
            "total_breakdowns": tot_breakdowns,
            "total_pm_replacements": tot_pms,
            "completed_operations": self.completed_operations,
            "progress_ratio": round(self.completed_operations / self.total_operations, 3),
            "action_mask": self.action_masks(),
        }


if __name__ == "__main__":
    print("=" * 80)
    print("Green-FJSP Gymnasium Environment Demonstration (Mk01 - 55 Operations)")
    print("=" * 80)

    env = GreenFJSPEnv(
        benchmark_path="data/benchmarks/Mk01_green.json",
        config_path="config/system_params.yaml",
        weights=(0.4, 0.4, 0.2),
        seed=42,
    )

    obs, info = env.reset(seed=42)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space     : {env.action_space}")
    print(f"Initial Cmax     : {info['cmax']:.2f} min")
    print(f"Active weights   : (w_cmax={env.w_cmax:.2f}, w_energy={env.w_energy:.2f}, w_tard={env.w_tardiness:.2f})")

    rng = np.random.default_rng(42)
    total_reward = 0.0
    step_num = 0

    print("-" * 80)
    print(f"{'Step':<6} | {'Action':<7} | {'Decoded (J, M, s, pm)':<22} | {'Reward':<10} | {'Cmax (m)':<10} | {'Cost ($)':<10}")
    print("-" * 80)

    terminated = False
    truncated = False

    while not (terminated or truncated):
        mask = env.action_masks()
        valid_actions = np.where(mask)[0]
        chosen_action = int(rng.choice(valid_actions))

        j_idx, m_idx, s_lvl, pm_f = env.decode_action(chosen_action)
        obs, reward, terminated, truncated, info = env.step(chosen_action)
        total_reward += reward
        step_num += 1

        if step_num <= 5 or step_num >= 50 or step_num % 10 == 0:
            print(
                f"{step_num:<6} | "
                f"{chosen_action:<7} | "
                f"{f'J{j_idx+1}, M{m_idx+1}, s={s_lvl}, pm={pm_f}':<22} | "
                f"{reward:<10.3f} | "
                f"{info['cmax']:<10.2f} | "
                f"{info['total_cost']:<10.3f}"
            )

    print("-" * 80)
    print(f"Episode Completed in {step_num} steps!")
    print(f"Cumulative Episode Reward       : {total_reward:.3f}")
    print(f"Final Makespan (C_max)          : {info['cmax']:.2f} min")
    print(f"Total Electricity Consumed      : {info['total_energy_kwh']:.3f} kWh")
    print(f"Total TOU Electricity Cost      : ${info['total_cost']:.3f}")
    print(f"Total Weighted Tardiness (TWT)  : {info['total_weighted_tardiness']:.2f} min")
    print(f"Catastrophic Tool Breakdowns    : {info['total_breakdowns']}")
    print(f"Opportunistic Tool Changes (PM) : {info['total_pm_replacements']}")
    print("=" * 80)

