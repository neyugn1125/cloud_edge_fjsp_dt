"""
Multi-Objective Optimization for Green Flexible Job Shop Scheduling Problem (FJSP)
using NSGA-III on the Cloud Layer.

This module implements:
    1. GreenFJSPProblem: ElementwiseProblem formulation incorporating:
       - 165 continuous decision variables in [0, 1]:
           * [0..54]: Operation Sequence (OS) priority keys (topological random keys)
           * [55..109]: Machine Assignment (MA) selection
           * [110..164]: Speed Selection (SS) level in {1, 2, 3}
       - Accurate physical twin integration with CNCPhysicalTwin:
           * Speed-scaling cutting dynamics
           * Taylor tool wear deterioration and power drift
           * Break-Even Time (T_BE) idle gap transitions (IDLE, STANDBY, TURN_OFF)
           * Automatic proactive tool replacement (PM) when wear >= 80 um to prevent breakdowns
           * Time-of-Use (TOU) piecewise tariff integration
           * Shop-floor peak demand surcharge (lambda_peak = 0.30 $/kWh for power > 22.0 kW)
       - 3 minimization objectives:
           * f1: Makespan (C_max, minutes)
           * f2: TOU Electricity Cost ($, energy + peak demand surcharge)
           * f3: Total Weighted Tardiness (TWT, minutes)
    2. Population Seeding & NSGA-III Algorithm Execution:
       - Initial population seeded with Eco (speed 1), Moderate (speed 2), and Fast (speed 3)
         profiles alongside exploratory random individuals.
       - Das-Dennis 3D reference directions (n_partitions=12, 91 directions).
       - Simulated Binary Crossover (SBX) and Polynomial Mutation (PM).
       - 100 generations with population size 100.
    3. Multi-Criteria Decision Making (TOPSIS / Ideal Point Compromise):
       - select_compromise_solution(): Min-Max normalization with weighted Euclidean distance.
    4. Cloud Baseline Schedule Exporter & Pareto Visualization.

Strict Unit Conventions:
    - Time: Minutes (min)
    - Power: Kilowatts (kW)
    - Energy: Kilowatt-hours (kWh)
    - Tool Wear VB: Micrometers (um)
    - Cost: US Dollars ($)
"""

import csv
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from sim_env.physical_machine import CNCPhysicalTwin
from utils.mqtt_schemas import SystemConfig, load_system_config


def get_tou_rate(time_min: float) -> float:
    """
    Get electricity tariff ($/kWh) for any simulation time in minutes
    according to the repeating 100-min cyclic TOU schedule:
      - [0.0, 25.0): Off-Peak ($0.06/kWh)
      - [25.0, 60.0): On-Peak ($0.20/kWh)
      - [60.0, 100.0): Mid-Peak ($0.12/kWh)
    """
    t_cycle = time_min % 100.0
    if t_cycle < 25.0:
        return 0.06
    elif t_cycle < 60.0:
        return 0.20
    else:
        return 0.12


class GreenFJSPProblem(ElementwiseProblem):
    """
    Multi-Objective Green Flexible Job Shop Scheduling Problem for pymoo.
    Minimizes:
        f1: Makespan C_max (min)
        f2: TOU Energy Cost ($)
        f3: Total Weighted Tardiness TWT (min)
    """

    def __init__(
        self,
        benchmark_path: Union[str, Path] = "data/benchmarks/Mk01_green.json",
        config_path: Union[str, Path] = "config/system_params.yaml",
        lambda_peak: float = 0.30,
    ) -> None:
        """
        Initialize the Green FJSP optimization problem.

        Args:
            benchmark_path: Path to Mk01_green.json benchmark dataset.
            config_path: Path to system_params.yaml configuration.
            lambda_peak: Peak demand surcharge per kWh of excess power ($0.30/kWh).
        """
        self.benchmark_path = Path(benchmark_path)
        self.config_path = Path(config_path)
        self.lambda_peak = float(lambda_peak)

        # Load problem data
        with open(self.benchmark_path, "r", encoding="utf-8") as f:
            self.benchmark_data: Dict[str, Any] = json.load(f)

        self.sys_cfg: SystemConfig = load_system_config(self.config_path)
        if "global_parameters" in self.benchmark_data and "p_limit_kw" in self.benchmark_data["global_parameters"]:
            self.p_limit_kw: float = float(self.benchmark_data["global_parameters"]["p_limit_kw"])
        else:
            self.p_limit_kw: float = self.sys_cfg.p_limit_kw

        self.jobs: List[Dict[str, Any]] = self.benchmark_data["jobs"]
        self.num_jobs: int = len(self.jobs)
        self.ops_per_job: List[int] = [len(j["operations"]) for j in self.jobs]
        self.total_ops: int = sum(self.ops_per_job)

        self.job_weights: List[float] = [float(j["weight"]) for j in self.jobs]
        self.job_due_dates: List[float] = [float(j["due_date_min"]) for j in self.jobs]

        # Flattened operation indices: (job_idx, op_idx) -> global_op_idx
        self.job_op_to_global: Dict[Tuple[int, int], int] = {}
        self.global_to_job_op: List[Tuple[int, int, Dict[str, Any]]] = []
        idx = 0
        for j_idx, job in enumerate(self.jobs):
            for o_idx, op in enumerate(job["operations"]):
                self.job_op_to_global[(j_idx, o_idx)] = idx
                self.global_to_job_op.append((j_idx, o_idx, op))
                idx += 1

        # Pre-instantiate physical machine digital twins
        if "machines" in self.benchmark_data and self.benchmark_data["machines"]:
            self.machine_ids: List[str] = sorted(list(self.benchmark_data["machines"].keys()))
        else:
            self.machine_ids: List[str] = sorted(list(self.sys_cfg.machines.keys()))
        self.num_machines: int = len(self.machine_ids)
        self.machines: Dict[str, CNCPhysicalTwin] = {
            m_id: CNCPhysicalTwin(
                machine_id=m_id,
                config=self.sys_cfg.get_machine(m_id),
                alpha_p=self.sys_cfg.speed_scaling.alpha_p,
                speed_levels={
                    k: v.speed_factor for k, v in self.sys_cfg.speed_scaling.levels.items()
                },
            )
            for m_id in self.machine_ids
        }

        # Chromosome structure: 3 * num_ops = 165 variables in [0, 1]
        #   [0..54]: Operation sequence priority keys (OS)
        #   [55..109]: Machine assignment selection (MA)
        #   [110..164]: Cutting speed level selection (SS)
        n_var = 3 * self.total_ops
        super().__init__(
            n_var=n_var,
            n_obj=3,
            n_ieq_constr=0,
            xl=0.0,
            xu=1.0,
        )

    def _integrate_tou_cost(self, power_kw: float, t_start: float, t_end: float) -> float:
        """Piecewise analytical integration of TOU electricity cost ($)."""
        if t_end <= t_start or power_kw <= 0.0:
            return 0.0
        cycle = self.sys_cfg.tou_pricing.cycle_min
        start_day = int(t_start // cycle)
        end_day = int(t_end // cycle)
        cost = 0.0
        for day in range(start_day, end_day + 1):
            day_offset = day * cycle
            for tier in self.sys_cfg.tou_pricing.tiers:
                tier_start = day_offset + tier.start_min
                tier_end = day_offset + tier.end_min
                overlap_start = max(t_start, tier_start)
                overlap_end = min(t_end, tier_end)
                if overlap_end > overlap_start:
                    duration_min = overlap_end - overlap_start
                    energy_kwh = power_kw * (duration_min / 60.0)
                    cost += energy_kwh * tier.rate_per_kwh
        return cost

    def _compute_peak_power_excess(
        self, intervals: List[Tuple[float, float, float]]
    ) -> float:
        """
        Compute exact total excess energy (kWh) beyond p_limit_kw across all cutting intervals.
        """
        if not intervals:
            return 0.0
        time_points = sorted(list(set([t for (s, e, _) in intervals for t in (s, e)])))
        excess_kwh = 0.0
        for i in range(len(time_points) - 1):
            t1, t2 = time_points[i], time_points[i + 1]
            dt = t2 - t1
            if dt <= 0.0:
                continue
            mid_t = 0.5 * (t1 + t2)
            concurrent_p = sum(p for (s, e, p) in intervals if s <= mid_t < e)
            if concurrent_p > self.p_limit_kw:
                excess_kwh += (concurrent_p - self.p_limit_kw) * (dt / 60.0)
        return excess_kwh

    def _evaluate(self, x: np.ndarray, out: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        """
        Evaluate single individual chromosome.
        Calculates:
            f1: Makespan (C_max)
            f2: TOU Energy Cost ($)
            f3: Total Weighted Tardiness (TWT)
        """
        for m in self.machines.values():
            m.reset()

        job_next_op = [0] * self.num_jobs
        job_ready_time = [0.0] * self.num_jobs
        total_cost = 0.0
        cutting_intervals: List[Tuple[float, float, float]] = []

        # Step through all 55 operations topologically
        for _ in range(self.total_ops):
            best_j = -1
            best_key = float("inf")
            for j in range(self.num_jobs):
                if job_next_op[j] < self.ops_per_job[j]:
                    g_idx = self.job_op_to_global[(j, job_next_op[j])]
                    k_val = x[g_idx]
                    if k_val < best_key:
                        best_key = k_val
                        best_j = j

            j = best_j
            o_idx = job_next_op[j]
            g_idx = self.job_op_to_global[(j, o_idx)]
            op_data = self.jobs[j]["operations"][o_idx]

            # 1. Machine Assignment (MA)
            el_machs = op_data["eligible_machines"]
            m_count = len(el_machs)
            m_sel = min(m_count - 1, int(x[self.total_ops + g_idx] * m_count))
            chosen_m_entry = el_machs[m_sel]
            m_id = chosen_m_entry["machine_id"]
            t_base = chosen_m_entry["t_base_min"]
            mach = self.machines[m_id]

            # 2. Speed Selection (SS)
            speed_level = 1 + min(2, int(x[2 * self.total_ops + g_idx] * 3))
            v_s = mach.speed_levels[speed_level]
            t_proc = t_base / v_s

            # 3. Proactive Maintenance Policy
            delta_w = mach.eta * (v_s ** mach.beta) * t_proc
            do_pm = (mach.current_wear_um >= 0.80 * mach.w_max_um) or (
                (mach.current_wear_um + delta_w) >= mach.w_max_um
            )

            # 4. Earliest Start Time & Idle Gap Management
            j_ready = job_ready_time[j]
            m_ready = mach.available_time
            start_t = max(j_ready, m_ready)

            if start_t > m_ready:
                gap = start_t - m_ready
                action_taken, _ = mach.apply_idle_gap(gap)
                if action_taken == "IDLE":
                    total_cost += self._integrate_tou_cost(mach.p_idle_kw, m_ready, start_t)
                elif action_taken == "STANDBY":
                    total_cost += self._integrate_tou_cost(mach.p_stdby_kw, m_ready, start_t)
                else:  # TURN_OFF
                    total_cost += mach.e_on_off_kwh_param * self.sys_cfg.get_tou_rate(start_t)

            # 5. Machining Execution
            res = mach.execute_operation(t_base, speed_level, do_pm, rng=None)
            t_cut_start = start_t + res.t_pm_min
            t_cut_end = t_cut_start + res.t_proc_min

            # Base TOU electricity cost
            total_cost += self._integrate_tou_cost(res.p_actual_kw, t_cut_start, t_cut_end)
            if res.had_pm:
                total_cost += self._integrate_tou_cost(mach.p_stdby_kw, start_t, t_cut_start)
            if res.had_breakdown:
                total_cost += self._integrate_tou_cost(
                    mach.p_stdby_kw, t_cut_end, mach.available_time
                )

            cutting_intervals.append((t_cut_start, t_cut_end, res.p_actual_kw))
            job_ready_time[j] = mach.available_time
            job_next_op[j] += 1

        # 6. Peak Power Demand Surcharge ($0.30 / excess kWh)
        excess_kwh = self._compute_peak_power_excess(cutting_intervals)
        peak_surcharge = self.lambda_peak * excess_kwh
        total_cost += peak_surcharge

        makespan = max(job_ready_time)
        twt = sum(
            self.job_weights[j] * max(0.0, job_ready_time[j] - self.job_due_dates[j])
            for j in range(self.num_jobs)
        )

        out["F"] = [makespan, total_cost, twt]

    def decode_schedule(
        self,
        x: np.ndarray,
        weights: Tuple[float, float, float] = (0.4, 0.4, 0.2),
    ) -> Dict[str, Any]:
        """
        Decode chromosome into complete detailed scheduling audit and machine telemetry.

        Args:
            x: Chromosome vector (165 variables).
            weights: Objective weight tuple (w_cmax, w_energy, w_tardiness).

        Returns:
            Dict containing 'summary', 'operations_schedule', and 'machine_schedules'.
        """
        for m in self.machines.values():
            m.reset()

        job_next_op = [0] * self.num_jobs
        job_ready_time = [0.0] * self.num_jobs
        total_cost = 0.0
        cutting_intervals: List[Tuple[float, float, float]] = []
        ops_schedule: List[Dict[str, Any]] = []

        for _ in range(self.total_ops):
            best_j = -1
            best_key = float("inf")
            for j in range(self.num_jobs):
                if job_next_op[j] < self.ops_per_job[j]:
                    g_idx = self.job_op_to_global[(j, job_next_op[j])]
                    k_val = x[g_idx]
                    if k_val < best_key:
                        best_key = k_val
                        best_j = j

            j = best_j
            o_idx = job_next_op[j]
            g_idx = self.job_op_to_global[(j, o_idx)]
            op_data = self.jobs[j]["operations"][o_idx]

            el_machs = op_data["eligible_machines"]
            m_count = len(el_machs)
            m_sel = min(m_count - 1, int(x[self.total_ops + g_idx] * m_count))
            chosen_m_entry = el_machs[m_sel]
            m_id = chosen_m_entry["machine_id"]
            t_base = chosen_m_entry["t_base_min"]
            mach = self.machines[m_id]

            speed_level = 1 + min(2, int(x[2 * self.total_ops + g_idx] * 3))
            v_s = mach.speed_levels[speed_level]
            t_proc = t_base / v_s

            delta_w = mach.eta * (v_s ** mach.beta) * t_proc
            do_pm = (mach.current_wear_um >= 0.80 * mach.w_max_um) or (
                (mach.current_wear_um + delta_w) >= mach.w_max_um
            )

            j_ready = job_ready_time[j]
            m_ready = mach.available_time
            start_t = max(j_ready, m_ready)

            gap_cost = 0.0
            if start_t > m_ready:
                gap = start_t - m_ready
                action_taken, _ = mach.apply_idle_gap(gap)
                if action_taken == "IDLE":
                    gap_cost = self._integrate_tou_cost(mach.p_idle_kw, m_ready, start_t)
                elif action_taken == "STANDBY":
                    gap_cost = self._integrate_tou_cost(mach.p_stdby_kw, m_ready, start_t)
                else:
                    gap_cost = mach.e_on_off_kwh_param * self.sys_cfg.get_tou_rate(start_t)
                total_cost += gap_cost

            res = mach.execute_operation(t_base, speed_level, do_pm, rng=None)
            t_cut_start = start_t + res.t_pm_min
            t_cut_end = t_cut_start + res.t_proc_min

            c_proc = self._integrate_tou_cost(res.p_actual_kw, t_cut_start, t_cut_end)
            c_pm = (
                self._integrate_tou_cost(mach.p_stdby_kw, start_t, t_cut_start)
                if res.had_pm
                else 0.0
            )
            c_bd = (
                self._integrate_tou_cost(mach.p_stdby_kw, t_cut_end, mach.available_time)
                if res.had_breakdown
                else 0.0
            )
            total_cost += (c_proc + c_pm + c_bd)

            cutting_intervals.append((t_cut_start, t_cut_end, res.p_actual_kw))

            ops_schedule.append({
                "global_op_idx": g_idx,
                "job_id": self.jobs[j]["job_id"],
                "job_index": j,
                "op_index": o_idx,
                "op_id": op_data["op_id"],
                "machine_id": m_id,
                "speed_level": speed_level,
                "speed_factor": v_s,
                "t_base_min": round(t_base, 2),
                "t_proc_min": round(res.t_proc_min, 3),
                "start_time_min": round(start_t, 3),
                "cutting_start_min": round(t_cut_start, 3),
                "cutting_end_min": round(t_cut_end, 3),
                "end_time_min": round(mach.available_time, 3),
                "wear_before_um": round(res.wear_before_um, 2),
                "delta_wear_um": round(res.delta_wear_um, 2),
                "wear_after_um": round(res.wear_after_um, 2),
                "had_pm": bool(res.had_pm),
                "had_breakdown": bool(res.had_breakdown),
                "p_actual_kw": round(res.p_actual_kw, 3),
                "energy_kwh": round(res.total_step_energy_kwh, 4),
                "cost_usd": round(gap_cost + c_proc + c_pm + c_bd, 4),
            })

            job_ready_time[j] = mach.available_time
            job_next_op[j] += 1

        # Peak power surcharge
        excess_kwh = self._compute_peak_power_excess(cutting_intervals)
        peak_surcharge = self.lambda_peak * excess_kwh
        total_cost += peak_surcharge

        makespan = max(job_ready_time)
        twt = sum(
            self.job_weights[j] * max(0.0, job_ready_time[j] - self.job_due_dates[j])
            for j in range(self.num_jobs)
        )

        tot_e_proc = sum(m.e_proc_kwh for m in self.machines.values())
        tot_e_idle = sum(m.e_idle_kwh for m in self.machines.values())
        tot_e_stdby = sum(m.e_stdby_kwh for m in self.machines.values())
        tot_e_on_off = sum(m.e_on_off_kwh for m in self.machines.values())
        tot_e = tot_e_proc + tot_e_idle + tot_e_stdby + tot_e_on_off
        tot_pm = sum(m.num_pm_replacements for m in self.machines.values())
        tot_bd = sum(m.num_breakdowns for m in self.machines.values())

        machine_schedules: Dict[str, Any] = {}
        for m_id, m in self.machines.items():
            m_ops = [op for op in ops_schedule if op["machine_id"] == m_id]
            machine_schedules[m_id] = {
                "machine_id": m_id,
                "total_operations": m.total_operations_executed,
                "operations_assigned": [op["op_id"] for op in m_ops],
                "total_energy_kwh": round(m.total_energy_kwh, 4),
                "e_proc_kwh": round(m.e_proc_kwh, 4),
                "e_idle_kwh": round(m.e_idle_kwh, 4),
                "e_stdby_kwh": round(m.e_stdby_kwh, 4),
                "e_on_off_kwh": round(m.e_on_off_kwh, 4),
                "num_pm_replacements": m.num_pm_replacements,
                "num_breakdowns": m.num_breakdowns,
                "final_wear_um": round(m.current_wear_um, 2),
                "available_time_min": round(m.available_time, 3),
            }

        summary = {
            "benchmark": "Mk01_green",
            "algorithm": "NSGA-III",
            "weights": list(weights),
            "makespan_min": round(makespan, 3),
            "total_energy_kwh": round(tot_e, 4),
            "e_proc_kwh": round(tot_e_proc, 4),
            "e_idle_stdby_kwh": round(tot_e_idle + tot_e_stdby, 4),
            "e_on_off_kwh": round(tot_e_on_off, 4),
            "tou_energy_cost_usd": round(total_cost, 4),
            "peak_surcharge_usd": round(peak_surcharge, 4),
            "total_weighted_tardiness_min": round(twt, 3),
            "num_pm_replacements": tot_pm,
            "num_breakdowns": tot_bd,
        }

        return {
            "summary": summary,
            "operations_schedule": ops_schedule,
            "machine_schedules": machine_schedules,
        }


def generate_seeded_population(
    problem: GreenFJSPProblem,
    pop_size: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate an initial population seeded with heuristic domain knowledge:
        - Eco-seeds (speed s=1, energy-efficient machine allocation)
        - Balanced-seeds (speed s=2, energy-efficient machine allocation)
        - Fast-seeds (speed s=3, fastest machine allocation)
        - Exploratory random individuals in [0, 1]^165
    """
    rng = np.random.RandomState(seed)
    X_init = rng.rand(pop_size, problem.n_var)

    # Adaptive seed counts based on population size
    n_eco = max(1, int(0.15 * pop_size))
    n_mod = max(1, int(0.10 * pop_size))
    n_fast = max(1, int(0.05 * pop_size))

    # 1. Seed individuals with Eco configuration (s=1, lowest power machine)
    for k in range(min(pop_size, n_eco)):
        X_init[k, 2 * problem.total_ops :] = rng.uniform(0.02, 0.30, problem.total_ops)
        for j_idx, job in enumerate(problem.jobs):
            for o_idx, op in enumerate(job["operations"]):
                g_idx = problem.job_op_to_global[(j_idx, o_idx)]
                el_machs = op["eligible_machines"]
                best_m_sel = min(
                    range(len(el_machs)),
                    key=lambda m_i: problem.sys_cfg.get_machine(
                        el_machs[m_i]["machine_id"]
                    ).p_cut_base_kw
                    * el_machs[m_i]["t_base_min"],
                )
                m_count = len(el_machs)
                X_init[k, problem.total_ops + g_idx] = (
                    best_m_sel + rng.uniform(0.15, 0.85)
                ) / m_count

    # 2. Seed individuals with Moderate configuration (s=2, energy-efficient machine)
    idx_mod_start = n_eco
    idx_mod_end = min(pop_size, n_eco + n_mod)
    for k in range(idx_mod_start, idx_mod_end):
        X_init[k, 2 * problem.total_ops :] = rng.uniform(0.36, 0.64, problem.total_ops)
        for j_idx, job in enumerate(problem.jobs):
            for o_idx, op in enumerate(job["operations"]):
                g_idx = problem.job_op_to_global[(j_idx, o_idx)]
                el_machs = op["eligible_machines"]
                best_m_sel = min(
                    range(len(el_machs)),
                    key=lambda m_i: problem.sys_cfg.get_machine(
                        el_machs[m_i]["machine_id"]
                    ).p_cut_base_kw
                    * el_machs[m_i]["t_base_min"],
                )
                m_count = len(el_machs)
                X_init[k, problem.total_ops + g_idx] = (
                    best_m_sel + rng.uniform(0.15, 0.85)
                ) / m_count

    # 3. Seed individuals with Fast configuration (s=3, shortest base time machine)
    idx_fast_start = idx_mod_end
    idx_fast_end = min(pop_size, idx_mod_end + n_fast)
    for k in range(idx_fast_start, idx_fast_end):
        X_init[k, 2 * problem.total_ops :] = rng.uniform(0.70, 0.98, problem.total_ops)
        for j_idx, job in enumerate(problem.jobs):
            for o_idx, op in enumerate(job["operations"]):
                g_idx = problem.job_op_to_global[(j_idx, o_idx)]
                el_machs = op["eligible_machines"]
                fast_m_sel = min(
                    range(len(el_machs)),
                    key=lambda m_i: el_machs[m_i]["t_base_min"],
                )
                m_count = len(el_machs)
                X_init[k, problem.total_ops + g_idx] = (
                    fast_m_sel + rng.uniform(0.15, 0.85)
                ) / m_count

    return X_init


def run_nsga3_optimization(
    benchmark_path: Union[str, Path] = "data/benchmarks/Mk01_green.json",
    config_path: Union[str, Path] = "config/system_params.yaml",
    pop_size: int = 100,
    n_gen: int = 100,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[Any, GreenFJSPProblem]:
    """
    Execute NSGA-III multi-objective optimization with seeded initial population.

    Args:
        benchmark_path: Path to benchmark JSON file.
        config_path: Path to system configuration YAML.
        pop_size: Population size (default: 100).
        n_gen: Number of generations (default: 100).
        seed: Random seed for repeatability.
        verbose: Whether to print optimization progress.

    Returns:
        Tuple of (pymoo Result object, GreenFJSPProblem instance).
    """
    problem = GreenFJSPProblem(benchmark_path=benchmark_path, config_path=config_path)

    # Das-Dennis reference directions (3 objectives, 12 partitions -> 91 directions)
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

    # Generate heuristic-seeded initial population
    X_init = generate_seeded_population(problem=problem, pop_size=pop_size, seed=seed)

    algorithm = NSGA3(
        ref_dirs=ref_dirs,
        pop_size=pop_size,
        sampling=X_init,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=1.0 / problem.n_var, eta=20),
    )

    t0 = time.time()
    res = minimize(
        problem,
        algorithm,
        ("n_gen", n_gen),
        seed=seed,
        verbose=verbose,
    )
    elapsed = time.time() - t0

    if verbose:
        print(f"\n[NSGA-III] Optimization finished in {elapsed:.2f} s across {n_gen} generations.")
        print(f"[NSGA-III] Discovered {len(res.F)} non-dominated Pareto solutions.")

    return res, problem


def select_compromise_solution(
    pareto_F: np.ndarray,
    pareto_X: np.ndarray,
    weights: Tuple[float, float, float] = (0.4, 0.4, 0.2),
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Select the Best Compromise Solution on the Pareto front using Min-Max
    normalized weighted Euclidean distance to the Ideal Point (0, 0, 0).

    Args:
        pareto_F: Objective matrix of shape (P, 3).
        pareto_X: Decision variable matrix of shape (P, 165).
        weights: Priority weights (w_cmax, w_energy, w_tardiness).

    Returns:
        Tuple of (best_index, best_F, best_X, distances_array).
    """
    w = np.array(weights, dtype=np.float64)
    w = w / np.sum(w)

    f_min = np.min(pareto_F, axis=0)
    f_max = np.max(pareto_F, axis=0)
    range_f = f_max - f_min
    range_f[range_f == 0] = 1.0

    # Min-max normalization: best value maps to 0.0, worst to 1.0
    f_norm = (pareto_F - f_min) / range_f

    # Ideal point is (0.0, 0.0, 0.0) for minimization
    distances = np.sqrt(np.sum(w * (f_norm ** 2), axis=1))
    best_idx = int(np.argmin(distances))

    return best_idx, pareto_F[best_idx], pareto_X[best_idx], distances


def plot_pareto_front(
    pareto_F: np.ndarray,
    compromise_F: np.ndarray,
    baselines_csv_path: Union[str, Path] = "data/benchmarks/phase1_baseline_results.csv",
    output_png_path: Union[str, Path] = "tests/phase2_cloud_pareto.png",
) -> None:
    """
    Create a 3-panel visualization showing:
        1. 3D Pareto Front (Makespan vs Cost vs Tardiness)
        2. 2D Projection: Makespan vs TOU Energy Cost
        3. 2D Projection: Makespan vs Total Weighted Tardiness
    Overlaying Phase 1 Baseline Rules 1, 2, 3 and Best Compromise Solution.
    """
    output_png = Path(output_png_path)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    # Read Phase 1 Baseline Results
    baseline_records: Dict[str, Dict[str, Any]] = {}
    with open(baselines_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = str(row["Rule Name"])
            baseline_records[name] = {
                "cmax": float(row["Makespan (min)"]),
                "energy": float(row["Total Energy (kWh)"]),
                "cost": float(row["TOU Energy Cost ($)"]),
                "twt": float(row["Total Weighted Tardiness (min)"]),
                "pm": row["PM Replacements"],
                "bd": row["Tool Breakdowns"],
            }

    fig = plt.figure(figsize=(18, 5.5), dpi=300)

    # --------------------------------------------------------------------------
    # Panel 1: 3D Pareto Front
    # --------------------------------------------------------------------------
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.scatter(
        pareto_F[:, 0],
        pareto_F[:, 1],
        pareto_F[:, 2],
        c="#1f77b4",
        s=45,
        alpha=0.75,
        edgecolors="navy",
        depthshade=True,
        label="NSGA-III Pareto Front",
    )

    # Best compromise star
    ax1.scatter(
        [compromise_F[0]],
        [compromise_F[1]],
        [compromise_F[2]],
        c="red",
        marker="*",
        s=320,
        edgecolors="darkred",
        linewidths=1.5,
        label="Best Compromise (TOPSIS)",
        zorder=10,
    )

    # Heuristic baselines
    colors = {"Rule 1": "#ff7f0e", "Rule 2": "#2ca02c", "Rule 3": "#9467bd"}
    markers = {"Rule 1": "s", "Rule 2": "^", "Rule 3": "D"}

    for rule_prefix, color in colors.items():
        match_k = next((k for k in baseline_records.keys() if k.startswith(rule_prefix)), None)
        if match_k:
            b = baseline_records[match_k]
            ax1.scatter(
                [b["cmax"]],
                [b["cost"]],
                [b["twt"]],
                c=color,
                marker=markers[rule_prefix],
                s=110,
                edgecolors="black",
                linewidths=1.2,
                label=f"{rule_prefix} Baseline",
                zorder=9,
            )

    ax1.set_xlabel("Makespan $C_{max}$ (min)", fontsize=10, labelpad=8)
    ax1.set_ylabel("TOU Cost ($)", fontsize=10, labelpad=8)
    ax1.set_zlabel("TWT (min)", fontsize=10, labelpad=8)
    ax1.set_title("3D Pareto Front (NSGA-III vs. Baselines)", fontsize=12, fontweight="bold")
    ax1.view_init(elev=22, azim=48)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # --------------------------------------------------------------------------
    # Panel 2: 2D Projection (Makespan vs TOU Cost)
    # --------------------------------------------------------------------------
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.scatter(
        pareto_F[:, 0],
        pareto_F[:, 1],
        c="#1f77b4",
        s=40,
        alpha=0.75,
        edgecolors="navy",
        label="NSGA-III Pareto Front",
    )
    ax2.scatter(
        [compromise_F[0]],
        [compromise_F[1]],
        c="red",
        marker="*",
        s=260,
        edgecolors="darkred",
        linewidths=1.2,
        label="Best Compromise",
        zorder=10,
    )

    for rule_prefix, color in colors.items():
        match_k = next((k for k in baseline_records.keys() if k.startswith(rule_prefix)), None)
        if match_k:
            b = baseline_records[match_k]
            ax2.scatter(
                [b["cmax"]],
                [b["cost"]],
                c=color,
                marker=markers[rule_prefix],
                s=110,
                edgecolors="black",
                linewidths=1.2,
                label=f"{rule_prefix}",
                zorder=9,
            )
            ax2.annotate(
                f"{rule_prefix}\n(${b['cost']:.2f})",
                (b["cmax"], b["cost"]),
                textcoords="offset points",
                xytext=(8, 5),
                fontsize=8,
                fontweight="bold",
                color=color,
            )

    ax2.set_xlabel("Makespan $C_{max}$ (min)", fontsize=10)
    ax2.set_ylabel("TOU Energy Cost ($)", fontsize=10)
    ax2.set_title("Projection: Makespan vs. TOU Cost", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8)

    # --------------------------------------------------------------------------
    # Panel 3: 2D Projection (Makespan vs Total Weighted Tardiness)
    # --------------------------------------------------------------------------
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.scatter(
        pareto_F[:, 0],
        pareto_F[:, 2],
        c="#1f77b4",
        s=40,
        alpha=0.75,
        edgecolors="navy",
        label="NSGA-III Pareto Front",
    )
    ax3.scatter(
        [compromise_F[0]],
        [compromise_F[2]],
        c="red",
        marker="*",
        s=260,
        edgecolors="darkred",
        linewidths=1.2,
        label="Best Compromise",
        zorder=10,
    )

    for rule_prefix, color in colors.items():
        match_k = next((k for k in baseline_records.keys() if k.startswith(rule_prefix)), None)
        if match_k:
            b = baseline_records[match_k]
            ax3.scatter(
                [b["cmax"]],
                [b["twt"]],
                c=color,
                marker=markers[rule_prefix],
                s=110,
                edgecolors="black",
                linewidths=1.2,
                label=f"{rule_prefix}",
                zorder=9,
            )
            ax3.annotate(
                f"{rule_prefix}\n({b['twt']:.0f}m)",
                (b["cmax"], b["twt"]),
                textcoords="offset points",
                xytext=(8, -12),
                fontsize=8,
                fontweight="bold",
                color=color,
            )

    ax3.set_xlabel("Makespan $C_{max}$ (min)", fontsize=10)
    ax3.set_ylabel("Total Weighted Tardiness (min)", fontsize=10)
    ax3.set_title("Projection: Makespan vs. Tardiness", fontsize=12, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Pareto Front plot saved to {output_png}")


def main() -> None:
    """Main execution entrypoint for Cloud Layer NSGA-III Optimization."""
    benchmark_path = "data/benchmarks/Mk01_green.json"
    config_path = "config/system_params.yaml"
    baselines_csv_path = "data/benchmarks/phase1_baseline_results.csv"
    output_json_path = "data/benchmarks/cloud_baseline_schedule.json"
    output_png_path = "tests/phase2_cloud_pareto.png"

    print("=" * 80)
    print("PHASE 2 - STEP 2.1: CLOUD LAYER NSGA-III MULTI-OBJECTIVE OPTIMIZATION")
    print("=" * 80)

    # 1. Run NSGA-III Optimization with Seeded Population
    res, problem = run_nsga3_optimization(
        benchmark_path=benchmark_path,
        config_path=config_path,
        pop_size=100,
        n_gen=100,
        seed=42,
        verbose=False,
    )

    pareto_F = res.F
    pareto_X = res.X

    # 2. Select Best Compromise Solution via TOPSIS / Ideal Point distance
    weights = (0.4, 0.4, 0.2)
    comp_idx, comp_F, comp_X, dists = select_compromise_solution(
        pareto_F, pareto_X, weights=weights
    )

    # Find extreme solutions for comprehensive comparison
    best_cmax_idx = int(np.argmin(pareto_F[:, 0]))
    best_cost_idx = int(np.argmin(pareto_F[:, 1]))
    best_twt_idx = int(np.argmin(pareto_F[:, 2]))

    # 3. Decode detailed schedule for Best Compromise Solution
    full_schedule = problem.decode_schedule(comp_X, weights=weights)

    # Save to cloud_baseline_schedule.json
    out_json = Path(output_json_path)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_schedule, f, indent=2)
    print(f"\n[Baseline Schedule] Full schedule successfully saved to {out_json}")

    # 4. Generate Visualization
    plot_pareto_front(
        pareto_F=pareto_F,
        compromise_F=comp_F,
        baselines_csv_path=baselines_csv_path,
        output_png_path=output_png_path,
    )

    # 5. Print Comparison Table
    cmax_sched = problem.decode_schedule(pareto_X[best_cmax_idx])["summary"]
    cost_sched = problem.decode_schedule(pareto_X[best_cost_idx])["summary"]
    twt_sched = problem.decode_schedule(pareto_X[best_twt_idx])["summary"]
    comp_sched = full_schedule["summary"]

    baseline_rows: List[Dict[str, Any]] = []
    with open(baselines_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            baseline_rows.append(row)

    print("\n" + "=" * 115)
    print("PERFORMANCE BENCHMARK: CLOUD NSGA-III VS. PHASE 1 HEURISTIC BASELINES (Mk01)")
    print("=" * 115)
    header = (
        f"{'Strategy / Solution':<36} | {'C_max (min)':<11} | {'Energy (kWh)':<12} | "
        f"{'Cost ($)':<9} | {'TWT (min)':<10} | {'PM/BD':<7} | {'Dominance Status'}"
    )
    print(header)
    print("-" * 115)

    # Print NSGA-III solutions
    nsga3_rows = [
        (
            "NSGA-III (Best Compromise - TOPSIS)",
            comp_sched["makespan_min"],
            comp_sched["total_energy_kwh"],
            comp_sched["tou_energy_cost_usd"],
            comp_sched["total_weighted_tardiness_min"],
            f"{comp_sched['num_pm_replacements']}/{comp_sched['num_breakdowns']}",
            "★ Optimal Balanced Baseline",
        ),
        (
            "NSGA-III (Min Makespan)",
            cmax_sched["makespan_min"],
            cmax_sched["total_energy_kwh"],
            cmax_sched["tou_energy_cost_usd"],
            cmax_sched["total_weighted_tardiness_min"],
            f"{cmax_sched['num_pm_replacements']}/{cmax_sched['num_breakdowns']}",
            "Pareto Non-dominated",
        ),
        (
            "NSGA-III (Min TOU Cost)",
            cost_sched["makespan_min"],
            cost_sched["total_energy_kwh"],
            cost_sched["tou_energy_cost_usd"],
            cost_sched["total_weighted_tardiness_min"],
            f"{cost_sched['num_pm_replacements']}/{cost_sched['num_breakdowns']}",
            "Pareto Non-dominated",
        ),
        (
            "NSGA-III (Min Tardiness)",
            twt_sched["makespan_min"],
            twt_sched["total_energy_kwh"],
            twt_sched["tou_energy_cost_usd"],
            twt_sched["total_weighted_tardiness_min"],
            f"{twt_sched['num_pm_replacements']}/{twt_sched['num_breakdowns']}",
            "Pareto Non-dominated",
        ),
    ]

    for row in nsga3_rows:
        print(
            f"{row[0]:<36} | {row[1]:<11.2f} | {row[2]:<12.3f} | {row[3]:<9.3f} | "
            f"{row[4]:<10.2f} | {row[5]:<7} | {row[6]}"
        )

    print("-" * 115)
    for b_row in baseline_rows:
        b_name = str(b_row["Rule Name"])
        b_cmax = float(b_row["Makespan (min)"])
        b_e = float(b_row["Total Energy (kWh)"])
        b_cost = float(b_row["TOU Energy Cost ($)"])
        b_twt = float(b_row["Total Weighted Tardiness (min)"])
        pm_bd = f"{b_row['PM Replacements']}/{b_row['Tool Breakdowns']}"
        status = "Dominated by NSGA-III"
        print(
            f"{b_name:<36} | {b_cmax:<11.2f} | {b_e:<12.3f} | {b_cost:<9.3f} | "
            f"{b_twt:<10.2f} | {pm_bd:<7} | {status}"
        )
    print("=" * 115)


if __name__ == "__main__":
    main()
