"""
Physical Machine Model and Digital Twin Representation for CNC Machine Tools.

This module implements CNCPhysicalTwin, simulating the thermo-mechanical and electrical
behavior of individual CNC machines:
    - Speed-scaling cutting dynamics: P_cut(v) = P_idle + P_cut_base * (v ^ alpha_p) * (1 + gamma * wear/w_max)
    - Extended Taylor tool wear: dVB/dt = eta * (v ^ beta)
    - Idle-gap energy-saving transitions (IDLE vs STANDBY vs TURN_OFF) based on Break-Even Time T_BE
    - Preventive maintenance (opportunistic PM) vs. catastrophic emergency breakdown events

Strict Unit Conventions:
    - Time: Minutes (min)
    - Power: Kilowatts (kW)
    - Energy: Kilowatt-hours (kWh)
    - Tool Wear VB: Micrometers (um)
"""

from dataclasses import dataclass, field
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from utils.mqtt_schemas import MachineConfig, MachineState, MachineTelemetryPayload, SystemConfig


@dataclass
class OperationExecutionResult:
    """Detailed telemetry and audit record of a single operation execution."""
    step_index: int
    t_base_min: float
    speed_level: int
    speed_factor: float
    t_proc_min: float
    p_actual_kw: float
    wear_before_um: float
    delta_wear_um: float
    wear_after_um: float
    e_proc_kwh: float
    had_pm: bool
    t_pm_min: float
    e_pm_kwh: float
    had_breakdown: bool
    t_breakdown_min: float
    e_breakdown_kwh: float
    total_step_time_min: float
    total_step_energy_kwh: float
    cumulative_available_time_min: float


class CNCPhysicalTwin:
    """
    Physical Twin of a CNC Machining Center.
    Encapsulates speed-scaling physics, flank tool wear deterioration,
    electrical energy consumption across 4 states, and maintenance policies.
    """

    def __init__(
        self,
        machine_id: str,
        config: Union[MachineConfig, Dict[str, Any]],
        alpha_p: float = 1.5,
        speed_levels: Optional[Dict[int, float]] = None,
    ) -> None:
        """
        Initialize the CNC Physical Twin.

        Args:
            machine_id: Machine identifier (e.g. 'M1').
            config: MachineConfig Pydantic model or parameter dictionary.
            alpha_p: Cutting power exponent with respect to speed (default: 1.5).
            speed_levels: Mapping of speed levels to speed multipliers v_s.
        """
        self.machine_id = machine_id.upper()

        if isinstance(config, MachineConfig):
            cfg_dict = config.model_dump()
        else:
            cfg_dict = config

        # Physical parameters
        self.name: str = cfg_dict.get("name", f"CNC Machine {self.machine_id}")
        self.machine_type: str = cfg_dict.get("machine_type", "CNC Milling Center")
        self.p_idle_kw: float = float(cfg_dict["p_idle_kw"])
        self.p_stdby_kw: float = float(cfg_dict["p_stdby_kw"])
        self.p_cut_base_kw: float = float(cfg_dict["p_cut_base_kw"])
        self.e_on_off_kwh_param: float = float(cfg_dict["e_on_off_kwh"])
        self.t_warmup_min: float = float(cfg_dict["t_warmup_min"])

        # Tool wear parameters
        tw = cfg_dict["tool_wear"]
        self.initial_wear_um: float = float(tw.get("initial_wear_um", 0.0))
        self.base_initial_wear_um: float = self.initial_wear_um
        self.eta: float = float(tw["eta"])
        self.base_eta: float = self.eta
        self.beta: float = float(tw["beta"])
        self.w_max_um: float = float(tw["w_max_um"])
        self.gamma: float = float(tw["gamma"])
        self.t_replace_min: float = float(tw["t_replace_min"])
        self.t_breakdown_min: float = float(tw["t_breakdown_min"])

        # Global scaling parameters
        self.alpha_p: float = alpha_p
        self.speed_levels: Dict[int, float] = speed_levels or {1: 1.0, 2: 1.25, 3: 1.5}

        # Energy Break-Even Time T_BE (min)
        net_power = self.p_idle_kw - self.p_stdby_kw
        if net_power <= 0:
            raise ValueError(f"p_idle_kw ({self.p_idle_kw}) must exceed p_stdby_kw ({self.p_stdby_kw})")
        self.break_even_time_min: float = (self.e_on_off_kwh_param * 60.0) / net_power

        # Dynamic state variables
        self.state: MachineState = MachineState.IDLE
        self.available_time: float = 0.0  # Simulation clock (min)
        self.current_wear_um: float = self.initial_wear_um  # Cumulative tool flank wear VB (um)

        # 4 Energy consumption components (kWh)
        self.e_proc_kwh: float = 0.0
        self.e_idle_kwh: float = 0.0
        self.e_stdby_kwh: float = 0.0
        self.e_on_off_kwh: float = 0.0

        # Maintenance and reliability counters
        self.num_pm_replacements: int = 0
        self.num_breakdowns: int = 0
        self.total_operations_executed: int = 0

        # Execution log history
        self.execution_history: List[OperationExecutionResult] = []

    @property
    def total_energy_kwh(self) -> float:
        """Total electrical energy consumed across all 4 components (kWh)."""
        return self.e_proc_kwh + self.e_idle_kwh + self.e_stdby_kwh + self.e_on_off_kwh

    @property
    def tool_wear_ratio(self) -> float:
        """Fraction of maximum tool wear consumed (0.0 to 1.0)."""
        return min(1.0, self.current_wear_um / self.w_max_um)

    def reset(
        self,
        initial_time: float = 0.0,
        initial_wear_um: Optional[float] = None,
        eta: Optional[float] = None,
    ) -> None:
        """Reset the physical twin to initial state."""
        self.state = MachineState.IDLE
        self.available_time = initial_time
        self.current_wear_um = initial_wear_um if initial_wear_um is not None else self.initial_wear_um
        self.eta = eta if eta is not None else self.base_eta
        self.e_proc_kwh = 0.0
        self.e_idle_kwh = 0.0
        self.e_stdby_kwh = 0.0
        self.e_on_off_kwh = 0.0
        self.num_pm_replacements = 0
        self.num_breakdowns = 0
        self.total_operations_executed = 0
        self.execution_history.clear()

    def evaluate_idle_gap(self, gap_duration_min: float) -> Tuple[str, float]:
        """
        Evaluate optimal energy-saving machine state transition during an idle gap.

        Decision Rules:
            - gap <= 2.0 min: Keep in IDLE state.
                Energy = P_idle * (gap / 60)
            - 2.0 < gap < T_BE: Transition to STANDBY state.
                Energy = P_stdby * (gap / 60)
            - gap >= T_BE: Transition to TURN_OFF (OFF -> Warmup).
                Energy = e_on_off_kwh (energy required for restart cycle)

        Args:
            gap_duration_min: Duration of waiting interval before next operation (min).

        Returns:
            Tuple[str, float]: (Chosen state ["IDLE", "STANDBY", "TURN_OFF"], Energy consumed in kWh)
        """
        if gap_duration_min < 0:
            raise ValueError(f"gap_duration_min cannot be negative, got {gap_duration_min}")

        if gap_duration_min <= 2.0:
            energy_kwh = self.p_idle_kw * (gap_duration_min / 60.0)
            return "IDLE", energy_kwh
        elif gap_duration_min < self.break_even_time_min:
            energy_kwh = self.p_stdby_kw * (gap_duration_min / 60.0)
            return "STANDBY", energy_kwh
        else:
            # When gap >= T_BE, machine is powered off and warmed up back
            energy_kwh = self.e_on_off_kwh_param
            return "TURN_OFF", energy_kwh

    def apply_idle_gap(self, gap_duration_min: float) -> Tuple[str, float]:
        """
        Apply the evaluated idle gap decision: advance available_time, update state,
        and accumulate the corresponding energy component.

        Args:
            gap_duration_min: Duration of idle gap (min).

        Returns:
            Tuple[str, float]: (Action taken, Energy consumed in kWh)
        """
        action, energy_kwh = self.evaluate_idle_gap(gap_duration_min)
        self.available_time += gap_duration_min

        if action == "IDLE":
            self.state = MachineState.IDLE
            self.e_idle_kwh += energy_kwh
        elif action == "STANDBY":
            self.state = MachineState.STANDBY
            self.e_stdby_kwh += energy_kwh
        elif action == "TURN_OFF":
            self.state = MachineState.OFF
            self.e_on_off_kwh += energy_kwh

        return action, energy_kwh

    def execute_operation(
        self,
        t_base_min: float,
        speed_level: int = 1,
        do_opportunistic_pm: bool = False,
        rng: Optional[random.Random] = None,
    ) -> OperationExecutionResult:
        """
        Execute a machining operation on this machine.

        Physical Steps:
            1. Opportunistic PM (if requested):
               Add t_replace_min, account for maintenance energy, reset tool wear to 0.
            2. Speed-scaling cutting time:
               t_proc = t_base_min / v_s
            3. Taylor tool wear increment with stochastic noise:
               delta_w = eta * (v_s ^ beta) * t_proc * noise
            4. Power drift accounting for average tool wear during cut:
               P_actual = P_idle + (P_cut_base * (v_s ^ alpha_p)) * (1 + gamma * (wear_avg / w_max))
            5. Energy calculation:
               e_proc = P_actual * (t_proc / 60)
            6. Tool breakdown check:
               If current_wear + delta_w >= w_max:
                   Trigger Breakdown: add t_breakdown_min penalty, standby energy, reset wear to 0.
               Else:
                   Update cumulative wear.

        Args:
            t_base_min: Base cutting time at nominal speed level 1 (min).
            speed_level: Selected speed level s (1, 2, or 3).
            do_opportunistic_pm: Whether to replace tool proactively prior to cutting.
            rng: Optional PRNG for Gaussian noise in tool wear.

        Returns:
            OperationExecutionResult: Comprehensive execution telemetry.
        """
        if speed_level not in self.speed_levels:
            raise ValueError(f"Invalid speed level {speed_level}. Available: {list(self.speed_levels.keys())}")

        v_s = self.speed_levels[speed_level]
        step_idx = self.total_operations_executed + 1

        # 1. Opportunistic Preventive Maintenance
        had_pm = False
        t_pm = 0.0
        e_pm = 0.0
        if do_opportunistic_pm:
            had_pm = True
            t_pm = self.t_replace_min
            # Machine in standby/auxiliary state during tool replacement
            e_pm = self.p_stdby_kw * (t_pm / 60.0)
            self.available_time += t_pm
            self.e_stdby_kwh += e_pm
            self.current_wear_um = 0.0
            self.num_pm_replacements += 1
            self.state = MachineState.TOOL_REPLACEMENT

        wear_before = self.current_wear_um

        # 2. Actual cutting time with speed-scaling
        t_proc = t_base_min / v_s

        # 3. Flank wear increment via Taylor equation
        if rng is not None:
            noise = max(0.5, rng.gauss(1.0, 0.03))
        else:
            noise = 1.0

        delta_w = self.eta * (v_s ** self.beta) * t_proc * noise

        # 4. Average wear and cutting power with wear drift
        wear_avg = wear_before + (0.5 * delta_w)
        wear_drift_factor = 1.0 + self.gamma * (wear_avg / self.w_max_um)
        p_cutting_scaled = self.p_cut_base_kw * (v_s ** self.alpha_p) * wear_drift_factor
        p_actual_kw = self.p_idle_kw + p_cutting_scaled

        # 5. Machining energy
        e_proc = p_actual_kw * (t_proc / 60.0)
        self.available_time += t_proc
        self.e_proc_kwh += e_proc
        self.state = MachineState.PROCESSING

        # 6. Tool breakdown threshold check
        had_breakdown = False
        t_breakdown = 0.0
        e_breakdown = 0.0

        if (wear_before + delta_w) >= self.w_max_um:
            had_breakdown = True
            t_breakdown = self.t_breakdown_min
            # Emergency downtime consumes standby power
            e_breakdown = self.p_stdby_kw * (t_breakdown / 60.0)
            self.available_time += t_breakdown
            self.e_stdby_kwh += e_breakdown
            self.num_breakdowns += 1
            self.current_wear_um = 0.0  # New tool installed after breakdown
            wear_after = 0.0
            self.state = MachineState.BREAKDOWN
        else:
            self.current_wear_um = wear_before + delta_w
            wear_after = self.current_wear_um
            self.state = MachineState.IDLE

        self.total_operations_executed += 1

        total_step_time = t_pm + t_proc + t_breakdown
        total_step_energy = e_pm + e_proc + e_breakdown

        result = OperationExecutionResult(
            step_index=step_idx,
            t_base_min=t_base_min,
            speed_level=speed_level,
            speed_factor=v_s,
            t_proc_min=t_proc,
            p_actual_kw=p_actual_kw,
            wear_before_um=wear_before,
            delta_wear_um=delta_w,
            wear_after_um=wear_after,
            e_proc_kwh=e_proc,
            had_pm=had_pm,
            t_pm_min=t_pm,
            e_pm_kwh=e_pm,
            had_breakdown=had_breakdown,
            t_breakdown_min=t_breakdown,
            e_breakdown_kwh=e_breakdown,
            total_step_time_min=total_step_time,
            total_step_energy_kwh=total_step_energy,
            cumulative_available_time_min=self.available_time,
        )

        self.execution_history.append(result)
        return result

    def get_telemetry_payload(self, timestamp: Optional[float] = None) -> MachineTelemetryPayload:
        """
        Generate a validated MachineTelemetryPayload snapshot of current physical state.
        Compatible with MQTT Edge Broker schema.
        """
        # Estimate vibration RMS proportional to cutting power and tool wear
        wear_ratio = self.tool_wear_ratio
        base_vib = 0.12 if self.state == MachineState.PROCESSING else 0.02
        vibration_rms = base_vib * (1.0 + 1.5 * wear_ratio)

        current_power = (
            self.execution_history[-1].p_actual_kw
            if self.execution_history and self.state == MachineState.PROCESSING
            else (
                self.p_idle_kw
                if self.state == MachineState.IDLE
                else self.p_stdby_kw
            )
        )

        return MachineTelemetryPayload(
            machine_id=self.machine_id,
            power_kw=round(current_power, 3),
            vibration_rms=round(vibration_rms, 3),
            estimated_wear_um=round(self.current_wear_um, 2),
            state=self.state,
            timestamp=timestamp if timestamp is not None else self.available_time,
        )
