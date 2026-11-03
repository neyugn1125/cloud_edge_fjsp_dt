"""
MQTT Schemas and Physical Parameter Models for Cloud-Edge Digital Twin.

This module defines Pydantic models for system configuration validation,
telemetry payloads from physical/edge machines (Breadboard/Edge Layer),
and global synchronization payloads from the Cloud Optimization Layer.

Strict Unit Conventions:
    - Time: Minutes (min)
    - Power: Kilowatts (kW)
    - Energy: Kilowatt-hours (kWh)
    - Tool Wear VB: Micrometers (um)
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ==============================================================================
# Enumerations
# ==============================================================================

class MachineState(str, Enum):
    """Operational states of a CNC machine."""
    OFF = "OFF"
    STANDBY = "STANDBY"
    WARMUP = "WARMUP"
    IDLE = "IDLE"
    PROCESSING = "PROCESSING"
    TOOL_REPLACEMENT = "TOOL_REPLACEMENT"
    BREAKDOWN = "BREAKDOWN"


class TOUTierType(str, Enum):
    """Time-of-Use electricity pricing tier types."""
    OFF_PEAK = "off_peak"
    MID_PEAK = "mid_peak"
    PEAK = "peak"


# ==============================================================================
# Configuration Models (Parsed from system_params.yaml)
# ==============================================================================

class ToolWearConfig(BaseModel):
    """
    Taylor tool wear model parameters and maintenance durations.
    Model: dVB/dt = eta * (v ^ beta)
    Power drift: P_cut_total = P_cut * (1 + gamma * (VB / w_max))
    """
    initial_wear_um: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Initial tool flank wear at shift start (um)"
    )
    eta: float = Field(
        ...,
        ge=0.1,
        le=3.0,
        description="Wear rate coefficient (um/min) at base speed v=1.0"
    )
    beta: float = Field(
        ...,
        ge=1.0,
        le=4.0,
        description="Taylor wear velocity exponent"
    )
    w_max_um: float = Field(
        100.0,
        gt=0.0,
        description="Maximum permissible flank wear threshold before replacement (um)"
    )
    gamma: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cutting power drift coefficient due to tool wear"
    )
    t_replace_min: float = Field(
        5.0,
        gt=0.0,
        description="Proactive tool replacement downtime (min)"
    )
    t_breakdown_min: float = Field(
        20.0,
        gt=0.0,
        description="Emergency downtime when tool breaks or wear exceeds limit (min)"
    )


class MachineConfig(BaseModel):
    """
    Physical energy and operational parameters for a single CNC machine.
    All energy parameters strictly use kW, kWh, and min.
    """
    name: Optional[str] = Field(None, description="Human-readable machine name")
    machine_type: Optional[str] = Field(None, description="CNC machine classification")
    p_idle_kw: float = Field(
        ...,
        ge=0.5,
        le=3.0,
        description="Power consumed in IDLE state (kW)"
    )
    p_stdby_kw: float = Field(
        ...,
        ge=0.05,
        le=1.0,
        description="Power consumed in STANDBY/SLEEP state (kW)"
    )
    p_cut_base_kw: float = Field(
        ...,
        ge=1.0,
        le=15.0,
        description="Base cutting power at nominal speed level 1 (kW)"
    )
    e_on_off_kwh: float = Field(
        ...,
        ge=0.01,
        le=1.0,
        description="Energy consumed during complete off-to-warmup cycle (kWh)"
    )
    t_warmup_min: float = Field(
        ...,
        ge=0.5,
        le=15.0,
        description="Warmup duration required before cutting can resume (min)"
    )
    tool_wear: ToolWearConfig = Field(
        ...,
        description="Tool deterioration model parameters"
    )

    @model_validator(mode="after")
    def validate_power_hierarchy(self) -> "MachineConfig":
        """Validate physical reality: P_cut_base > P_idle > P_stdby."""
        if self.p_idle_kw <= self.p_stdby_kw:
            raise ValueError(
                f"p_idle_kw ({self.p_idle_kw} kW) must be greater than "
                f"p_stdby_kw ({self.p_stdby_kw} kW)"
            )
        if self.p_cut_base_kw <= self.p_idle_kw:
            raise ValueError(
                f"p_cut_base_kw ({self.p_cut_base_kw} kW) must be greater than "
                f"p_idle_kw ({self.p_idle_kw} kW)"
            )
        return self

    @property
    def break_even_time_min(self) -> float:
        """
        Energy Break-Even Time T_BE in minutes.

        Definition:
            The threshold idle interval duration above which turning off /
            transitioning to standby mode saves more energy than remaining in idle.

        Formula:
            E_idle(T_BE) = E_on_off + E_stdby(T_BE)
            P_idle * (T_BE / 60) = E_on_off + P_stdby * (T_BE / 60)
            => T_BE (min) = (E_on_off * 60) / (P_idle - P_stdby)

        Units:
            E_on_off: kWh
            P_idle, P_stdby: kW
            T_BE: Minutes
        """
        net_power_saving_kw = self.p_idle_kw - self.p_stdby_kw
        return (self.e_on_off_kwh * 60.0) / net_power_saving_kw


class SpeedLevelConfig(BaseModel):
    """Cutting speed scaling level definition."""
    speed_factor: float = Field(..., gt=0.0, description="Relative cutting speed multiplier v")
    description: Optional[str] = Field(None, description="Level description")


class SpeedScalingConfig(BaseModel):
    """Cutting speed levels and power exponent."""
    alpha_p: float = Field(
        1.5,
        gt=0.0,
        description="Power exponent for cutting power scaling: P_cut(v) = P_base * (v ^ alpha_p)"
    )
    levels: Dict[int, SpeedLevelConfig] = Field(
        ...,
        description="Available speed levels (e.g. 1, 2, 3)"
    )


class TOUTierConfig(BaseModel):
    """A single Time-Of-Use pricing interval within the 24-hour cycle."""
    name: str = Field(..., description="Interval name identifier")
    tier_type: TOUTierType = Field(..., description="Type of pricing tier")
    start_min: float = Field(..., ge=0.0, description="Interval start minute [0, 1440)")
    end_min: float = Field(..., le=1440.0, description="Interval end minute (0, 1440]")
    rate_per_kwh: float = Field(..., gt=0.0, description="Electricity price per kWh ($/kWh)")
    description: Optional[str] = Field(None, description="Tariff description")

    @model_validator(mode="after")
    def validate_interval_order(self) -> "TOUTierConfig":
        if self.start_min >= self.end_min:
            raise ValueError(f"start_min ({self.start_min}) must be less than end_min ({self.end_min})")
        return self


class TOUPricingConfig(BaseModel):
    """24-Hour (1440 min) repeating Time-of-Use tariff structure."""
    cycle_min: float = Field(1440.0, gt=0.0, description="Tariff repeat cycle in minutes")
    tiers: List[TOUTierConfig] = Field(..., description="Ordered list of pricing intervals")

    def get_rate_at_minute(self, minute: float) -> float:
        """
        Lookup the electricity tariff ($/kWh) for any simulation time in minutes.
        Automatically wraps time within the 24h cycle (modulo 1440 min).
        """
        normalized_min = minute % self.cycle_min
        for tier in self.tiers:
            if tier.start_min <= normalized_min < tier.end_min:
                return tier.rate_per_kwh
        # Boundary condition: exactly at cycle_min
        if normalized_min == 0.0 and self.tiers:
            return self.tiers[0].rate_per_kwh
        raise ValueError(f"No TOU tier found covering minute {normalized_min:.2f}")


class SystemConfig(BaseModel):
    """
    Root configuration model encapsulating all physical and economic parameters.
    """
    unit_conventions: Dict[str, str] = Field(
        default_factory=lambda: {
            "time": "min",
            "power": "kW",
            "energy": "kWh",
            "tool_wear": "um",
        },
        description="Explicit measurement units"
    )
    p_limit_kw: float = Field(
        22.0,
        gt=0.0,
        description="Peak power threshold for the entire shop floor (kW)"
    )
    speed_scaling: SpeedScalingConfig = Field(
        ...,
        description="Speed levels and power exponent configuration"
    )
    tou_pricing: TOUPricingConfig = Field(
        ...,
        description="24-hour TOU pricing structure"
    )
    machines: Dict[str, MachineConfig] = Field(
        ...,
        description="CNC Machine physical profiles keyed by machine_id (M1..M6)"
    )

    def get_machine(self, machine_id: str) -> MachineConfig:
        """Retrieve machine profile by ID."""
        if machine_id not in self.machines:
            raise KeyError(f"Machine '{machine_id}' not found in system configuration.")
        return self.machines[machine_id]

    def get_break_even_times(self) -> Dict[str, float]:
        """Compute Energy Break-Even Time T_BE (in minutes) for all machines."""
        return {m_id: cfg.break_even_time_min for m_id, cfg in self.machines.items()}

    def get_tou_rate(self, time_min: float) -> float:
        """Get electricity tariff ($/kWh) at simulation minute."""
        return self.tou_pricing.get_rate_at_minute(time_min)


# ==============================================================================
# MQTT Topics & Payload Models
# ==============================================================================

TOPIC_EDGE_TELEMETRY = "dt/fjsp/edge/telemetry"
TOPIC_EDGE_TRIGGER_RESCHEDULE = "dt/fjsp/edge/trigger_reschedule"
TOPIC_CLOUD_SCHEDULE_SYNC = "dt/fjsp/cloud/schedule_sync"


class MachineTelemetryPayload(BaseModel):
    """
    Real-time telemetry payload transmitted from physical Breadboard / Edge node
    to Edge/Cloud MQTT Broker (Topic: `dt/fjsp/edge/telemetry`).
    """
    model_config = ConfigDict(use_enum_values=True)

    machine_id: str = Field(..., description="Unique machine identifier (e.g. 'M1')")
    power_kw: float = Field(..., ge=0.0, description="Real-time electrical power (kW)")
    vibration_rms: float = Field(..., ge=0.0, description="Vibration root-mean-square sensor reading (g or mm/s)")
    estimated_wear_um: float = Field(..., ge=0.0, description="Current estimated tool wear VB (um)")
    state: Union[MachineState, str] = Field(..., description="Operational status of machine")
    timestamp: float = Field(..., description="Telemetry epoch timestamp or simulation time (min)")
    cdi: float = Field(0.0, ge=0.0, description="Cumulative Disturbance Index CDI(t)")

    # Optional hardware DAQ specific readings
    v_bus: Optional[float] = Field(None, ge=0.0, description="Bus voltage (V)")
    i_amp: Optional[float] = Field(None, ge=0.0, description="Current (A)")
    p_raw_w: Optional[float] = Field(None, ge=0.0, description="Raw power (W)")
    p_filt_w: Optional[float] = Field(None, ge=0.0, description="Kalman filtered power (W)")

    @field_validator("machine_id")
    @classmethod
    def validate_machine_id(cls, v: str) -> str:
        v_clean = v.strip().upper()
        if not v_clean:
            raise ValueError("machine_id cannot be empty")
        return v_clean


class RescheduleTriggerPayload(BaseModel):
    """
    Emergency rescheduling trigger payload transmitted from Edge to Cloud
    when shop-floor disturbance CDI(t) exceeds threshold or tool breaks
    (Topic: `dt/fjsp/edge/trigger_reschedule`).
    """
    model_config = ConfigDict(use_enum_values=True)

    machine_id: str = Field(..., description="Machine ID triggering the reschedule")
    reason: str = Field(..., description="Trigger reason (e.g. 'CDI_THRESHOLD_EXCEEDED', 'BREAKDOWN')")
    cdi: float = Field(..., ge=0.0, description="Current Cumulative Disturbance Index")
    cdi_threshold: float = Field(0.25, ge=0.0, description="Threshold that was exceeded")
    current_wear_um: float = Field(..., ge=0.0, description="Flank wear at trigger moment (um)")
    timestamp: float = Field(..., description="Trigger epoch timestamp or simulation minute")
    details: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Diagnostic metadata")


class CloudSyncPayload(BaseModel):
    """
    Global control and synchronization payload published from Cloud Optimization
    Layer down to Edge Dispatching Layer (Topic: `dt/fjsp/cloud/schedule_sync`).
    """
    weights: Dict[str, float] = Field(
        ...,
        description="Objective trade-off weights (e.g. makespan, energy, cost, tool_wear)"
    )
    baseline_cmax: float = Field(
        ...,
        ge=0.0,
        description="Global baseline makespan from Cloud solver (min)"
    )
    baseline_energy_kwh: float = Field(
        ...,
        ge=0.0,
        description="Global baseline total energy consumption from Cloud solver (kWh)"
    )
    p_limit_kw: float = Field(
        22.0,
        gt=0.0,
        description="Peak power threshold constraint (kW)"
    )
    timestamp: Optional[float] = Field(None, description="Sync epoch or simulation minute")
    iteration: Optional[int] = Field(None, ge=0, description="Cloud optimization round/epoch")
    baseline_schedule: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="List of operation assignments from Cloud solver"
    )

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, w: Dict[str, float]) -> Dict[str, float]:
        if not w:
            raise ValueError("weights dictionary cannot be empty")
        for k, v in w.items():
            if v < 0:
                raise ValueError(f"Weight '{k}' must be non-negative, got {v}")
        return w


# ==============================================================================
# Helper Functions
# ==============================================================================

def load_system_config(config_path: Union[str, Path] = "config/system_params.yaml") -> SystemConfig:
    """
    Load, parse, and validate system configuration from YAML file.

    Args:
        config_path: Filepath to system_params.yaml.

    Returns:
        SystemConfig: Fully validated Pydantic model instance.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return SystemConfig.model_validate(data)


if __name__ == "__main__":
    import json

    print("=" * 78)
    print("Cloud-Edge Digital Twin - System Configuration & Break-Even Analysis")
    print("=" * 78)

    cfg = load_system_config()

    print(f"\n[OK] Configuration loaded successfully!")
    print(f"Shop-Floor Power Limit: {cfg.p_limit_kw:.2f} kW")
    print(f"Speed Scaling Exponent (alpha_p): {cfg.speed_scaling.alpha_p}")
    print(f"Speed Levels: {', '.join(f's={k}: v={v.speed_factor}' for k, v in cfg.speed_scaling.levels.items())}")
    print(f"TOU Cycle: {cfg.tou_pricing.cycle_min:.0f} min ({len(cfg.tou_pricing.tiers)} intervals)")

    print("\n" + "-" * 78)
    print(f"{'Machine':<8} | {'P_idle (kW)':<11} | {'P_stdby (kW)':<12} | {'E_on_off (kWh)':<14} | {'T_warmup (m)':<12} | {'T_BE (min)':<10}")
    print("-" * 78)

    break_even_times = cfg.get_break_even_times()
    for m_id, m_cfg in cfg.machines.items():
        t_be = break_even_times[m_id]
        print(
            f"{m_id:<8} | "
            f"{m_cfg.p_idle_kw:<11.2f} | "
            f"{m_cfg.p_stdby_kw:<12.2f} | "
            f"{m_cfg.e_on_off_kwh:<14.2f} | "
            f"{m_cfg.t_warmup_min:<12.1f} | "
            f"{t_be:<10.2f}"
        )
    print("-" * 78)

    # Validate Telemetry and CloudSync serialization
    sample_telemetry = MachineTelemetryPayload(
        machine_id="M1",
        power_kw=4.82,
        vibration_rms=0.18,
        estimated_wear_um=24.5,
        state=MachineState.PROCESSING,
        timestamp=120.0
    )

    sample_sync = CloudSyncPayload(
        weights={"cmax": 0.4, "energy": 0.3, "cost": 0.2, "tool_wear": 0.1},
        baseline_cmax=185.5,
        baseline_energy_kwh=142.3,
        p_limit_kw=cfg.p_limit_kw,
        timestamp=0.0,
        iteration=1
    )

    print("\n[OK] Sample MachineTelemetryPayload JSON:")
    print(json.dumps(sample_telemetry.model_dump(), indent=2))

    print("\n[OK] Sample CloudSyncPayload JSON:")
    print(json.dumps(sample_sync.model_dump(), indent=2))
    print("=" * 78)

