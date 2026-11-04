"""
Unit tests for system configuration loading, physical parameter constraints,
break-even time calculation, and MQTT schema validation.
"""

import pytest
from pydantic import ValidationError

from utils.mqtt_schemas import (
    load_system_config,
    SystemConfig,
    MachineConfig,
    ToolWearConfig,
    MachineTelemetryPayload,
    CloudSyncPayload,
    MachineState,
    TOUTierType,
)


@pytest.fixture
def system_config() -> SystemConfig:
    """Fixture returning validated SystemConfig from config/system_params.yaml."""
    return load_system_config("config/system_params.yaml")


def test_system_config_structure(system_config: SystemConfig):
    """Test global configuration parameters and units."""
    assert system_config.p_limit_kw == 22.0
    assert system_config.unit_conventions["time"] == "min"
    assert system_config.unit_conventions["power"] == "kW"
    assert system_config.unit_conventions["energy"] == "kWh"
    assert system_config.unit_conventions["tool_wear"] == "um"

    # Speed levels
    assert system_config.speed_scaling.alpha_p == 1.5
    levels = system_config.speed_scaling.levels
    assert set(levels.keys()) == {1, 2, 3}
    assert levels[1].speed_factor == 1.00
    assert levels[2].speed_factor == 1.25
    assert levels[3].speed_factor == 1.50


def test_all_machines_present_and_in_ranges(system_config: SystemConfig):
    """Test that all machines (M1 to M8) are loaded and strictly follow specified physical parameter ranges."""
    expected_ids = {f"M{i}" for i in range(1, len(system_config.machines) + 1)}
    assert set(system_config.machines.keys()) == expected_ids
    assert len(expected_ids) >= 6

    for m_id, m in system_config.machines.items():
        # Power & Warmup checks
        assert 0.8 <= m.p_idle_kw <= 1.5, f"{m_id}: p_idle_kw out of bounds ({m.p_idle_kw})"
        assert 0.15 <= m.p_stdby_kw <= 0.25, f"{m_id}: p_stdby_kw out of bounds ({m.p_stdby_kw})"
        assert 3.0 <= m.p_cut_base_kw <= 6.0, f"{m_id}: p_cut_base_kw out of bounds ({m.p_cut_base_kw})"
        assert 0.12 <= m.e_on_off_kwh <= 0.20, f"{m_id}: e_on_off_kwh out of bounds ({m.e_on_off_kwh})"
        assert 2.0 <= m.t_warmup_min <= 4.0, f"{m_id}: t_warmup_min out of bounds ({m.t_warmup_min})"

        # Physical hierarchy: P_cut_base > P_idle > P_stdby
        assert m.p_cut_base_kw > m.p_idle_kw > m.p_stdby_kw

        # Tool wear parameters
        tw = m.tool_wear
        assert 0.9 <= tw.eta <= 1.4, f"{m_id}: eta out of bounds ({tw.eta})"
        assert 2.2 <= tw.beta <= 2.8, f"{m_id}: beta out of bounds ({tw.beta})"
        assert tw.w_max_um == 100.0, f"{m_id}: w_max_um must be 100.0 ({tw.w_max_um})"
        assert 0.20 <= tw.gamma <= 0.30, f"{m_id}: gamma out of bounds ({tw.gamma})"
        assert tw.t_replace_min == 5.0, f"{m_id}: t_replace_min must be 5.0 ({tw.t_replace_min})"
        assert tw.t_breakdown_min == 20.0, f"{m_id}: t_breakdown_min must be 20.0 ({tw.t_breakdown_min})"
        assert 40.0 <= tw.initial_wear_um <= 75.0, f"{m_id}: initial_wear_um out of bounds ({tw.initial_wear_um})"


def test_break_even_time_calculation(system_config: SystemConfig):
    """
    Test Break-Even Time T_BE:
    T_BE (min) = (e_on_off_kwh * 60) / (p_idle_kw - p_stdby_kw)
    Also verify physical consistency: T_BE > t_warmup.
    """
    be_times = system_config.get_break_even_times()
    assert len(be_times) == len(system_config.machines)

    for m_id, m in system_config.machines.items():
        expected_t_be = (m.e_on_off_kwh * 60.0) / (m.p_idle_kw - m.p_stdby_kw)
        assert pytest.approx(be_times[m_id], rel=1e-5) == expected_t_be
        assert pytest.approx(m.break_even_time_min, rel=1e-5) == expected_t_be
        # Feasibility check: T_BE should be larger than warmup time
        assert m.break_even_time_min > m.t_warmup_min


def test_tou_pricing_cycle(system_config: SystemConfig):
    """Test TOU tariff cycle lookup and periodicity."""
    tou = system_config.tou_pricing
    assert tou.cycle_min == 1440.0

    # Test rate at 10 min -> Off-peak (0.06)
    assert system_config.get_tou_rate(10.0) == 0.06
    # Test rate at 40 min -> On-peak (0.20)
    assert system_config.get_tou_rate(40.0) == 0.20
    # Test rate at 80 min -> Mid-peak (0.12)
    assert system_config.get_tou_rate(80.0) == 0.12
    # Test rate at 150 min -> On-peak (0.20)
    assert system_config.get_tou_rate(150.0) == 0.20
    # Test multi-day periodicity (Day 2 at 40 min = 1440 + 40 = 1480 min)
    assert system_config.get_tou_rate(1480.0) == 0.20


def test_machine_telemetry_payload_validation():
    """Test valid and invalid MachineTelemetryPayload creation."""
    payload = MachineTelemetryPayload(
        machine_id="m2",
        power_kw=5.4,
        vibration_rms=0.22,
        estimated_wear_um=45.0,
        state=MachineState.PROCESSING,
        timestamp=350.5
    )
    assert payload.machine_id == "M2"  # Auto uppercase
    assert payload.state == "PROCESSING"

    # Validation failure: negative power
    with pytest.raises(ValidationError):
        MachineTelemetryPayload(
            machine_id="M1",
            power_kw=-1.0,
            vibration_rms=0.1,
            estimated_wear_um=10.0,
            state="IDLE",
            timestamp=10.0
        )


def test_cloud_sync_payload_validation():
    """Test CloudSyncPayload creation and validation."""
    sync = CloudSyncPayload(
        weights={"makespan": 0.5, "energy": 0.5},
        baseline_cmax=210.0,
        baseline_energy_kwh=95.0,
        p_limit_kw=22.0
    )
    assert sync.baseline_cmax == 210.0
    assert sync.weights["makespan"] == 0.5

    # Validation failure: negative weight
    with pytest.raises(ValidationError):
        CloudSyncPayload(
            weights={"makespan": -0.5},
            baseline_cmax=200.0,
            baseline_energy_kwh=100.0
        )

