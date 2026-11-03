"""
Test Suite for Task 3.2: Breadboard DAQ Interface and Real-Time Tool Wear Estimator.
Verifies:
1. Dual-sensor fusion (INA226 Power + ADXL345 Vibration) achieves RMSE < 2.5 um and R^2 > 0.97.
2. Dual-sensor fusion outperforms single-sensor estimation (RMSE_fusion < RMSE_power and RMSE_fusion < RMSE_vib).
3. Inrush current spike (2.2x steady-state power for the first 200ms of startup transition).
4. Serial telemetry JSON schema compliance with firmware_esp32/main.cpp.
"""

import json
import math
import numpy as np
import pytest

from edge_layer.breadboard_daq import (
    BreadboardDAQInterface,
    RealTimeToolWearEstimator,
)


def test_dual_sensor_fusion_performance_and_ablation():
    """
    Test 500 telemetry samples across all 3 cutting speeds (s = 1, 2, 3)
    with actual tool wear increasing from 0.0 to 95.0 um.
    Asserts:
    - RMSE_fusion < 2.5 um and R^2 > 0.97.
    - RMSE_fusion < RMSE_power_only and RMSE_fusion < RMSE_vib_only.
    Prints comparison table 'Single Sensor vs. Dual-Sensor Fusion (INA226 + ADXL345)'.
    """
    np.random.seed(42)
    N = 500
    true_wear = np.linspace(0.0, 95.0, N)

    # 3 operational stages across cutting speeds s in {1, 2, 3}
    speeds = np.zeros(N, dtype=int)
    speeds[:166] = 1
    speeds[166:333] = 2
    speeds[333:] = 3

    daq = BreadboardDAQInterface(mode="hardware_emulator")
    estimator = RealTimeToolWearEstimator()

    daq.send_command("CMD:CUTTING")

    # Generate telemetry stream
    telemetries = []
    for s, w in zip(speeds, true_wear):
        daq.set_operating_conditions(speed_level=s, true_wear_um=w)
        tel = daq.read_telemetry(dt_ms=100)
        telemetries.append(tel)

    # 1. Dual-Sensor Fusion (INA226 + ADXL345)
    estimator.reset(0.0)
    wear_fusion = [
        estimator.estimate_wear(tel, s, sensor_mode="fusion", use_filter=True)
        for tel, s in zip(telemetries, speeds)
    ]

    # 2. Single Sensor: INA226 Power Only
    estimator.reset(0.0)
    wear_power_only = [
        estimator.estimate_wear(tel, s, sensor_mode="power_only", use_filter=True)
        for tel, s in zip(telemetries, speeds)
    ]

    # 3. Single Sensor: ADXL345 Vibration Only
    estimator.reset(0.0)
    wear_vib_only = [
        estimator.estimate_wear(tel, s, sensor_mode="vib_only", use_filter=True)
        for tel, s in zip(telemetries, speeds)
    ]

    wear_fusion_arr = np.array(wear_fusion)
    wear_p_arr = np.array(wear_power_only)
    wear_v_arr = np.array(wear_vib_only)

    # Compute Error Metrics
    rmse_fusion = float(np.sqrt(np.mean((wear_fusion_arr - true_wear) ** 2)))
    rmse_power = float(np.sqrt(np.mean((wear_p_arr - true_wear) ** 2)))
    rmse_vib = float(np.sqrt(np.mean((wear_v_arr - true_wear) ** 2)))

    mae_fusion = float(np.mean(np.abs(wear_fusion_arr - true_wear)))
    mae_power = float(np.mean(np.abs(wear_p_arr - true_wear)))
    mae_vib = float(np.mean(np.abs(wear_v_arr - true_wear)))

    ss_res_f = float(np.sum((true_wear - wear_fusion_arr) ** 2))
    ss_tot = float(np.sum((true_wear - np.mean(true_wear)) ** 2))
    r2_fusion = 1.0 - (ss_res_f / ss_tot)

    r2_power = 1.0 - (float(np.sum((true_wear - wear_p_arr) ** 2)) / ss_tot)
    r2_vib = 1.0 - (float(np.sum((true_wear - wear_v_arr) ** 2)) / ss_tot)

    # Print Formatted Comparison Table to Terminal
    print("\n" + "=" * 80)
    print("TASK 3.2: BREADBOARD SENSOR FUSION VS. SINGLE SENSOR ABLATION BENCHMARK")
    print("=" * 80)
    print(f"Sample Count              : {N} continuous telemetry frames (10 Hz, 50.0 s)")
    print(f"Tool Wear Horizon         : 0.0 um -> 95.0 um (Speed Levels s in {{1, 2, 3}})")
    print("-" * 80)
    print(f"{'Estimation Method':<38} | {'RMSE (um)':<10} | {'MAE (um)':<10} | {'R^2 Score':<10}")
    print("-" * 80)
    print(f"{'Single Sensor (ADXL345 Vibration Only)':<38} | {rmse_vib:<10.3f} | {mae_vib:<10.3f} | {r2_vib:<10.4f}")
    print(f"{'Single Sensor (INA226 Power Only)':<38} | {rmse_power:<10.3f} | {mae_power:<10.3f} | {r2_power:<10.4f}")
    print(f"{'Dual-Sensor Fusion (INA226 + ADXL345)':<38} | {rmse_fusion:<10.3f} | {mae_fusion:<10.3f} | {r2_fusion:<10.4f}")
    print("=" * 80)
    print(f"Fusion RMSE Improvement over Vibration Only: {((rmse_vib - rmse_fusion) / rmse_vib) * 100:.2f}%")
    print(f"Fusion RMSE Improvement over Power Only    : {((rmse_power - rmse_fusion) / rmse_power) * 100:.2f}%")
    print("=" * 80)

    # Core Requirements Assertions
    assert rmse_fusion < 2.5, f"Fusion RMSE target failed: {rmse_fusion:.3f} um >= 2.5 um"
    assert r2_fusion > 0.97, f"Fusion R^2 target failed: {r2_fusion:.4f} <= 0.97"
    assert rmse_fusion < rmse_power, f"Fusion should outperform Power Only: {rmse_fusion:.3f} >= {rmse_power:.3f}"
    assert rmse_fusion < rmse_vib, f"Fusion should outperform Vibration Only: {rmse_fusion:.3f} >= {rmse_vib:.3f}"


def test_inrush_current_spike():
    """
    Verify Inrush Current Spike during transition from OFF/STANDBY -> IDLE/CUTTING.
    Power must be ~2.2x steady-state power during the first 200 ms,
    then return to normal steady-state.
    """
    daq = BreadboardDAQInterface(mode="hardware_emulator")

    # 1. Start in STANDBY
    daq.send_command("CMD:STANDBY")
    tel_standby = daq.read_telemetry(dt_ms=100)
    assert tel_standby["ssr_state"] == "STANDBY"
    assert tel_standby["p_filt_w"] < 5.0

    # 2. Transition to CUTTING at speed s=1, wear=0 um (base power = 45.0 W)
    daq.send_command("CMD:CUTTING")
    daq.set_operating_conditions(speed_level=1, true_wear_um=0.0)

    # First 100 ms (Inrush Active)
    t1 = daq.read_telemetry(dt_ms=100)
    # Second 100 ms (Inrush Active, total 200ms)
    t2 = daq.read_telemetry(dt_ms=100)
    # Third 100 ms (Inrush Expired -> Steady State)
    t3 = daq.read_telemetry(dt_ms=100)
    t4 = daq.read_telemetry(dt_ms=100)

    # Base power = 45.0W; 2.2x inrush = ~99.0W
    expected_inrush_power = 45.0 * 2.2
    assert t1["p_filt_w"] > 70.0, f"Inrush spike not observed in frame 1: {t1['p_filt_w']} W"
    assert t2["p_filt_w"] > 85.0, f"Inrush spike not observed in frame 2: {t2['p_filt_w']} W"
    assert pytest.approx(expected_inrush_power, rel=0.15) == t2["p_filt_w"]

    # Post-inrush power settles to nominal steady-state (45.0 W)
    assert t4["p_filt_w"] < 55.0, f"Power did not settle back to steady-state: {t4['p_filt_w']} W"
    assert pytest.approx(45.0, rel=0.15) == t4["p_filt_w"]


def test_telemetry_schema_and_commands():
    """
    Verify JSON telemetry packet schema and command processing.
    """
    daq = BreadboardDAQInterface(mode="hardware_emulator")

    # Check command transitions
    assert daq.send_command("CMD:CUTTING") is True
    assert daq.ssr_state == "CUTTING"

    assert daq.send_command("CMD:IDLE") is True
    assert daq.ssr_state == "IDLE"

    assert daq.send_command("CMD:STANDBY") is True
    assert daq.ssr_state == "STANDBY"

    assert daq.send_command("CMD:OFF") is True
    assert daq.ssr_state == "OFF"

    # In OFF state, telemetry should be zero
    off_tel = daq.read_telemetry()
    assert off_tel["ssr_state"] == "OFF"
    assert off_tel["p_raw_w"] == 0.0
    assert off_tel["v_bus"] == 0.0
    assert off_tel["i_amp"] == 0.0

    # Test invalid command
    assert daq.send_command("CMD:INVALID") is False


def test_estimator_update_diagnostics():
    """
    Verify detailed diagnostics dictionary returned by estimator.update().
    """
    estimator = RealTimeToolWearEstimator()
    sample_tel = {
        "ts_ms": 1000,
        "v_bus": 24.05,
        "i_amp": 2.40,
        "p_raw_w": 52.0,
        "p_filt_w": 50.6,
        "vib_rms_g": 1.05,
        "ssr_state": "CUTTING",
    }
    diag = estimator.update(sample_tel, speed_level=1, sensor_mode="fusion")
    assert "estimated_wear_um" in diag
    assert "power_wear_est" in diag
    assert "vib_wear_est" in diag
    assert "estimation_variance" in diag
    assert diag["sensor_mode"] == "fusion"
    assert 0.0 <= diag["estimated_wear_um"] <= 100.0


def test_real_serial_mode_error_handling():
    """
    Verify that BreadboardDAQInterface with mode='real_serial' raises appropriate
    ConnectionError when no physical USB device is present.
    """
    with pytest.raises(ConnectionError):
        BreadboardDAQInterface(mode="real_serial", port="/dev/ttyNonExistentPort")

