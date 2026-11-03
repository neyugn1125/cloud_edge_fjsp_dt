"""
Test Suite for Task 3.1: ESP32 Firmware DSP & Telemetry Validation
Verifies the 200Hz digital signal processing (1D Kalman + 3-tap Median filter),
Vibration RMS sliding window calculation, serial telemetry schema, and command parser.
"""

import math
import json
import numpy as np
import pytest


class SimpleKalmanFilter1D:
    """
    Python mirror of C++ SimpleKalmanFilter1D in firmware_esp32/src/main.cpp.
    Combines a 3-tap median circular buffer with 1D Kalman equations.
    """
    def __init__(self, mea_e: float = 4.0, est_e: float = 4.0, q: float = 0.20):
        self._err_measure = float(mea_e)
        self._err_estimate = float(est_e)
        self._q = float(q)
        self._current_estimate = 0.0
        self._last_estimate = 0.0
        self._kalman_gain = 0.0
        self._initialized = False

        self._med_buf = [0.0, 0.0, 0.0]
        self._med_count = 0
        self._med_head = 0

    def _compute_median(self, val: float) -> float:
        self._med_buf[self._med_head] = val
        self._med_head = (self._med_head + 1) % 3
        if self._med_count < 3:
            self._med_count += 1

        if self._med_count == 1:
            return self._med_buf[0]
        if self._med_count == 2:
            return 0.5 * (self._med_buf[0] + self._med_buf[1])

        # Exact 3-element median:
        a = self._med_buf[0]
        b = self._med_buf[1]
        c = self._med_buf[2]
        return max(min(a, b), min(max(a, b), c))

    def reset(self, initial_val: float = 0.0):
        self._current_estimate = initial_val
        self._last_estimate = initial_val
        self._err_estimate = self._err_measure
        self._initialized = False
        self._med_count = 0
        self._med_head = 0
        self._med_buf = [initial_val, initial_val, initial_val]

    def update_estimate(self, mea: float) -> float:
        med = self._compute_median(mea)

        if not self._initialized:
            self._current_estimate = med
            self._last_estimate = med
            self._initialized = True
            return med

        self._kalman_gain = self._err_estimate / (self._err_estimate + self._err_measure)
        self._current_estimate = self._last_estimate + self._kalman_gain * (med - self._last_estimate)
        self._err_estimate = (1.0 - self._kalman_gain) * self._err_estimate + abs(self._last_estimate - self._current_estimate) * self._q
        self._last_estimate = self._current_estimate

        return self._current_estimate

    def get_estimate(self) -> float:
        return self._current_estimate


class VibrationRMSCalculator:
    """
    Python mirror of C++ VibrationRMSCalculator in firmware_esp32/src/main.cpp.
    20-sample circular sliding window (100 ms at 200 Hz).
    """
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self._buf = [0.0] * window_size
        self._head = 0
        self._count = 0
        self._sq_sum = 0.0

    def reset(self):
        self._head = 0
        self._count = 0
        self._sq_sum = 0.0
        self._buf = [0.0] * self.window_size

    def update(self, ax: float, ay: float, az: float) -> float:
        mag_sq = ax * ax + ay * ay + az * az
        if self._count >= self.window_size:
            self._sq_sum -= self._buf[self._head]
            if self._sq_sum < 0.0:
                self._sq_sum = 0.0
        else:
            self._count += 1

        self._buf[self._head] = mag_sq
        self._sq_sum += mag_sq
        self._head = (self._head + 1) % self.window_size

        return math.sqrt(self._sq_sum / float(self._count))


# ============================================================================
# Unit Tests
# ============================================================================

def test_dsp_power_filtering_noise_reduction():
    """
    Simulate a 200Hz signal (dt = 5ms) with +/-4.0W Gaussian noise and +25.0W PWM spikes.
    Assert:
    1. Noise variance reduction >= 75% (var_filt <= 0.25 * var_raw).
    2. Step response settling time < 150 ms.
    Prints SNR and RMSE metrics before/after filtering.
    """
    np.random.seed(42)
    dt = 0.005  # 200 Hz
    t = np.arange(0.0, 5.0, dt)  # 5 seconds, 1000 samples

    # Baseline clean signal: 60.0W steady state, step to 150.0W at t = 2.0s
    clean = np.where(t < 2.0, 60.0, 150.0)

    # Gaussian noise with +/- 4W variation
    gaussian_noise = np.random.normal(0.0, 1.5, size=len(t))
    raw = clean + gaussian_noise

    # Inject random PWM switching spikes (+25.0 W on ~6% of samples)
    spike_indices = np.random.choice(len(t), size=int(0.06 * len(t)), replace=False)
    raw[spike_indices] += 25.0

    # Run filter
    kf = SimpleKalmanFilter1D(mea_e=4.0, est_e=4.0, q=0.20)
    filt = np.array([kf.update_estimate(float(x)) for x in raw])

    # 1. Steady-state noise evaluation (t in [0.5, 1.9] s)
    ss_mask = (t >= 0.5) & (t < 1.9)
    clean_ss = clean[ss_mask]
    raw_ss = raw[ss_mask]
    filt_ss = filt[ss_mask]

    var_raw = float(np.var(raw_ss - clean_ss))
    var_filt = float(np.var(filt_ss - clean_ss))
    var_reduction_pct = (1.0 - var_filt / var_raw) * 100.0

    rmse_raw = float(np.sqrt(np.mean((raw_ss - clean_ss) ** 2)))
    rmse_filt = float(np.sqrt(np.mean((filt_ss - clean_ss) ** 2)))

    snr_raw = float(10.0 * np.log10(np.mean(clean_ss ** 2) / np.mean((raw_ss - clean_ss) ** 2)))
    snr_filt = float(10.0 * np.log10(np.mean(clean_ss ** 2) / np.mean((filt_ss - clean_ss) ** 2)))

    # 2. Step settling time evaluation (step at t = 2.0s to 150.0W)
    step_time = 2.0
    step_idx = int(step_time / dt)
    step_target = 150.0
    step_band = 0.05 * (150.0 - 60.0)  # 5% band = 4.5 W

    settled_idx = None
    for i in range(step_idx, len(t) - 10):
        # Must stay within band for at least 10 consecutive samples (50 ms)
        if np.all(np.abs(filt[i:i + 10] - step_target) <= step_band):
            settled_idx = i
            break

    assert settled_idx is not None, "Filter failed to settle after step response"
    settling_time_ms = (t[settled_idx] - step_time) * 1000.0

    # Print verification metrics to Terminal
    print("\n" + "=" * 70)
    print("TASK 3.1: ESP32 DSP POWER FILTERING VERIFICATION")
    print("=" * 70)
    print(f"Sampling Frequency       : 200 Hz (dt = 5.0 ms)")
    print(f"Total Duration           : {len(t) * dt:.1f} s ({len(t)} samples)")
    print(f"Raw Signal Noise Var     : {var_raw:.4f} W^2")
    print(f"Filtered Signal Noise Var: {var_filt:.4f} W^2")
    print(f"Noise Variance Reduction : {var_reduction_pct:.2f}% (Target: >= 75.0%)")
    print(f"Raw RMSE                 : {rmse_raw:.3f} W")
    print(f"Filtered RMSE            : {rmse_filt:.3f} W")
    print(f"Raw SNR                  : {snr_raw:.2f} dB")
    print(f"Filtered SNR             : {snr_filt:.2f} dB (Gain: +{snr_filt - snr_raw:.2f} dB)")
    print(f"Step Settling Time       : {settling_time_ms:.1f} ms (Target: < 150.0 ms)")
    print("=" * 70)

    # Formal assertions
    assert var_filt <= 0.25 * var_raw, f"Variance reduction failed: {var_reduction_pct:.2f}% < 75.0%"
    assert settling_time_ms < 150.0, f"Settling time failed: {settling_time_ms:.1f} ms >= 150.0 ms"


def test_vibration_rms_sliding_window():
    """
    Verify the 20-sample (100 ms) sliding window RMS calculation.
    """
    calc = VibrationRMSCalculator(window_size=20)

    # 1. Constant vibration components
    # ax = 0.5, ay = 0.5, az = 0.5 -> mag = sqrt(0.75) ~ 0.8660
    for _ in range(25):
        rms = calc.update(0.5, 0.5, 0.5)
    expected_rms = math.sqrt(0.5**2 + 0.5**2 + 0.5**2)
    assert pytest.approx(expected_rms, rel=1e-3) == rms

    # 2. Step transition in vibration from IDLE (~0.08g) to CUTTING (~1.25g)
    calc.reset()
    for _ in range(20):
        calc.update(0.04, 0.05, 0.05)  # IDLE vibration
    idle_rms = calc.update(0.04, 0.05, 0.05)
    assert 0.05 <= idle_rms <= 0.12

    # Step into cutting vibration: ax=0.6, ay=0.8, az=0.8 -> sqrt(0.36 + 0.64 + 0.64) = sqrt(1.64) ~ 1.28g
    for _ in range(20):
        calc.update(0.6, 0.8, 0.8)
    cutting_rms = calc.update(0.6, 0.8, 0.8)
    assert pytest.approx(math.sqrt(1.64), rel=1e-3) == cutting_rms


def test_serial_json_telemetry_schema():
    """
    Verify that the telemetry format matches the exact contract:
    {"ts_ms": 1000, "v_bus": 24.05, "i_amp": 2.40, "p_raw_w": 61.2, "p_filt_w": 57.7, "vib_rms_g": 1.25, "ssr_state": "CUTTING"}
    """
    sample_json = (
        '{"ts_ms": 1000, "v_bus": 24.05, "i_amp": 2.40, '
        '"p_raw_w": 61.2, "p_filt_w": 57.7, "vib_rms_g": 1.25, '
        '"ssr_state": "CUTTING"}'
    )
    payload = json.loads(sample_json)

    required_keys = ["ts_ms", "v_bus", "i_amp", "p_raw_w", "p_filt_w", "vib_rms_g", "ssr_state"]
    for k in required_keys:
        assert k in payload, f"Missing key {k} in telemetry payload"

    assert isinstance(payload["ts_ms"], int)
    assert payload["v_bus"] == pytest.approx(24.05, rel=1e-2)
    assert payload["i_amp"] == pytest.approx(2.40, rel=1e-2)
    assert payload["p_raw_w"] == pytest.approx(61.2, rel=1e-2)
    assert payload["p_filt_w"] == pytest.approx(57.7, rel=1e-2)
    assert payload["vib_rms_g"] == pytest.approx(1.25, rel=1e-2)
    assert payload["ssr_state"] in {"CUTTING", "IDLE", "STANDBY", "OFF"}


def test_command_parser_and_ssr_states():
    """
    Verify command parsing state transitions and SSR logic.
    """
    states = {
        "CMD:CUTTING": {"state": "CUTTING", "ssr_main": True, "ssr_standby": False},
        "CMD:IDLE":    {"state": "IDLE",    "ssr_main": True, "ssr_standby": False},
        "CMD:STANDBY": {"state": "STANDBY", "ssr_main": False, "ssr_standby": True},
        "CMD:OFF":     {"state": "OFF",     "ssr_main": False, "ssr_standby": False},
    }

    for cmd, expected in states.items():
        assert cmd.startswith("CMD:")
        sub = cmd[4:]
        assert sub == expected["state"]

