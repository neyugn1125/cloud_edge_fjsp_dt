"""
Breadboard DAQ Interface and Real-Time Tool Wear Estimator.
Implements:
1. BreadboardDAQInterface: Supports 'real_serial' (UART /dev/ttyUSB0) and 'hardware_emulator'
   (High-Fidelity Electro-Mechanical Emulator with Unmodeled Material & Thermal Disturbances,
   inrush current spike matching firmware_esp32/main.cpp, and realistic flank wear degradation).
2. RealTimeToolWearEstimator: Dual-sensor fusion (INA226 Power + ADXL345 Vibration)
   using Extended Kalman Filtering (EKF) and optimal weighted regression.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import yaml

try:
    import serial
except ImportError:
    serial = None  # Handled gracefully if pyserial is not available in test mocks


class SimpleKalmanFilter1D:
    """
    Python mirror of firmware_esp32/src/main.cpp SimpleKalmanFilter1D.
    Used within the hardware emulator to produce realistic filtered power (p_filt_w).
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

        a, b, c = self._med_buf[0], self._med_buf[1], self._med_buf[2]
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


class BreadboardDAQInterface:
    """
    Data Acquisition (DAQ) Interface for the Breadboard Physical Twin.
    Supports:
    - 'real_serial': Connects to ESP32 via physical UART (/dev/ttyUSB0 at 115200 baud).
    - 'hardware_emulator': High-fidelity electro-mechanical simulation of the CNC Twin node:
        + Inrush current spike (2.2x steady-state power for first 200ms of startup).
        + Speed-dependent cutting power and vibration.
        + Flank wear degradation: P_cut increases with gamma * (VB / w_max),
          vibration increases non-linearly with (VB / w_max)^1.35.
        + 200 Hz internal subsampling with realistic Gaussian noise and intermittent PWM spikes.
    """

    def __init__(
        self,
        mode: str = "hardware_emulator",
        port: str = "/dev/ttyUSB0",
        baudrate: int = 115200,
        timeout: float = 1.0,
        config_path: str = "config/system_params.yaml",
        machine_id: str = "M1",
        p_cut_base_w: float = 45.0,
        vib_base_g: float = 0.85,
    ):
        self.mode = mode
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.machine_id = machine_id

        # Load system configuration
        self.config = self._load_config(config_path)
        m_cfg = self.config.get("machines", {}).get(machine_id, {})
        tw_cfg = m_cfg.get("tool_wear", {})

        self.gamma = float(tw_cfg.get("gamma", 0.25))
        self.w_max_um = float(tw_cfg.get("w_max_um", 100.0))
        self.alpha_p = float(self.config.get("speed_scaling", {}).get("alpha_p", 1.5))
        self.speed_factors = {1: 1.00, 2: 1.25, 3: 1.50}

        # Bench-scale baseline parameters
        self.p_cut_base_w = p_cut_base_w
        self.vib_base_g = vib_base_g
        self.p_idle_w = 15.6
        self.p_stdby_w = 3.6
        self.alpha_v = 1.20
        self.kappa = 0.48
        self.alpha_vib_wear = 1.35

        # Unmodeled electro-mechanical dynamics and physical calibration mismatch:
        # In a real physical CNC machine, the true mechanical response slightly deviates
        # from the nominal model constants known to the estimator:
        self.p_cut_base_real = self.p_cut_base_w * 1.003   # Calibration mismatch: +0.3%
        self.gamma_real = self.gamma * 0.988              # Real wear power slope: -1.2%
        self.vib_base_real = self.vib_base_g * 1.004      # Real base vibration: +0.4%
        self.kappa_real = self.kappa * 0.985              # Real wear vibration slope: -1.5%

        # Dynamic disturbance states
        self.delta_mat = 0.0          # Workpiece hardness fluctuation AR(1) state
        self.ar_phi = 0.70             # Autoregressive memory coefficient (correlation time ~300ms)
        self.ar_amp_max = 0.04         # Max hardness fluctuation amplitude (+/- 4%)
        self.chip_load_fraction = 0.28 # Fraction of cutting power attributable to chip shear
        self.accumulated_cut_time_s = 0.0  # Spindle thermal run-time accumulator

        # Hardware emulator internal state
        self.ssr_state = "IDLE"
        self.current_speed = 1
        self.true_wear_um = 0.0
        self.sim_time_ms = 0
        self.inrush_remaining_ms = 0.0
        self.serial_conn = None

        self._filter = SimpleKalmanFilter1D(mea_e=4.0, est_e=4.0, q=0.20)
        self._filter.reset(self.p_idle_w)

        if self.mode == "real_serial":
            self._init_serial()

    def _load_config(self, path: str) -> dict:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _init_serial(self):
        if serial is None:
            raise ImportError("pyserial is not installed. Install via pip install pyserial.")
        try:
            self.serial_conn = serial.Serial(self.port, baudrate=self.baudrate, timeout=self.timeout)
            time.sleep(0.2)
        except Exception as e:
            raise ConnectionError(f"Failed to open serial port {self.port}: {e}")

    def send_command(self, cmd: str) -> bool:
        """
        Sends command to ESP32 (e.g. CMD:IDLE, CMD:STANDBY, CMD:OFF, CMD:CUTTING).
        In emulator mode, updates SSR states and triggers inrush current when applicable.
        """
        cmd_clean = cmd.strip()
        if self.mode == "real_serial":
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.write((cmd_clean + "\n").encode("utf-8"))
                return True
            return False

        # Hardware Emulator Mode
        if cmd_clean.startswith("CMD:"):
            target_state = cmd_clean[4:].upper()
            if target_state in {"IDLE", "STANDBY", "OFF", "CUTTING"}:
                old_state = self.ssr_state
                self.ssr_state = target_state

                # Reset dynamic disturbance state when not cutting
                if target_state in {"OFF", "STANDBY", "IDLE"}:
                    self.delta_mat = 0.0
                if target_state == "STANDBY":
                    # Tool swap / preventive maintenance allows spindle cooling
                    self.accumulated_cut_time_s = max(0.0, self.accumulated_cut_time_s - 15.0)

                # Check for inrush transition: from OFF/STANDBY to IDLE/CUTTING
                if old_state in {"OFF", "STANDBY"} and target_state in {"IDLE", "CUTTING"}:
                    self.inrush_remaining_ms = 200.0  # 200 ms inrush duration
                return True
        return False

    def set_operating_conditions(self, speed_level: int, true_wear_um: float):
        """Sets current cutting speed level and true tool wear for emulator."""
        self.current_speed = int(speed_level)
        self.true_wear_um = float(np.clip(true_wear_um, 0.0, 105.0))

    def read_telemetry(self, dt_ms: int = 100) -> Dict[str, Any]:
        """
        Reads one telemetry packet (at 10Hz = 100ms interval).
        In emulator mode, runs the 200Hz internal DSP subsampling loop (20 samples per 100ms)
        matching the exact behavior of firmware_esp32/main.cpp.
        Incorporates unmodeled physical disturbances:
        - Workpiece hardness fluctuation AR(1) process (+/- 4%).
        - Sensor calibration mismatch and slow thermal drift (+0.6%).
        - Intermittent non-linear chatter bursts when tool flank wear VB > 65 um.
        Returns JSON dictionary matching ESP32 firmware schema:
        {"ts_ms": int, "v_bus": float, "i_amp": float, "p_raw_w": float,
         "p_filt_w": float, "vib_rms_g": float, "ssr_state": str}
        """
        if self.mode == "real_serial":
            if not self.serial_conn or not self.serial_conn.is_open:
                raise ConnectionError("Serial connection is not active.")
            line = self.serial_conn.readline().decode("utf-8", errors="ignore").strip()
            if line.startswith("{") and line.endswith("}"):
                return json.loads(line)
            raise ValueError(f"Invalid telemetry line received: {line}")

        # Hardware Emulator Mode
        self.sim_time_ms += dt_ms
        inrush_active = self.inrush_remaining_ms > 0.0
        if inrush_active:
            self.inrush_remaining_ms = max(0.0, self.inrush_remaining_ms - dt_ms)

        v_bus = 24.0 + random.gauss(0.0, 0.03)

        if self.ssr_state == "OFF":
            return {
                "ts_ms": int(self.sim_time_ms),
                "v_bus": 0.0,
                "i_amp": 0.0,
                "p_raw_w": 0.0,
                "p_filt_w": 0.0,
                "vib_rms_g": 0.0,
                "ssr_state": "OFF",
                "sensor_ok": True,
                "sim_fallback": False,
            }

        if self.ssr_state == "STANDBY":
            # Run subsamples at 200Hz (dt = 5ms)
            subsamples = max(1, dt_ms // 5)
            last_p_raw = 0.0
            for _ in range(subsamples):
                p_steady = self.p_stdby_w + random.gauss(0.0, 0.1)
                last_p_raw = max(0.0, p_steady)
                self._filter.update_estimate(last_p_raw)

            p_filt = self._filter.get_estimate()
            return {
                "ts_ms": int(self.sim_time_ms),
                "v_bus": round(float(v_bus), 2),
                "i_amp": round(float(p_filt / v_bus), 2),
                "p_raw_w": round(float(last_p_raw), 1),
                "p_filt_w": round(float(p_filt), 1),
                "vib_rms_g": 0.01,
                "ssr_state": "STANDBY",
                "sensor_ok": True,
                "sim_fallback": False,
            }

        if self.ssr_state == "IDLE":
            p_steady = self.p_idle_w
            if inrush_active:
                p_steady *= 2.2  # Inrush spike 2.2x

            subsamples = max(1, dt_ms // 5)
            last_p_raw = 0.0
            for _ in range(subsamples):
                last_p_raw = max(0.0, p_steady + random.gauss(0.0, 0.3))
                self._filter.update_estimate(last_p_raw)

            p_filt = self._filter.get_estimate()
            vib_rms = max(0.01, 0.08 + random.gauss(0.0, 0.005))
            return {
                "ts_ms": int(self.sim_time_ms),
                "v_bus": round(float(v_bus), 2),
                "i_amp": round(float(p_filt / v_bus), 2),
                "p_raw_w": round(float(last_p_raw), 1),
                "p_filt_w": round(float(p_filt), 1),
                "vib_rms_g": round(float(vib_rms), 2),
                "ssr_state": "IDLE",
                "sensor_ok": True,
                "sim_fallback": False,
            }

        # CUTTING State
        self.accumulated_cut_time_s += dt_ms / 1000.0

        # 1. Unmodeled Workpiece Hardness Fluctuation: AR(1) process (+/- 4% max amplitude)
        self.delta_mat = self.ar_phi * self.delta_mat + random.gauss(0.0, self.ar_amp_max * 0.25)
        self.delta_mat = float(np.clip(self.delta_mat, -self.ar_amp_max, self.ar_amp_max))

        # 2. Unmodeled Thermal Drift (+0.0% -> +0.6% gradual drift over prolonged cutting)
        thermal_drift = min(0.006, self.accumulated_cut_time_s * 0.00012)

        speed_factor = self.speed_factors.get(self.current_speed, 1.0)
        p_base = self.p_cut_base_real * (speed_factor ** self.alpha_p)

        # Flank wear increases cutting power by gamma_real * (VB / w_max),
        # combined with unmodeled workpiece hardness and thermal drift on the mechanical load:
        wear_ratio = self.true_wear_um / self.w_max_um
        p_cut_steady = p_base * (1.0 + thermal_drift + self.chip_load_fraction * self.delta_mat + self.gamma_real * wear_ratio)

        if inrush_active:
            p_cut_steady *= 2.2  # Inrush spike 2.2x

        # Emulate 200Hz sampling (subsamples = dt_ms // 5 = 20) with PWM spikes and noise
        subsamples = max(1, dt_ms // 5)
        last_p_raw = 0.0
        for _ in range(subsamples):
            pwm_spike = 25.0 if (random.random() < 0.05) else 0.0
            last_p_raw = p_cut_steady + random.gauss(0.0, 1.2) + pwm_spike
            self._filter.update_estimate(last_p_raw)

        p_filt = self._filter.get_estimate()

        # 3. Vibration model: base vibration + material hardness + wear degradation
        # Plus unmodeled intermittent chatter bursts when tool wear VB > 65 um:
        vib_base = self.vib_base_real * (speed_factor ** self.alpha_v)
        chatter_burst = 0.0
        if self.true_wear_um > 65.0 and random.random() < 0.10:
            chatter_burst = random.uniform(0.04, 0.12) * ((self.true_wear_um - 65.0) / 35.0)

        vib_steady = vib_base * (1.0 + self.chip_load_fraction * 0.8 * self.delta_mat + self.kappa_real * (wear_ratio ** self.alpha_vib_wear)) + chatter_burst
        vib_rms = max(0.01, vib_steady + random.gauss(0.0, 0.025))

        return {
            "ts_ms": int(self.sim_time_ms),
            "v_bus": round(float(v_bus), 2),
            "i_amp": round(float(p_filt / v_bus), 2),
            "p_raw_w": round(float(last_p_raw), 1),
            "p_filt_w": round(float(p_filt), 1),
            "vib_rms_g": round(float(vib_rms), 2),
            "ssr_state": "CUTTING",
            "sensor_ok": True,
            "sim_fallback": False,
        }

    def close(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class RealTimeToolWearEstimator:
    """
    Real-Time Dual-Sensor Fusion Estimator for CNC Tool Flank Wear (VB in micrometers).
    Fuses:
    - INA226 Power telemetry (p_filt_w)
    - ADXL345 Vibration RMS telemetry (vib_rms_g)
    Using an Extended Kalman Filter (EKF) and optimal weighted regression.
    """

    def __init__(
        self,
        config_path: str = "config/system_params.yaml",
        machine_id: str = "M1",
        p_cut_base_w: float = 45.0,
        vib_base_g: float = 0.85,
    ):
        self.machine_id = machine_id
        self.config = self._load_config(config_path)
        m_cfg = self.config.get("machines", {}).get(machine_id, {})
        tw_cfg = m_cfg.get("tool_wear", {})

        self.gamma = float(tw_cfg.get("gamma", 0.25))
        self.w_max_um = float(tw_cfg.get("w_max_um", 100.0))
        self.alpha_p = float(self.config.get("speed_scaling", {}).get("alpha_p", 1.5))
        self.speed_factors = {1: 1.00, 2: 1.25, 3: 1.50}

        self.p_cut_base_w = p_cut_base_w
        self.vib_base_g = vib_base_g
        self.alpha_v = 1.20
        self.kappa = 0.48
        self.alpha_vib_wear = 1.35

        # Sensor measurement noise characteristics (accounting for effective observation uncertainty)
        self.sigma_p_filt = 0.40  # Watts (effective measurement standard deviation under unmodeled load variations)
        self.sigma_vib_rms = 0.025  # g

        # EKF State variables
        self.x_est = 0.0  # Estimated wear (um)
        self.P_est = 4.0  # Estimation covariance
        self.q_proc = 0.06  # Process noise variance (optimal tracking damping)

    @property
    def estimated_wear_um(self) -> float:
        """Returns the current EKF estimated flank wear in micrometers."""
        return float(self.x_est)

    @property
    def sigma_ekf(self) -> float:
        """Returns standard deviation of estimation error sqrt(P_est) in micrometers."""
        return float(math.sqrt(max(1e-6, self.P_est)))

    def get_confidence_bounds(self, k_sigma: float = 2.0) -> Tuple[float, float]:
        """Returns [lower, upper] confidence bounds in micrometers (default 95% = 2*sigma)."""
        std = self.sigma_ekf
        return max(0.0, self.x_est - k_sigma * std), min(105.0, self.x_est + k_sigma * std)

    def _load_config(self, path: str) -> dict:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def reset(self, initial_wear_um: float = 0.0):
        """Resets the estimator state to initial wear."""
        self.x_est = float(np.clip(initial_wear_um, 0.0, 100.0))
        self.P_est = 4.0

    def _get_base_values(self, speed_level: int) -> Tuple[float, float]:
        speed_factor = self.speed_factors.get(int(speed_level), 1.0)
        p_base = self.p_cut_base_w * (speed_factor ** self.alpha_p)
        vib_base = self.vib_base_g * (speed_factor ** self.alpha_v)
        return p_base, vib_base

    def invert_power_wear(self, p_filt_w: float, speed_level: int) -> Tuple[float, float]:
        """
        Calculates tool wear estimate and error variance from INA226 power telemetry.
        Model: P = P_base * (1 + gamma * (w / w_max))
        => w = w_max * (P - P_base) / (gamma * P_base)
        """
        p_base, _ = self._get_base_values(speed_level)
        w_est = self.w_max_um * ((p_filt_w - p_base) / (self.gamma * p_base))
        w_clamped = float(np.clip(w_est, 0.0, 105.0))

        # Variance propagation: sigma_w^2 = (w_max / (gamma * P_base))^2 * sigma_p^2
        var_w = ((self.w_max_um / (self.gamma * p_base)) ** 2) * (self.sigma_p_filt ** 2)
        return w_clamped, max(0.01, var_w)

    def invert_vib_wear(self, vib_rms_g: float, speed_level: int) -> Tuple[float, float]:
        """
        Calculates tool wear estimate and error variance from ADXL345 vibration telemetry.
        Model: Vib = Vib_base * (1 + kappa * (w / w_max)^alpha)
        => w = w_max * max(0, (Vib - Vib_base) / (kappa * Vib_base))^(1 / alpha)
        """
        _, vib_base = self._get_base_values(speed_level)
        ratio = (vib_rms_g - vib_base) / (self.kappa * vib_base)
        ratio = max(0.0001, ratio)
        w_est = self.w_max_um * (ratio ** (1.0 / self.alpha_vib_wear))
        w_clamped = float(np.clip(w_est, 0.0, 105.0))

        # Sensitivity: dv/dw = Vib_base * kappa * alpha / w_max * (w / w_max)^(alpha - 1)
        w_ref = max(0.05, w_clamped / self.w_max_um)
        dvdz = max(0.001, vib_base * self.kappa * self.alpha_vib_wear / self.w_max_um * (w_ref ** (self.alpha_vib_wear - 1.0)))
        var_w = (self.sigma_vib_rms ** 2) / (dvdz ** 2)
        return w_clamped, max(0.01, var_w)

    def estimate_wear(
        self,
        telemetry: Dict[str, Any],
        speed_level: int,
        sensor_mode: str = "fusion",
        use_filter: bool = True,
        return_std: bool = False,
    ) -> Union[float, Tuple[float, float]]:
        """
        Estimates tool wear (um) from a telemetry packet.
        Parameters:
            telemetry: Dict containing 'p_filt_w', 'vib_rms_g', 'ssr_state'.
            speed_level: Cutting speed level (1, 2, or 3).
            sensor_mode: 'fusion' (both), 'power_only' (INA226), or 'vib_only' (ADXL345).
            use_filter: True to use EKF tracking, False for instantaneous inversion.
            return_std: True to return (estimated_wear_um, sigma_ekf) tuple.
        """
        ssr_state = telemetry.get("ssr_state", "CUTTING")
        p_filt = float(telemetry.get("p_filt_w", 0.0))
        vib_rms = float(telemetry.get("vib_rms_g", 0.0))

        p_base, _ = self._get_base_values(speed_level)
        # If machine is not actively cutting or inrush transient spike is active, hold current estimate
        if ssr_state != "CUTTING":
            return (self.x_est, self.sigma_ekf) if return_std else self.x_est
        if p_filt > 1.5 * p_base * (1.0 + self.gamma):
            return (self.x_est, self.sigma_ekf) if return_std else self.x_est

        w_p, var_p = self.invert_power_wear(p_filt, speed_level)
        w_v, var_v = self.invert_vib_wear(vib_rms, speed_level)

        if sensor_mode == "power_only":
            z_meas = w_p
            r_meas = var_p
        elif sensor_mode == "vib_only":
            z_meas = w_v
            r_meas = var_v
        else:  # 'fusion'
            # Optimal Best Linear Unbiased Estimator (BLUE) fusion
            inv_p = 1.0 / var_p
            inv_v = 1.0 / var_v
            z_meas = (w_p * inv_p + w_v * inv_v) / (inv_p + inv_v)
            r_meas = 1.0 / (inv_p + inv_v)

        if not use_filter:
            inst_est = float(np.clip(z_meas, 0.0, 100.0))
            return (inst_est, math.sqrt(r_meas)) if return_std else inst_est

        # EKF Time Update (Prediction: random-walk progression)
        x_pred = self.x_est
        P_pred = self.P_est + self.q_proc

        # EKF Measurement Update
        kalman_gain = P_pred / (P_pred + r_meas)
        self.x_est = x_pred + kalman_gain * (z_meas - x_pred)
        self.P_est = (1.0 - kalman_gain) * P_pred
        self.x_est = float(np.clip(self.x_est, 0.0, 100.0))

        return (self.x_est, self.sigma_ekf) if return_std else self.x_est

    def update(
        self,
        telemetry: Dict[str, Any],
        speed_level: int,
        sensor_mode: str = "fusion",
    ) -> Dict[str, Any]:
        """
        Processes telemetry, updates EKF, and returns detailed estimation metrics.
        """
        p_filt = float(telemetry.get("p_filt_w", 0.0))
        vib_rms = float(telemetry.get("vib_rms_g", 0.0))

        w_p, var_p = self.invert_power_wear(p_filt, speed_level)
        w_v, var_v = self.invert_vib_wear(vib_rms, speed_level)

        est_wear = self.estimate_wear(telemetry, speed_level, sensor_mode=sensor_mode, use_filter=True)

        return {
            "estimated_wear_um": round(est_wear, 2),
            "power_wear_est": round(w_p, 2),
            "vib_wear_est": round(w_v, 2),
            "power_var": round(var_p, 3),
            "vib_var": round(var_v, 3),
            "estimation_variance": round(self.P_est, 4),
            "estimation_std_um": round(self.sigma_ekf, 3),
            "sensor_mode": sensor_mode,
        }
