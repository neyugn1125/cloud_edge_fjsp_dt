"""
Edge Digital Twin Node with GRBL Controller, Breadboard DAQ, and Cumulative Disturbance Index (CDI).

Responsibilities:
1. Multi-Subsystem Integration:
   - BreadboardDAQInterface: High-frequency telemetry acquisition (INA226 Power + ADXL345 Vibration).
   - RealTimeToolWearEstimator: Dual-sensor EKF tool flank wear estimation (VB_hat in um).
   - EdgeONNXInferenceEngine: Sub-millisecond deterministic PPO policy inference on CPU.
   - GRBLHardwareController: G-code streaming and real-time feed/spindle override speed scaling.
   - EdgeMQTTClient: Industrial communication to Cloud for telemetry streaming and reschedule triggers.
2. Disturbance Tracking & Closed-Loop Control:
   - Cumulative Disturbance Index (CDI(t)):
     CDI(t) = alpha * |C_actual(t) - C_base(t)| / C_base(t) + (1 - alpha) * |E_actual(t) - E_base(t)| / E_base(t)
     Default: alpha = 0.5, cdi_threshold = 0.12.
   - If CDI(t) <= cdi_threshold: Autonomous edge compensation using ONNX policy (speed scaling s in {1, 2, 3}
     or opportunistic PM) without calling Cloud.
   - If CDI(t) > cdi_threshold: Signals emergency cloud rescheduling (trigger_cloud_reschedule = True)
     with full machine and operation progress snapshot.
   - Upon receiving updated baseline schedule from Cloud: resets CDI(t) = 0.0 and trigger_cloud_reschedule = False.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from edge_layer.breadboard_daq import BreadboardDAQInterface, RealTimeToolWearEstimator
from edge_layer.grbl_controller import GRBLHardwareController
from edge_layer.mqtt_client import EdgeMQTTClient
from edge_layer.onnx_exporter import EdgeONNXInferenceEngine
from utils.mqtt_schemas import (
    CloudSyncPayload,
    MachineState,
    MachineTelemetryPayload,
    RescheduleTriggerPayload,
    TOPIC_EDGE_TRIGGER_RESCHEDULE,
)

logger = logging.getLogger("EdgeDigitalTwinNode")


class EdgeDigitalTwinNode:
    """
    Physical Edge Digital Twin Node executing on shop-floor IPCs / micro-PCs.

    Orchestrates:
    - Real-time sensor acquisition & tool flank wear estimation (VB_hat).
    - Sub-millisecond local PPO/ONNX decision making for speed scaling and PM.
    - Motion control and override commands to GRBL CNC hardware.
    - Continuous tracking of Cumulative Disturbance Index (CDI(t)).
    - Event-driven emergency rescheduling triggers sent to Cloud Optimization Layer.
    """

    def __init__(
        self,
        machine_id: str = "M1",
        daq_interface: Optional[BreadboardDAQInterface] = None,
        grbl_controller: Optional[GRBLHardwareController] = None,
        onnx_engine: Optional[EdgeONNXInferenceEngine] = None,
        mqtt_client: Optional[EdgeMQTTClient] = None,
        alpha: float = 0.5,
        cdi_threshold: float = 0.12,
        baseline_schedule_path: Optional[str] = "data/benchmarks/cloud_baseline_schedule.json",
        w_max_um: float = 100.0,
        pm_wear_threshold_um: float = 80.0,
        p_cut_base_kw: float = 4.5,
    ):
        """
        Initialize the Edge Digital Twin Node.

        Args:
            machine_id: CNC machine identifier (e.g., 'M1').
            daq_interface: Hardware DAQ interface (emulator or real serial).
            grbl_controller: GRBL motion and override controller.
            onnx_engine: Pre-trained PPO policy inference engine.
            mqtt_client: Optional MQTT client for bidirectional Cloud synchronization.
            alpha: Weight for makespan deviation in CDI formula (1 - alpha for energy).
            cdi_threshold: Trigger threshold for emergency Cloud rescheduling (default 0.12).
            baseline_schedule_path: Path to baseline JSON schedule from Cloud solver.
            w_max_um: Flank wear limit (um) before tool failure.
            pm_wear_threshold_um: Tool wear threshold (um) triggering proactive replacement.
            p_cut_base_kw: Base cutting power in kW for this machine.
        """
        self.machine_id = machine_id
        self.alpha = float(alpha)
        self.cdi_threshold = float(cdi_threshold)
        self.w_max_um = float(w_max_um)
        self.pm_wear_threshold_um = float(pm_wear_threshold_um)
        self.p_cut_base_kw = float(p_cut_base_kw)

        # 1. Hardware DAQ Interface
        if daq_interface is not None:
            self.daq = daq_interface
        else:
            self.daq = BreadboardDAQInterface(mode="hardware_emulator", machine_id=machine_id)

        # 2. Dual-Sensor Fusion Tool Wear Estimator
        self.wear_estimator = RealTimeToolWearEstimator(machine_id=machine_id)

        # 3. GRBL Hardware Motion Controller
        if grbl_controller is not None:
            self.grbl = grbl_controller
        else:
            self.grbl = GRBLHardwareController(mode="grbl_emulator", esp32_interface=self.daq)

        # 4. ONNX Inference Engine (CPU-optimized)
        self.onnx_engine = onnx_engine
        if self.onnx_engine is None:
            default_onnx = Path("edge_layer/models/ppo_edge_policy.onnx")
            if default_onnx.exists():
                try:
                    self.onnx_engine = EdgeONNXInferenceEngine(str(default_onnx))
                except Exception as e:
                    logger.warning(f"Could not initialize ONNX engine from {default_onnx}: {e}")

        # 5. MQTT Industrial Communication
        self.mqtt_client = mqtt_client
        if self.mqtt_client is not None:
            self.attach_mqtt_client(self.mqtt_client)

        # 6. Disturbance and CDI Tracking
        self.cdi: float = 0.0
        self.trigger_cloud_reschedule: bool = False
        self.last_trigger_payload: Optional[RescheduleTriggerPayload] = None

        # Cumulative tracking at progress point t
        self.c_actual: float = 0.0  # Actual makespan (min)
        self.c_base: float = 0.0    # Baseline expected makespan (min)
        self.e_actual: float = 0.0  # Actual energy consumption (kWh)
        self.e_base: float = 0.0    # Baseline expected energy consumption (kWh)

        # Flank wear status (um)
        self.current_wear_um: float = 0.0
        self.estimated_wear_um: float = 0.0

        # Host Telemetry Heartbeat Watchdog (300 ms = 3 missed 10 Hz frames)
        self.telemetry_watchdog_timeout_ms: float = 300.0
        self.last_valid_telemetry_ts_ms: Optional[float] = None
        self.watchdog_tripped: bool = False

        # Continuous Background Telemetry Monitor (10 Hz) & Thread Synchronization
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_stop_event = threading.Event()
        self._state_lock = threading.RLock()
        self.latest_telemetry: Optional[Dict[str, Any]] = None
        self._last_rx_monotonic_s: Optional[float] = None
        self._safety_interlock_active: bool = False

        # Baseline schedule storage
        self.baseline_schedule: Optional[Dict[str, Any]] = None
        self.operations_schedule: List[Dict[str, Any]] = []
        self.completed_operations: List[Dict[str, Any]] = []

        if baseline_schedule_path and os.path.exists(baseline_schedule_path):
            self.load_baseline_schedule_file(baseline_schedule_path)

    def attach_mqtt_client(self, client: EdgeMQTTClient) -> None:
        """Attach MQTT client and subscribe to cloud schedule synchronization."""
        self.mqtt_client = client
        self.mqtt_client.subscribe_schedule_sync(self.update_baseline_schedule)

    def load_baseline_schedule_file(self, path: str) -> None:
        """Load baseline schedule from JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.baseline_schedule = data
            self.operations_schedule = data.get("operations_schedule", [])
            logger.info(
                f"[{self.machine_id}] Loaded baseline schedule with {len(self.operations_schedule)} ops."
            )
        except Exception as e:
            logger.warning(f"Failed to load baseline schedule from {path}: {e}")

    # ==========================================================================
    # Cumulative Disturbance Index (CDI) Calculation
    # ==========================================================================

    def compute_cdi(
        self,
        c_actual: Optional[float] = None,
        c_base: Optional[float] = None,
        e_actual: Optional[float] = None,
        e_base: Optional[float] = None,
    ) -> float:
        """
        Calculate Cumulative Disturbance Index (CDI(t)):
            CDI(t) = alpha * |C_actual(t) - C_base(t)| / C_base(t) +
                     (1 - alpha) * |E_actual(t) - E_base(t)| / E_base(t)

        Evaluates whether cumulative disturbances exceed cdi_threshold (0.12).
        If exceeded: sets trigger_cloud_reschedule = True and prepares trigger payload.

        Args:
            c_actual: Optional override for actual makespan (min).
            c_base: Optional override for baseline makespan (min).
            e_actual: Optional override for actual energy (kWh).
            e_base: Optional override for baseline energy (kWh).

        Returns:
            float: Computed CDI value.
        """
        if c_actual is not None:
            self.c_actual = float(c_actual)
        if c_base is not None:
            self.c_base = float(c_base)
        if e_actual is not None:
            self.e_actual = float(e_actual)
        if e_base is not None:
            self.e_base = float(e_base)

        c_base_safe = max(float(self.c_base), 1e-6)
        e_base_safe = max(float(self.e_base), 1e-6)

        delta_c_rel = abs(self.c_actual - self.c_base) / c_base_safe
        delta_e_rel = abs(self.e_actual - self.e_base) / e_base_safe

        cdi_val = self.alpha * delta_c_rel + (1.0 - self.alpha) * delta_e_rel
        self.cdi = float(cdi_val)

        if self.cdi > self.cdi_threshold:
            self.trigger_cloud_reschedule = True
            self.last_trigger_payload = RescheduleTriggerPayload(
                machine_id=self.machine_id,
                reason="CDI_EXCEEDED",
                cdi=self.cdi,
                cdi_threshold=self.cdi_threshold,
                current_wear_um=self.estimated_wear_um,
                timestamp=float(self.c_actual),
                details={
                    "c_actual": self.c_actual,
                    "c_base": self.c_base,
                    "e_actual": self.e_actual,
                    "e_base": self.e_base,
                    "delta_c_rel": delta_c_rel,
                    "delta_e_rel": delta_e_rel,
                    "completed_ops_count": len(self.completed_operations),
                },
            )

            # Publish via MQTT if connected
            if self.mqtt_client is not None and getattr(self.mqtt_client, "is_connected", False):
                self.mqtt_client.trigger_reschedule(
                    machine_id=self.machine_id,
                    reason="CDI_EXCEEDED",
                    cdi=self.cdi,
                    cdi_threshold=self.cdi_threshold,
                    current_wear_um=self.estimated_wear_um,
                    details=self.last_trigger_payload.details,
                )
        else:
            self.trigger_cloud_reschedule = False

        return self.cdi

    def update_baseline_schedule(self, new_schedule: Union[Dict[str, Any], CloudSyncPayload]) -> bool:
        """
        Synchronize with new baseline schedule received from Cloud Optimization Layer.

        Actions:
        1. Updates internal baseline schedule structures.
        2. Resets CDI(t) to 0.0.
        3. Clears trigger_cloud_reschedule flag (False).
        4. Re-anchors baseline tracking to current progress state.

        Args:
            new_schedule: Dict or CloudSyncPayload from Cloud.

        Returns:
            bool: True on successful update.
        """
        if hasattr(new_schedule, "model_dump"):
            data = new_schedule.model_dump()
        elif isinstance(new_schedule, dict):
            data = new_schedule
        else:
            raise ValueError(f"Invalid new_schedule format: {type(new_schedule)}")

        self.baseline_schedule = data
        if "operations_schedule" in data and data["operations_schedule"]:
            self.operations_schedule = data["operations_schedule"]
        elif "baseline_schedule" in data and data["baseline_schedule"]:
            self.operations_schedule = data["baseline_schedule"]

        # Re-anchor baseline metrics to current progress point to prevent lingering drift
        self.c_base = float(self.c_actual)
        self.e_base = float(self.e_actual)

        # Reset disturbance metrics
        self.cdi = 0.0
        self.trigger_cloud_reschedule = False
        self.last_trigger_payload = None

        logger.info(
            f"[{self.machine_id}] Baseline schedule updated from Cloud. CDI successfully reset to 0.0."
        )
        return True

    # ==========================================================================
    # Telemetry Heartbeat Watchdog & Safety Emergency Stop
    # ==========================================================================

    def _handle_safety_interlock(
        self,
        reason: str,
        details: Dict[str, Any],
        call_emergency_stop: bool = True,
        is_watchdog: bool = False,
    ) -> None:
        """
        Execute GRBL Stop Category 1 emergency stop ('!' Feed Hold over Serial)
        and emit Cloud reschedule trigger on sensor fault or telemetry watchdog trip.
        """
        with self._state_lock:
            if is_watchdog:
                self.watchdog_tripped = True
            self._safety_interlock_active = True
            self.trigger_cloud_reschedule = True
            self.last_trigger_payload = RescheduleTriggerPayload(
                machine_id=self.machine_id,
                reason=reason,
                cdi=self.cdi,
                cdi_threshold=self.cdi_threshold,
                current_wear_um=self.estimated_wear_um,
                timestamp=float(self.c_actual),
                details=details,
            )

        if call_emergency_stop:
            self.grbl.emergency_stop()

        if self.mqtt_client is not None and getattr(self.mqtt_client, "is_connected", False):
            self.mqtt_client.trigger_reschedule(
                machine_id=self.machine_id,
                reason=reason,
                cdi=self.cdi,
                cdi_threshold=self.cdi_threshold,
                current_wear_um=self.estimated_wear_um,
                details=details,
            )

    def _trigger_safety_emergency_stop(
        self,
        reason: str,
        details: Dict[str, Any],
        call_emergency_stop: bool = True,
        is_watchdog: bool = False,
    ) -> None:
        """Backward-compatible alias for _handle_safety_interlock()."""
        self._handle_safety_interlock(
            reason=reason,
            details=details,
            call_emergency_stop=call_emergency_stop,
            is_watchdog=is_watchdog,
        )

    def start_telemetry_monitor(self, poll_interval_s: float = 0.05) -> None:
        """
        Start continuous 10 Hz background telemetry and 300 ms watchdog monitoring thread.
        """
        with self._state_lock:
            if self._monitor_thread is not None and self._monitor_thread.is_alive():
                return
            self._monitor_stop_event.clear()
            self._safety_interlock_active = False
            self._last_rx_monotonic_s = time.monotonic()
            self._monitor_thread = threading.Thread(
                target=self._telemetry_monitor_loop,
                args=(float(poll_interval_s),),
                name=f"EdgeTelemetryMonitor-{self.machine_id}",
                daemon=True,
            )
            self._monitor_thread.start()

    def stop_telemetry_monitor(self) -> None:
        """Signal the background telemetry monitor thread to exit cleanly and join it."""
        self._monitor_stop_event.set()
        with self._state_lock:
            thread_ref = self._monitor_thread
        if thread_ref is not None and thread_ref.is_alive() and threading.current_thread() is not thread_ref:
            thread_ref.join(timeout=2.0)
        with self._state_lock:
            self._monitor_thread = None

    def _telemetry_monitor_loop(self, poll_interval_s: float = 0.05) -> None:
        """Background loop polling DAQ telemetry and enforcing 300 ms watchdog mid-cut."""
        while not self._monitor_stop_event.is_set():
            try:
                telemetry = self.daq.read_telemetry(dt_ms=100)
            except (TimeoutError, ConnectionError) as exc:
                if not self._safety_interlock_active:
                    logger.error(
                        f"[{self.machine_id}] Background monitor ESP32 failure "
                        f"({type(exc).__name__}: {exc}). Triggering GRBL emergency stop."
                    )
                    self._handle_safety_interlock(
                        reason="ESP32_WATCHDOG_TIMEOUT",
                        details={
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "timeout_ms": self.telemetry_watchdog_timeout_ms,
                        },
                        call_emergency_stop=True,
                        is_watchdog=True,
                    )
                self._monitor_stop_event.wait(poll_interval_s)
                continue
            except Exception as exc:
                logger.warning(f"[{self.machine_id}] Telemetry monitor unexpected error: {exc}")
                self._monitor_stop_event.wait(poll_interval_s)
                continue

            now_mono_s = time.monotonic()
            with self._state_lock:
                last_rx_s = self._last_rx_monotonic_s
                already_interlocked = self._safety_interlock_active

            if last_rx_s is not None:
                elapsed_ms = (now_mono_s - last_rx_s) * 1000.0
                if elapsed_ms > self.telemetry_watchdog_timeout_ms:
                    if not already_interlocked:
                        logger.error(
                            f"[{self.machine_id}] Background watchdog timeout: elapsed={elapsed_ms:.1f} ms "
                            f"> {self.telemetry_watchdog_timeout_ms:.1f} ms. Triggering GRBL emergency stop."
                        )
                        self._handle_safety_interlock(
                            reason="ESP32_WATCHDOG_TIMEOUT",
                            details={
                                "elapsed_ms": elapsed_ms,
                                "timeout_ms": self.telemetry_watchdog_timeout_ms,
                            },
                            call_emergency_stop=True,
                            is_watchdog=True,
                        )
                    self._monitor_stop_event.wait(poll_interval_s)
                    continue

            if telemetry is None or not isinstance(telemetry, dict):
                self._monitor_stop_event.wait(poll_interval_s)
                continue

            packet_ts_ms = telemetry.get("ts_ms")
            if isinstance(packet_ts_ms, (int, float)) and self.last_valid_telemetry_ts_ms is not None:
                ts_delta_ms = float(packet_ts_ms) - float(self.last_valid_telemetry_ts_ms)
                if ts_delta_ms < 0.0 or ts_delta_ms > self.telemetry_watchdog_timeout_ms:
                    if not already_interlocked:
                        self._handle_safety_interlock(
                            reason="ESP32_WATCHDOG_TIMEOUT",
                            details={
                                "elapsed_ms": ts_delta_ms,
                                "last_valid_telemetry_ts_ms": self.last_valid_telemetry_ts_ms,
                                "current_time_ms": float(packet_ts_ms),
                                "timeout_ms": self.telemetry_watchdog_timeout_ms,
                            },
                            call_emergency_stop=True,
                            is_watchdog=True,
                        )
                    self._monitor_stop_event.wait(poll_interval_s)
                    continue

            with self._state_lock:
                self._last_rx_monotonic_s = now_mono_s
                self.latest_telemetry = telemetry

            if not telemetry.get("sensor_ok", True) or telemetry.get("fault_latched", False):
                if not already_interlocked:
                    self._handle_safety_interlock(
                        reason="ESP32_SENSOR_FAULT",
                        details={
                            "sensor_ok": telemetry.get("sensor_ok", False),
                            "fault_latched": telemetry.get("fault_latched", False),
                            "i2c_err_cnt": telemetry.get("i2c_err_cnt", 0),
                        },
                        call_emergency_stop=True,
                        is_watchdog=False,
                    )
            else:
                with self._state_lock:
                    if isinstance(packet_ts_ms, (int, float)):
                        self.last_valid_telemetry_ts_ms = float(packet_ts_ms)
                    self.watchdog_tripped = False
                    self._safety_interlock_active = False
                    speed_lvl = int(getattr(self.grbl, "current_speed_level", 1))
                    self.estimated_wear_um = self.wear_estimator.estimate_wear(
                        telemetry,
                        speed_level=speed_lvl,
                    )

            self._monitor_stop_event.wait(poll_interval_s)

    def check_telemetry_watchdog(self, current_time_ms: float) -> bool:
        """
        Verify ESP32 10 Hz telemetry heartbeat against telemetry_watchdog_timeout_ms (300 ms).

        If elapsed time since the last valid telemetry exceeds telemetry_watchdog_timeout_ms
        (3 missed consecutive 10 Hz frames) or ts_ms regresses due to unannounced MCU reset,
        immediately commands GRBL emergency_stop() ('!' Feed Hold) directly over Serial.

        Args:
            current_time_ms: Timestamp (ms) of current observation or monotonic clock.

        Returns:
            bool: True if watchdog tripped, False if heartbeat is healthy.
        """
        if self.last_valid_telemetry_ts_ms is None:
            return False

        elapsed_ms = float(current_time_ms) - float(self.last_valid_telemetry_ts_ms)
        if elapsed_ms < 0.0 or elapsed_ms > self.telemetry_watchdog_timeout_ms:
            logger.error(
                f"[{self.machine_id}] Host telemetry watchdog tripped: gap={elapsed_ms:.1f} ms "
                f"(timeout={self.telemetry_watchdog_timeout_ms:.1f} ms). Issuing GRBL emergency_stop()."
            )
            self._trigger_safety_emergency_stop(
                reason="TELEMETRY_WATCHDOG_TIMEOUT",
                details={
                    "elapsed_ms": elapsed_ms,
                    "last_valid_telemetry_ts_ms": self.last_valid_telemetry_ts_ms,
                    "current_time_ms": float(current_time_ms),
                    "timeout_ms": self.telemetry_watchdog_timeout_ms,
                },
                call_emergency_stop=True,
                is_watchdog=True,
            )
            return True
        return False

    # ==========================================================================
    # Action Decoding and Local Control Inference
    # ==========================================================================

    @staticmethod
    def decode_action(
        action: int,
        num_jobs: int = 10,
        num_machines: int = 6,
        num_speed_levels: int = 3,
        num_pm_options: int = 2,
    ) -> Tuple[int, int, int, int]:
        """
        Decode integer action into (job_idx, machine_idx, speed_level, pm_flag).
        Consistent with sim_env.fjsp_gym_env.GreenFJSPEnv encoding scheme.
        """
        machine_stride = num_speed_levels * num_pm_options
        job_stride = num_machines * machine_stride

        job_idx, rem = divmod(action, job_stride)
        machine_idx, rem = divmod(rem, machine_stride)
        speed_idx, pm_flag = divmod(rem, num_pm_options)
        speed_level = speed_idx + 1

        return job_idx, machine_idx, speed_level, pm_flag

    def infer_local_control(
        self,
        obs: Optional[np.ndarray] = None,
        action_mask: Optional[np.ndarray] = None,
        op_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Infer local speed level and PM replacement decision.
        Utilizes Edge ONNX policy when available; otherwise applies reactive local heuristic.
        """
        action_idx: Optional[int] = None
        speed_level: int = 1
        pm_flag: int = 0

        # If ONNX engine and environment inputs are provided, execute sub-ms policy inference
        if self.onnx_engine is not None and obs is not None and action_mask is not None:
            action_idx = self.onnx_engine.predict_action(obs, action_mask)
            _, _, speed_level, pm_flag = self.decode_action(action_idx)
        else:
            # Local edge surrogate decision based on wear state and schedule delay
            if self.estimated_wear_um >= self.pm_wear_threshold_um:
                pm_flag = 1

            base_speed = op_data.get("speed_level", 1) if op_data else 1
            # If actual makespan is lagging behind baseline, attempt speed boost
            if self.c_actual > self.c_base and self.estimated_wear_um < 75.0:
                speed_level = min(3, base_speed + 1)
            elif self.estimated_wear_um >= 75.0:
                speed_level = 1  # Conserve worn tool
            else:
                speed_level = base_speed

        return {
            "action_idx": action_idx,
            "speed_level": speed_level,
            "pm_flag": pm_flag,
        }

    # ==========================================================================
    # Operation Dispatch and Physical Twin Simulation
    # ==========================================================================

    def dispatch_operation(
        self,
        op_data: Dict[str, Any],
        disturbance_factor: float = 1.0,
        is_breakdown: bool = False,
        breakdown_delay_min: float = 15.0,
        obs: Optional[np.ndarray] = None,
        action_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Dispatch a machining operation on the Edge Node.

        Steps:
        1. Infer optimal speed level s in {1, 2, 3} and PM replacement via ONNX / local logic.
        2. If PM required: command PM state (SSR STANDBY), reset tool wear.
        3. Set GRBL real-time override (s in {1, 2, 3}).
        4. Send operation G-code (G1 X.. Y.. Z.. F.. S..) to GRBL controller.
        5. Read DAQ telemetry, update tool wear estimate via dual-sensor fusion.
        6. Compute Cumulative Disturbance Index CDI(t).
        7. If CDI(t) <= 0.12: Edge handles disturbance locally.
           If CDI(t) > 0.12: Edge signals emergency reschedule trigger to Cloud.

        Args:
            op_data: Operation metadata (t_base_min, energy_kwh, feed, spindle, coordinates).
            disturbance_factor: Multiplier simulating tool wear or hard workpiece (e.g. 1.10, 1.25).
            is_breakdown: True to simulate sudden machine breakdown event.
            breakdown_delay_min: Emergency repair downtime in minutes.
            obs: Optional observation vector for ONNX policy.
            action_mask: Optional action mask for ONNX policy.

        Returns:
            Dict containing execution summary and CDI status.
        """
        # Step 1: Infer control action
        decision = self.infer_local_control(obs, action_mask, op_data)
        speed_level = decision["speed_level"]
        pm_done = False

        # Step 2: Handle Preventive Tool Replacement
        if decision["pm_flag"] == 1:
            pm_done = True
            self.grbl.set_machine_state("PM")
            pm_downtime = 5.0  # minutes
            self.c_actual += pm_downtime
            self.c_base += pm_downtime
            # Tool wear reset
            self.current_wear_um = 0.0
            self.estimated_wear_um = 0.0
            self.wear_estimator.reset(0.0)
            self.daq.set_operating_conditions(speed_level=1, true_wear_um=0.0)

        # Step 3: Apply GRBL Speed Override
        override_info = self.grbl.set_speed_override(speed_level)

        # Step 4: Dispatch G-code to GRBL
        self.grbl.set_machine_state("CUTTING")
        x_target = float(op_data.get("x", 50.0))
        y_target = float(op_data.get("y", 50.0))
        z_target = float(op_data.get("z", -2.0))
        feed_base = float(op_data.get("feed", 500.0))
        spindle_base = float(op_data.get("spindle", 10000.0))

        gcode_cmd = self.grbl.format_operation_gcode(
            x=x_target,
            y=y_target,
            z=z_target,
            feed=feed_base * self.grbl.speed_factor,
            spindle=spindle_base * self.grbl.speed_factor,
        )
        gcode_resp = self.grbl.send_gcode(gcode_cmd)

        # Step 5: Physical Twin Simulation & Metric Updates
        base_speed = int(op_data.get("speed_level", 1))
        v_s_base = self.grbl.SPEED_FACTORS.get(base_speed, 1.0)
        v_s = self.grbl.speed_factor

        t_base_op = float(op_data.get("t_proc_min", op_data.get("t_base_min", 4.0)))

        alpha_p = 1.5
        gamma = 0.25

        if "energy_kwh" in op_data:
            e_base_op = float(op_data["energy_kwh"])
        else:
            base_p = self.p_cut_base_kw * (v_s_base**alpha_p)
            e_base_op = base_p * (t_base_op / 60.0)

        # Actual duration adjusted for speed level choice and physical disturbance
        t_actual = (t_base_op * (v_s_base / v_s)) * float(disturbance_factor)
        if is_breakdown:
            t_actual += float(breakdown_delay_min)

        # Cutting power and energy calculation
        wear_ratio = self.current_wear_um / self.w_max_um
        power_kw = (
            self.p_cut_base_kw
            * (v_s**alpha_p)
            * (1.0 + gamma * wear_ratio)
        )
        # Energy scales with power and duration
        e_actual = power_kw * (t_actual / 60.0)

        # Flank wear progression
        delta_wear = 0.8 * (v_s**2.0) * (t_actual) * float(disturbance_factor)
        self.current_wear_um = min(self.w_max_um, self.current_wear_um + delta_wear)

        def _package_step_result(
            cdi_value: float,
            sensor_fault_stop: bool = False,
            watchdog_tripped: bool = False,
        ) -> Dict[str, Any]:
            res = {
                "op_id": op_data.get("op_id", "OP_DEFAULT"),
                "speed_level": speed_level,
                "speed_factor": v_s,
                "pm_done": pm_done,
                "t_actual_min": t_actual,
                "e_actual_kwh": e_actual,
                "c_actual": self.c_actual,
                "c_base": self.c_base,
                "e_actual": self.e_actual,
                "e_base": self.e_base,
                "cdi": cdi_value,
                "trigger_cloud_reschedule": self.trigger_cloud_reschedule,
                "sensor_fault_stop": sensor_fault_stop,
                "watchdog_tripped": watchdog_tripped,
                "estimated_wear_um": self.estimated_wear_um,
                "override_info": override_info,
                "gcode_sent": gcode_cmd,
                "gcode_resp": gcode_resp,
            }
            self.completed_operations.append(res)
            return res

        # Update DAQ hardware emulator & guard with 300 ms Host Telemetry Watchdog
        self.daq.set_operating_conditions(speed_level=speed_level, true_wear_um=self.current_wear_um)
        with self._state_lock:
            monitor_running = self._monitor_thread is not None and self._monitor_thread.is_alive()
            cached_telemetry = self.latest_telemetry if monitor_running else None
            cached_rx_s = self._last_rx_monotonic_s if monitor_running else None
            cached_watchdog_tripped = self.watchdog_tripped if monitor_running else False

        if monitor_running:
            if cached_watchdog_tripped:
                return _package_step_result(self.cdi, sensor_fault_stop=False, watchdog_tripped=True)

            if cached_rx_s is not None:
                rx_age_ms = (time.monotonic() - cached_rx_s) * 1000.0
                if rx_age_ms > self.telemetry_watchdog_timeout_ms:
                    self._handle_safety_interlock(
                        reason="ESP32_WATCHDOG_TIMEOUT",
                        details={
                            "elapsed_ms": rx_age_ms,
                            "timeout_ms": self.telemetry_watchdog_timeout_ms,
                            "op_id": op_data.get("op_id", "OP_DEFAULT"),
                        },
                        call_emergency_stop=True,
                        is_watchdog=True,
                    )
                    return _package_step_result(self.cdi, sensor_fault_stop=False, watchdog_tripped=True)

            if cached_telemetry is not None:
                if not cached_telemetry.get("sensor_ok", True) or cached_telemetry.get("fault_latched", False):
                    return _package_step_result(self.cdi, sensor_fault_stop=True, watchdog_tripped=False)

                # Monitor already runs 10 Hz EKF updates under _state_lock
                with self._state_lock:
                    self.estimated_wear_um = float(self.wear_estimator.x_est)
            else:
                monitor_running = False

        if not monitor_running:
            t_read_start_ms = time.monotonic() * 1000.0
            try:
                telemetry = self.daq.read_telemetry(dt_ms=100)
            except (TimeoutError, ConnectionError) as exc:
                read_elapsed_ms = (time.monotonic() * 1000.0) - t_read_start_ms
                logger.error(
                    f"[{self.machine_id}] ESP32 telemetry failure ({type(exc).__name__}: {exc}). "
                    "Host watchdog executing GRBL Feed Hold ('!') over Serial."
                )
                self._trigger_safety_emergency_stop(
                    reason="TELEMETRY_WATCHDOG_TIMEOUT",
                    details={
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "read_elapsed_ms": read_elapsed_ms,
                        "timeout_ms": self.telemetry_watchdog_timeout_ms,
                        "op_id": op_data.get("op_id", "OP_DEFAULT"),
                    },
                    call_emergency_stop=True,
                    is_watchdog=True,
                )
                return _package_step_result(self.cdi, sensor_fault_stop=False, watchdog_tripped=True)

            read_elapsed_ms = (time.monotonic() * 1000.0) - t_read_start_ms
            if read_elapsed_ms > self.telemetry_watchdog_timeout_ms:
                logger.error(
                    f"[{self.machine_id}] Telemetry read duration {read_elapsed_ms:.1f} ms exceeded "
                    f"{self.telemetry_watchdog_timeout_ms:.1f} ms watchdog limit. Issuing GRBL Feed Hold."
                )
                self._trigger_safety_emergency_stop(
                    reason="TELEMETRY_WATCHDOG_TIMEOUT",
                    details={
                        "read_elapsed_ms": read_elapsed_ms,
                        "timeout_ms": self.telemetry_watchdog_timeout_ms,
                        "op_id": op_data.get("op_id", "OP_DEFAULT"),
                    },
                    call_emergency_stop=True,
                    is_watchdog=True,
                )
                return _package_step_result(self.cdi, sensor_fault_stop=False, watchdog_tripped=True)

            packet_ts_ms = telemetry.get("ts_ms")
            current_ts_ms = (
                float(packet_ts_ms)
                if isinstance(packet_ts_ms, (int, float))
                else (time.monotonic() * 1000.0)
            )
            if self.check_telemetry_watchdog(current_ts_ms):
                if self.last_trigger_payload is not None:
                    self.last_trigger_payload.details["op_id"] = op_data.get("op_id", "OP_DEFAULT")
                return _package_step_result(self.cdi, sensor_fault_stop=False, watchdog_tripped=True)

            if not telemetry.get("sensor_ok", True) or telemetry.get("fault_latched", False):
                self._trigger_safety_emergency_stop(
                    reason="SENSOR_FAULT",
                    details={
                        "sensor_ok": telemetry.get("sensor_ok", False),
                        "fault_latched": telemetry.get("fault_latched", False),
                        "i2c_err_cnt": telemetry.get("i2c_err_cnt", 0),
                        "op_id": op_data.get("op_id", "OP_DEFAULT"),
                    },
                    call_emergency_stop=True,
                    is_watchdog=False,
                )
                return _package_step_result(self.cdi, sensor_fault_stop=True, watchdog_tripped=False)

            with self._state_lock:
                self.last_valid_telemetry_ts_ms = current_ts_ms
                self.watchdog_tripped = False
                self.estimated_wear_um = self.wear_estimator.estimate_wear(telemetry, speed_level=speed_level)

        # Step 6: Cumulative Tracking and CDI Computation
        self.c_actual += t_actual
        self.c_base += t_base_op
        self.e_actual += e_actual
        self.e_base += e_base_op

        cdi_val = self.compute_cdi()

        # Step 7: Return machine to standby
        self.grbl.set_machine_state("STANDBY")

        # Step 8: Package record
        return _package_step_result(cdi_val, sensor_fault_stop=False, watchdog_tripped=False)
