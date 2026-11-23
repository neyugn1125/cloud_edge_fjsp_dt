"""
Test Suite for Task 4.2: GRBL Hardware Controller, Cumulative Disturbance Index (CDI(t)), and ONNX Closed-Loop Edge Twin.

Verifies:
1. GRBLHardwareController:
   - Emulation and serial communication handling.
   - G-code command transmission (G1 X.. Y.. Z.. F.. S..) with motion parsing.
   - Real-time Speed Overrides (s in {1, 2, 3}):
     * s = 1 (v = 1.00): 100% Feed & Spindle overrides (0x90, 0x99).
     * s = 2 (v = 1.25): +25% Speed scaling via real-time override bytes (0x91, 0x9A).
     * s = 3 (v = 1.50): +50% Speed scaling with ESP32 SSR relay switching (CMD:CUTTING).
   - ESP32 SSR relay states (CMD:CUTTING, CMD:STANDBY, CMD:OFF) and emergency stop.
2. EdgeDigitalTwinNode:
   - Cumulative Disturbance Index CDI(t) calculation:
     CDI(t) = alpha * |C_actual(t) - C_base(t)| / C_base(t) + (1 - alpha) * |E_actual(t) - E_base(t)| / E_base(t)
   - Mild disturbance (+10%): CDI(t) < 0.12, Edge ONNX adapts locally without calling Cloud.
   - Severe disturbance (> 15% deviation / breakdown): CDI(t) > 0.12, triggers emergency reschedule.
   - Cloud schedule synchronization: resets CDI(t) = 0.0 and trigger_cloud_reschedule = False.
3. Integration with EdgeONNXInferenceEngine (ppo_edge_policy.onnx) and BreadboardDAQInterface.
"""

from pathlib import Path
from typing import Any, Dict
import numpy as np
import pytest

from edge_layer.breadboard_daq import BreadboardDAQInterface
from edge_layer.edge_twin_node import EdgeDigitalTwinNode
from edge_layer.grbl_controller import GRBLHardwareController
from edge_layer.onnx_exporter import EdgeONNXInferenceEngine
from utils.mqtt_schemas import RescheduleTriggerPayload


# ==============================================================================
# 1. GRBL Hardware Controller Unit Tests
# ==============================================================================

def test_grbl_controller_modes_and_gcode_execution():
    """
    Test GRBL controller initialization, G-code transmission, and motion parsing in emulation mode.
    """
    controller = GRBLHardwareController(mode="grbl_emulator")
    assert controller.is_connected is True

    # 1. Initial Status Query
    st0 = controller.query_status()
    assert st0["state"] == "Idle"
    assert st0["mpos"] == [0.0, 0.0, 0.0]
    assert st0["ov_feed"] == 100
    assert st0["ov_spindle"] == 100

    # 2. Format and Send G-code Cutting Command
    gcode = controller.format_operation_gcode(x=25.4, y=10.2, z=-1.5, feed=600.0, spindle=12000.0)
    assert gcode == "G1 X25.400 Y10.200 Z-1.500 F600.0 S12000"

    resp = controller.send_gcode(gcode)
    assert resp.strip() == "ok"
    assert len(controller.command_history) == 1

    # 3. Verify Emulator Motion State Update
    st1 = controller.query_status()
    assert st1["state"] == "Run"
    assert st1["mpos"] == [25.4, 10.2, -1.5]
    assert st1["feed"] == 600.0
    assert st1["spindle"] == 12000.0

    # 4. End Program
    controller.send_gcode("M30")
    st2 = controller.query_status()
    assert st2["state"] == "Idle"

    # 5. Serial connection error handling
    with pytest.raises(ConnectionError):
        # Invalid serial port to verify robust exception handling
        GRBLHardwareController(mode="real_serial", port="/dev/non_existent_tty_device_999")


def test_grbl_controller_real_time_speed_overrides_and_ssr():
    """
    Test GRBL v1.1 real-time override bytes (0x90, 0x91, 0x99, 0x9A) and SSR commands for s in {1, 2, 3}.
    """
    controller = GRBLHardwareController(mode="grbl_emulator")

    # 1. Speed Level s = 1 (v = 1.00, 100%)
    ov1 = controller.set_speed_override(1)
    assert ov1["speed_level"] == 1
    assert ov1["speed_factor"] == 1.00
    assert ov1["feed_override_pct"] == 100
    assert ov1["spindle_override_pct"] == 100
    assert ov1["sent_bytes"] == [0x90, 0x99]
    assert ov1["ssr_cmd"] is None

    # 2. Speed Level s = 2 (v = 1.25, 125%: reset to 100%, +2x10%, +5x1% -> 16 bytes)
    ov2 = controller.set_speed_override(2)
    assert ov2["speed_level"] == 2
    assert ov2["speed_factor"] == 1.25
    assert ov2["feed_override_pct"] == 125
    assert ov2["spindle_override_pct"] == 125
    assert ov2["sent_bytes"] == [
        0x90, 0x91, 0x91, 0x93, 0x93, 0x93, 0x93, 0x93,
        0x99, 0x9A, 0x9A, 0x9C, 0x9C, 0x9C, 0x9C, 0x9C,
    ]
    assert ov2["ssr_cmd"] is None

    # 3. Speed Level s = 3 (v = 1.50, 150%: reset to 100%, +5x10% -> 12 bytes + SSR CMD:CUTTING)
    ov3 = controller.set_speed_override(3)
    assert ov3["speed_level"] == 3
    assert ov3["speed_factor"] == 1.50
    assert ov3["feed_override_pct"] == 150
    assert ov3["spindle_override_pct"] == 150
    assert ov3["sent_bytes"] == [
        0x90, 0x91, 0x91, 0x91, 0x91, 0x91,
        0x99, 0x9A, 0x9A, 0x9A, 0x9A, 0x9A,
    ]
    assert ov3["ssr_cmd"] == "CMD:CUTTING"
    assert controller.current_state == "CUTTING"

    # 4. SSR State Switching
    assert controller.set_machine_state("STANDBY") == "CMD:STANDBY"
    assert controller.set_machine_state("OFF") == "CMD:OFF"
    assert controller.set_machine_state("PM") == "CMD:STANDBY"

    # 5. Emergency Stop
    alarm = controller.emergency_stop()
    assert "EMERGENCY STOP" in alarm
    assert controller.grbl_state == "Hold"
    assert controller.current_state == "OFF"
    assert controller.last_ssr_cmd == "CMD:OFF"

    # 6. Invalid speed level
    with pytest.raises(ValueError):
        controller.set_speed_override(4)


# ==============================================================================
# 2. Cumulative Disturbance Index (CDI(t)) Unit Tests
# ==============================================================================

def test_cdi_calculation_mild_disturbance_autonomous_edge():
    """
    Assert: When tool wear / duration disturbance is mild (+10%),
    CDI(t) remains < 0.12 and Edge Node adapts locally without triggering Cloud reschedule.
    """
    node = EdgeDigitalTwinNode(alpha=0.5, cdi_threshold=0.12)

    # 1. Zero initial deviation
    cdi_zero = node.compute_cdi(c_actual=10.0, c_base=10.0, e_actual=1.0, e_base=1.0)
    assert cdi_zero == 0.0
    assert node.trigger_cloud_reschedule is False
    assert node.last_trigger_payload is None

    # 2. Mild disturbance (+10% on both makespan and energy)
    # CDI = 0.5 * |11.0 - 10.0| / 10.0 + 0.5 * |1.10 - 1.0| / 1.0 = 0.5 * 0.10 + 0.5 * 0.10 = 0.10
    cdi_mild = node.compute_cdi(c_actual=11.0, c_base=10.0, e_actual=1.10, e_base=1.0)
    assert pytest.approx(cdi_mild, abs=1e-5) == 0.10
    assert cdi_mild < 0.12
    assert node.trigger_cloud_reschedule is False
    assert node.last_trigger_payload is None

    # 3. Autonomous operation dispatch with mild disturbance (+8% to +10%)
    op_data = {
        "op_id": "J1_O1",
        "t_proc_min": 4.0,
        "speed_level": 1,
        "energy_kwh": 0.30,
        "feed": 500.0,
        "spindle": 10000.0,
    }
    # Reset tracking
    node.c_actual = 0.0
    node.c_base = 0.0
    node.e_actual = 0.0
    node.e_base = 0.0

    step_res = node.dispatch_operation(op_data, disturbance_factor=1.08)
    assert step_res["cdi"] < 0.12
    assert step_res["trigger_cloud_reschedule"] is False
    assert node.trigger_cloud_reschedule is False

    # Check G-code dispatched
    assert "G1" in step_res["gcode_sent"]
    assert step_res["gcode_resp"].strip() == "ok"


def test_cdi_calculation_severe_disturbance_triggers_reschedule_and_cloud_sync_reset():
    """
    Assert: When large disturbance occurs (> 15% deviation or machine breakdown),
    CDI(t) exceeds 0.12, triggers emergency reschedule to Cloud, and after receiving
    new baseline schedule from Cloud, CDI(t) is reset to 0.0.
    """
    node = EdgeDigitalTwinNode(alpha=0.5, cdi_threshold=0.12)

    # 1. Severe Disturbance (+25% deviation due to ultra-hard workpiece)
    # CDI = 0.5 * |12.5 - 10.0| / 10.0 + 0.5 * |1.25 - 1.0| / 1.0 = 0.25 > 0.12
    cdi_severe = node.compute_cdi(c_actual=12.5, c_base=10.0, e_actual=1.25, e_base=1.0)
    assert pytest.approx(cdi_severe, abs=1e-5) == 0.25
    assert cdi_severe > 0.12
    assert node.trigger_cloud_reschedule is True

    # Verify RescheduleTriggerPayload structure
    payload = node.last_trigger_payload
    assert isinstance(payload, RescheduleTriggerPayload)
    assert payload.machine_id == "M1"
    assert payload.reason == "CDI_EXCEEDED"
    assert payload.cdi == pytest.approx(0.25, abs=1e-4)
    assert payload.cdi_threshold == 0.12
    assert payload.details["c_actual"] == 12.5
    assert payload.details["c_base"] == 10.0

    # 2. Cloud Rescheduling Synchronization: Cloud sends updated schedule
    cloud_sync_data = {
        "summary": {"makespan_min": 78.5, "total_energy_kwh": 24.2},
        "baseline_cmax": 78.5,
        "baseline_energy_kwh": 24.2,
        "operations_schedule": [
            {"op_id": "J1_O2", "machine_id": "M1", "speed_level": 2, "t_proc_min": 3.2}
        ],
    }

    sync_success = node.update_baseline_schedule(cloud_sync_data)
    assert sync_success is True

    # 3. Assert CDI is reset to 0.0 and reschedule flag is cleared
    assert node.cdi == 0.0
    assert node.trigger_cloud_reschedule is False
    assert node.last_trigger_payload is None
    assert node.c_base == node.c_actual
    assert node.e_base == node.e_actual


# ==============================================================================
# 3. Integrated Closed-Loop Edge Digital Twin Node Tests
# ==============================================================================

def test_edge_digital_twin_full_closed_loop_with_onnx_and_grbl():
    """
    Test full closed-loop execution integrating:
    - Real ONNX model (ppo_edge_policy.onnx)
    - GRBLHardwareController
    - BreadboardDAQInterface
    - Tool wear estimation and CDI tracking
    """
    onnx_path = Path("edge_layer/models/ppo_edge_policy.onnx")
    assert onnx_path.exists(), "Pre-trained ONNX policy must exist for edge integration test."

    onnx_engine = EdgeONNXInferenceEngine(str(onnx_path))
    daq = BreadboardDAQInterface(mode="hardware_emulator", machine_id="M1")
    grbl = GRBLHardwareController(mode="grbl_emulator", esp32_interface=daq)

    node = EdgeDigitalTwinNode(
        machine_id="M1",
        daq_interface=daq,
        grbl_controller=grbl,
        onnx_engine=onnx_engine,
        alpha=0.5,
        cdi_threshold=0.12,
    )

    # 1. Step 1: Normal operation with ONNX inference and mild wear
    obs_sample = np.zeros(87, dtype=np.float32)
    mask_sample = np.ones(360, dtype=bool)

    op_step1 = {
        "op_id": "J2_O1",
        "t_proc_min": 3.0,
        "speed_level": 1,
        "energy_kwh": 0.225,
        "feed": 500.0,
        "spindle": 10000.0,
        "x": 30.0,
        "y": 45.0,
        "z": -2.0,
    }

    res1 = node.dispatch_operation(
        op_step1,
        disturbance_factor=1.05,
        obs=obs_sample,
        action_mask=mask_sample,
    )

    assert res1["speed_level"] in (1, 2, 3)
    assert res1["gcode_resp"].strip() == "ok"
    assert res1["estimated_wear_um"] >= 0.0
    assert grbl.current_mpos == [30.0, 45.0, -2.0]

    # 2. Step 2: Severe disturbance with machine breakdown (+15.0 min delay)
    op_step2 = {
        "op_id": "J2_O2",
        "t_proc_min": 2.5,
        "speed_level": 2,
        "energy_kwh": 0.25,
        "feed": 600.0,
        "spindle": 12000.0,
        "x": 60.0,
        "y": 80.0,
        "z": -3.0,
    }

    res2 = node.dispatch_operation(
        op_step2,
        disturbance_factor=1.30,
        is_breakdown=True,
        breakdown_delay_min=15.0,
    )

    # Assert CDI exceeds threshold and triggers reschedule
    assert res2["cdi"] > 0.12
    assert res2["trigger_cloud_reschedule"] is True
    assert node.trigger_cloud_reschedule is True
    assert node.last_trigger_payload is not None

    # 3. Step 3: Receive Cloud baseline update and verify reset
    node.update_baseline_schedule(
        {"baseline_cmax": node.c_actual, "baseline_energy_kwh": node.e_actual}
    )
    assert node.cdi == 0.0
    assert node.trigger_cloud_reschedule is False


def test_grbl_real_serial_query_status_parsing_and_ov_sync():
    """
    Verify GRBLHardwareController.query_status() in real_serial mode skips non-frame
    lines (such as 'ok') and parses State, MPos, FS, and Ov:<feed>,<rapid>,<spindle>.
    Also verify set_speed_override updates internal override pcts in real_serial mode.
    """
    class MockGRBLSerial:
        def __init__(self, responses):
            self._responses = list(responses)
            self.written = []

        def write(self, data):
            self.written.append(data)

        def flush(self):
            pass

        def readline(self):
            return self._responses.pop(0) if self._responses else b""

        def close(self):
            pass

    grbl = GRBLHardwareController(mode="grbl_emulator")
    grbl.mode = "real_serial"
    grbl.serial_conn = MockGRBLSerial([
        b"ok\r\n",
        b"<Run|MPos:12.500,34.000,-1.500|FS:625,12500|Ov:125,100,125>\r\n",
    ])

    st = grbl.query_status()
    assert st["state"] == "Run"
    assert st["mpos"] == [12.5, 34.0, -1.5]
    assert st["feed"] == pytest.approx(625.0)
    assert st["spindle"] == pytest.approx(12500.0)
    assert st["ov_feed"] == 125
    assert st["ov_spindle"] == 125

    # Verify set_speed_override syncs shadow state in real_serial mode too
    res_ov = grbl.set_speed_override(3)
    assert res_ov["feed_override_pct"] == 150
    assert res_ov["spindle_override_pct"] == 150
    assert grbl.feed_override_pct == 150
    assert grbl.spindle_override_pct == 150


def test_edge_node_sensor_fault_triggers_emergency_stop():
    """
    Verify EdgeDigitalTwinNode.dispatch_operation() triggers immediate emergency stop
    (Feed Hold before SSR cutoff), flags sensor_fault_stop=True and Cloud reschedule,
    and preserves wear_estimator.x_est when DAQ reports sensor_ok=False.
    """
    class MockGRBLSerial:
        def __init__(self):
            self.writes = []
            self._pending_status = False

        def write(self, data):
            self.writes.append(data)
            if data == b"?":
                self._pending_status = True

        def flush(self):
            pass

        def readline(self):
            if self._pending_status:
                self._pending_status = False
                return b"<Hold:0|MPos:0.000,0.000,0.000|FS:0,10000|Ov:100,100,100>\r\n"
            return b"ok\r\n"

        def close(self):
            pass

    daq = BreadboardDAQInterface(mode="esp32_emulator")
    daq_cmds = []
    orig_send_cmd = daq.send_command

    def tracking_send_cmd(cmd):
        daq_cmds.append(cmd)
        return orig_send_cmd(cmd)

    daq.send_command = tracking_send_cmd
    daq.read_telemetry = lambda dt_ms=100: {
        "ts_ms": 2000,
        "v_bus": 0.0,
        "i_amp": 0.0,
        "p_raw_w": 0.0,
        "p_filt_w": 0.0,
        "vib_rms_g": -1.0,
        "ssr_state": "OFF",
        "sensor_ok": False,
        "fault_latched": True,
        "i2c_err_cnt": 3,
    }

    grbl = GRBLHardwareController(mode="grbl_emulator", esp32_interface=daq)
    grbl.mode = "real_serial"
    mock_serial = MockGRBLSerial()
    grbl.serial_conn = mock_serial

    node = EdgeDigitalTwinNode(
        machine_id="M0",
        daq_interface=daq,
        grbl_controller=grbl,
        cdi_threshold=0.12,
    )
    node.wear_estimator.reset(32.5)
    node.estimated_wear_um = 32.5

    op = {
        "op_id": "J1_O1",
        "t_proc_min": 2.0,
        "speed_level": 1,
        "energy_kwh": 0.15,
        "feed": 500.0,
        "spindle": 10000.0,
        "x": 10.0,
        "y": 15.0,
        "z": -1.0,
    }

    res = node.dispatch_operation(op)
    assert res["sensor_fault_stop"] is True
    assert res["trigger_cloud_reschedule"] is True
    assert node.trigger_cloud_reschedule is True
    assert node.last_trigger_payload is not None
    assert node.last_trigger_payload.reason == "SENSOR_FAULT"
    assert node.wear_estimator.x_est == pytest.approx(32.5)
    assert res["estimated_wear_um"] == pytest.approx(32.5)

    # Verify Feed Hold ('!') was sent before Soft Reset ('\x18') and CMD:OFF
    assert b"!" in mock_serial.writes
    assert b"\x18" in mock_serial.writes
    assert mock_serial.writes.index(b"!") < mock_serial.writes.index(b"\x18")
    assert "CMD:OFF" in daq_cmds
    assert grbl.emergency_stop_log == [
        "FEED_HOLD_SENT",
        "AXES_STOPPED_VERIFIED",
        "SSR_OFF_SENT",
        "SOFT_RESET_SENT",
    ]


def test_grbl_emergency_stop_waits_for_hold0_before_ssr_and_reset():
    """
    Verify Stop Category 1 sequence in GRBLHardwareController.emergency_stop():
    1. Immediately sends Feed Hold ('!') and flushes.
    2. Polls '?' while GRBL reports Hold:1 with non-zero feed (axes still decelerating).
    3. Only after GRBL reports Hold:0 (or feed <= 1e-3 / Idle) does it send CMD:OFF to ESP32
       and then Soft Reset (0x18) to GRBL.
    4. Records exact event sequence in emergency_stop_log.
    """
    timeline = []

    class DeceleratingMockGRBLSerial:
        def __init__(self, status_sequence):
            self._status_sequence = list(status_sequence)
            self._pending_status = False

        def write(self, data):
            timeline.append(("GRBL", data))
            if data == b"?":
                self._pending_status = True

        def flush(self):
            pass

        def readline(self):
            if self._pending_status:
                self._pending_status = False
                if self._status_sequence:
                    return self._status_sequence.pop(0)
            return b"ok\r\n"

        def close(self):
            pass

    class TrackingESP32:
        def send_command(self, cmd):
            timeline.append(("ESP32", cmd))
            return True

    esp32 = TrackingESP32()
    grbl = GRBLHardwareController(mode="grbl_emulator", esp32_interface=esp32)
    grbl.mode = "real_serial"
    grbl.grbl_state = "Run"
    grbl.serial_conn = DeceleratingMockGRBLSerial([
        b"<Hold:1|MPos:10.500,20.000,-1.500|FS:420,10000|Ov:100,100,100>\r\n",
        b"<Hold:1|MPos:10.800,20.000,-1.500|FS:110,10000|Ov:100,100,100>\r\n",
        b"<Hold:0|MPos:10.850,20.000,-1.500|FS:0,10000|Ov:100,100,100>\r\n",
    ])

    alarm = grbl.emergency_stop(max_poll_s=1.0, poll_interval_s=0.01)
    assert "EMERGENCY STOP" in alarm
    assert grbl.emergency_stop_log == [
        "FEED_HOLD_SENT",
        "AXES_STOPPED_VERIFIED",
        "SSR_OFF_SENT",
        "SOFT_RESET_SENT",
    ]

    # Exact cross-device timeline: '!' -> 3 status polls ('?') -> ESP32 CMD:OFF -> Soft Reset (0x18)
    assert timeline == [
        ("GRBL", b"!"),
        ("GRBL", b"?"),
        ("GRBL", b"?"),
        ("GRBL", b"?"),
        ("ESP32", "CMD:OFF"),
        ("GRBL", b"\x18"),
    ]


def test_grbl_emergency_stop_timeout_still_executes_ssr_off_and_reset():
    """
    Verify safety timeout fallback in emergency_stop() if GRBL stays in Hold:1
    beyond max_poll_s: SSR OFF and Soft Reset still execute rather than hanging.
    """
    timeline = []

    class StuckHold1Serial:
        def __init__(self):
            self._pending_status = False

        def write(self, data):
            timeline.append(("GRBL", data))
            if data == b"?":
                self._pending_status = True

        def flush(self):
            pass

        def readline(self):
            if self._pending_status:
                self._pending_status = False
                return b"<Hold:1|MPos:5.000,5.000,-1.000|FS:250,10000|Ov:100,100,100>\r\n"
            return b"ok\r\n"

    class TrackingESP32:
        def send_command(self, cmd):
            timeline.append(("ESP32", cmd))
            return True

    grbl = GRBLHardwareController(mode="grbl_emulator", esp32_interface=TrackingESP32())
    grbl.mode = "real_serial"
    grbl.grbl_state = "Run"
    grbl.serial_conn = StuckHold1Serial()

    grbl.emergency_stop(max_poll_s=0.06, poll_interval_s=0.02)
    assert grbl.emergency_stop_log == [
        "FEED_HOLD_SENT",
        "AXES_STOP_TIMEOUT",
        "SSR_OFF_SENT",
        "SOFT_RESET_SENT",
    ]
    assert timeline[0] == ("GRBL", b"!")
    assert timeline[-2] == ("ESP32", "CMD:OFF")
    assert timeline[-1] == ("GRBL", b"\x18")


def test_grbl_compute_decel_timeout_from_grbl_params():
    """
    Verify GRBLHardwareController.compute_decel_timeout_s() computes kinematic deceleration
    timeout from GRBL $110 (max_rate_mm_min=1000.0) and $120 (accel_mm_s2=10.0) + 0.833 s
    margin (~2.50 s), synchronized with ESP32 SPINDLE_DECEL_WAIT_MS = 2500 ms.
    """
    timeout_s = GRBLHardwareController.compute_decel_timeout_s(1000.0, 10.0, 0.833)
    assert timeout_s == pytest.approx(2.50, abs=1e-3)

    ctrl = GRBLHardwareController(mode="grbl_emulator")
    assert ctrl.grbl_max_rate_mm_min == pytest.approx(1000.0)
    assert ctrl.grbl_accel_mm_s2 == pytest.approx(10.0)
    assert ctrl.compute_decel_timeout_s(ctrl.grbl_max_rate_mm_min, ctrl.grbl_accel_mm_s2) == pytest.approx(
        2.50, abs=1e-3
    )

    with pytest.raises(ValueError):
        GRBLHardwareController.compute_decel_timeout_s(1000.0, 0.0, 0.833)


def test_edge_node_telemetry_watchdog_trips_grbl_emergency_stop_on_esp32_timeout():
    """
    Verify Host Telemetry Heartbeat Watchdog (300 ms) in EdgeDigitalTwinNode:
    1. When daq.read_telemetry() raises TimeoutError (ESP32 lost power / hung > 300 ms),
       EdgeDigitalTwinNode immediately invokes grbl.emergency_stop() sending '!' (Feed Hold)
       over Serial to GRBL and sets watchdog_tripped=True.
    2. When telemetry ts_ms gap between packets exceeds 300 ms (e.g. 450 - 100 = 350 ms),
       check_telemetry_watchdog() trips and dispatches '!' to GRBL as well.
    """
    class WatchdogMockGRBLSerial:
        def __init__(self):
            self.writes = []
            self._pending_status = False

        def write(self, data):
            self.writes.append(data)
            if data == b"?":
                self._pending_status = True

        def flush(self):
            pass

        def readline(self):
            if self._pending_status:
                self._pending_status = False
                return b"<Hold:0|MPos:0.000,0.000,0.000|FS:0,10000|Ov:100,100,100>\r\n"
            return b"ok\r\n"

        def close(self):
            pass

    # Case 1: daq.read_telemetry() times out (> 300 ms due to dead ESP32)
    daq_timeout = BreadboardDAQInterface(mode="hardware_emulator", machine_id="M0")

    def dead_esp32_read(dt_ms=100):
        raise TimeoutError("ESP32 telemetry UART timeout after 0.300s")

    daq_timeout.read_telemetry = dead_esp32_read

    mock_serial_1 = WatchdogMockGRBLSerial()
    grbl_1 = GRBLHardwareController(mode="grbl_emulator", esp32_interface=daq_timeout)
    grbl_1.mode = "real_serial"
    grbl_1.serial_conn = mock_serial_1

    node_1 = EdgeDigitalTwinNode(
        machine_id="M0",
        daq_interface=daq_timeout,
        grbl_controller=grbl_1,
        cdi_threshold=0.12,
    )
    assert node_1.telemetry_watchdog_timeout_ms == pytest.approx(300.0)
    assert node_1.last_valid_telemetry_ts_ms is None

    op = {
        "op_id": "J1_O1",
        "t_proc_min": 2.0,
        "speed_level": 1,
        "energy_kwh": 0.15,
        "feed": 500.0,
        "spindle": 10000.0,
        "x": 10.0,
        "y": 15.0,
        "z": -1.0,
    }
    res_1 = node_1.dispatch_operation(op)
    assert res_1["watchdog_tripped"] is True
    assert node_1.watchdog_tripped is True
    assert res_1["trigger_cloud_reschedule"] is True
    assert node_1.last_trigger_payload is not None
    assert node_1.last_trigger_payload.reason == "TELEMETRY_WATCHDOG_TIMEOUT"
    assert b"!" in mock_serial_1.writes
    assert grbl_1.grbl_state == "Hold"

    # Case 2: Packet gap > 300 ms triggers check_telemetry_watchdog()
    daq_gap = BreadboardDAQInterface(mode="hardware_emulator", machine_id="M0")
    packets = iter([
        {
            "ts_ms": 100,
            "v_bus": 24.0,
            "i_amp": 1.9,
            "p_raw_w": 45.6,
            "p_filt_w": 45.5,
            "vib_rms_g": 0.88,
            "ssr_state": "CUTTING",
            "sensor_ok": True,
            "fault_latched": False,
            "i2c_err_cnt": 0,
        },
        {
            "ts_ms": 450,
            "v_bus": 24.0,
            "i_amp": 1.9,
            "p_raw_w": 45.6,
            "p_filt_w": 45.5,
            "vib_rms_g": 0.88,
            "ssr_state": "CUTTING",
            "sensor_ok": True,
            "fault_latched": False,
            "i2c_err_cnt": 0,
        },
    ])
    daq_gap.read_telemetry = lambda dt_ms=100: next(packets)

    mock_serial_2 = WatchdogMockGRBLSerial()
    grbl_2 = GRBLHardwareController(mode="grbl_emulator", esp32_interface=daq_gap)
    grbl_2.mode = "real_serial"
    grbl_2.serial_conn = mock_serial_2

    node_2 = EdgeDigitalTwinNode(
        machine_id="M0",
        daq_interface=daq_gap,
        grbl_controller=grbl_2,
        cdi_threshold=0.12,
    )
    res_ok = node_2.dispatch_operation(op)
    assert res_ok["watchdog_tripped"] is False
    assert node_2.last_valid_telemetry_ts_ms == pytest.approx(100.0)
    assert b"!" not in mock_serial_2.writes

    res_gap = node_2.dispatch_operation(op)
    assert res_gap["watchdog_tripped"] is True
    assert node_2.watchdog_tripped is True
    assert b"!" in mock_serial_2.writes
    assert grbl_2.grbl_state == "Hold"




