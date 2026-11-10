"""
Test Suite for Task 3.2: GRBL v1.1 Hardware Controller & Real-Time Driver.
Validates:
1. Exact byte sequences for GRBL v1.1 real-time speed overrides:
   - s = 1 (100%): [0x90, 0x99] (2 bytes)
   - s = 2 (125%): [0x90, 0x91, 0x91, 0x93, 0x93, 0x93, 0x93, 0x93, 0x99, 0x9A, 0x9A, 0x9C, 0x9C, 0x9C, 0x9C, 0x9C] (16 bytes)
   - s = 3 (150%): [0x90, 0x91, 0x91, 0x91, 0x91, 0x91, 0x99, 0x9A, 0x9A, 0x9A, 0x9A, 0x9A] (12 bytes)
2. Anti-cumulation guarantee: Repeated calls to set_speed_override(2) or set_speed_override(3)
   never accumulate past the intended percentage (strictly locked at 125% and 150%).
3. Multi-level transition consistency (1 -> 2 -> 3 -> 1 -> 3 -> 2).
4. Status report string formatting and real-time query parsing.
5. Emergency stop, SSR commands, and G-code motion execution.
"""

import pytest
from edge_layer.grbl_controller import GRBLHardwareController


def test_grbl_override_bytes_exact_protocol():
    """
    Verify exact GRBL v1.1 byte sequences for speed levels 1, 2, and 3.
    """
    controller = GRBLHardwareController(mode="grbl_emulator")

    # 1. Level 1: 100% (2 bytes: feed reset 0x90, spindle reset 0x99)
    res1 = controller.set_speed_override(1)
    assert res1["speed_level"] == 1
    assert res1["speed_factor"] == 1.00
    assert res1["feed_override_pct"] == 100
    assert res1["spindle_override_pct"] == 100
    assert res1["sent_bytes"] == [0x90, 0x99]
    assert len(res1["sent_bytes"]) == 2
    assert res1["ssr_cmd"] is None

    # 2. Level 2: 125% (16 bytes: reset to 100% + 2x10% + 5x1% for feed and spindle)
    expected_bytes_s2 = [
        0x90, 0x91, 0x91, 0x93, 0x93, 0x93, 0x93, 0x93,
        0x99, 0x9A, 0x9A, 0x9C, 0x9C, 0x9C, 0x9C, 0x9C,
    ]
    res2 = controller.set_speed_override(2)
    assert res2["speed_level"] == 2
    assert res2["speed_factor"] == 1.25
    assert res2["feed_override_pct"] == 125
    assert res2["spindle_override_pct"] == 125
    assert res2["sent_bytes"] == expected_bytes_s2
    assert len(res2["sent_bytes"]) == 16
    assert res2["ssr_cmd"] is None

    # 3. Level 3: 150% (12 bytes: reset to 100% + 5x10% for feed and spindle)
    expected_bytes_s3 = [
        0x90, 0x91, 0x91, 0x91, 0x91, 0x91,
        0x99, 0x9A, 0x9A, 0x9A, 0x9A, 0x9A,
    ]
    res3 = controller.set_speed_override(3)
    assert res3["speed_level"] == 3
    assert res3["speed_factor"] == 1.50
    assert res3["feed_override_pct"] == 150
    assert res3["spindle_override_pct"] == 150
    assert res3["sent_bytes"] == expected_bytes_s3
    assert len(res3["sent_bytes"]) == 12
    assert res3["ssr_cmd"] == "CMD:CUTTING"
    assert controller.current_state == "CUTTING"


def test_grbl_override_anti_cumulation_repeated_calls():
    """
    Ensure that calling set_speed_override repeatedly does NOT accumulate past target percentage.
    In raw GRBL without resets, calling +10% repeatedly would saturate at 200%.
    With the reset byte prefix (0x90, 0x99), repeated calls must remain strictly at 125% and 150%.
    """
    controller = GRBLHardwareController(mode="grbl_emulator")

    # Call speed level 2 consecutively 10 times
    for i in range(10):
        res = controller.set_speed_override(2)
        assert res["feed_override_pct"] == 125, f"Iteration {i}: feed override drifted to {res['feed_override_pct']}"
        assert res["spindle_override_pct"] == 125, f"Iteration {i}: spindle override drifted to {res['spindle_override_pct']}"
        assert controller.feed_override_pct == 125
        assert controller.spindle_override_pct == 125

    # Call speed level 3 consecutively 10 times
    for i in range(10):
        res = controller.set_speed_override(3)
        assert res["feed_override_pct"] == 150, f"Iteration {i}: feed override drifted to {res['feed_override_pct']}"
        assert res["spindle_override_pct"] == 150, f"Iteration {i}: spindle override drifted to {res['spindle_override_pct']}"
        assert controller.feed_override_pct == 150
        assert controller.spindle_override_pct == 150

    # Call speed level 1 consecutively 10 times
    for i in range(10):
        res = controller.set_speed_override(1)
        assert res["feed_override_pct"] == 100, f"Iteration {i}: feed override drifted to {res['feed_override_pct']}"
        assert res["spindle_override_pct"] == 100, f"Iteration {i}: spindle override drifted to {res['spindle_override_pct']}"
        assert controller.feed_override_pct == 100
        assert controller.spindle_override_pct == 100


def test_grbl_override_arbitrary_transitions():
    """
    Verify transitions across arbitrary speed sequence: 1 -> 2 -> 3 -> 1 -> 3 -> 2 -> 1.
    """
    controller = GRBLHardwareController(mode="grbl_emulator")
    transitions = [
        (1, 100, 1.00),
        (2, 125, 1.25),
        (3, 150, 1.50),
        (1, 100, 1.00),
        (3, 150, 1.50),
        (2, 125, 1.25),
        (1, 100, 1.00),
    ]

    for lvl, expected_pct, expected_factor in transitions:
        res = controller.set_speed_override(lvl)
        assert res["speed_level"] == lvl
        assert res["speed_factor"] == expected_factor
        assert res["feed_override_pct"] == expected_pct
        assert res["spindle_override_pct"] == expected_pct
        assert controller.feed_override_pct == expected_pct
        assert controller.spindle_override_pct == expected_pct


def test_grbl_status_reporting_and_query():
    """
    Test real-time status query formatting matching GRBL v1.1 <Idle|MPos:...|FS:...|Ov:...> schema.
    """
    controller = GRBLHardwareController(mode="grbl_emulator")

    controller.set_speed_override(2)
    st = controller.query_status()

    assert st["state"] == "Idle"
    assert st["ov_feed"] == 125
    assert st["ov_spindle"] == 125
    assert "Ov:125,125,100" in st["raw"]

    controller.set_speed_override(3)
    st3 = controller.query_status()
    assert st3["ov_feed"] == 150
    assert st3["ov_spindle"] == 150
    assert "Ov:150,150,100" in st3["raw"]


def test_grbl_motion_gcode_and_emergency_stop():
    """
    Verify G-code parsing, motion tracking, and fail-safe emergency stop.
    """
    controller = GRBLHardwareController(mode="grbl_emulator")

    # Send linear cut G-code
    gcode = controller.format_operation_gcode(x=45.0, y=30.0, z=-3.5, feed=600.0, spindle=12000.0)
    assert gcode == "G1 X45.000 Y30.000 Z-3.500 F600.0 S12000"

    resp = controller.send_gcode(gcode)
    assert "ok" in resp
    assert controller.current_mpos == [45.0, 30.0, -3.5]
    assert controller.current_f == 600.0
    assert controller.current_s == 12000.0
    assert controller.grbl_state == "Run"

    # Emergency stop triggers Hold and powers off SSRs
    alarm = controller.emergency_stop()
    assert "EMERGENCY STOP" in alarm
    assert controller.grbl_state == "Hold"
    assert controller.current_state == "OFF"
    assert controller.last_ssr_cmd == "CMD:OFF"

    # Reset emulator restores initial idle state
    controller.reset_emulator()
    assert controller.current_state == "IDLE"
    assert controller.grbl_state == "Idle"
    assert controller.feed_override_pct == 100
    assert controller.spindle_override_pct == 100
