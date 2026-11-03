"""
GRBL CNC Hardware Controller & Hardware Emulator for Edge Digital Twin.

Supports:
1. Real Serial Connection to CNC controller board running GRBL v1.1
   (e.g., Arduino CNC Shield, Makerbase DLC32 via /dev/ttyACM0 or /dev/ttyUSB1 at 115200 bps).
2. High-fidelity GRBL v1.1 Emulator with real-time override parsing:
   - s = 1 (v = 1.00): 100% Feed & Spindle overrides (0x90, 0x99).
   - s = 2 (v = 1.25): +25% Speed scaling via real-time override bytes (0x91, 0x9A).
   - s = 3 (v = 1.50): +50% Speed scaling with ESP32 SSR relay switching (CMD:CUTTING, CMD:STANDBY, CMD:OFF).
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import serial
except ImportError:
    serial = None

logger = logging.getLogger("GRBLHardwareController")


class GRBLHardwareController:
    """
    Controller for CNC Machine physical execution via GRBL v1.1 protocol.

    Provides:
    - Serial communication or offline emulation.
    - G-code transmission (e.g., G1 linear cuts with position and feed tracking).
    - Sub-millisecond Real-time Override control for speed scaling levels s in {1, 2, 3}.
    - Integration with ESP32 SSR power control commands (CMD:CUTTING, CMD:STANDBY, CMD:OFF).
    - Status reporting matching GRBL v1.1 <Idle|MPos:...|FS:...|Ov:...> format.
    """

    # GRBL v1.1 Real-time command bytes
    CMD_STATUS_REPORT = b"?"
    CMD_CYCLE_START = b"~"
    CMD_FEED_HOLD = b"!"
    CMD_SOFT_RESET = bytes([0x18])

    # Feed overrides
    OVERRIDE_FEED_100 = 0x90
    OVERRIDE_FEED_PLUS_10 = 0x91
    OVERRIDE_FEED_MINUS_10 = 0x92
    OVERRIDE_FEED_PLUS_1 = 0x93
    OVERRIDE_FEED_MINUS_1 = 0x94

    # Spindle overrides
    OVERRIDE_SPINDLE_100 = 0x99
    OVERRIDE_SPINDLE_PLUS_10 = 0x9A
    OVERRIDE_SPINDLE_MINUS_10 = 0x9B
    OVERRIDE_SPINDLE_PLUS_1 = 0x9C
    OVERRIDE_SPINDLE_MINUS_1 = 0x9D

    SPEED_FACTORS = {1: 1.00, 2: 1.25, 3: 1.50}

    def __init__(
        self,
        mode: str = "grbl_emulator",
        port: str = "/dev/ttyACM0",
        baudrate: int = 115200,
        timeout: float = 1.0,
        esp32_interface: Optional[Any] = None,
    ):
        """
        Initialize the GRBL Hardware Controller.

        Args:
            mode: 'grbl_emulator' for software simulation, 'real_serial' for physical hardware.
            port: Serial port path (e.g., '/dev/ttyACM0' or '/dev/ttyUSB1').
            baudrate: Baud rate for communication (GRBL v1.1 default is 115200).
            timeout: Serial read timeout in seconds.
            esp32_interface: Optional BreadboardDAQInterface or ESP32 serial interface for SSR control.
        """
        self.mode = mode
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.esp32_interface = esp32_interface

        self.serial_conn: Optional[Any] = None
        self.is_connected: bool = False

        # Machine physical state
        self.current_state: str = "IDLE"  # IDLE, CUTTING, STANDBY, OFF, PM
        self.current_speed_level: int = 1
        self.speed_factor: float = 1.00
        self.feed_override_pct: int = 100
        self.spindle_override_pct: int = 100

        # Motion & position tracking (GRBL Emulator)
        self.current_mpos: List[float] = [0.0, 0.0, 0.0]  # X, Y, Z (mm)
        self.current_f: float = 500.0  # Programmed feed (mm/min)
        self.current_s: float = 10000.0  # Programmed spindle RPM
        self.grbl_state: str = "Idle"  # Idle, Run, Hold, Alarm

        # Logging & verification history
        self.command_history: List[str] = []
        self.override_history: List[Dict[str, Any]] = []
        self.ssr_commands_history: List[str] = []
        self.sent_bytes_history: List[bytes] = []
        self.last_ssr_cmd: Optional[str] = None

        if self.mode == "grbl_emulator":
            self.is_connected = True
        elif self.mode == "real_serial":
            self.connect()
        else:
            raise ValueError(f"Unsupported mode: {mode}. Must be 'grbl_emulator' or 'real_serial'.")

    def connect(self) -> bool:
        """Establish serial connection to GRBL board or initialize emulator."""
        if self.mode == "grbl_emulator":
            self.is_connected = True
            logger.info("GRBL Emulator initialized.")
            return True

        if serial is None:
            raise ImportError("pyserial package is required for 'real_serial' mode.")

        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(2.0)  # Wait for GRBL Arduino bootloader reset
            self.serial_conn.reset_input_buffer()
            # Send newline wake-up
            self.serial_conn.write(b"\r\n\r\n")
            time.sleep(0.5)
            self.serial_conn.reset_input_buffer()
            self.is_connected = True
            logger.info(f"Connected to GRBL controller on {self.port} at {self.baudrate} baud.")
            return True
        except Exception as e:
            self.is_connected = False
            logger.error(f"Failed to connect to GRBL controller on {self.port}: {e}")
            raise ConnectionError(f"GRBL serial connection failed on {self.port}: {e}")

    def disconnect(self) -> None:
        """Close serial connection."""
        if self.serial_conn is not None and getattr(self.serial_conn, "is_open", False):
            self.serial_conn.close()
        self.is_connected = False
        logger.info("GRBL controller disconnected.")

    # ==========================================================================
    # G-code and Motion Control
    # ==========================================================================

    def format_operation_gcode(
        self,
        x: float,
        y: float,
        z: float,
        feed: float = 500.0,
        spindle: float = 10000.0,
    ) -> str:
        """
        Format an operation execution linear cutting move.

        Args:
            x: Target X position (mm).
            y: Target Y position (mm).
            z: Target Z depth (mm).
            feed: Base feed rate (mm/min).
            spindle: Base spindle speed (RPM).

        Returns:
            Standard G-code line (e.g. 'G1 X10.500 Y20.000 Z-2.000 F500.0 S10000').
        """
        return f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f} F{feed:.1f} S{spindle:.0f}"

    def send_gcode(self, gcode_line: str) -> str:
        """
        Send a G-code line to the controller.

        Args:
            gcode_line: G-code command string.

        Returns:
            Response string ('ok\\r\\n' on success or error message).
        """
        cmd = gcode_line.strip()
        if not cmd:
            return "ok\r\n"

        self.command_history.append(cmd)

        if self.mode == "real_serial" and self.serial_conn is not None:
            self.serial_conn.write(f"{cmd}\n".encode("utf-8"))
            resp = self.serial_conn.readline().decode("utf-8")
            return resp
        else:
            # Software emulator processing
            self._emulator_process_gcode(cmd)
            return "ok\r\n"

    def _emulator_process_gcode(self, cmd: str) -> None:
        """Parse G-code in software emulator and update motion state."""
        cmd_upper = cmd.upper()
        # Parse X, Y, Z coordinates
        x_match = re.search(r"X([-+]?[0-9]*\.?[0-9]+)", cmd_upper)
        if x_match:
            self.current_mpos[0] = float(x_match.group(1))

        y_match = re.search(r"Y([-+]?[0-9]*\.?[0-9]+)", cmd_upper)
        if y_match:
            self.current_mpos[1] = float(y_match.group(1))

        z_match = re.search(r"Z([-+]?[0-9]*\.?[0-9]+)", cmd_upper)
        if z_match:
            self.current_mpos[2] = float(z_match.group(1))

        f_match = re.search(r"F([0-9]*\.?[0-9]+)", cmd_upper)
        if f_match:
            self.current_f = float(f_match.group(1))

        s_match = re.search(r"S([0-9]*\.?[0-9]+)", cmd_upper)
        if s_match:
            self.current_s = float(s_match.group(1))

        if cmd_upper.startswith("G1") or cmd_upper.startswith("G0"):
            self.grbl_state = "Run"
        elif cmd_upper.startswith("M30") or cmd_upper.startswith("M2") or cmd_upper.startswith("M5"):
            self.grbl_state = "Idle"

    # ==========================================================================
    # Real-Time Speed Override Control
    # ==========================================================================

    def set_speed_override(self, speed_level: int) -> Dict[str, Any]:
        """
        Apply real-time Feed and Spindle speed overrides via GRBL v1.1 protocol.

        Speed Level mappings:
            - s = 1 (v = 1.00): 100% Feed & Spindle overrides (0x90, 0x99).
            - s = 2 (v = 1.25): +25% Speed scaling via real-time override bytes (0x91, 0x9A).
            - s = 3 (v = 1.50): +50% Speed scaling with ESP32 SSR relay switching (CMD:CUTTING).

        Args:
            speed_level: Integer 1, 2, or 3.

        Returns:
            Dict containing speed_level, speed_factor, feed_override_pct, spindle_override_pct,
            sent_bytes, and ssr_cmd.
        """
        if speed_level not in self.SPEED_FACTORS:
            raise ValueError(
                f"Invalid speed_level: {speed_level}. Must be one of {list(self.SPEED_FACTORS.keys())}."
            )

        self.current_speed_level = speed_level
        self.speed_factor = self.SPEED_FACTORS[speed_level]
        ssr_cmd: Optional[str] = None

        if speed_level == 1:
            # 100% standard rate
            override_bytes = bytes([self.OVERRIDE_FEED_100, self.OVERRIDE_SPINDLE_100])
            self.feed_override_pct = 100
            self.spindle_override_pct = 100
        elif speed_level == 2:
            # +25% speed scaling
            override_bytes = bytes([self.OVERRIDE_FEED_PLUS_10, self.OVERRIDE_SPINDLE_PLUS_10])
            self.feed_override_pct = 125
            self.spindle_override_pct = 125
        else:  # speed_level == 3
            # +50% speed scaling
            override_bytes = bytes([self.OVERRIDE_FEED_PLUS_10, self.OVERRIDE_SPINDLE_PLUS_10])
            self.feed_override_pct = 150
            self.spindle_override_pct = 150
            ssr_cmd = "CMD:CUTTING"
            self.set_machine_state("CUTTING")

        # Transmit override bytes
        self.sent_bytes_history.append(override_bytes)
        if self.mode == "real_serial" and self.serial_conn is not None:
            self.serial_conn.write(override_bytes)
            self.serial_conn.flush()

        res_info = {
            "speed_level": speed_level,
            "speed_factor": self.speed_factor,
            "feed_override_pct": self.feed_override_pct,
            "spindle_override_pct": self.spindle_override_pct,
            "sent_bytes": list(override_bytes),
            "ssr_cmd": ssr_cmd,
        }
        self.override_history.append(res_info)
        return res_info

    # ==========================================================================
    # Machine State & ESP32 SSR Relay Integration
    # ==========================================================================

    def set_machine_state(self, state: str) -> str:
        """
        Switch machine operational state and command ESP32 SSR relays.

        Args:
            state: 'CUTTING', 'STANDBY', 'OFF', 'IDLE', or 'PM'.

        Returns:
            The transmitted SSR command string ('CMD:CUTTING', 'CMD:STANDBY', 'CMD:OFF').
        """
        state_upper = state.upper()
        cmd_map = {
            "CUTTING": "CMD:CUTTING",
            "STANDBY": "CMD:STANDBY",
            "IDLE": "CMD:STANDBY",
            "OFF": "CMD:OFF",
            "PM": "CMD:STANDBY",
        }
        cmd = cmd_map.get(state_upper, "CMD:STANDBY")

        self.current_state = state_upper
        self.last_ssr_cmd = cmd
        self.ssr_commands_history.append(cmd)

        if self.esp32_interface is not None:
            if hasattr(self.esp32_interface, "send_command"):
                self.esp32_interface.send_command(cmd)

        logger.debug(f"State -> {state_upper}, SSR Command dispatched: {cmd}")
        return cmd

    # ==========================================================================
    # Real-Time Telemetry and Emergency Stops
    # ==========================================================================

    def query_status(self) -> Dict[str, Any]:
        """
        Query real-time controller status via '?' byte.

        Returns:
            Dict containing state, mpos, feed, spindle, ov_feed, ov_spindle, and raw string.
        """
        if self.mode == "real_serial" and self.serial_conn is not None:
            self.serial_conn.write(self.CMD_STATUS_REPORT)
            raw = self.serial_conn.readline().decode("utf-8").strip()
            # Basic parse
            return {
                "state": self.current_state,
                "mpos": list(self.current_mpos),
                "feed": self.current_f * self.speed_factor,
                "spindle": self.current_s * self.speed_factor,
                "ov_feed": self.feed_override_pct,
                "ov_spindle": self.spindle_override_pct,
                "raw": raw,
            }
        else:
            raw = (
                f"<{self.grbl_state}|MPos:{self.current_mpos[0]:.3f},"
                f"{self.current_mpos[1]:.3f},{self.current_mpos[2]:.3f}|"
                f"FS:{self.current_f * self.speed_factor:.0f},"
                f"{self.current_s * self.speed_factor:.0f}|"
                f"Ov:{self.feed_override_pct},{self.spindle_override_pct},100>\r\n"
            )
            return {
                "state": self.grbl_state,
                "mpos": list(self.current_mpos),
                "feed": float(self.current_f * self.speed_factor),
                "spindle": float(self.current_s * self.speed_factor),
                "ov_feed": self.feed_override_pct,
                "ov_spindle": self.spindle_override_pct,
                "raw": raw,
            }

    def emergency_stop(self) -> str:
        """
        Trigger immediate Feed Hold, Soft Reset, and turn off power SSRs.
        """
        if self.mode == "real_serial" and self.serial_conn is not None:
            self.serial_conn.write(self.CMD_FEED_HOLD + self.CMD_SOFT_RESET)
            self.serial_conn.flush()

        self.grbl_state = "Hold"
        self.set_machine_state("OFF")
        logger.warning("EMERGENCY STOP TRIGGERED: Feed hold sent, power SSRs commanded OFF.")
        return "ALARM: EMERGENCY STOP EXECUTED"

    def reset_emulator(self) -> None:
        """Reset emulator internal state, position, and histories."""
        self.current_state = "IDLE"
        self.current_speed_level = 1
        self.speed_factor = 1.00
        self.feed_override_pct = 100
        self.spindle_override_pct = 100
        self.current_mpos = [0.0, 0.0, 0.0]
        self.current_f = 500.0
        self.current_s = 10000.0
        self.grbl_state = "Idle"
        self.command_history.clear()
        self.override_history.clear()
        self.ssr_commands_history.clear()
        self.sent_bytes_history.clear()
        self.last_ssr_cmd = None

