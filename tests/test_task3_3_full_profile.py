"""
Test Suite and Full 60-Second Hardware Telemetry Benchmark (Task 3.3).
Executes the comprehensive 60-second hardware verification scenario (600 JSON telemetry frames at 10Hz):
1. 0s - 5s   : Machine OFF (P ~ 0W).
2. 5s - 12s  : Turn-on Inrush -> Steady-state IDLE (15.6W).
3. 12s - 22s : CUTTING Speed s=1 (slow wear progression 30 um -> 38 um).
4. 22s - 42s : CUTTING Speed s=3 (fast wear 38 um -> 84 um crossing PM threshold 80 um with power/vib drift).
5. 42s - 50s : SSR STANDBY & Opportunistic Tool Replacement (P_stdby ~ 3.6W, wear reset to 0 um).
6. 50s - 60s : Wake-up & Fresh Tool CUTTING s=3 (clean nominal cutting power & smooth vibration).

Exports:
- CSV: data/benchmarks/phase3_hardware_telemetry.csv
- Plot: tests/phase3_hardware_verification.png (3 publication-grade synchronized subplots)
"""

import csv
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytest

from edge_layer.breadboard_daq import (
    BreadboardDAQInterface,
    RealTimeToolWearEstimator,
)


def run_60s_hardware_verification_scenario(
    csv_out_path: str = "data/benchmarks/phase3_hardware_telemetry.csv",
    plot_out_path: str = "tests/phase3_hardware_verification.png",
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Executes the 60-second scenario, saves telemetry to CSV, and generates the verification plot.
    Returns:
        records: List of 600 telemetry dictionaries.
        metrics: Dictionary of evaluation metrics (RMSE, R^2, max_power, etc.).
    """
    os.makedirs(os.path.dirname(csv_out_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_out_path), exist_ok=True)

    dt_s = 0.1  # 10 Hz sampling
    times = np.arange(0.0, 60.0, dt_s)
    n_samples = len(times)  # 600

    true_wear = np.zeros(n_samples)
    speeds = np.ones(n_samples, dtype=int)
    states = []

    # Construct the 6-phase trajectory
    for i, t in enumerate(times):
        if t < 5.0:
            # Phase 1: OFF (0s - 5s)
            true_wear[i] = 30.0  # Carryover wear from previous shift
            speeds[i] = 1
            states.append("OFF")
        elif t < 12.0:
            # Phase 2: Inrush -> IDLE (5s - 12s)
            true_wear[i] = 30.0
            speeds[i] = 1
            states.append("IDLE")
        elif t < 22.0:
            # Phase 3: CUTTING Speed s=1 (12s - 22s: wear 30 -> 38 um)
            frac = (t - 12.0) / 10.0
            true_wear[i] = 30.0 + frac * (38.0 - 30.0)
            speeds[i] = 1
            states.append("CUTTING")
        elif t < 42.0:
            # Phase 4: CUTTING Speed s=3 (22s - 42s: wear 38 -> 84 um, crossing PM threshold 80 um)
            frac = (t - 22.0) / 20.0
            true_wear[i] = 38.0 + frac * (84.0 - 38.0)
            speeds[i] = 3
            states.append("CUTTING")
        elif t < 50.0:
            # Phase 5: STANDBY & PM Tool Replacement (42s - 50s: wear reset to 0 um)
            true_wear[i] = 0.0
            speeds[i] = 1
            states.append("STANDBY")
        else:
            # Phase 6: Wake-up & Fresh Tool CUTTING s=3 (50s - 60s: fresh wear 0 -> 2.5 um)
            frac = (t - 50.0) / 10.0
            true_wear[i] = 0.0 + frac * 2.5
            speeds[i] = 3
            states.append("CUTTING")

    daq = BreadboardDAQInterface(mode="hardware_emulator")
    estimator = RealTimeToolWearEstimator()
    estimator.reset(30.0)  # Initial carryover wear

    daq.send_command("CMD:OFF")
    records: List[Dict[str, Any]] = []
    last_state = "OFF"

    for i, t in enumerate(times):
        st = states[i]
        if st != last_state:
            daq.send_command(f"CMD:{st}")
            if st == "STANDBY":
                # Opportunistic tool replacement resets tool wear to brand new
                estimator.reset(0.0)
            last_state = st

        daq.set_operating_conditions(speed_level=speeds[i], true_wear_um=true_wear[i])
        tel = daq.read_telemetry(dt_ms=100)

        # Estimate wear using Dual-Sensor Fusion
        ew = estimator.estimate_wear(
            tel, speed_level=speeds[i], sensor_mode="fusion", use_filter=True
        )

        records.append({
            "time_s": round(float(t), 2),
            "ts_ms": int(tel["ts_ms"]),
            "ssr_state": tel["ssr_state"],
            "speed_level": int(speeds[i]),
            "v_bus": float(tel["v_bus"]),
            "i_amp": float(tel["i_amp"]),
            "p_raw_w": float(tel["p_raw_w"]),
            "p_filt_w": float(tel["p_filt_w"]),
            "vib_rms_g": float(tel["vib_rms_g"]),
            "true_wear_um": round(float(true_wear[i]), 2),
            "estimated_wear_um": round(float(ew), 2),
        })

    # Save to CSV
    fieldnames = [
        "time_s", "ts_ms", "ssr_state", "speed_level",
        "v_bus", "i_amp", "p_raw_w", "p_filt_w",
        "vib_rms_g", "true_wear_um", "estimated_wear_um",
    ]
    with open(csv_out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # Compute Statistical Metrics
    y_true = np.array([r["true_wear_um"] for r in records])
    y_est = np.array([r["estimated_wear_um"] for r in records])

    rmse = float(np.sqrt(np.mean((y_est - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_est - y_true)))
    ss_res = float(np.sum((y_true - y_est) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot)

    metrics = {
        "rmse_um": rmse,
        "mae_um": mae,
        "r2_score": r2,
        "total_samples": len(records),
    }

    # Generate Publication-Quality 3-Panel Verification Plot
    generate_verification_plot(records, metrics, plot_out_path)

    return records, metrics


def generate_verification_plot(
    records: List[Dict[str, Any]],
    metrics: Dict[str, float],
    plot_out_path: str,
):
    """
    Renders 3 vertically stacked, synchronized subplots (Time 0..60s)
    with shaded operational phase bands and detailed annotations.
    """
    t = np.array([r["time_s"] for r in records])
    p_raw = np.array([r["p_raw_w"] for r in records])
    p_filt = np.array([r["p_filt_w"] for r in records])
    vib_rms = np.array([r["vib_rms_g"] for r in records])
    true_wear = np.array([r["true_wear_um"] for r in records])
    est_wear = np.array([r["estimated_wear_um"] for r in records])

    phases = [
        (0.0, 5.0, "Phase 1: OFF", "#F5F5F5"),
        (5.0, 12.0, "Phase 2: Inrush -> IDLE", "#FFF9C4"),
        (12.0, 22.0, "Phase 3: CUTTING (s=1)", "#E8F5E9"),
        (22.0, 42.0, "Phase 4: CUTTING (s=3) Fast Wear & Power Drift", "#FFEBEE"),
        (42.0, 50.0, "Phase 5: STANDBY & PM Replace", "#EDE7F6"),
        (50.0, 60.0, "Phase 6: Fresh Tool CUTTING (s=3)", "#E0F7FA"),
    ]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    plt.subplots_adjust(hspace=0.18)

    # Shaded operational background spans
    for ax in (ax1, ax2, ax3):
        for start, end, label, color in phases:
            ax.axvspan(start, end, color=color, alpha=0.65, zorder=0)

    # -------------------------------------------------------------------------
    # Subplot 1: Power Profile (INA226 Telemetry)
    # -------------------------------------------------------------------------
    ax1.plot(t, p_raw, color="#90CAF9", alpha=0.55, linewidth=1.1, label="Raw Power $P_{\\text{raw}}$ (INA226 + PWM noise)", zorder=2)
    ax1.plot(t, p_filt, color="#0D47A1", linewidth=2.2, label="Filtered Power $P_{\\text{filt}}$ (1D Kalman + Median)", zorder=3)
    ax1.set_ylabel("Power (W)", fontsize=11, fontweight="bold")
    ax1.set_ylim(-5.0, 210.0)
    ax1.set_title("Breadboard Physical Twin Telemetry & Dual-Sensor Tool Wear Verification (60s Profile)", fontsize=13, fontweight="bold", pad=12)
    ax1.grid(True, linestyle="--", alpha=0.5, zorder=1)
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=9.5)

    # Annotate Inrush Spikes
    ax1.annotate(
        "Turn-on Inrush Spike\n(2.2x $P_{\\text{idle}} \\approx 34.3$ W)",
        xy=(5.1, 34.0), xytext=(7.0, 75.0),
        arrowprops=dict(facecolor="#E65100", edgecolor="#E65100", arrowstyle="->", lw=1.5),
        fontsize=9, fontweight="bold", color="#E65100",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3E0", ec="#E65100", alpha=0.9)
    )
    ax1.annotate(
        "Wake-up Inrush Spike\n(2.2x $P_{\\text{cut}} \\approx 181.9$ W)",
        xy=(50.1, 182.0), xytext=(38.0, 170.0),
        arrowprops=dict(facecolor="#C62828", edgecolor="#C62828", arrowstyle="->", lw=1.5),
        fontsize=9, fontweight="bold", color="#C62828",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFEBEE", ec="#C62828", alpha=0.9)
    )

    # -------------------------------------------------------------------------
    # Subplot 2: Vibration RMS Profile (ADXL345 Telemetry)
    # -------------------------------------------------------------------------
    ax2.plot(t, vib_rms, color="#D84315", linewidth=2.0, label="3-Axis Vibration RMS $Vib_{\\text{RMS}}$ (ADXL345)", zorder=3)
    ax2.set_ylabel("Vibration RMS (g)", fontsize=11, fontweight="bold")
    ax2.set_ylim(0.0, 2.2)
    ax2.grid(True, linestyle="--", alpha=0.5, zorder=1)
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=9.5)

    ax2.annotate(
        "Severe Chatter & Power Drift\nDue to Worn Flank ($VB > 80\\,\\mu$m)",
        xy=(41.5, 1.83), xytext=(24.0, 1.85),
        arrowprops=dict(facecolor="#B71C1C", edgecolor="#B71C1C", arrowstyle="->", lw=1.5),
        fontsize=9, fontweight="bold", color="#B71C1C",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFCDD2", ec="#B71C1C", alpha=0.9)
    )
    ax2.annotate(
        "Vibration Drops to Smooth Baseline\nAfter Tool Replacement",
        xy=(55.0, 1.39), xytext=(45.0, 0.70),
        arrowprops=dict(facecolor="#00695C", edgecolor="#00695C", arrowstyle="->", lw=1.5),
        fontsize=9, fontweight="bold", color="#00695C",
        bbox=dict(boxstyle="round,pad=0.3", fc="#E0F2F1", ec="#00695C", alpha=0.9)
    )

    # -------------------------------------------------------------------------
    # Subplot 3: Tool Wear Tracking & PM Threshold (Sensor Fusion)
    # -------------------------------------------------------------------------
    ax3.plot(t, true_wear, color="#2E7D32", linewidth=2.4, label="True Flank Wear $VB_{\\text{true}}$ ($\\mu$m)", zorder=3)
    ax3.plot(t, est_wear, color="#6A1B9A", linewidth=2.0, linestyle="--", label="Dual-Sensor Fusion Estimate $\\hat{VB}$ (INA226 + ADXL345)", zorder=4)
    ax3.axhline(80.0, color="#C62828", linestyle=":", linewidth=2.0, label="Proactive PM Threshold (80.0 $\\mu$m)", zorder=2)

    ax3.set_ylabel("Tool Wear $VB$ ($\\mu$m)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Time (s)", fontsize=11, fontweight="bold")
    ax3.set_xlim(0.0, 60.0)
    ax3.set_ylim(-2.0, 105.0)
    ax3.grid(True, linestyle="--", alpha=0.5, zorder=1)
    ax3.legend(loc="upper left", framealpha=0.9, fontsize=9.5)

    # Metrics Badge
    metrics_text = (
        f"Sensor Fusion Accuracy:\n"
        f"• RMSE = {metrics['rmse_um']:.3f} $\\mu$m\n"
        f"• MAE  = {metrics['mae_um']:.3f} $\\mu$m\n"
        f"• $R^2$   = {metrics['r2_score']:.4f}"
    )
    ax3.text(
        0.83, 0.45, metrics_text, transform=ax3.transAxes,
        fontsize=9.5, fontweight="bold", color="#1A237E",
        bbox=dict(boxstyle="round,pad=0.4", fc="#E8EAF6", ec="#3F51B5", alpha=0.95),
        zorder=5
    )

    # PM Event Annotation
    ax3.annotate(
        "PM Trigger Threshold Reached\n($VB = 80.0\\,\\mu$m at $t \\approx 40.3$s)",
        xy=(40.3, 80.0), xytext=(22.0, 92.0),
        arrowprops=dict(facecolor="#C62828", edgecolor="#C62828", arrowstyle="->", lw=1.5),
        fontsize=9, fontweight="bold", color="#C62828",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFEBEE", ec="#C62828", alpha=0.9)
    )
    ax3.annotate(
        "Tool Replaced (PM)\nWear Reset to 0.0 $\\mu$m",
        xy=(43.0, 0.0), xytext=(35.0, 20.0),
        arrowprops=dict(facecolor="#2E7D32", edgecolor="#2E7D32", arrowstyle="->", lw=1.5),
        fontsize=9, fontweight="bold", color="#2E7D32",
        bbox=dict(boxstyle="round,pad=0.3", fc="#E8F5E9", ec="#2E7D32", alpha=0.9)
    )

    # Phase Labels at Top of Subplot 1
    for start, end, label, _ in phases:
        mid = (start + end) / 2.0
        ax1.text(
            mid, 196.0, label, ha="center", va="top",
            fontsize=8.5, fontweight="bold", color="#37474F",
            bbox=dict(boxstyle="square,pad=0.15", fc="#FFFFFF", ec="#B0BEC5", alpha=0.7)
        )

    plt.tight_layout()
    fig.savefig(plot_out_path, dpi=300)
    plt.close(fig)


# =============================================================================
# Pytest Verification Functions
# =============================================================================

def test_full_60s_scenario_generation_and_metrics():
    """
    Executes the full 60-second scenario, validates generated artifacts, and checks RMSE / R^2.
    """
    csv_path = "data/benchmarks/phase3_hardware_telemetry.csv"
    plot_path = "tests/phase3_hardware_verification.png"

    records, metrics = run_60s_hardware_verification_scenario(csv_path, plot_path)

    # 1. Artifact existence and integrity
    assert os.path.exists(csv_path), "Telemetry CSV file was not created"
    assert os.path.exists(plot_path), "Verification PNG plot was not created"
    assert os.path.getsize(plot_path) > 50000, "Verification plot file is unusually small"
    assert len(records) == 600, f"Expected 600 telemetry records, got {len(records)}"

    # 2. Tracking Accuracy Assertions
    print("\n" + "=" * 70)
    print("TASK 3.3: 60-SECOND HARDWARE DAQ BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Total Telemetry Frames  : {metrics['total_samples']} (10 Hz, 60.0 s)")
    print(f"Full Profile Wear RMSE  : {metrics['rmse_um']:.3f} um (Target: < 2.5 um)")
    print(f"Full Profile Wear MAE   : {metrics['mae_um']:.3f} um")
    print(f"Full Profile Wear R^2   : {metrics['r2_score']:.4f} (Target: > 0.95)")
    print("=" * 70)

    assert metrics["rmse_um"] < 2.5, f"RMSE failed: {metrics['rmse_um']:.3f} um >= 2.5 um"
    assert metrics["r2_score"] > 0.95, f"R^2 failed: {metrics['r2_score']:.4f} <= 0.95"


def test_inrush_and_phase_dynamics():
    """
    Validates physical inrush spikes and power / vibration transitions across phases.
    """
    csv_path = "data/benchmarks/phase3_hardware_telemetry.csv"
    assert os.path.exists(csv_path), "CSV must exist before running phase tests"

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    # Phase 1: OFF (0s - 5s)
    off_records = [r for r in reader if float(r["time_s"]) < 5.0]
    for r in off_records:
        assert r["ssr_state"] == "OFF"
        assert float(r["p_raw_w"]) == 0.0

    # Phase 2: Inrush Spike at t=5.0s
    inrush_idle = [r for r in reader if 5.0 <= float(r["time_s"]) < 5.3]
    max_inrush_idle = max(float(r["p_filt_w"]) for r in inrush_idle)
    assert max_inrush_idle > 25.0, f"Expected inrush spike > 25W on IDLE startup, got {max_inrush_idle}"

    # Phase 4: Power and Vibration Drift (22s - 42s)
    p_start_p4 = float([r for r in reader if float(r["time_s"]) == 22.0][0]["p_filt_w"])
    p_end_p4 = float([r for r in reader if float(r["time_s"]) == 41.9][0]["p_filt_w"])
    assert p_end_p4 > p_start_p4 + 8.0, "Cutting power did not drift upward as wear increased"

    vib_start_p4 = float([r for r in reader if float(r["time_s"]) == 22.0][0]["vib_rms_g"])
    vib_end_p4 = float([r for r in reader if float(r["time_s"]) == 41.9][0]["vib_rms_g"])
    assert vib_end_p4 > vib_start_p4 + 0.25, "Vibration RMS did not increase significantly with wear"

    # Phase 4: PM threshold crossing (80.0 um reached before 42s)
    wear_p4 = [float(r["true_wear_um"]) for r in reader if 22.0 <= float(r["time_s"]) < 42.0]
    assert max(wear_p4) >= 80.0, "Tool wear must cross 80.0 um threshold before PM"

    # Phase 5: STANDBY & PM Tool replacement (42s - 50s)
    standby_records = [r for r in reader if 42.0 <= float(r["time_s"]) < 50.0]
    for r in standby_records:
        assert r["ssr_state"] == "STANDBY"
        assert float(r["p_filt_w"]) < 6.0
        assert float(r["true_wear_um"]) == 0.0

    # Phase 6: Post-PM Fresh Tool cutting power and vibration drop (52s - 60s)
    fresh_records = [r for r in reader if 53.0 <= float(r["time_s"]) < 60.0]
    avg_fresh_power = np.mean([float(r["p_filt_w"]) for r in fresh_records])
    avg_fresh_vib = np.mean([float(r["vib_rms_g"]) for r in fresh_records])

    # Must be significantly lower than worn cutting at t = 41.9s
    assert avg_fresh_power < p_end_p4 - 10.0, "Fresh tool power should be substantially lower than worn tool"
    assert avg_fresh_vib < vib_end_p4 - 0.25, "Fresh tool vibration should be substantially lower than worn tool"


if __name__ == "__main__":
    csv_file = "data/benchmarks/phase3_hardware_telemetry.csv"
    plot_file = "tests/phase3_hardware_verification.png"
    records, metrics = run_60s_hardware_verification_scenario(csv_file, plot_file)
    print(f"Generated {len(records)} records in {csv_file}")
    print(f"Generated verification plot in {plot_file}")
    print(f"Metrics: RMSE = {metrics['rmse_um']:.3f} um, R^2 = {metrics['r2_score']:.4f}")

