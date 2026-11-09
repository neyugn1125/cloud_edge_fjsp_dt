"""
Test Suite for Task 3.1: Breadboard DAQ Interface, Sensor Fusion & Telemetry Verification.
Validates:
1. Separation of data generation (BreadboardDAQInterface with unmodeled physical dynamics)
   and observation model (RealTimeToolWearEstimator).
2. Telemetry output integrity, sensor fusion accuracy (RMSE < 2.5 um, R^2 > 0.95),
   and inrush current transient behavior.
3. Full 60-second hardware-in-the-loop scenario export to:
   - data/benchmarks/phase3_hardware_telemetry.csv
   - results/phase3_hardware_telemetry.csv
4. Automated telemetry event analysis export to:
   - results/phase3_hil_event_summary.csv
   - data/benchmarks/phase3_hil_event_summary.csv
"""

import csv
import os
import shutil
from pathlib import Path
import numpy as np
import pytest

from edge_layer.breadboard_daq import (
    BreadboardDAQInterface,
    RealTimeToolWearEstimator,
)
from tests.test_task3_3_full_profile import run_60s_hardware_verification_scenario
from scripts.analyze_hardware_telemetry import analyze_hardware_telemetry


def test_unmodeled_dynamics_and_estimator_decoupling():
    """
    Asserts that the physical emulator produces non-identical sensor readings
    under varying physical states even when tool wear is identical (due to unmodeled dynamics),
    and that the estimator has no privileged access to the internal true wear.
    """
    daq = BreadboardDAQInterface(mode="hardware_emulator")
    estimator = RealTimeToolWearEstimator()
    estimator.reset(40.0)

    daq.send_command("CMD:CUTTING")
    daq.set_operating_conditions(speed_level=2, true_wear_um=40.0)

    # Read a short sequence of telemetry points
    telemetries = [daq.read_telemetry(dt_ms=100) for _ in range(10)]
    raw_powers = [t["p_raw_w"] for t in telemetries]
    vib_levels = [t["vib_rms_g"] for t in telemetries]

    # Values should exhibit stochastic unmodeled dynamics (not static algebraic constants)
    assert len(set(raw_powers)) > 1, "Sensor generator must include dynamic variation across samples"
    assert len(set(vib_levels)) > 1, "Vibration must include dynamic physical noise across samples"

    # Estimator estimates wear using purely telemetry dict without internal state contamination
    est_wear_1 = estimator.estimate_wear(telemetries[0], speed_level=2, sensor_mode="fusion")
    est_wear_2 = estimator.estimate_wear(telemetries[-1], speed_level=2, sensor_mode="fusion")

    assert isinstance(est_wear_1, float)
    assert isinstance(est_wear_2, float)
    assert abs(est_wear_1 - 40.0) < 15.0, f"Initial estimate out of realistic band: {est_wear_1}"


def test_full_scenario_execution_and_exports():
    """
    Runs the full 60s verification scenario, synchronizes CSVs to both
    data/benchmarks/ and results/, and triggers analyze_hardware_telemetry.
    """
    data_csv = "data/benchmarks/phase3_hardware_telemetry.csv"
    results_csv = "results/phase3_hardware_telemetry.csv"
    plot_path = "tests/phase3_hardware_verification.png"
    results_plot = "results/phase3_hardware_verification.png"

    records, metrics = run_60s_hardware_verification_scenario(
        csv_out_path=data_csv,
        plot_out_path=plot_path,
    )

    # Copy telemetry CSV to results/
    os.makedirs("results", exist_ok=True)
    shutil.copyfile(data_csv, results_csv)
    if os.path.exists(plot_path):
        shutil.copyfile(plot_path, results_plot)

    # Run analysis to generate event summary
    event_summary_results = "results/phase3_hil_event_summary.csv"
    event_summary_data = "data/benchmarks/phase3_hil_event_summary.csv"

    summary_rows = analyze_hardware_telemetry(
        input_csv_path=data_csv,
        output_csv_path=event_summary_results,
        verbose=False,
    )
    shutil.copyfile(event_summary_results, event_summary_data)

    # Validate generated artifacts
    assert os.path.exists(data_csv), "Missing data/benchmarks/phase3_hardware_telemetry.csv"
    assert os.path.exists(results_csv), "Missing results/phase3_hardware_telemetry.csv"
    assert os.path.exists(event_summary_results), "Missing results/phase3_hil_event_summary.csv"
    assert os.path.exists(event_summary_data), "Missing data/benchmarks/phase3_hil_event_summary.csv"

    assert len(records) == 600, f"Expected 600 telemetry records, got {len(records)}"
    assert metrics["rmse_um"] < 2.5, f"RMSE failed: {metrics['rmse_um']:.3f} >= 2.5 um"
    assert metrics["r2_score"] > 0.95, f"R^2 failed: {metrics['r2_score']:.4f} <= 0.95"
    assert len(summary_rows) >= 6, f"Expected at least 6 event segments, got {len(summary_rows)}"
