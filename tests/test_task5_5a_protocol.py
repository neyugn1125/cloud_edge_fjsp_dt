"""
Unit and Verification Tests for Task 5.5A:
Evaluation Protocol, Disjoint Seed Sets, Shift-Normalized CDI, and Reproducibility Specifications.
"""

import json
from pathlib import Path
import pytest
import numpy as np

from utils.eval_protocol import (
    TRAIN_SEEDS,
    VAL_SEEDS,
    TEST_SEEDS,
    EDGE_ONNX_LATENCY_MEAN_MS,
    EDGE_ONNX_LATENCY_MIN_MS,
    EDGE_ONNX_LATENCY_MAX_MS,
    CLOUD_SOLVER_LATENCY_MS,
    MQTT_RTT_MEAN_MS,
    SPEEDUP_FACTOR,
    DEFAULT_ALPHA_CDI,
    DEFAULT_CDI_THRESHOLD,
    REF_POINT_HV,
    compute_normalized_cdi,
    export_reproducibility_specs,
    get_canonical_hyperparameters,
)


def test_seed_sets_disjointness_and_counts():
    """Verify that training, validation, and test seed sets are strictly disjoint and have correct counts."""
    # 1. Verify counts
    assert len(TRAIN_SEEDS) == 3, f"Expected 3 training seeds, got {len(TRAIN_SEEDS)}"
    assert len(VAL_SEEDS) == 10, f"Expected 10 validation seeds, got {len(VAL_SEEDS)}"
    assert len(TEST_SEEDS) == 15, f"Expected 15 testing seeds, got {len(TEST_SEEDS)}"

    # 2. Verify explicit values
    assert TRAIN_SEEDS == [42, 101, 2026]
    assert VAL_SEEDS == list(range(201, 211))
    assert TEST_SEEDS == list(range(301, 316))

    # 3. Verify strict pairwise disjointness (Zero Data Leakage)
    train_set = set(TRAIN_SEEDS)
    val_set = set(VAL_SEEDS)
    test_set = set(TEST_SEEDS)

    assert train_set.isdisjoint(val_set), "TRAIN_SEEDS and VAL_SEEDS must not overlap!"
    assert val_set.isdisjoint(test_set), "VAL_SEEDS and TEST_SEEDS must not overlap!"
    assert train_set.isdisjoint(test_set), "TRAIN_SEEDS and TEST_SEEDS must not overlap!"


def test_normalized_cdi_calculation_first_operation():
    """
    Verify that compute_normalized_cdi eliminates early-shift division-by-zero singularities.
    Even under a large initial disturbance at op_idx = 1, CDI must remain strictly finite and < 0.50.
    """
    # Nominal shift totals (e.g. Mk01 baseline schedule: ~63.8 min, ~18.6 kWh)
    c_base_total = 63.8
    e_base_total = 18.6

    # Op 1 nominal values (early in shift, c_base ~ 1.2 min, e_base ~ 0.35 kWh)
    c_base_1 = 1.2
    e_base_1 = 0.35

    # Simulated micro-disturbance at Op 1: +1.0 min delay and +0.15 kWh excess energy
    c_act_1 = c_base_1 + 1.0  # 2.2 min
    e_act_1 = e_base_1 + 0.15  # 0.50 kWh

    cdi_norm = compute_normalized_cdi(
        c_actual_t=c_act_1,
        c_base_t=c_base_1,
        c_base_total=c_base_total,
        e_actual_t=e_act_1,
        e_base_t=e_base_1,
        e_base_total=e_base_total,
        alpha=0.5,
    )

    # 1. Assert strictly finite and bounded
    assert np.isfinite(cdi_norm), "Normalized CDI must be finite."
    assert 0.0 <= cdi_norm < 0.50, f"Expected normalized CDI < 0.50 at op 1, got {cdi_norm:.4f}"

    # Specifically, for +1.0 min on a 63.8 min shift, c_dev = 1.0/63.8 = 0.01567
    # For +0.15 kWh on an 18.6 kWh shift, e_dev = 0.15/18.6 = 0.00806
    # cdi = 0.5 * 0.01567 + 0.5 * 0.00806 = 0.01187 (well below 0.12 threshold and far below 0.50)
    assert cdi_norm < 0.05, f"Expected early shift CDI < 0.05, got {cdi_norm:.4f}"

    # Severe macro stoppage (+8.0 min stoppage at Op 1)
    c_act_severe = c_base_1 + 8.0
    cdi_severe = compute_normalized_cdi(
        c_actual_t=c_act_severe,
        c_base_t=c_base_1,
        c_base_total=c_base_total,
        e_actual_t=e_act_1,
        e_base_t=e_base_1,
        e_base_total=e_base_total,
        alpha=0.5,
    )
    # 0.5 * (8.0/63.8) + 0.5 * (0.15/18.6) = 0.5 * 0.1254 + 0.0040 = 0.0667 < 0.50
    assert 0.0 < cdi_severe < 0.50, f"Severe stoppage at op 1 must still give normalized CDI < 0.50, got {cdi_severe:.4f}"


def test_normalized_cdi_boundary_and_validation_errors():
    """Verify input validation and boundary conditions for compute_normalized_cdi."""
    # Zero disturbance gives exactly 0.0
    cdi_zero = compute_normalized_cdi(
        c_actual_t=10.0,
        c_base_t=10.0,
        c_base_total=60.0,
        e_actual_t=5.0,
        e_base_t=5.0,
        e_base_total=20.0,
    )
    assert cdi_zero == 0.0, f"Identical execution must give CDI = 0.0, got {cdi_zero}"

    # Invalid c_base_total <= 0 raises ValueError
    with pytest.raises(ValueError, match="c_base_total must be strictly positive"):
        compute_normalized_cdi(10.0, 10.0, 0.0, 5.0, 5.0, 20.0)

    # Invalid e_base_total <= 0 raises ValueError
    with pytest.raises(ValueError, match="e_base_total must be strictly positive"):
        compute_normalized_cdi(10.0, 10.0, 60.0, 5.0, 5.0, -1.0)

    # Invalid alpha raises ValueError
    with pytest.raises(ValueError, match="alpha must be in"):
        compute_normalized_cdi(10.0, 10.0, 60.0, 5.0, 5.0, 20.0, alpha=1.5)


def test_hardware_latency_constants():
    """Verify hardware-measured latency constants and calculated speedup."""
    assert EDGE_ONNX_LATENCY_MEAN_MS == 0.0214
    assert EDGE_ONNX_LATENCY_MIN_MS == 0.0180
    assert EDGE_ONNX_LATENCY_MAX_MS == 0.0270
    assert CLOUD_SOLVER_LATENCY_MS in (25000.0, 25400.0)
    assert MQTT_RTT_MEAN_MS == 0.2240

    # Speedup: 25.4 s / 0.0214 ms = 1,186,915x (~1.19 x 10^6x)
    assert 1.15e6 <= SPEEDUP_FACTOR <= 1.20e6, f"Speedup factor {SPEEDUP_FACTOR} outside [1.15e6, 1.20e6]"


def test_reproducibility_specs_files_exist_and_valid():
    """Verify that reproducibility_specs.json and .md exist, parse correctly, and contain all required sections."""
    json_path = Path("data/benchmarks/reproducibility_specs.json")
    md_path = Path("data/benchmarks/reproducibility_specs.md")

    # Generate if not exists
    if not json_path.exists() or not md_path.exists():
        export_reproducibility_specs(json_path, md_path)

    assert json_path.exists(), "reproducibility_specs.json must exist."
    assert md_path.exists(), "reproducibility_specs.md must exist."

    with open(json_path, "r", encoding="utf-8") as f:
        specs = json.load(f)

    # Verify JSON structure
    assert specs["evaluation_protocol"]["train_seeds"] == [42, 101, 2026]
    assert specs["evaluation_protocol"]["val_seeds"] == list(range(201, 211))
    assert specs["evaluation_protocol"]["test_seeds"] == list(range(301, 316))
    assert specs["hardware_latencies"]["edge_onnx_latency_mean_ms"] == 0.0214
    assert specs["cdi_formulation"]["knee_point_threshold"] == 0.12
    assert specs["observation_space"]["dimension"] == 87
    assert specs["action_space"]["dimension"] == 360
    assert "reward_function" in specs
    assert "hyperparameters" in specs

    # Verify Markdown content
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    assert "TRAIN_SEEDS" in md_text
    assert "VAL_SEEDS" in md_text
    assert "TEST_SEEDS" in md_text
    assert "0.0214 ms" in md_text
    assert "CDI(t)" in md_text
    assert "NSGA-III" in md_text
    assert "MaskablePPO" in md_text

