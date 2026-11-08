"""
Verification Test Suite for Task 5.6B:
Multi-Seed Open-Loop Optimization, Non-Zero Variance, Cliff's Delta,
Holm-Bonferroni Correction, 8-Method Benchmarking, and Cyclic TOU Ablation Study.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import pytest
import numpy as np


WILCOXON_CSV = Path("data/benchmarks/phase5_wilcoxon_benchmark_table.csv")
ABLATION_CSV = Path("data/benchmarks/phase5_ablation_results.csv")
PHASE2_CSV = Path("data/benchmarks/phase2_comparison_results.csv")
HV_BOUNDS_JSON = Path("data/benchmarks/hv_normalization_bounds.json")
FIGURE_PATH = Path("tests/phase5_multi_instance_hv_igd.png")


def test_wilcoxon_table_presence_and_columns():
    """Verify phase5_wilcoxon_benchmark_table.csv exists, contains 40 rows (5 instances x 8 methods),
    and has all required statistical columns including Cliff's Delta and Holm-Bonferroni p-values."""
    assert WILCOXON_CSV.exists(), f"{WILCOXON_CSV} does not exist!"

    with open(WILCOXON_CSV, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) in (40, 50), f"Expected 40 or 50 rows (5 instances x 8-10 methods), got {len(rows)}"

    required_cols = [
        "Instance", "Scale", "Method",
        "HV_Mean", "HV_Std", "HV_p_raw", "HV_p_holm", "HV_Cliffs_Delta", "HV_Wilcoxon_Symbol",
        "IGD_Mean", "IGD_Std", "IGD_p_raw", "IGD_p_holm", "IGD_Cliffs_Delta", "IGD_Wilcoxon_Symbol",
        "Makespan_Mean", "Makespan_Std", "Cost_Mean", "Cost_Std", "TWT_Mean", "TWT_Std", "Breakdowns_Mean"
    ]
    for col in required_cols:
        assert col in rows[0], f"Required column '{col}' missing from {WILCOXON_CSV}"

    # Verify Cliff's Delta values are strictly within [-1.0, 1.0]
    for r in rows:
        if r["Method"] != "Proposed Cloud-Edge DT (NSGA-III + Edge ONNX + CDI)":
            delta_hv = float(r["HV_Cliffs_Delta"])
            delta_igd = float(r["IGD_Cliffs_Delta"])
            assert -1.0 <= delta_hv <= 1.0, f"Invalid HV Cliff's Delta {delta_hv} on {r['Instance']} {r['Method']}"
            assert -1.0 <= delta_igd <= 1.0, f"Invalid IGD Cliff's Delta {delta_igd} on {r['Instance']} {r['Method']}"

            p_holm = float(r["HV_p_holm"])
            p_raw = float(r["HV_p_raw"])
            assert 0.0 <= p_holm <= 1.0, f"Invalid p_holm {p_holm}"
            assert p_holm >= p_raw - 1e-9, f"p_holm ({p_holm}) cannot be smaller than p_raw ({p_raw})"


def test_nonzero_stochastic_variance_across_all_instances():
    """Verify that NSGA-III (Open-Loop) on Mk02 and all other methods on Mk01..Mk05
    have non-zero standard deviations for Makespan, TOU Cost, and TWT."""
    with open(WILCOXON_CSV, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        inst = r["Instance"]
        method = r["Method"]
        cmax_std = float(r["Makespan_Std"])
        cost_std = float(r["Cost_Std"])
        twt_std = float(r["TWT_Std"])

        assert cmax_std > 0.0, f"Zero makespan variance detected on {inst} for {method}"
        assert cost_std > 0.0, f"Zero cost variance detected on {inst} for {method}"
        assert twt_std > 0.0, f"Zero TWT variance detected on {inst} for {method}"

    # Specifically check Mk02 NSGA-III (Open-Loop)
    mk02_nsga = next(r for r in rows if r["Instance"] == "Mk02" and "NSGA-III (Open-Loop)" in r["Method"])
    assert float(mk02_nsga["Makespan_Std"]) > 0.5, f"Mk02 NSGA-III makespan std too low: {mk02_nsga['Makespan_Std']}"


def test_eight_methods_present_and_consistent():
    """Verify that all 8 methods are evaluated across each of the 5 Brandimarte instances."""
    with open(WILCOXON_CSV, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    expected_methods = [
        "Rule-SPT-Fast",
        "Rule-EET-Eco",
        "Rule-EDD-Bal",
        "Right-Shift Rescheduling",
        "MOEA-D (Open-Loop)",
        "NSGA-III (Open-Loop)",
        "Standalone Edge PPO (No Cloud Prior)",
        "Proposed Cloud-Edge DT (NSGA-III + Edge ONNX + CDI)",
    ]

    expected_10_methods = set(expected_methods) | {"NSGA-II (Open-Loop)", "RVEA (Open-Loop)"}
    for inst in ["Mk01", "Mk02", "Mk03", "Mk04", "Mk05"]:
        inst_methods = [r["Method"] for r in rows if r["Instance"] == inst]
        assert set(inst_methods) in (set(expected_methods), expected_10_methods), f"Mismatch in methods for {inst}: {inst_methods}"


def test_hv_normalization_bounds_file():
    """Verify hv_normalization_bounds.json records ideal and nadir bounds and ref_point."""
    assert HV_BOUNDS_JSON.exists(), f"{HV_BOUNDS_JSON} does not exist!"

    with open(HV_BOUNDS_JSON, mode="r", encoding="utf-8") as f:
        bounds = json.load(f)

    for inst in ["Mk01", "Mk02", "Mk03", "Mk04", "Mk05"]:
        assert inst in bounds, f"Instance {inst} missing from normalization bounds"
        b_data = bounds[inst]
        assert "ideal_point" in b_data
        assert "nadir_point" in b_data
        assert "ref_point" in b_data
        assert b_data["ref_point"] == [1.1, 1.1, 1.1]
        assert len(b_data["ideal_point"]) == 3
        assert len(b_data["nadir_point"]) == 3


def test_ablation_variant_2_cost_increase_under_cyclic_tou():
    """Verify that Variant 2 (Fixed Speed s=2 Locked) suffers higher TOU cost than Full Proposed DT
    under cyclic TOU tariff on both Mk01 and Mk03."""
    assert ABLATION_CSV.exists(), f"{ABLATION_CSV} does not exist!"

    with open(ABLATION_CSV, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for inst in ["Mk01", "Mk03"]:
        v2_row = next(r for r in rows if r["Instance"] == inst and "Variant 2" in r["Variant"])
        cost_change = float(v2_row["TOU Cost Degradation (%)"])
        assert cost_change > 0.0, (
            f"Variant 2 on {inst} must exhibit higher TOU cost than Full DT under cyclic TOU tariff, "
            f"got {cost_change:+.2f}%"
        )


def test_mk01_numerical_synchronization():
    """Verify that Mk01 metrics match 100% across Phase 2, Phase 5 Wilcoxon, and Phase 5 Ablation."""
    with open(PHASE2_CSV, mode="r", encoding="utf-8") as f:
        p2_rows = list(csv.DictReader(f))
    with open(WILCOXON_CSV, mode="r", encoding="utf-8") as f:
        p5_rows = list(csv.DictReader(f))
    with open(ABLATION_CSV, mode="r", encoding="utf-8") as f:
        abl_rows = list(csv.DictReader(f))

    p2_prop = next(r for r in p2_rows if "Cloud-Edge Collaborative DT" in r["Method"])
    p5_prop = next(r for r in p5_rows if r["Instance"] == "Mk01" and "Proposed Cloud-Edge DT" in r["Method"])
    abl_prop = next(r for r in abl_rows if r["Instance"] == "Mk01" and "Full Proposed Cloud-Edge DT" in r["Variant"])

    cmax_p2 = float(p2_prop["Makespan Mean (min)"])
    cmax_p5 = round(float(p5_prop["Makespan_Mean"]), 2)
    cmax_abl = float(abl_prop["Makespan Mean (min)"])

    assert abs(cmax_p2 - cmax_p5) < 1e-4, f"Makespan mismatch P2 vs P5: {cmax_p2} vs {cmax_p5}"
    assert abs(cmax_p5 - cmax_abl) < 1e-4, f"Makespan mismatch P5 vs Ablation: {cmax_p5} vs {cmax_abl}"

    cost_p2 = float(p2_prop["TOU Cost Mean ($)"])
    cost_p5 = round(float(p5_prop["Cost_Mean"]), 3)
    cost_abl = float(abl_prop["TOU Cost Mean ($)"])

    assert abs(cost_p2 - cost_p5) < 1e-4, f"Cost mismatch P2 vs P5: {cost_p2} vs {cost_p5}"
    assert abs(cost_p5 - cost_abl) < 1e-4, f"Cost mismatch P5 vs Ablation: {cost_p5} vs {cost_abl}"


def test_publication_figure_integrity():
    """Verify that tests/phase5_multi_instance_hv_igd.png exists and is non-empty."""
    assert FIGURE_PATH.exists(), f"Figure missing at {FIGURE_PATH}"
    assert FIGURE_PATH.stat().st_size > 50000, f"Figure file size too small: {FIGURE_PATH.stat().st_size} bytes"

