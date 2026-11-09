#!/usr/bin/env python3
"""
Automated Numerical Integrity and Cross-Table Consistency Verification Script.
Asserts that all simulation outputs, statistical tables, figures, and benchmark CSVs
satisfy strict mathematical and reproducibility invariants across the entire pipeline:

1. Boundary Identity at tau_trans = 0.0:
   Makespan, TOU Cost, TWT, and Tool Breakdowns match identically (< 1e-6) across:
   - data/benchmarks/phase5_transport_sensitivity.csv (tau = 0.0)
   - data/benchmarks/phase2_comparison_results.csv
   - data/benchmarks/phase5_wilcoxon_benchmark_table.csv (Mk01 Proposed DT)
   - data/benchmarks/phase5_ablation_results.csv (Mk01 Full Proposed DT)
2. Holm-Bonferroni Statistical Integrity:
   - Every benchmark problem (Mk01..Mk05) corrects all 9 baselines simultaneously (K = 9).
   - No row in phase5_wilcoxon_benchmark_table.csv with p_holm >= 0.05 carries a '+' or '-' sign.
   - p_holm >= p_raw for all rows, bounded in [0.0, 1.0].
3. Realistic Discrete-Event CDI Sensitivity:
   - Makespan and TOU Cost vary realistically across all 10 thresholds in phase5_cdi_sensitivity.csv.
   - Global minimum of Total System Cost Index is attained at theta_th* = 0.12.
4. Cross-Directory Synchronization:
   - Exact binary/string consistency of all benchmark CSVs between data/benchmarks/ and results/.
   - All 9 publication figures (.png) exist with valid sizes (> 50 KB).
"""

import csv
import filecmp
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Any


def verify_file_synchronization():
    """Verify that all CSV files in data/benchmarks/ and results/ are synchronized."""
    print("\n[CHECK 1/5] Verifying Cross-Directory File Synchronization...")
    csv_names = [
        "cloud_pareto_front_metrics.csv",
        "phase2_comparison_results.csv",
        "phase3_hardware_telemetry.csv",
        "phase3_hil_event_summary.csv",
        "phase4_hil_metrics.csv",
        "phase5_ablation_results.csv",
        "phase5_cdi_sensitivity.csv",
        "phase5_transport_sensitivity.csv",
        "phase5_wilcoxon_benchmark_table.csv",
    ]
    for name in csv_names:
        p_data = Path(f"data/benchmarks/{name}")
        p_res = Path(f"results/{name}")
        assert p_data.exists(), f"Missing file: {p_data}"
        assert p_res.exists(), f"Missing file: {p_res}"
        assert p_data.stat().st_size > 100, f"Empty file: {p_data}"
        assert filecmp.cmp(p_data, p_res), f"CSV mismatch between {p_data} and {p_res}"

    png_names = [
        "phase1_sample_gantt.png",
        "phase2_cloud_pareto.png",
        "phase2_ppo_training_curve.png",
        "phase3_hardware_verification.png",
        "phase4_closed_loop_hil.png",
        "phase5_ablation_and_sensitivity.png",
        "phase5_multi_instance_hv_igd.png",
        "phase5_transport_sensitivity.png",
        "single_machine_verification.png",
    ]
    for name in png_names:
        p_tests = Path(f"tests/{name}")
        p_res = Path(f"results/{name}")
        assert p_tests.exists(), f"Missing plot: {p_tests}"
        assert p_res.exists(), f"Missing plot: {p_res}"
        assert p_tests.stat().st_size > 50000, f"Plot unusually small: {p_tests}"
        assert p_res.stat().st_size > 50000, f"Plot unusually small: {p_res}"

    print("  -> PASSED: All 9 CSVs and 9 publication plots are strictly synchronized!")


def verify_tau_zero_boundary_invariance():
    """Verify exact numerical agreement for Mk01 Proposed DT at tau_trans = 0.0 across 4 tables."""
    print("\n[CHECK 2/5] Verifying Boundary Identity at tau_trans = 0.0 across 4 Benchmark Tables...")
    with open("data/benchmarks/phase5_transport_sensitivity.csv") as f:
        t_rows = list(csv.DictReader(f))
    with open("data/benchmarks/phase2_comparison_results.csv") as f:
        p2_rows = list(csv.DictReader(f))
    with open("data/benchmarks/phase5_wilcoxon_benchmark_table.csv") as f:
        w_rows = list(csv.DictReader(f))
    with open("data/benchmarks/phase5_ablation_results.csv") as f:
        a_rows = list(csv.DictReader(f))

    t_p = next(r for r in t_rows if r["Instance"] == "Mk01" and r["Transport_Time_min"] == "0.0" and "Proposed" in r["Method"])
    p2_p = next(r for r in p2_rows if "Cloud-Edge" in r["Method"])
    w_p = next(r for r in w_rows if r["Instance"] == "Mk01" and "Proposed" in r["Method"])
    a_p = next(r for r in a_rows if r["Instance"] == "Mk01" and "Full Proposed" in r["Variant"])

    cmax_t = float(t_p["Makespan_Mean"])
    cmax_p2 = float(p2_p["Makespan Mean (min)"])
    cmax_w = float(w_p["Makespan_Mean"])
    cmax_a = float(a_p["Makespan Mean (min)"])

    cost_t = float(t_p["TOU_Cost_Mean"])
    cost_p2 = float(p2_p["TOU Cost Mean ($)"])
    cost_w = float(w_p["Cost_Mean"])
    cost_a = float(a_p["TOU Cost Mean ($)"])

    twt_t = float(t_p["TWT_Mean"])
    twt_p2 = float(p2_p["TWT Mean (min)"])
    twt_w = float(w_p["TWT_Mean"])
    twt_a = float(a_p["TWT Mean (min)"])

    bds_t = float(t_p["Breakdowns_Mean"])
    bds_p2 = float(p2_p["Tool Breakdowns Mean"])
    bds_w = float(w_p["Breakdowns_Mean"])
    bds_a = float(a_p["Breakdowns Mean"])

    print(f"  Transport Table (tau=0.0): Cmax={cmax_t:.2f}, Cost={cost_t:.3f}, TWT={twt_t:.2f}, BDs={bds_t:.2f}")
    print(f"  Phase 2 Table            : Cmax={cmax_p2:.2f}, Cost={cost_p2:.3f}, TWT={twt_p2:.2f}, BDs={bds_p2:.2f}")
    print(f"  Wilcoxon Benchmark Table : Cmax={cmax_w:.2f}, Cost={cost_w:.3f}, TWT={twt_w:.2f}, BDs={bds_w:.2f}")
    print(f"  Ablation Study Table     : Cmax={cmax_a:.2f}, Cost={cost_a:.3f}, TWT={twt_a:.2f}, BDs={bds_a:.2f}")

    # Assert absolute numerical identity on rounded publication metrics (< 1e-6)
    assert abs(cmax_t - cmax_p2) < 1e-6, f"Cmax mismatch: Transport {cmax_t} vs Phase2 {cmax_p2}"
    assert abs(cmax_t - round(cmax_w, 2)) < 1e-6, f"Cmax mismatch: Transport {cmax_t} vs Wilcoxon {cmax_w}"
    assert abs(cmax_t - cmax_a) < 1e-6, f"Cmax mismatch: Transport {cmax_t} vs Ablation {cmax_a}"

    assert abs(cost_t - cost_p2) < 1e-6, f"Cost mismatch: Transport {cost_t} vs Phase2 {cost_p2}"
    assert abs(cost_t - round(cost_w, 3)) < 1e-6, f"Cost mismatch: Transport {cost_t} vs Wilcoxon {cost_w}"
    assert abs(cost_t - cost_a) < 1e-6, f"Cost mismatch: Transport {cost_t} vs Ablation {cost_a}"

    assert abs(twt_t - twt_p2) < 1e-6, f"TWT mismatch: Transport {twt_t} vs Phase2 {twt_p2}"
    assert abs(twt_t - round(twt_w, 2)) < 1e-6, f"TWT mismatch: Transport {twt_t} vs Wilcoxon {twt_w}"
    assert abs(twt_t - twt_a) < 1e-6, f"TWT mismatch: Transport {twt_t} vs Ablation {twt_a}"

    assert abs(bds_t - bds_p2) < 1e-6, f"Breakdowns mismatch: Transport {bds_t} vs Phase2 {bds_p2}"
    assert abs(bds_t - bds_w) < 1e-6, f"Breakdowns mismatch: Transport {bds_t} vs Wilcoxon {bds_w}"
    assert abs(bds_t - bds_a) < 1e-6, f"Breakdowns mismatch: Transport {bds_t} vs Ablation {bds_a}"

    print("  -> PASSED: Boundary condition at tau_trans = 0.0 matches identically across all 4 tables!")


def verify_wilcoxon_holm_integrity():
    """Verify Holm-Bonferroni correction properties and absence of invalid markers."""
    print("\n[CHECK 3/5] Verifying Statistical Invariants in Phase 5 Wilcoxon Benchmark Table...")
    with open("data/benchmarks/phase5_wilcoxon_benchmark_table.csv") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 50, f"Expected 50 rows (5 instances x 10 methods), got {len(rows)}"

    instances = ["Mk01", "Mk02", "Mk03", "Mk04", "Mk05"]
    for inst in instances:
        inst_rows = [r for r in rows if r["Instance"] == inst and "Proposed" not in r["Method"]]
        assert len(inst_rows) == 9, f"Expected exactly 9 baseline comparisons per instance (K=9), got {len(inst_rows)}"

        # Check p-value ordering and Holm adjustment
        p_raws = [float(r["HV_p_raw"]) for r in inst_rows]
        p_holms = [float(r["HV_p_holm"]) for r in inst_rows]

        for p_r, p_h in zip(p_raws, p_holms):
            assert p_h >= p_r - 1e-9, f"Holm p-value {p_h} cannot be smaller than raw p-value {p_r}"
            assert 0.0 <= p_h <= 1.0, f"p_holm out of [0, 1]: {p_h}"

        min_raw = min(p_raws)
        matching_holm = [h for r, h in zip(p_raws, p_holms) if r == min_raw][0]
        expected_min_holm = min(1.0, 9.0 * min_raw)
        assert abs(matching_holm - expected_min_holm) < 1e-4, f"Lowest p-value not adjusted by K=9: {matching_holm} vs {expected_min_holm}"

    # Verify that NO row with p_holm >= 0.05 has a '+' or '-' sign
    for r in rows:
        if "Proposed" in r["Method"]:
            continue
        p_h_hv = float(r.get("HV_p_holm", 1.0))
        sig_hv = r.get("HV_Significance", "")
        if p_h_hv >= 0.05:
            assert sig_hv not in ["+", "-"], f"Violation: {r['Instance']} {r['Method']} has p_holm={p_h_hv} >= 0.05 but marker='{sig_hv}'"
            assert sig_hv in ["≈", ""], f"Unexpected non-significant marker: {sig_hv}"

        p_h_igd = float(r.get("IGD_p_holm", 1.0))
        sig_igd = r.get("IGD_Significance", "")
        if p_h_igd >= 0.05:
            assert sig_igd not in ["+", "-"], f"Violation: {r['Instance']} {r['Method']} has IGD p_holm={p_h_igd} >= 0.05 but marker='{sig_igd}'"
            assert sig_igd in ["≈", ""], f"Unexpected non-significant marker: {sig_igd}"

    print("  -> PASSED: Holm-Bonferroni correction (K=9) and honest significance markers strictly verified!")


def verify_cdi_sensitivity_variation():
    """Verify that CDI sensitivity table has realistic variance across thresholds."""
    print("\n[CHECK 4/5] Verifying CDI Threshold Sensitivity Table Dynamics...")
    with open("data/benchmarks/phase5_cdi_sensitivity.csv") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 10, f"Expected 10 thresholds in grid, got {len(rows)}"

    thresholds = [float(r["CDI_Threshold"]) for r in rows]
    makespans = [float(r["Makespan_Mean (min)"]) for r in rows]
    costs = [float(r["TOU_Cost_Mean ($)"]) for r in rows]
    sys_costs = [float(r["Total_System_Cost_Index"]) for r in rows]
    calls = [float(r["Cloud_Calls_Mean"]) for r in rows]

    # Verify variance across grid
    assert len(set(makespans)) >= 7, f"Makespans appear unphysically collapsed: {set(makespans)}"
    assert len(set(costs)) >= 7, f"TOU costs appear unphysically collapsed: {set(costs)}"

    # Check non-flat plateau: values between 0.02 and 0.12 must differ
    row_002 = next(r for r in rows if abs(float(r["CDI_Threshold"]) - 0.02) < 1e-3)
    row_012 = next(r for r in rows if abs(float(r["CDI_Threshold"]) - 0.12) < 1e-3)
    row_050 = next(r for r in rows if abs(float(r["CDI_Threshold"]) - 0.50) < 1e-3)

    assert float(row_002["Makespan_Mean (min)"]) != float(row_012["Makespan_Mean (min)"]), "Makespan at 0.02 and 0.12 cannot be identical"
    assert float(row_002["TOU_Cost_Mean ($)"]) != float(row_012["TOU_Cost_Mean ($)"]), "Cost at 0.02 and 0.12 cannot be identical"
    assert float(row_002["Cloud_Calls_Mean"]) > 40.0, f"Expected heavy chatter at theta=0.02, got {row_002['Cloud_Calls_Mean']}"
    assert float(row_050["Cloud_Calls_Mean"]) == 0.0, f"Expected zero calls at theta=0.50, got {row_050['Cloud_Calls_Mean']}"

    # Global minimum of Total System Cost Index must be at theta_th* = 0.12
    min_sys_idx = min(sys_costs)
    opt_idx = sys_costs.index(min_sys_idx)
    opt_theta = thresholds[opt_idx]
    assert opt_theta == 0.12, f"Optimal threshold should be 0.12, got {opt_theta} with index {min_sys_idx}"

    print(f"  Optimal Operating Point: theta_th* = {opt_theta:.2f} (Total_System_Cost_Index = {min_sys_idx:.4f})")
    print("  -> PASSED: Discrete-event CDI sensitivity exhibits realistic non-flat U-shaped trade-off!")


def verify_phase3_and_phase4_metrics():
    """Verify Phase 3 hardware telemetry and Phase 4 HIL metrics integrity."""
    print("\n[CHECK 5/5] Verifying Phase 3 DAQ & Phase 4 HIL Simulation Metrics...")
    with open("data/benchmarks/phase3_hardware_telemetry.csv") as f:
        t_rows = list(csv.DictReader(f))
    assert len(t_rows) == 600, f"Expected 600 telemetry records, got {len(t_rows)}"

    with open("data/benchmarks/phase3_hil_event_summary.csv") as f:
        s_rows = list(csv.DictReader(f))
    assert len(s_rows) >= 6, f"Expected at least 6 event summary segments, got {len(s_rows)}"

    with open("data/benchmarks/phase4_hil_metrics.csv") as f:
        p4_rows = list(csv.DictReader(f))
    assert len(p4_rows) == 3, f"Expected 3 HIL modes, got {len(p4_rows)}"

    mode1 = next(r for r in p4_rows if r["mode_id"] == "Mode 1")
    mode2 = next(r for r in p4_rows if r["mode_id"] == "Mode 2")
    mode3 = next(r for r in p4_rows if r["mode_id"] == "Mode 3")

    # Mode 1 has high periodic cloud calls and high network/solver overhead
    assert int(mode1["cloud_reschedule_calls"]) >= 8, "Mode 1 (Periodic Cloud) must invoke periodic reschedulings"
    assert float(mode1["total_solver_latency_s"]) > 20.0, "Mode 1 must accumulate high Cloud solver latency"

    # Mode 2 has zero cloud calls but suffers uncorrected drift and higher makespan/cost
    assert int(mode2["cloud_reschedule_calls"]) == 0, "Mode 2 must have 0 cloud calls"
    assert float(mode2["final_makespan_min"]) > float(mode3["final_makespan_min"]), "Mode 2 makespan must be higher than Mode 3"
    assert float(mode2["tou_energy_cost_usd"]) > float(mode3["tou_energy_cost_usd"]), "Mode 2 cost must be higher than Mode 3"

    # Mode 3 (Proposed Closed-Loop HIL) achieves ~80% reduction in cloud calls vs Mode 1 with lowest makespan
    assert int(mode3["cloud_reschedule_calls"]) <= 3, "Mode 3 must trigger only event-based cloud calls"
    assert float(mode3["reduction_in_cloud_calls_pct"]) >= 70.0, "Mode 3 must achieve >= 70% call reduction vs Mode 1"
    assert int(mode3["num_breakdowns"]) == 0, "Mode 3 must achieve 0 emergency breakdowns"

    print("  -> PASSED: Phase 3 DAQ tracking and Phase 4 HIL closed-loop metrics verified!")


def main():
    print("=" * 80)
    print("AUTOMATED NUMERICAL INTEGRITY AND CROSS-TABLE AUDIT SUITE")
    print("=" * 80)
    try:
        verify_file_synchronization()
        verify_tau_zero_boundary_invariance()
        verify_wilcoxon_holm_integrity()
        verify_cdi_sensitivity_variation()
        verify_phase3_and_phase4_metrics()
        print("\n" + "=" * 80)
        print("ALL NUMERICAL INTEGRITY AND CROSS-TABLE INVARIANTS PASSED 100%!")
        print("=" * 80 + "\n")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n[FAIL] Numerical Integrity Assertion Failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
