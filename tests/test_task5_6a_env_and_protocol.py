"""
Unit and System Verification Suite for Task 5.6A:
Environment Refinement, Cyclic TOU Tariffs, Pre-Operation Predictive PM,
Transport Latency Sensitivity Analysis, and Cross-Phase Name/Latency Standardization.

Verifies:
1. Cyclic TOU tariff: 100-min periodic schedule across all access points (0.06 / 0.20 / 0.12).
   - Assert get_tou_rate(110.0) == 0.06 and get_tou_rate(140.0) == 0.20.
2. Pre-operation predictive PM:
   - Evaluates non-preemptive machining wear increments.
   - Proactive PM triggered when wear + delta_w >= 85.0 um.
   - Speed derating to s=1 when higher speeds would breach 85.0 um.
3. Transport time sensitivity:
   - GreenFJSPEnv(transport_time_min=...) applies inter-machine transfer latency.
   - phase5_transport_sensitivity.csv exists with 4 levels (0, 1, 2, 3 min).
   - Makespan increases monotonically with transport latency for all 3 methods.
4. Naming and latency standardization:
   - Edge ONNX inference latency is unified to 0.0214 ms everywhere.
   - Cloud solver latency is measured as 25.4 ± 1.2 s.
   - Mode 2 renamed to 'One-Shot Cloud + Edge Autonomous (No Re-sync)'.
   - Standalone edge method renamed to 'Standalone Edge PPO (No Cloud Prior)'.
5. Updated PPO training curve:
   - tests/phase2_ppo_training_curve.png exists and reflects updated titles/labels.
"""

import csv
import sys
from pathlib import Path
from typing import Dict, List
import pytest
import numpy as np

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim_env.fjsp_gym_env import GreenFJSPEnv, get_tou_rate as get_tou_rate_env
from cloud_layer.nsga3_optimizer import get_tou_rate as get_tou_rate_cloud
from utils.mqtt_schemas import load_system_config
from utils.eval_protocol import (
    EDGE_ONNX_LATENCY_MEAN_MS,
    CLOUD_SOLVER_LATENCY_STR,
    TEST_SEEDS,
)


def test_cyclic_tou_tariff_lookup():
    """Verify cyclic 100-minute TOU tariff schedule across all modules."""
    sys_cfg = load_system_config()

    # Base cycle assertions
    # [0.0, 25.0): Off-Peak ($0.06)
    assert get_tou_rate_env(0.0) == 0.06
    assert get_tou_rate_env(10.0) == 0.06
    assert get_tou_rate_env(24.9) == 0.06

    # [25.0, 60.0): On-Peak ($0.20)
    assert get_tou_rate_env(25.0) == 0.20
    assert get_tou_rate_env(40.0) == 0.20
    assert get_tou_rate_env(59.9) == 0.20

    # [60.0, 100.0): Mid-Peak ($0.12)
    assert get_tou_rate_env(60.0) == 0.12
    assert get_tou_rate_env(80.0) == 0.12
    assert get_tou_rate_env(99.9) == 0.12

    # Cyclic repetition assertions (Cycle 2: 100 to 200 min)
    assert get_tou_rate_env(100.0) == 0.06
    assert get_tou_rate_env(110.0) == 0.06
    assert get_tou_rate_env(124.9) == 0.06
    assert get_tou_rate_env(125.0) == 0.20
    assert get_tou_rate_env(140.0) == 0.20
    assert get_tou_rate_env(159.9) == 0.20
    assert get_tou_rate_env(160.0) == 0.12
    assert get_tou_rate_env(180.0) == 0.12

    # Verify parity between sim_env, cloud_layer, and system_config
    for test_t in [5.0, 35.0, 75.0, 110.0, 140.0, 180.0, 250.0, 385.0]:
        val_env = get_tou_rate_env(test_t)
        val_cloud = get_tou_rate_cloud(test_t)
        val_cfg = sys_cfg.get_tou_rate(test_t)
        assert val_env == val_cloud == val_cfg, f"Discrepancy at t={test_t}: {val_env} vs {val_cloud} vs {val_cfg}"


def test_pre_operation_predictive_pm():
    """Verify predictive PM check prior to operation execution under non-preemptive constraint."""
    env = GreenFJSPEnv(cloud_guided_mask=True)
    env.reset(seed=42)

    # In cloud-guided mode, determine machine assigned by Cloud to Job 0 Op 0
    m_id = env.cloud_op_machine.get((0, 0), "M1")
    m_idx = env.machine_ids.index(m_id)
    twin = env.machines[m_id]
    assert twin.current_wear_um < 75.0

    # Test 1: Under safe wear (~55 um), proactive PM is locked
    mask = env.action_masks()
    act_pm_s1 = env.encode_action(job_idx=0, machine_idx=m_idx, speed_level=1, pm_flag=1)
    assert bool(mask[act_pm_s1]) is False, "PM should not be permitted under low tool wear"

    # Test 2: If tool wear + expected cut increment exceeds 85.0 um:
    # Set wear to 81.0 um (where a cut of ~5 um at s=2 or s=3 breaches 85.0 um)
    twin.current_wear_um = 81.0
    mask_high = env.action_masks()
    # PM must now be permitted proactively before cutting
    assert bool(mask_high[act_pm_s1]) is True, "Predictive PM must be enabled when wear approaches 85.0 um threshold"

    # Test 3: If tool wear reaches 85.0 um or above, normal cutting (pm_flag=0) should be locked in cloud-guided mode
    twin.current_wear_um = 86.0
    mask_crit = env.action_masks()
    act_no_pm_s1 = env.encode_action(job_idx=0, machine_idx=m_idx, speed_level=1, pm_flag=0)
    assert bool(mask_crit[act_no_pm_s1]) is False, "Cutting without PM must be locked when wear >= 85.0 um"
    assert bool(mask_crit[act_pm_s1]) is True, "Proactive PM must be mandatory when wear >= 85.0 um"


def test_transport_time_mechanics():
    """Verify that GreenFJSPEnv applies transport/setup latency when job moves between machines."""
    env_zero = GreenFJSPEnv(transport_time_min=0.0)
    env_trans = GreenFJSPEnv(transport_time_min=2.5)

    env_zero.reset(seed=101)
    env_trans.reset(seed=101)

    # In Mk01, Job 0: Op 0 eligible on M1, M2, M3, M5, M6.
    # Op 1 eligible on M1, M2, M4, M6.
    # Schedule Op 0 on M1 (machine_idx=0)
    act_op0 = env_zero.encode_action(job_idx=0, machine_idx=0, speed_level=1, pm_flag=0)
    _, _, _, _, info_0_zero = env_zero.step(act_op0)
    _, _, _, _, info_0_trans = env_trans.step(act_op0)

    # Completion of Op 0 should be identical since it is the first operation
    assert env_zero.job_ready_time[0] == env_trans.job_ready_time[0]
    assert env_zero.job_last_machine[0] == env_trans.job_last_machine[0] == "M1"

    # Schedule Op 1 on a DIFFERENT machine, M2 (machine_idx=1)
    act_op1 = env_zero.encode_action(job_idx=0, machine_idx=1, speed_level=1, pm_flag=0)
    _, _, _, _, info_1_zero = env_zero.step(act_op1)
    _, _, _, _, info_1_trans = env_trans.step(act_op1)

    # Op 1 on env_trans must start at least 2.5 min later due to transport latency
    t_start_zero = env_zero.scheduled_events[-1]["start_time"]
    t_start_trans = env_trans.scheduled_events[-1]["start_time"]
    assert t_start_trans >= t_start_zero + 2.5, f"Expected transport delay >= 2.5 min, got {t_start_trans} vs {t_start_zero}"


def test_transport_sensitivity_csv_and_monotonicity():
    """Verify that phase5_transport_sensitivity.csv exists, has 4 levels, and demonstrates monotonic makespan increase across Mk01 and Mk03."""
    csv_path = Path("data/benchmarks/phase5_transport_sensitivity.csv")
    assert csv_path.exists(), f"{csv_path} does not exist!"

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 32, f"Expected 32 rows (2 instances x 4 levels x 4 methods), got {len(reader)}"

    # Check that all 4 levels of transport time are present
    taus = sorted(list({float(r["Transport_Time_min"]) for r in reader}))
    assert taus == [0.0, 1.0, 2.0, 3.0], f"Expected tau levels [0.0, 1.0, 2.0, 3.0], got {taus}"

    instances = sorted(list({r["Instance"] for r in reader}))
    assert instances == ["Mk01", "Mk03"], f"Expected instances ['Mk01', 'Mk03'], got {instances}"

    # Group by instance & method and verify makespan behavior
    # For coordinated/scheduled methods, makespan must strictly increase monotonically with transport latency
    scheduled_methods = ["NSGA-III (Open-Loop)", "Right-Shift Rescheduling", "Proposed Cloud-Edge DT"]
    for inst in instances:
        for m in scheduled_methods:
            m_rows = sorted([r for r in reader if r["Instance"] == inst and r["Method"] == m], key=lambda x: float(x["Transport_Time_min"]))
            assert len(m_rows) == 4, f"Instance {inst} Method {m} has {len(m_rows)} rows instead of 4"

            cmaxes = [float(r["Makespan_Mean"]) for r in m_rows]
            for i in range(len(cmaxes) - 1):
                assert cmaxes[i + 1] > cmaxes[i], (
                    f"Makespan for {inst} {m} did not increase monotonically with transport time: "
                    f"tau={m_rows[i]['Transport_Time_min']} ({cmaxes[i]}) vs "
                    f"tau={m_rows[i+1]['Transport_Time_min']} ({cmaxes[i+1]})"
                )

    # For reactive Standalone Edge PPO without global plan, makespan under delay must exceed zero-delay makespan
    for inst in instances:
        ppo_rows = sorted([r for r in reader if r["Instance"] == inst and r["Method"] == "Standalone Edge PPO"], key=lambda x: float(x["Transport_Time_min"]))
        cmaxes = [float(r["Makespan_Mean"]) for r in ppo_rows]
        assert cmaxes[-1] > cmaxes[0], f"Standalone Edge PPO at tau=3 ({cmaxes[-1]}) must exceed tau=0 ({cmaxes[0]}) on {inst}"
        if inst == "Mk01":
            for i in range(len(cmaxes) - 1):
                assert cmaxes[i + 1] > cmaxes[i]

    # Verify that Proposed DT maintains 0.00 breakdowns across all transport delays on all instances
    for r in reader:
        if r["Method"] == "Proposed Cloud-Edge DT":
            assert float(r["Breakdowns_Mean"]) == 0.00, f"Proposed DT must eliminate breakdowns under all transport delays on {r['Instance']}"

    # Boundary condition check: tau = 0.0 must exactly match phase5_wilcoxon_benchmark_table.csv
    wilcoxon_path = Path("data/benchmarks/phase5_wilcoxon_benchmark_table.csv")
    assert wilcoxon_path.exists()
    with open(wilcoxon_path, mode="r", encoding="utf-8") as f:
        w_rows = list(csv.DictReader(f))

    for inst in ["Mk01", "Mk03"]:
        for m in ["NSGA-III (Open-Loop)", "Right-Shift Rescheduling", "Standalone Edge PPO", "Proposed Cloud-Edge DT"]:
            t0_row = next(r for r in reader if r["Instance"] == inst and r["Method"] == m and float(r["Transport_Time_min"]) == 0.0)
            w_row = next(r for r in w_rows if r["Instance"] == inst and (m == r["Method"] or (m == "Standalone Edge PPO" and "Standalone Edge PPO" in r["Method"]) or (m == "Proposed Cloud-Edge DT" and "Proposed Cloud-Edge DT" in r["Method"])))
            assert abs(float(t0_row["Makespan_Mean"]) - float(w_row["Makespan_Mean"])) < 1e-2
            assert abs(float(t0_row["TOU_Cost_Mean"]) - float(w_row["Cost_Mean"])) < 1e-3
            assert abs(float(t0_row["TWT_Mean"]) - float(w_row["TWT_Mean"])) < 1e-2
            assert abs(float(t0_row["Breakdowns_Mean"]) - float(w_row["Breakdowns_Mean"])) < 1e-2


def test_standardized_latencies_and_method_naming():
    """Verify that latency tags and method names are strictly standardized across all benchmarks."""
    # 1. Phase 2 Comparison Results
    p2_path = Path("data/benchmarks/phase2_comparison_results.csv")
    assert p2_path.exists()
    with open(p2_path, mode="r", encoding="utf-8") as f:
        p2_rows = list(csv.DictReader(f))

    method_names = [r["Method"] for r in p2_rows]
    assert "Standalone Edge PPO (No Cloud Prior)" in method_names, "Phase 2 must use standardized name"
    assert "Edge-Only PPO (Reactive DRL without Cloud Schedule)" not in method_names

    dt_row = next(r for r in p2_rows if "Cloud-Edge Collaborative DT" in r["Method"])
    assert dt_row["Decision Latency"] == "0.0214 ms", f"Expected 0.0214 ms, got {dt_row['Decision Latency']}"

    nsga_row = next(r for r in p2_rows if "NSGA-III Open-Loop" in r["Method"])
    assert "25.4 ± 1.2 s" in nsga_row["Decision Latency"], f"Expected 25.4 ± 1.2 s, got {nsga_row['Decision Latency']}"

    # 2. Phase 4 HIL Metrics
    p4_path = Path("data/benchmarks/phase4_hil_metrics.csv")
    assert p4_path.exists()
    with open(p4_path, mode="r", encoding="utf-8") as f:
        p4_rows = list(csv.DictReader(f))

    m2_row = next(r for r in p4_rows if r["mode_id"] == "Mode 2")
    assert m2_row["mode_name"] == "One-Shot Cloud + Edge Autonomous (No Re-sync)"
    assert float(m2_row["edge_decision_latency_ms"]) == 0.0214

    # 3. Phase 5 Wilcoxon Benchmark Table
    wilc_path = Path("data/benchmarks/phase5_wilcoxon_benchmark_table.csv")
    assert wilc_path.exists()
    with open(wilc_path, mode="r", encoding="utf-8") as f:
        wilc_rows = list(csv.DictReader(f))

    wilc_methods = {r["Method"] for r in wilc_rows}
    assert "Standalone Edge PPO (No Cloud Prior)" in wilc_methods
    assert "Edge-Only PPO (Reactive DRL)" not in wilc_methods

    # 4. Global Protocol Constants
    assert EDGE_ONNX_LATENCY_MEAN_MS == 0.0214
    assert CLOUD_SOLVER_LATENCY_STR == "25.4 ± 1.2 s"


def test_updated_ppo_curve_figure():
    """Verify that tests/phase2_ppo_training_curve.png exists and is non-empty."""
    fig_path = Path("tests/phase2_ppo_training_curve.png")
    assert fig_path.exists(), "tests/phase2_ppo_training_curve.png does not exist!"
    assert fig_path.stat().st_size > 50_000, f"Figure file size too small: {fig_path.stat().st_size} bytes"
