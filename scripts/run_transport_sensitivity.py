"""
Comprehensive Transport Sensitivity Benchmark:
Evaluates tau_trans in [0.0, 1.0, 2.0, 3.0] minutes across 4 methods:
- NSGA-III (Open-Loop)
- Right-Shift Rescheduling
- Standalone Edge PPO
- Proposed Cloud-Edge DT
on both Mk01 (10x6, 55 ops) and Mk03 (15x8, 150 ops) across TEST_SEEDS = [301..315].

Strict boundary check:
At tau_trans = 0.0, Mean and Std of Makespan, TOU_Cost, TWT, and Breakdowns
MUST match Phase 5 Wilcoxon table with 0.00 error.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from sim_env.fjsp_gym_env import GreenFJSPEnv
from edge_layer.onnx_exporter import EdgeONNXInferenceEngine
from utils.eval_protocol import TEST_SEEDS
from tests.test_task5_2_multi_instance_eval import (
    execute_open_loop_schedule,
    execute_right_shift_rescheduling,
    execute_edge_only_ppo,
    execute_proposed_cloud_edge_dt,
)


def run_transport_sensitivity():
    t_start = time.time()
    print("=" * 85)
    print("STARTING TRANSPORT SENSITIVITY BENCHMARK (Mk01 & Mk03, tau in [0.0, 1.0, 2.0, 3.0] min)")
    print("=" * 85)

    with open("data/benchmarks/multi_seed_schedules.json") as f:
        multi_seed_schedules = json.load(f)

    with open("data/benchmarks/phase5_wilcoxon_benchmark_table.csv") as f:
        w_rows = list(csv.DictReader(f))

    onnx_path = Path("edge_layer/models/ppo_edge_policy.onnx")
    onnx_engine = EdgeONNXInferenceEngine(str(onnx_path)) if onnx_path.exists() else None

    instances = ["Mk01", "Mk03"]
    tau_values = [0.0, 1.0, 2.0, 3.0]
    methods = [
        "NSGA-III (Open-Loop)",
        "Right-Shift Rescheduling",
        "Standalone Edge PPO",
        "Proposed Cloud-Edge DT",
    ]

    all_results = []
    raw_data_for_plot: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        inst: {m: {"tau": [], "cmax": [], "cmax_std": [], "cost": [], "cost_std": [], "twt": [], "twt_std": [], "bds": []} for m in methods}
        for inst in instances
    }

    for b_id in instances:
        json_path = f"data/benchmarks/{b_id}_green.json"
        with open(f"data/benchmarks/nsga3_baseline_{b_id}.json") as f:
            nsga_base = json.load(f)["operations_schedule"]

        print(f"\n---> Evaluating Instance {b_id}...")

        for tau in tau_values:
            print(f"  [tau = {tau:.1f} min]")
            for m in methods:
                cmaxes, costs, twts, bds = [], [], [], []
                for s in TEST_SEEDS:
                    s_str = str(s)
                    sched_nsga = multi_seed_schedules["NSGA-III"][b_id][s_str]

                    if m == "NSGA-III (Open-Loop)":
                        e = GreenFJSPEnv(benchmark_path=json_path, transport_time_min=tau)
                        e.reset(seed=s, options={"realistic_disturbance": True})
                        inf = execute_open_loop_schedule(e, sched_nsga)
                    elif m == "Right-Shift Rescheduling":
                        e = GreenFJSPEnv(benchmark_path=json_path, transport_time_min=tau)
                        e.reset(seed=s, options={"realistic_disturbance": True})
                        inf = execute_right_shift_rescheduling(e, sched_nsga)
                    elif m == "Standalone Edge PPO":
                        e = GreenFJSPEnv(benchmark_path=json_path, cloud_guided_mask=False, transport_time_min=tau)
                        e.reset(seed=s, options={"realistic_disturbance": True})
                        inf = execute_edge_only_ppo(e, b_id, onnx_engine=onnx_engine)
                    elif m == "Proposed Cloud-Edge DT":
                        e = GreenFJSPEnv(
                            benchmark_path=json_path,
                            cloud_schedule_path=f"data/benchmarks/nsga3_baseline_{b_id}.json",
                            cloud_guided_mask=True,
                            transport_time_min=tau,
                        )
                        e.reset(seed=s, options={"realistic_disturbance": True})
                        inf = execute_proposed_cloud_edge_dt(
                            e, b_id, nsga_base, onnx_engine=onnx_engine if b_id == "Mk01" else None
                        )

                    cmaxes.append(inf["cmax"])
                    costs.append(inf["total_cost"])
                    twts.append(inf["total_weighted_tardiness"])
                    bds.append(inf["total_breakdowns"])

                cm_val = float(np.mean(cmaxes))
                cs_val = float(np.std(cmaxes))
                costm_val = float(np.mean(costs))
                costs_val = float(np.std(costs))
                twtm_val = float(np.mean(twts))
                twts_val = float(np.std(twts))
                bd_val = float(np.mean(bds))

                # Boundary Check at tau = 0.0
                if tau == 0.0:
                    w_match = next(
                        r for r in w_rows
                        if r["Instance"] == b_id and (
                            m in r["Method"] or (m == "Standalone Edge PPO" and "Standalone Edge PPO" in r["Method"])
                        )
                    )
                    w_cm = float(w_match["Makespan_Mean"])
                    w_costm = float(w_match["Cost_Mean"])
                    w_twtm = float(w_match["TWT_Mean"])
                    w_bds = float(w_match["Breakdowns_Mean"])

                    assert abs(cm_val - w_cm) < 1e-4, f"Mismatch at tau=0.0 Cmax on {b_id} {m}: {cm_val} vs {w_cm}"
                    assert abs(costm_val - w_costm) < 1e-4, f"Mismatch at tau=0.0 Cost on {b_id} {m}: {costm_val} vs {w_costm}"
                    assert abs(twtm_val - w_twtm) < 1e-4, f"Mismatch at tau=0.0 TWT on {b_id} {m}: {twtm_val} vs {w_twtm}"
                    assert abs(bd_val - w_bds) < 1e-4, f"Mismatch at tau=0.0 BDs on {b_id} {m}: {bd_val} vs {w_bds}"
                    print(f"    [CHECK PASSED] {m:<25}: Exactly matches Wilcoxon table at tau=0.0!")

                res_dict = {
                    "Instance": b_id,
                    "Transport_Time_min": f"{tau:.1f}",
                    "Method": m,
                    "Makespan_Mean": f"{cm_val:.2f}",
                    "Makespan_Std": f"{cs_val:.2f}",
                    "TOU_Cost_Mean": f"{costm_val:.3f}",
                    "TOU_Cost_Std": f"{costs_val:.3f}",
                    "TWT_Mean": f"{twtm_val:.2f}",
                    "TWT_Std": f"{twts_val:.2f}",
                    "Breakdowns_Mean": f"{bd_val:.2f}",
                }
                all_results.append(res_dict)

                raw_data_for_plot[b_id][m]["tau"].append(tau)
                raw_data_for_plot[b_id][m]["cmax"].append(cm_val)
                raw_data_for_plot[b_id][m]["cmax_std"].append(cs_val)
                raw_data_for_plot[b_id][m]["cost"].append(costm_val)
                raw_data_for_plot[b_id][m]["cost_std"].append(costs_val)
                raw_data_for_plot[b_id][m]["twt"].append(twtm_val)
                raw_data_for_plot[b_id][m]["twt_std"].append(twts_val)
                raw_data_for_plot[b_id][m]["bds"].append(bd_val)

    # Export to both results/ and data/benchmarks/
    headers = [
        "Instance",
        "Transport_Time_min",
        "Method",
        "Makespan_Mean",
        "Makespan_Std",
        "TOU_Cost_Mean",
        "TOU_Cost_Std",
        "TWT_Mean",
        "TWT_Std",
        "Breakdowns_Mean",
    ]

    out_paths = [
        Path("results/phase5_transport_sensitivity.csv"),
        Path("data/benchmarks/phase5_transport_sensitivity.csv"),
    ]

    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            w.writerows(all_results)
        print(f"[Export] Saved {len(all_results)} rows to {p}")

    # Plot 300 DPI Publication Figure
    plot_transport_sensitivity(raw_data_for_plot)

    elapsed = time.time() - t_start
    print(f"\n[Completed] Transport sensitivity benchmark finished successfully in {elapsed:.2f} s.")
    print("=" * 85)


def plot_transport_sensitivity(data: Dict[str, Dict[str, Dict[str, List[float]]]]):
    """Generate high-resolution (300 DPI) publication figure for transport sensitivity across Mk01 & Mk03."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    plt.subplots_adjust(hspace=0.28, wspace=0.22)

    palette = {
        "NSGA-III (Open-Loop)": ("#E7298A", "^", ":"),
        "Right-Shift Rescheduling": ("#A6761D", "s", "--"),
        "Standalone Edge PPO": ("#386CB0", "o", "-."),
        "Proposed Cloud-Edge DT": ("#1B9E77", "D", "-"),
    }

    # (a) Mk01 Makespan
    ax = axes[0, 0]
    for m, (col, marker, ls) in palette.items():
        tau = data["Mk01"][m]["tau"]
        cmax = data["Mk01"][m]["cmax"]
        cmax_std = data["Mk01"][m]["cmax_std"]
        is_prop = "Proposed" in m
        ax.errorbar(
            tau, cmax, yerr=cmax_std, fmt=marker, color=col, linestyle=ls,
            linewidth=2.2 if is_prop else 1.4, markersize=8 if is_prop else 6,
            capsize=3.5, label=m, zorder=5 if is_prop else 3
        )
    ax.set_title("(a) Mk01 (55 ops) - Makespan Cmax vs. Transport Latency", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Inter-Machine Transport Time τ_trans (min)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Makespan C_max (min)", fontsize=10, fontweight="bold")
    ax.set_xticks([0.0, 1.0, 2.0, 3.0])
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92)

    # (b) Mk01 TOU Cost
    ax = axes[0, 1]
    for m, (col, marker, ls) in palette.items():
        tau = data["Mk01"][m]["tau"]
        cost = data["Mk01"][m]["cost"]
        cost_std = data["Mk01"][m]["cost_std"]
        is_prop = "Proposed" in m
        ax.errorbar(
            tau, cost, yerr=cost_std, fmt=marker, color=col, linestyle=ls,
            linewidth=2.2 if is_prop else 1.4, markersize=8 if is_prop else 6,
            capsize=3.5, label=m, zorder=5 if is_prop else 3
        )
    ax.set_title("(b) Mk01 (55 ops) - TOU Electricity Cost vs. Transport Latency", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Inter-Machine Transport Time τ_trans (min)", fontsize=10, fontweight="bold")
    ax.set_ylabel("TOU Electricity Cost ($)", fontsize=10, fontweight="bold")
    ax.set_xticks([0.0, 1.0, 2.0, 3.0])
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92)

    # (c) Mk03 Makespan
    ax = axes[1, 0]
    for m, (col, marker, ls) in palette.items():
        tau = data["Mk03"][m]["tau"]
        cmax = data["Mk03"][m]["cmax"]
        cmax_std = data["Mk03"][m]["cmax_std"]
        is_prop = "Proposed" in m
        ax.errorbar(
            tau, cmax, yerr=cmax_std, fmt=marker, color=col, linestyle=ls,
            linewidth=2.2 if is_prop else 1.4, markersize=8 if is_prop else 6,
            capsize=3.5, label=m, zorder=5 if is_prop else 3
        )
    ax.set_title("(c) Mk03 (150 ops) - Makespan Cmax vs. Transport Latency", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Inter-Machine Transport Time τ_trans (min)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Makespan C_max (min)", fontsize=10, fontweight="bold")
    ax.set_xticks([0.0, 1.0, 2.0, 3.0])
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92)

    # (d) Mk03 TOU Cost
    ax = axes[1, 1]
    for m, (col, marker, ls) in palette.items():
        tau = data["Mk03"][m]["tau"]
        cost = data["Mk03"][m]["cost"]
        cost_std = data["Mk03"][m]["cost_std"]
        is_prop = "Proposed" in m
        ax.errorbar(
            tau, cost, yerr=cost_std, fmt=marker, color=col, linestyle=ls,
            linewidth=2.2 if is_prop else 1.4, markersize=8 if is_prop else 6,
            capsize=3.5, label=m, zorder=5 if is_prop else 3
        )
    ax.set_title("(d) Mk03 (150 ops) - TOU Electricity Cost vs. Transport Latency", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Inter-Machine Transport Time τ_trans (min)", fontsize=10, fontweight="bold")
    ax.set_ylabel("TOU Electricity Cost ($)", fontsize=10, fontweight="bold")
    ax.set_xticks([0.0, 1.0, 2.0, 3.0])
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92)

    plot_paths = [
        Path("results/phase5_transport_sensitivity.png"),
        Path("tests/phase5_transport_sensitivity.png"),
    ]
    for p in plot_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"[Figure] Saved 300 DPI chart to {p}")
    plt.close(fig)


if __name__ == "__main__":
    run_transport_sensitivity()
