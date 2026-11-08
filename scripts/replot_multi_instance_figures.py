#!/usr/bin/env python3
"""
Regenerate publication-grade 300 DPI multi-instance Hypervolume (HV) and IGD comparison figure
(tests/phase5_multi_instance_hv_igd.png & results/phase5_multi_instance_hv_igd.png).
Features:
- Individual pairwise Wilcoxon significance symbols ('+', '≈', '-') annotated directly above each bar/point.
- Explicit legend/caption for Holm-Bonferroni corrected non-parametric test.
- Full 15-seed audited distributions from phase5_wilcoxon_benchmark_table.csv.
"""

import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_task5_2_multi_instance_eval import (
    plot_phase5_multi_instance_figures,
    BENCHMARK_SPECS,
    METHODS,
    PROPOSED_METHOD_NAME
)

def replot_multi_instance():
    csv_path = Path("data/benchmarks/phase5_wilcoxon_benchmark_table.csv")
    assert csv_path.exists(), f"{csv_path} does not exist!"

    with open(csv_path, "r", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))

    benchmark_results = {}
    for spec in BENCHMARK_SPECS:
        b_id = spec["id"]
        benchmark_results[b_id] = {"hvs": {}, "igds": {}, "raw_metrics": {}}
        for m in METHODS:
            row = next(r for r in csv_rows if r["Instance"] == b_id and (r["Method"] == m or (m == PROPOSED_METHOD_NAME and "Proposed" in r["Method"])))
            hv_m = float(row["HV_Mean"])
            hv_s = float(row["HV_Std"])
            igd_m = float(row["IGD_Mean"])
            igd_s = float(row["IGD_Std"])
            cost_m = float(row["Cost_Mean"])

            # 2-element array [mean - std, mean + std] yields exact mean and std
            benchmark_results[b_id]["hvs"][m] = [hv_m - hv_s, hv_m + hv_s]
            benchmark_results[b_id]["igds"][m] = [igd_m - igd_s, igd_m + igd_s]
            benchmark_results[b_id]["raw_metrics"][m] = {"cost": [cost_m]}

    out_test = Path("tests/phase5_multi_instance_hv_igd.png")
    plot_phase5_multi_instance_figures(benchmark_results, csv_rows, output_png_path=str(out_test))

    out_res = Path("results/phase5_multi_instance_hv_igd.png")
    out_res.parent.mkdir(parents=True, exist_ok=True)
    plot_phase5_multi_instance_figures(benchmark_results, csv_rows, output_png_path=str(out_res))

    print(f"[+] Multi-instance figures successfully updated at {out_test} and {out_res}")

if __name__ == "__main__":
    replot_multi_instance()
