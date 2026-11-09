#!/usr/bin/env python3
"""
Level 1 Evaluation: Cloud Pareto Front Quality.

Evaluates the 4 Cloud Multi-Objective Evolutionary Algorithms:
    1. NSGA-II
    2. MOEA-D
    3. RVEA
    4. NSGA-III

Across all 5 Brandimarte benchmark instances (Mk01 - Mk05) on 15 test seeds (TEST_SEEDS = [301..315]).
For each algorithm and seed, captures the full population Pareto front P_s (up to 100 individuals).
Computes:
    - Population Hypervolume (HV_pop) with reference point r = (1.1, 1.1, 1.1)
    - Population Inverted Generational Distance (IGD_pop) relative to the unified empirical Pareto front P*_cloud
    - Pareto front size (number of non-dominated solutions)
    - Paired Wilcoxon Signed-Rank Test and Holm-Bonferroni correction vs NSGA-III (Cloud default)
    - Paired Rank-Biserial Correlation (r_rb) and Cliff's Delta

Exports to:
    - data/benchmarks/cloud_pareto_populations.json
    - data/benchmarks/cloud_pareto_front_metrics.csv
    - results/cloud_pareto_front_metrics.csv
"""

from __future__ import annotations

import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from cloud_layer.generate_moead_baselines import NormalizedGreenFJSPProblem
from cloud_layer.nsga3_optimizer import (
    GreenFJSPProblem,
    generate_seeded_population,
    run_nsga3_optimization,
)
from utils.eval_protocol import TEST_SEEDS
from utils.stat_metrics import (
    calculate_hypervolume,
    calculate_igd,
    compute_cliffs_delta,
    compute_paired_cohens_d,
    compute_paired_rank_biserial,
    extract_non_dominated_front,
    holm_bonferroni_correction,
    run_wilcoxon_test,
)


BENCHMARK_SPECS: List[Dict[str, Any]] = [
    {"id": "Mk01", "scale": "10x6, 55 ops", "ops": 55, "jobs": 10, "machines": 6},
    {"id": "Mk02", "scale": "10x6, 58 ops", "ops": 58, "jobs": 10, "machines": 6},
    {"id": "Mk03", "scale": "15x8, 150 ops", "ops": 150, "jobs": 15, "machines": 8},
    {"id": "Mk04", "scale": "15x8, 90 ops", "ops": 90, "jobs": 15, "machines": 8},
    {"id": "Mk05", "scale": "15x4, 106 ops", "ops": 106, "jobs": 15, "machines": 4},
]

CLOUD_MOEAS: List[str] = [
    "NSGA-II",
    "MOEA-D",
    "RVEA",
    "NSGA-III",
]


def _solve_single_moea(args: Tuple[str, str, int, int]) -> Tuple[str, str, int, List[List[float]]]:
    """Worker task to solve one MOEA on one instance for one seed."""
    algo_name, b_id, seed, n_gen = args
    json_path = f"data/benchmarks/{b_id}_green.json"
    prob = GreenFJSPProblem(benchmark_path=json_path)

    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
    pop_size = len(ref_dirs)  # 91

    if algo_name == "NSGA-III":
        res, _ = run_nsga3_optimization(
            benchmark_path=json_path,
            pop_size=pop_size,
            n_gen=n_gen,
            seed=seed,
            verbose=False,
        )
        front = extract_non_dominated_front(res.F)

    elif algo_name == "MOEA-D":
        X_init = generate_seeded_population(prob, pop_size=pop_size, seed=seed)
        F_init = []
        for x in X_init:
            out_eval: Dict[str, Any] = {}
            prob._evaluate(x, out_eval)
            F_init.append(out_eval["F"])
        F_arr = np.array(F_init)
        ideal_pt = np.min(F_arr, axis=0) * 0.90
        nadir_pt = np.max(F_arr, axis=0) * 1.10

        norm_prob = NormalizedGreenFJSPProblem(json_path, ideal_pt, nadir_pt)
        algo = MOEAD(
            ref_dirs=ref_dirs,
            n_neighbors=15,
            prob_neighbor_mating=0.7,
            sampling=X_init,
        )
        res = minimize(norm_prob, algo, ("n_gen", n_gen), seed=seed, verbose=False)
        raw_F = (res.F * norm_prob.denom) + norm_prob.ideal_point
        front = extract_non_dominated_front(raw_F)

    elif algo_name == "NSGA-II":
        X_init = generate_seeded_population(problem=prob, pop_size=pop_size, seed=seed)
        algo = NSGA2(
            pop_size=pop_size,
            sampling=X_init,
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(prob=1.0 / prob.n_var, eta=20),
        )
        res = minimize(prob, algo, ("n_gen", n_gen), seed=seed, verbose=False)
        front = extract_non_dominated_front(res.F)

    elif algo_name == "RVEA":
        X_init = generate_seeded_population(problem=prob, pop_size=pop_size, seed=seed)
        algo = RVEA(
            ref_dirs=ref_dirs,
            sampling=X_init,
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(prob=1.0 / prob.n_var, eta=20),
        )
        res = minimize(prob, algo, ("n_gen", n_gen), seed=seed, verbose=False)
        front = extract_non_dominated_front(res.F)

    else:
        raise ValueError(f"Unknown algo {algo_name}")

    return (algo_name, b_id, seed, front.tolist())


def compute_or_load_cloud_pareto_populations(
    cache_path: str = "data/benchmarks/cloud_pareto_populations.json",
    seeds: List[int] = TEST_SEEDS,
    n_gen: int = 20,
    force_recompute: bool = False,
) -> Dict[str, Dict[str, Dict[str, List[List[float]]]]]:
    """
    Compute or load Pareto populations for all 4 MOEAs across instances and seeds.
    Returns nested dict: populations[b_id][algo_name][seed_str] -> list of 3D points.
    """
    cache_file = Path(cache_path)
    if not force_recompute and cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Check completeness
        complete = True
        for spec in BENCHMARK_SPECS:
            b_id = spec["id"]
            if b_id not in data:
                complete = False
                break
            for algo in CLOUD_MOEAS:
                if algo not in data[b_id]:
                    complete = False
                    break
                for s in seeds:
                    if str(s) not in data[b_id][algo]:
                        complete = False
                        break
        if complete:
            print(f"[+] Loaded existing Cloud Pareto populations from: {cache_file}")
            return data

    print("[*] Computing Cloud Pareto populations across 4 MOEAs x 5 Benchmarks x 15 Seeds...")
    t0 = time.time()
    tasks = []
    for spec in BENCHMARK_SPECS:
        b_id = spec["id"]
        for algo in CLOUD_MOEAS:
            for s in seeds:
                tasks.append((algo, b_id, s, n_gen))

    populations: Dict[str, Dict[str, Dict[str, List[List[float]]]]] = {
        spec["id"]: {algo: {} for algo in CLOUD_MOEAS}
        for spec in BENCHMARK_SPECS
    }

    # Parallel execution using 16 workers
    with ProcessPoolExecutor(max_workers=16) as executor:
        results = executor.map(_solve_single_moea, tasks)
        for algo, b_id, s, front in results:
            populations[b_id][algo][str(s)] = front

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(populations, f, indent=2)

    elapsed = time.time() - t0
    print(f"[+] Cloud Pareto populations computed in {elapsed:.2f}s and saved to: {cache_file}")
    return populations


def evaluate_cloud_pareto_fronts(
    populations: Dict[str, Dict[str, Dict[str, List[List[float]]]]],
    output_csv_path: str = "data/benchmarks/cloud_pareto_front_metrics.csv",
    seeds: List[int] = TEST_SEEDS,
) -> List[Dict[str, Any]]:
    """
    Level 1 Evaluation:
    Computes HV_pop and IGD_pop for all 4 Cloud MOEAs.
    Saves results table to data/ and results/.
    """
    csv_rows: List[Dict[str, Any]] = []
    ref_pt = np.array([1.1, 1.1, 1.1])

    for spec in BENCHMARK_SPECS:
        b_id = spec["id"]
        scale_str = spec["scale"]

        # Pool all population solutions across all 4 MOEAs and 15 seeds for this benchmark
        all_pts_list = []
        for algo in CLOUD_MOEAS:
            for s in seeds:
                pts = populations[b_id][algo][str(s)]
                if len(pts) > 0:
                    all_pts_list.append(np.array(pts, dtype=np.float64))

        pooled = np.vstack(all_pts_list)
        ideal_cloud = np.min(pooled, axis=0)
        nadir_cloud = np.max(pooled, axis=0)
        denom_cloud = np.where((nadir_cloud - ideal_cloud) > 1e-9, nadir_cloud - ideal_cloud, 1.0)

        # Build unified empirical reference front across all 4 MOEAs
        all_norm = (pooled - ideal_cloud) / denom_cloud
        ref_front_cloud = extract_non_dominated_front(all_norm)

        # Compute metrics per seed for each MOEA
        algo_metrics: Dict[str, Dict[str, List[float]]] = {
            algo: {"hv_pop": [], "igd_pop": [], "size": []}
            for algo in CLOUD_MOEAS
        }

        for algo in CLOUD_MOEAS:
            for s in seeds:
                pts = np.array(populations[b_id][algo][str(s)], dtype=np.float64)
                if len(pts) == 0:
                    norm_pts = np.empty((0, 3))
                else:
                    norm_pts = np.clip((pts - ideal_cloud) / denom_cloud, 0.0, 1.0)
                    norm_pts = extract_non_dominated_front(norm_pts)

                hv_val = calculate_hypervolume(norm_pts, ref_point=ref_pt)
                igd_val = calculate_igd(norm_pts, ref_front_cloud)
                algo_metrics[algo]["hv_pop"].append(hv_val)
                algo_metrics[algo]["igd_pop"].append(igd_val)
                algo_metrics[algo]["size"].append(float(len(norm_pts)))

        # Paired Wilcoxon tests vs NSGA-III (Cloud reference optimizer)
        ref_algo = "NSGA-III"
        other_algos = [a for a in CLOUD_MOEAS if a != ref_algo]

        raw_p_hvs: Dict[str, float] = {}
        raw_p_igds: Dict[str, float] = {}
        r_rbs_hvs: Dict[str, float] = {}
        r_rbs_igds: Dict[str, float] = {}
        cliffs_hvs: Dict[str, float] = {}
        cliffs_igds: Dict[str, float] = {}

        for a in other_algos:
            p_hv, _ = run_wilcoxon_test(algo_metrics[ref_algo]["hv_pop"], algo_metrics[a]["hv_pop"], higher_is_better=True)
            p_igd, _ = run_wilcoxon_test(algo_metrics[ref_algo]["igd_pop"], algo_metrics[a]["igd_pop"], higher_is_better=False)
            raw_p_hvs[a] = p_hv
            raw_p_igds[a] = p_igd
            r_rbs_hvs[a] = compute_paired_rank_biserial(algo_metrics[ref_algo]["hv_pop"], algo_metrics[a]["hv_pop"])
            r_rbs_igds[a] = compute_paired_rank_biserial(algo_metrics[ref_algo]["igd_pop"], algo_metrics[a]["igd_pop"])
            cliffs_hvs[a] = compute_cliffs_delta(algo_metrics[ref_algo]["hv_pop"], algo_metrics[a]["hv_pop"])
            cliffs_igds[a] = compute_cliffs_delta(algo_metrics[ref_algo]["igd_pop"], algo_metrics[a]["igd_pop"])

        adj_p_hvs = holm_bonferroni_correction([raw_p_hvs[a] for a in other_algos])
        adj_p_igds = holm_bonferroni_correction([raw_p_igds[a] for a in other_algos])
        holm_p_hvs = {a: adj_p_hvs[idx] for idx, a in enumerate(other_algos)}
        holm_p_igds = {a: adj_p_igds[idx] for idx, a in enumerate(other_algos)}

        # Format CSV Rows
        for algo in CLOUD_MOEAS:
            hv_mean = float(np.mean(algo_metrics[algo]["hv_pop"]))
            hv_std = float(np.std(algo_metrics[algo]["hv_pop"]))
            igd_mean = float(np.mean(algo_metrics[algo]["igd_pop"]))
            igd_std = float(np.std(algo_metrics[algo]["igd_pop"]))
            size_mean = float(np.mean(algo_metrics[algo]["size"]))
            size_std = float(np.std(algo_metrics[algo]["size"]))

            if algo == ref_algo:
                hv_str = f"{hv_mean:.4f} ± {hv_std:.4f} (Base)"
                igd_str = f"{igd_mean:.4f} ± {igd_std:.4f} (Base)"
                hv_sym, hv_p_raw, hv_p_holm, hv_r_rb, hv_delta = "Base", 1.0, 1.0, 0.0, 0.0
                igd_sym, igd_p_raw, igd_p_holm, igd_r_rb, igd_delta = "Base", 1.0, 1.0, 0.0, 0.0
            else:
                hv_p_raw = raw_p_hvs[algo]
                hv_p_holm = holm_p_hvs[algo]
                hv_r_rb = r_rbs_hvs[algo]
                hv_delta = cliffs_hvs[algo]
                if hv_p_holm < 0.05:
                    hv_sym = "+" if float(np.mean(algo_metrics[ref_algo]["hv_pop"])) > hv_mean else "-"
                else:
                    hv_sym = "≈"

                igd_p_raw = raw_p_igds[algo]
                igd_p_holm = holm_p_igds[algo]
                igd_r_rb = r_rbs_igds[algo]
                igd_delta = cliffs_igds[algo]
                if igd_p_holm < 0.05:
                    igd_sym = "+" if float(np.mean(algo_metrics[ref_algo]["igd_pop"])) < igd_mean else "-"
                else:
                    igd_sym = "≈"

                hv_str = f"{hv_mean:.4f} ± {hv_std:.4f} ({hv_sym}, p={hv_p_holm:.4f})"
                igd_str = f"{igd_mean:.4f} ± {igd_std:.4f} ({igd_sym}, p={igd_p_holm:.4f})"

            row = {
                "Instance": b_id,
                "Scale": scale_str,
                "Algorithm": algo,
                "HV_pop Mean ± Std": hv_str,
                "IGD_pop Mean ± Std": igd_str,
                "Pareto_Size Mean ± Std": f"{size_mean:.1f} ± {size_std:.1f}",
                "HV_pop_Mean": hv_mean,
                "HV_pop_Std": hv_std,
                "HV_p_raw": hv_p_raw,
                "HV_p_holm": hv_p_holm,
                "HV_Wilcoxon_Symbol": hv_sym,
                "HV_Rank_Biserial": hv_r_rb,
                "HV_Cliffs_Delta": hv_delta,
                "IGD_pop_Mean": igd_mean,
                "IGD_pop_Std": igd_std,
                "IGD_p_raw": igd_p_raw,
                "IGD_p_holm": igd_p_holm,
                "IGD_Wilcoxon_Symbol": igd_sym,
                "IGD_Rank_Biserial": igd_r_rb,
                "IGD_Cliffs_Delta": igd_delta,
                "Pareto_Size_Mean": size_mean,
                "Pareto_Size_Std": size_std,
            }
            csv_rows.append(row)

    headers = [
        "Instance",
        "Scale",
        "Algorithm",
        "HV_pop Mean ± Std",
        "IGD_pop Mean ± Std",
        "Pareto_Size Mean ± Std",
        "HV_pop_Mean",
        "HV_pop_Std",
        "HV_p_raw",
        "HV_p_holm",
        "HV_Wilcoxon_Symbol",
        "HV_Rank_Biserial",
        "HV_Cliffs_Delta",
        "IGD_pop_Mean",
        "IGD_pop_Std",
        "IGD_p_raw",
        "IGD_p_holm",
        "IGD_Wilcoxon_Symbol",
        "IGD_Rank_Biserial",
        "IGD_Cliffs_Delta",
        "Pareto_Size_Mean",
        "Pareto_Size_Std",
    ]

    out_file = Path(output_csv_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(csv_rows)

    # Mirror to results/
    res_file = Path("results/cloud_pareto_front_metrics.csv")
    res_file.parent.mkdir(parents=True, exist_ok=True)
    with open(res_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"[+] Level 1 Cloud Pareto Front metrics saved to: {out_file} and {res_file}")
    return csv_rows


if __name__ == "__main__":
    pops = compute_or_load_cloud_pareto_populations()
    rows = evaluate_cloud_pareto_fronts(pops)
    print("\nSample Cloud Pareto Metrics Summary:")
    for r in rows[:8]:
        print(f"[{r['Instance']}] {r['Algorithm']:<10} | HV_pop={r['HV_pop Mean ± Std']} | IGD_pop={r['IGD_pop Mean ± Std']} | Size={r['Pareto_Size Mean ± Std']}")
