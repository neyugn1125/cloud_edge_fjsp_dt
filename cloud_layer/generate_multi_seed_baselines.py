"""
Generate Multi-Seed Open-Loop Schedules for NSGA-III and MOEA-D.
Evaluates 15 independent test seeds (TEST_SEEDS = [301..315]) across Mk01 - Mk05.
Saves to data/benchmarks/multi_seed_schedules.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from cloud_layer.generate_moead_baselines import NormalizedGreenFJSPProblem
from cloud_layer.nsga3_optimizer import (
    GreenFJSPProblem,
    generate_seeded_population,
    run_nsga3_optimization,
    select_compromise_solution,
)
from utils.eval_protocol import TEST_SEEDS


def generate_all_multi_seed_schedules(
    seeds: List[int] = TEST_SEEDS,
    output_path: str = "data/benchmarks/multi_seed_schedules.json",
    n_gen: int = 20,
) -> Dict[str, Any]:
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    benchmarks = ["Mk01", "Mk02", "Mk03", "Mk04", "Mk05"]
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
    pop_size = len(ref_dirs)  # 91 directions

    schedules: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = {
        "NSGA-III": {b: {} for b in benchmarks},
        "MOEA-D": {b: {} for b in benchmarks},
    }

    t_start = time.time()
    print("=" * 80)
    print(f"Generating Multi-Seed Schedules for {len(benchmarks)} instances across {len(seeds)} seeds...")
    print("=" * 80)

    for b in benchmarks:
        b_start = time.time()
        json_path = f"data/benchmarks/{b}_green.json"
        temp_prob = GreenFJSPProblem(benchmark_path=json_path)

        for s in seeds:
            s_str = str(s)

            # 1. NSGA-III Optimization for seed s
            res_nsga, prob_nsga = run_nsga3_optimization(
                benchmark_path=json_path,
                pop_size=92,
                n_gen=n_gen,
                seed=s,
                verbose=False,
            )
            _, _, comp_X_nsga, _ = select_compromise_solution(res_nsga.F, res_nsga.X)
            sched_nsga = prob_nsga.decode_schedule(comp_X_nsga)
            schedules["NSGA-III"][b][s_str] = sched_nsga["operations_schedule"]

            # 2. MOEA-D Optimization for seed s
            X_init_moead = generate_seeded_population(temp_prob, pop_size=pop_size, seed=s)
            F_init = []
            for x in X_init_moead:
                out_eval: Dict[str, Any] = {}
                temp_prob._evaluate(x, out_eval)
                F_init.append(out_eval["F"])
            F_arr = np.array(F_init)
            ideal_pt = np.min(F_arr, axis=0) * 0.90
            nadir_pt = np.max(F_arr, axis=0) * 1.10

            norm_prob = NormalizedGreenFJSPProblem(json_path, ideal_pt, nadir_pt)
            algo_moead = MOEAD(
                ref_dirs=ref_dirs,
                n_neighbors=15,
                prob_neighbor_mating=0.7,
                sampling=X_init_moead,
            )
            res_moead = minimize(norm_prob, algo_moead, ("n_gen", n_gen), seed=s, verbose=False)
            final_raw_F = (res_moead.F * norm_prob.denom) + norm_prob.ideal_point
            _, _, comp_X_moead, _ = select_compromise_solution(final_raw_F, res_moead.X)
            sched_moead = norm_prob.decode_schedule(comp_X_moead)
            schedules["MOEA-D"][b][s_str] = sched_moead["operations_schedule"]

        elapsed_b = time.time() - b_start
        print(f"[{b}] Completed {len(seeds)} seeds for NSGA-III & MOEA-D in {elapsed_b:.2f} s")

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(schedules, f)

    total_elapsed = time.time() - t_start
    print(f"All multi-seed schedules saved to {output_path} in {total_elapsed:.2f} s")
    return schedules


if __name__ == "__main__":
    generate_all_multi_seed_schedules()
