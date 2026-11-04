"""
Generate Normalized MOEA/D Baselines for Mk01 - Mk05.
Uses adaptive ideal and nadir point normalization and heuristic seeded population.
"""

import json
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from cloud_layer.nsga3_optimizer import (
    GreenFJSPProblem,
    generate_seeded_population,
    select_compromise_solution,
)


class NormalizedGreenFJSPProblem(GreenFJSPProblem):
    def __init__(self, benchmark_path: str, ideal_point: np.ndarray, nadir_point: np.ndarray):
        super().__init__(benchmark_path=benchmark_path)
        self.ideal_point = np.array(ideal_point, dtype=np.float64)
        self.nadir_point = np.array(nadir_point, dtype=np.float64)
        self.denom = np.maximum(self.nadir_point - self.ideal_point, 1e-4)

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        super()._evaluate(x, out, *args, **kwargs)
        raw_f = np.copy(out["F"])
        out["F"] = (raw_f - self.ideal_point) / self.denom
        out["raw_F"] = raw_f


def generate_all_moead_baselines():
    benchmarks = ["Mk01", "Mk02", "Mk03", "Mk04", "Mk05"]
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)  # 91 ref dirs
    pop_size = len(ref_dirs)

    for b in benchmarks:
        t0 = time.time()
        json_path = f"data/benchmarks/{b}_green.json"
        temp_prob = GreenFJSPProblem(benchmark_path=json_path)

        # 1. Evaluate seeded population for initial ideal and nadir bounds
        X_init = generate_seeded_population(temp_prob, pop_size=pop_size, seed=42)
        F_init = []
        for x in X_init:
            out = {}
            temp_prob._evaluate(x, out)
            F_init.append(out["F"])
        F_init = np.array(F_init)
        ideal_pt = np.min(F_init, axis=0) * 0.90
        nadir_pt = np.max(F_init, axis=0) * 1.10

        norm_prob = NormalizedGreenFJSPProblem(json_path, ideal_pt, nadir_pt)
        algo = MOEAD(
            ref_dirs=ref_dirs,
            n_neighbors=15,
            prob_neighbor_mating=0.7,
            sampling=X_init,
        )

        n_gen = 60
        res = minimize(norm_prob, algo, ("n_gen", n_gen), seed=42, verbose=False)

        # Denormalize
        final_raw_F = (res.F * norm_prob.denom) + norm_prob.ideal_point
        comp_idx, comp_F, comp_X, dists = select_compromise_solution(final_raw_F, res.X)
        sched = norm_prob.decode_schedule(comp_X)

        out_file = f"data/benchmarks/moead_baseline_{b}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(sched, f, indent=2)

        summary = sched["summary"]
        cmax = summary["makespan_min"]
        cost = summary["tou_energy_cost_usd"]
        pms = summary["num_pm_replacements"]
        print(f"[{b}] MOEA/D baseline generated in {time.time() - t0:.2f}s | Cmax={cmax} min | Cost=${cost:.3f} | PMs={pms}")


if __name__ == "__main__":
    generate_all_moead_baselines()
