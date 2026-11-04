"""
Test Suite for Task 5.1: Multi-Scale Benchmark Expansion (Mk01 - Mk05) & Statistical Indicators (HV, IGD, Wilcoxon).

Verifies:
1. Generation and physical parameter standardization for 5 Brandimarte benchmarks:
   - Mk01_green.json: 10 Jobs x 6 Machines (55 Ops, P_limit=22.0 kW)
   - Mk02_green.json: 10 Jobs x 6 Machines (58 Ops, P_limit=24.0 kW)
   - Mk03_green.json: 15 Jobs x 8 Machines (150 Ops, P_limit=32.0 kW)
   - Mk04_green.json: 15 Jobs x 8 Machines (90 Ops, P_limit=30.0 kW)
   - Mk05_green.json: 15 Jobs x 4 Machines (106 Ops, P_limit=16.0 kW)
2. 100% successful rollout of full episodes in GreenFJSPEnv across all 5 benchmark scales.
3. Multi-objective Min-Max normalization, Hypervolume (HV) calculation with ref_point=[1.1, 1.1, 1.1],
   and Inverted Generational Distance (IGD).
4. Wilcoxon Signed-Rank test returning (p_value, symbol) with '+', '-', '≈'.
"""

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pytest

from data.benchmark_parser import BENCHMARK_CONFIGS, generate_all_green_benchmarks
from sim_env.fjsp_gym_env import GreenFJSPEnv
from utils.stat_metrics import (
    calculate_hypervolume,
    calculate_igd,
    normalize_multiobjective_fronts,
    run_wilcoxon_test,
)


def test_benchmark_generation_and_properties():
    """
    Verify generation and standardization of all 5 Brandimarte benchmark scales (Mk01 - Mk05).
    Prints summary table to Terminal.
    """
    benchmarks = generate_all_green_benchmarks()
    assert len(benchmarks) == 5

    expected_specs = {
        "Mk01": {"jobs": 10, "machines": 6, "ops": 55, "p_limit": 22.0, "m_list": [f"M{i}" for i in range(1, 7)]},
        "Mk02": {"jobs": 10, "machines": 6, "ops": 58, "p_limit": 24.0, "m_list": [f"M{i}" for i in range(1, 7)]},
        "Mk03": {"jobs": 15, "machines": 8, "ops": 150, "p_limit": 32.0, "m_list": [f"M{i}" for i in range(1, 9)]},
        "Mk04": {"jobs": 15, "machines": 8, "ops": 90, "p_limit": 30.0, "m_list": [f"M{i}" for i in range(1, 9)]},
        "Mk05": {"jobs": 15, "machines": 4, "ops": 106, "p_limit": 16.0, "m_list": [f"M{i}" for i in range(1, 5)]},
    }

    print("\n" + "=" * 90)
    print("TASK 5.1: MULTI-SCALE BRANDIMARTE BENCHMARK SUITE SUMMARY (Mk01 - Mk05)")
    print("=" * 90)
    print(f"{'Instance':<12} | {'Scale Category':<24} | {'Jobs':<5} | {'Machines':<8} | {'Total Ops':<10} | {'P_limit (kW)':<12}")
    print("-" * 90)

    for inst_id, exp in expected_specs.items():
        data = benchmarks[inst_id]
        gp = data["global_parameters"]
        mach_keys = sorted(list(data["machines"].keys()))

        # Assertions on dataset structure
        assert gp["num_jobs"] == exp["jobs"]
        assert gp["num_machines"] == exp["machines"]
        assert gp["total_operations"] == exp["ops"]
        assert gp["p_limit_kw"] == exp["p_limit"]
        assert mach_keys == exp["m_list"]

        json_file = Path(f"data/benchmarks/{inst_id}_green.json")
        assert json_file.exists(), f"File {json_file} must exist."

        scale_cat = BENCHMARK_CONFIGS[inst_id]["scale_category"]
        print(f"{inst_id + '_green':<12} | {scale_cat:<24} | {gp['num_jobs']:<5} | {gp['num_machines']:<8} | {gp['total_operations']:<10} | {gp['p_limit_kw']:<12.1f}")

    print("=" * 90)


def test_green_fjsp_env_full_rollout_all_benchmarks():
    """
    Assert: All 5 benchmark files (Mk01 to Mk05) successfully initialize GreenFJSPEnv
    and complete 100% of operations without error.
    """
    for inst_id in ["Mk01", "Mk02", "Mk03", "Mk04", "Mk05"]:
        json_path = f"data/benchmarks/{inst_id}_green.json"
        env = GreenFJSPEnv(benchmark_path=json_path)

        exp_ops = BENCHMARK_CONFIGS[inst_id]["total_operations"]
        exp_jobs = BENCHMARK_CONFIGS[inst_id]["num_jobs"]
        exp_machs = BENCHMARK_CONFIGS[inst_id]["num_machines"]

        assert env.num_jobs == exp_jobs
        assert env.num_machines == exp_machs
        assert env.total_operations == exp_ops
        assert env.observation_space.shape == (env.obs_dim,)

        obs, info = env.reset(seed=42)
        rng = np.random.default_rng(42)
        step_count = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            mask = env.action_masks()
            valid_actions = np.where(mask)[0]
            assert len(valid_actions) > 0, f"Deadlock! No valid actions at step {step_count} for {inst_id}"

            action = int(rng.choice(valid_actions))
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

        # Must complete 100% of operations
        assert terminated is True
        assert env.completed_operations == exp_ops
        assert step_count == exp_ops
        assert info["cmax"] > 0.0
        assert info["total_energy_kwh"] > 0.0
        assert info["total_cost"] > 0.0


def test_multiobjective_normalization_and_indicators():
    """
    Verify Min-Max normalization of 3 objectives, Hypervolume calculation, and IGD indicator.
    """
    # 1. Multi-Objective Front Normalization
    f_proposed = np.array([
        [50.0, 10.0, 20.0],
        [60.0, 8.0, 30.0],
        [70.0, 6.0, 40.0],
    ])
    f_baseline = np.array([
        [55.0, 12.0, 25.0],
        [65.0, 9.0, 35.0],
        [75.0, 7.0, 50.0],
    ])

    norm_dict, ideal, nadir = normalize_multiobjective_fronts({
        "proposed": f_proposed,
        "baseline": f_baseline,
    })

    assert np.allclose(ideal, [50.0, 6.0, 20.0])
    assert np.allclose(nadir, [75.0, 12.0, 50.0])

    p_norm = norm_dict["proposed"]
    b_norm = norm_dict["baseline"]

    assert p_norm.shape == (3, 3)
    assert np.all(p_norm >= 0.0) and np.all(p_norm <= 1.0)
    assert np.all(b_norm >= 0.0) and np.all(b_norm <= 1.0)

    # 2. Hypervolume (ref_point = [1.1, 1.1, 1.1])
    hv_proposed = calculate_hypervolume(p_norm, ref_point=np.array([1.1, 1.1, 1.1]))
    hv_baseline = calculate_hypervolume(b_norm, ref_point=np.array([1.1, 1.1, 1.1]))

    assert 0.0 < hv_proposed <= 1.1**3
    assert 0.0 < hv_baseline <= 1.1**3
    # Proposed strictly dominates Baseline in this setup -> HV_proposed > HV_baseline
    assert hv_proposed > hv_baseline

    # 3. Inverted Generational Distance (IGD)
    # Distance to itself must be exactly 0.0
    igd_self = calculate_igd(p_norm, p_norm)
    assert pytest.approx(igd_self, abs=1e-6) == 0.0

    # IGD to reference front
    igd_baseline_to_prop = calculate_igd(b_norm, p_norm)
    assert igd_baseline_to_prop > 0.0


def test_wilcoxon_signed_rank_test():
    """
    Verify Wilcoxon signed-rank test across all hypothesis scenarios:
    - Minimization (Makespan/Cost/IGD): lower is superior (+)
    - Maximization (Hypervolume): higher is superior (+)
    - Non-significant difference: (≈)
    - Identical arrays: (≈)
    """
    # 1. Minimization: Proposed is significantly lower (superior) -> '+'
    prop_cmax = [50.1, 51.2, 49.8, 50.5, 50.0, 49.5, 50.2, 50.8, 51.0, 49.9]
    base_cmax = [58.2, 59.1, 57.5, 58.0, 59.0, 57.8, 58.5, 58.9, 59.2, 58.0]
    p_val, sym = run_wilcoxon_test(prop_cmax, base_cmax, higher_is_better=False)
    assert p_val < 0.05
    assert sym == "+"

    # 2. Minimization: Proposed is significantly higher (inferior) -> '-'
    p_val_inv, sym_inv = run_wilcoxon_test(base_cmax, prop_cmax, higher_is_better=False)
    assert p_val_inv < 0.05
    assert sym_inv == "-"

    # 3. Maximization: Proposed Hypervolume is significantly higher (superior) -> '+'
    prop_hv = [0.85, 0.86, 0.84, 0.87, 0.85, 0.86, 0.88, 0.85, 0.84, 0.86]
    base_hv = [0.72, 0.73, 0.71, 0.74, 0.72, 0.73, 0.75, 0.72, 0.71, 0.73]
    p_val_hv, sym_hv = run_wilcoxon_test(prop_hv, base_hv, higher_is_better=True)
    assert p_val_hv < 0.05
    assert sym_hv == "+"

    # 4. Statistically equivalent samples -> '≈'
    s1 = [50.1, 50.3, 49.9, 50.2, 50.0, 50.1, 49.8, 50.4, 50.0, 50.2]
    s2 = [50.2, 50.1, 50.0, 50.3, 49.9, 50.0, 50.1, 50.2, 50.1, 50.0]
    p_val_eq, sym_eq = run_wilcoxon_test(s1, s2, higher_is_better=False)
    assert p_val_eq >= 0.05
    assert sym_eq == "≈"

    # 5. Exactly identical samples -> '≈' with p_val = 1.0
    p_val_ident, sym_ident = run_wilcoxon_test(s1, s1, higher_is_better=False)
    assert p_val_ident == 1.0
    assert sym_ident == "≈"

