"""
Unit and Integration Tests for Cloud Layer NSGA-III Multi-Objective Optimization.

Verifies:
    1. Problem initialization and variable dimensions (165 variables, 3 objectives).
    2. Single chromosome evaluation logic (non-negative objectives, no errors).
    3. Operation precedence feasibility and machine constraint satisfaction in decoding.
    4. Proactive maintenance policy (zero unexpected breakdowns when tool wear >= 80 um).
    5. Multi-criteria compromise selection (TOPSIS / Ideal Point distance).
    6. Integrity of the generated cloud_baseline_schedule.json.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from cloud_layer.nsga3_optimizer import (
    GreenFJSPProblem,
    run_nsga3_optimization,
    select_compromise_solution,
)


@pytest.fixture
def green_problem() -> GreenFJSPProblem:
    """Fixture providing initialized GreenFJSPProblem."""
    return GreenFJSPProblem(
        benchmark_path="data/benchmarks/Mk01_green.json",
        config_path="config/system_params.yaml",
    )


def test_problem_initialization(green_problem: GreenFJSPProblem) -> None:
    """Verify problem formulation and variable counts."""
    assert green_problem.n_var == 165
    assert green_problem.n_obj == 3
    assert green_problem.total_ops == 55
    assert green_problem.num_jobs == 10
    assert green_problem.num_machines == 6
    assert np.all(green_problem.xl == 0.0)
    assert np.all(green_problem.xu == 1.0)


def test_single_chromosome_evaluation(green_problem: GreenFJSPProblem) -> None:
    """Test evaluation of a random chromosome."""
    np.random.seed(42)
    x = np.random.rand(165)
    out: dict = {}
    green_problem._evaluate(x, out)

    assert "F" in out
    f = out["F"]
    assert len(f) == 3

    cmax, cost, twt = f
    assert cmax > 0.0, "Makespan must be strictly positive"
    assert cost > 0.0, "TOU Energy Cost must be strictly positive"
    assert twt >= 0.0, "TWT must be non-negative"


def test_schedule_decoding_and_precedence(green_problem: GreenFJSPProblem) -> None:
    """Verify that decoded schedule strictly adheres to job precedence and valid machines."""
    np.random.seed(123)
    x = np.random.rand(165)
    decoded = green_problem.decode_schedule(x)

    assert "summary" in decoded
    assert "operations_schedule" in decoded
    assert "machine_schedules" in decoded

    ops = decoded["operations_schedule"]
    assert len(ops) == 55, "Decoded schedule must contain all 55 operations"

    # Verify job precedence constraints:
    # For any job j, operation k+1 start_time_min must be >= operation k end_time_min
    job_last_end: dict = {}
    for op in ops:
        j_id = op["job_id"]
        o_idx = op["op_index"]
        start_t = op["start_time_min"]
        end_t = op["end_time_min"]
        cut_start = op["cutting_start_min"]
        cut_end = op["cutting_end_min"]

        # Interval ordering
        assert start_t <= cut_start <= cut_end <= end_t

        if j_id in job_last_end:
            last_idx, last_t = job_last_end[j_id]
            assert o_idx == last_idx + 1, f"Operations out of order for {j_id}"
            assert start_t >= last_t - 1e-6, (
                f"Job precedence violated for {j_id}: op {o_idx} started at {start_t} "
                f"before op {last_idx} finished at {last_t}"
            )
        else:
            assert o_idx == 0, f"First scheduled operation for {j_id} must be op 0"

        job_last_end[j_id] = (o_idx, end_t)

    # Verify machine assignment validity
    for op in ops:
        m_id = op["machine_id"]
        assert m_id in green_problem.machine_ids
        assert op["speed_level"] in [1, 2, 3]


def test_proactive_pm_eliminates_breakdowns(green_problem: GreenFJSPProblem) -> None:
    """Verify that proactive PM policy keeps tool wear <= w_max and eliminates breakdowns."""
    np.random.seed(999)
    # Evaluate 10 random schedules
    for _ in range(10):
        x = np.random.rand(165)
        decoded = green_problem.decode_schedule(x)
        summary = decoded["summary"]

        assert summary["num_breakdowns"] == 0, (
            f"Found {summary['num_breakdowns']} breakdowns; proactive PM should avoid breakdowns"
        )
        assert summary["num_pm_replacements"] >= 0


def test_select_compromise_solution() -> None:
    """Verify TOPSIS / Ideal Point selection mathematical correctness."""
    # Synthetic Pareto front with 3 solutions
    # Solution 0: [10, 100, 50]  (balanced)
    # Solution 1: [5, 200, 100]  (extreme cmax)
    # Solution 2: [20, 50, 10]   (extreme cost/twt)
    pareto_F = np.array([
        [10.0, 100.0, 50.0],
        [5.0, 200.0, 100.0],
        [20.0, 50.0, 10.0],
    ])
    pareto_X = np.zeros((3, 165))
    pareto_X[0, 0] = 0.5
    pareto_X[1, 0] = 1.0
    pareto_X[2, 0] = 2.0

    best_idx, best_F, best_X, dists = select_compromise_solution(
        pareto_F, pareto_X, weights=(0.4, 0.4, 0.2)
    )

    assert 0 <= best_idx < 3
    assert len(best_F) == 3
    assert len(best_X) == 165
    assert len(dists) == 3
    assert dists[best_idx] == np.min(dists)


def test_quick_nsga3_run(green_problem: GreenFJSPProblem) -> None:
    """Verify that run_nsga3_optimization executes cleanly for 2 generations."""
    res, prob = run_nsga3_optimization(
        benchmark_path="data/benchmarks/Mk01_green.json",
        config_path="config/system_params.yaml",
        pop_size=20,
        n_gen=2,
        seed=123,
        verbose=False,
    )
    assert res.F is not None
    assert len(res.F) > 0
    assert res.F.shape[1] == 3


def test_cloud_baseline_schedule_file() -> None:
    """Verify that the generated cloud_baseline_schedule.json is complete and valid."""
    path = Path("data/benchmarks/cloud_baseline_schedule.json")
    assert path.exists(), "cloud_baseline_schedule.json must exist"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data["summary"]
    assert summary["benchmark"] == "Mk01_green"
    assert summary["algorithm"] == "NSGA-III"
    assert summary["makespan_min"] > 0
    assert summary["total_energy_kwh"] > 0
    assert summary["tou_energy_cost_usd"] > 0
    assert summary["total_weighted_tardiness_min"] >= 0
    assert summary["num_breakdowns"] == 0

    assert len(data["operations_schedule"]) == 55
    assert len(data["machine_schedules"]) == 6
