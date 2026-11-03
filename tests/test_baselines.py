"""
Unit tests for baseline dispatching rules and comparative benchmark execution.
"""

import csv
from pathlib import Path
import pytest

from sim_env.fjsp_gym_env import GreenFJSPEnv
from sim_env.run_baselines import (
    execute_deterministic_rule,
    execute_random_baseline,
    rule_spt_fast,
    rule_eet_eco,
    rule_edd_balanced,
    run_all_baselines,
)


@pytest.fixture
def env() -> GreenFJSPEnv:
    return GreenFJSPEnv(
        benchmark_path="data/benchmarks/Mk01_green.json",
        config_path="config/system_params.yaml",
        seed=42,
    )


def test_rule_spt_fast_performance(env: GreenFJSPEnv):
    """Verify Rule 1 (SPT + Fast): completes 55 ops, all ops at speed level 3, suffers breakdowns without PM."""
    res = execute_deterministic_rule(env, rule_spt_fast, seed=42)

    assert len(res["scheduled_events"]) == 55
    assert res["cmax"] < 100.0  # Around 90.67 min
    assert res["bds"] > 0       # Suffers tool breakdowns due to wear without PM
    # Verify all executed cutting operations were at speed level 3
    for ev in res["scheduled_events"]:
        assert ev["speed_level"] == 3


def test_rule_eet_eco_performance(env: GreenFJSPEnv):
    """Verify Rule 2 (EET + Eco): completes 55 ops, all ops at speed level 1, lowest energy, 0 breakdowns with PM."""
    res = execute_deterministic_rule(env, rule_eet_eco, seed=42)

    assert len(res["scheduled_events"]) == 55
    assert res["e_total"] < 20.0  # Very energy efficient (< 20 kWh)
    assert res["bds"] == 0        # Proactive PM prevents breakdowns
    for ev in res["scheduled_events"]:
        assert ev["speed_level"] == 1


def test_rule_edd_balanced_performance(env: GreenFJSPEnv):
    """Verify Rule 3 (EDD + Balanced): completes 55 ops, records timeline events properly, 0 breakdowns with PM."""
    res = execute_deterministic_rule(env, rule_edd_balanced, seed=42)

    assert len(res["scheduled_events"]) == 55
    assert res["cmax"] > 0.0
    assert res["cost"] > 0.0
    assert res["bds"] == 0
    assert res["twt"] < 450.0  # Significantly lower tardiness than random (734 min)


def test_full_baseline_pipeline_and_csv(tmp_path: Path):
    """Verify end-to-end execution of run_all_baselines and CSV output generation."""
    test_csv = tmp_path / "test_baseline_results.csv"
    test_gantt = tmp_path / "test_gantt.png"

    results = run_all_baselines(
        benchmark_path="data/benchmarks/Mk01_green.json",
        config_path="config/system_params.yaml",
        csv_output_path=str(test_csv),
        gantt_output_path=str(test_gantt),
    )

    assert len(results) == 4
    assert test_csv.is_file()
    assert test_gantt.is_file()

    # Verify CSV format and rows
    with open(test_csv, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    assert len(reader) == 5  # Header + 4 rules
    assert reader[0][0] == "Rule Name"
    assert reader[1][0] == "Rule 1 (SPT + Fast Speed)"

