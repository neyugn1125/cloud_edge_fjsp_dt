"""
Unit tests for standard FJSP benchmark parser and Green-FJSP enrichment.
"""

import json
from pathlib import Path
import pytest

from data.benchmark_parser import parse_raw_fjs, parse_and_enrich_fjs


def test_parse_raw_fjs():
    """Verify raw parsing of Brandimarte Mk01.fjs instance."""
    fjs_file = "data/benchmarks/Mk01.fjs"
    num_jobs, num_machines, declared_avg, raw_jobs = parse_raw_fjs(fjs_file)

    assert num_jobs == 10
    assert num_machines == 6
    assert declared_avg == 2.0
    assert len(raw_jobs) == 10

    # Verify operation count for each job
    expected_ops = [6, 5, 5, 5, 6, 6, 5, 5, 6, 6]
    actual_ops = [len(job) for job in raw_jobs]
    assert actual_ops == expected_ops
    assert sum(actual_ops) == 55


def test_parse_and_enrich_fjs(tmp_path: Path):
    """Verify enrichment with physical energy parameters, due dates, and weights."""
    fjs_file = "data/benchmarks/Mk01.fjs"
    yaml_config = "config/system_params.yaml"
    out_json = tmp_path / "test_Mk01_green.json"

    dataset = parse_and_enrich_fjs(
        fjs_path=fjs_file,
        yaml_config_path=yaml_config,
        output_json_path=out_json,
        tightness_factor=1.5,
        seed=42,
    )

    # Check top-level keys
    assert "instance_name" in dataset
    assert "global_parameters" in dataset
    assert "machines" in dataset
    assert "jobs" in dataset

    # Global parameters check
    gp = dataset["global_parameters"]
    assert gp["num_jobs"] == 10
    assert gp["num_machines"] == 6
    assert gp["total_operations"] == 55
    assert pytest.approx(gp["avg_machines_per_op"], rel=1e-2) == 2.09
    assert gp["tightness_factor"] == 1.5
    assert gp["p_limit_kw"] == 22.0

    # Machines check
    machines = dataset["machines"]
    assert set(machines.keys()) == {"M1", "M2", "M3", "M4", "M5", "M6"}
    for m_id, m in machines.items():
        assert "break_even_time_min" in m
        assert m["break_even_time_min"] > 0
        assert "tool_wear" in m

    # Expected Due Dates (with TF = 1.5)
    expected_due_dates = {
        "J1": 30.75,
        "J2": 28.50,
        "J3": 32.00,
        "J4": 25.50,
        "J5": 43.25,
        "J6": 34.25,
        "J7": 21.25,
        "J8": 36.50,
        "J9": 33.50,
        "J10": 31.25,
    }

    jobs = dataset["jobs"]
    assert len(jobs) == 10
    for job in jobs:
        j_id = job["job_id"]
        assert j_id in expected_due_dates
        assert pytest.approx(job["due_date_min"], rel=1e-3) == expected_due_dates[j_id]
        assert job["weight"] in {1, 2, 3}

    # Verify JSON file on disk
    assert out_json.is_file()
    with open(out_json, "r", encoding="utf-8") as f:
        loaded_json = json.load(f)
    assert loaded_json["instance_name"] == "Mk01_green"


def test_deterministic_weights_reproducibility():
    """Verify that using the same random seed produces identical weights."""
    fjs_file = "data/benchmarks/Mk01.fjs"
    yaml_config = "config/system_params.yaml"

    d1 = parse_and_enrich_fjs(fjs_file, yaml_config, output_json_path=None, seed=42)
    d2 = parse_and_enrich_fjs(fjs_file, yaml_config, output_json_path=None, seed=42)

    weights1 = [j["weight"] for j in d1["jobs"]]
    weights2 = [j["weight"] for j in d2["jobs"]]
    assert weights1 == weights2
    assert weights1 == [3, 1, 1, 3, 2, 1, 1, 1, 3, 1]

