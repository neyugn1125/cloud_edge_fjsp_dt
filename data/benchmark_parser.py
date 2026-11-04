"""
Benchmark Generator and Parser for Green Flexible Job Shop Scheduling (Green-FJSP).

This module parses standard international benchmark files (.fjs), enriches them with
machine physical energy parameters from config/system_params.yaml, computes realistic
Job Due Dates (D_i) based on average operation workloads and tightness factors,
assigns priority weights (w_i) for Total Weighted Tardiness (TWT), and exports
a comprehensive Green Benchmark instance in JSON format.

Strict Unit Conventions:
    - Time: Minutes (min)
    - Power: Kilowatts (kW)
    - Energy: Kilowatt-hours (kWh)
    - Tool Wear VB: Micrometers (um)
"""

import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from utils.mqtt_schemas import SystemConfig, load_system_config

# Standard Brandimarte benchmark scales and configurations
BENCHMARK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "Mk01": {
        "num_jobs": 10,
        "num_machines": 6,
        "total_operations": 55,
        "p_limit_kw": 22.0,
        "tightness_factor": 1.5,
        "scale_category": "Small/Medium scale",
    },
    "Mk02": {
        "num_jobs": 10,
        "num_machines": 6,
        "total_operations": 58,
        "p_limit_kw": 24.0,
        "tightness_factor": 1.5,
        "scale_category": "High flexibility",
    },
    "Mk03": {
        "num_jobs": 15,
        "num_machines": 8,
        "total_operations": 150,
        "p_limit_kw": 32.0,
        "tightness_factor": 1.5,
        "scale_category": "Medium/Large scale",
    },
    "Mk04": {
        "num_jobs": 15,
        "num_machines": 8,
        "total_operations": 90,
        "p_limit_kw": 30.0,
        "tightness_factor": 1.5,
        "scale_category": "Complex routing",
    },
    "Mk05": {
        "num_jobs": 15,
        "num_machines": 4,
        "total_operations": 106,
        "p_limit_kw": 16.0,
        "tightness_factor": 1.5,
        "scale_category": "High bottleneck/congestion",
    },
}


def parse_raw_fjs(fjs_path: Union[str, Path]) -> Tuple[int, int, Optional[float], List[List[List[Tuple[int, float]]]]]:
    """
    Parse a standard FJSP benchmark instance file (.fjs).

    Format:
        Line 1: <num_jobs> <num_machines> [<avg_machines_per_op>]
        Each subsequent job line:
            <num_ops>  <nb_m_1> <m_11> <t_11> ... <nb_m_2> <m_21> <t_21> ...

    Returns:
        num_jobs (int): Total number of jobs.
        num_machines (int): Total number of machines.
        avg_machines (float | None): Benchmark declared average machines per op (if present).
        raw_jobs (List): Nested list: raw_jobs[job_idx][op_idx] = [(mach_idx, proc_time), ...]
    """
    path = Path(fjs_path)
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark file not found at: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        # Filter comments and empty lines
        raw_lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    if not raw_lines:
        raise ValueError(f"Benchmark file {path} is empty or invalid.")

    # Parse header
    header_tokens = raw_lines[0].split()
    num_jobs = int(header_tokens[0])
    num_machines = int(header_tokens[1])
    avg_machines = float(header_tokens[2]) if len(header_tokens) > 2 else None

    # Determine token stream for jobs
    job_lines = raw_lines[1:]
    raw_jobs: List[List[List[Tuple[int, float]]]] = []

    for line_idx, line in enumerate(job_lines):
        tokens = [int(tok) if tok.isdigit() else float(tok) for tok in line.split()]
        if not tokens:
            continue

        num_ops = int(tokens[0])
        token_cursor = 1
        ops_list: List[List[Tuple[int, float]]] = []

        for _ in range(num_ops):
            nb_eligible = int(tokens[token_cursor])
            token_cursor += 1
            alternatives: List[Tuple[int, float]] = []

            for _ in range(nb_eligible):
                mach_idx = int(tokens[token_cursor])
                proc_time = float(tokens[token_cursor + 1])
                token_cursor += 2
                alternatives.append((mach_idx, proc_time))

            ops_list.append(alternatives)

        raw_jobs.append(ops_list)

    if len(raw_jobs) != num_jobs:
        raise ValueError(
            f"Header specifies {num_jobs} jobs, but found {len(raw_jobs)} job lines in {path}"
        )

    return num_jobs, num_machines, avg_machines, raw_jobs


def parse_and_enrich_fjs(
    fjs_path: Union[str, Path] = "data/benchmarks/Mk01.fjs",
    yaml_config_path: Union[str, Path] = "config/system_params.yaml",
    output_json_path: Optional[Union[str, Path]] = "data/benchmarks/Mk01_green.json",
    tightness_factor: float = 1.5,
    seed: int = 42,
    default_arrival_time: float = 0.0,
    p_limit_override: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Parse a standard .fjs file and enrich it with physical energy/wear parameters.

    Computations:
        - Calculates mean processing time \\bar{P}_{ij} for each operation.
        - Calculates total expected workload W_i = \\sum_j \\bar{P}_{ij}.
        - Computes Job Due Date D_i = r_i + tightness_factor * W_i.
        - Assigns priority weight w_i \\in {1, 2, 3} deterministically using PRNG seed.
        - Integrates machine specifications (p_idle_kw, p_stdby_kw, e_on_off_kwh, T_BE, etc.)
          from system_params.yaml.

    Args:
        fjs_path: Path to .fjs benchmark instance.
        yaml_config_path: Path to system_params.yaml.
        output_json_path: Path to output enriched JSON file (if None, file is not written).
        tightness_factor: Due date tightness multiplier (default: 1.5).
        seed: Random seed for deterministic priority weight assignment (default: 42).
        default_arrival_time: Default arrival time r_i in minutes (default: 0.0).

    Returns:
        Dict[str, Any]: Fully structured Green-FJSP benchmark dataset.
    """
    # 1. Parse raw benchmark
    num_jobs, num_machines, declared_avg_mach, raw_jobs = parse_raw_fjs(fjs_path)

    # 2. Load physical system configuration
    system_config: SystemConfig = load_system_config(yaml_config_path)

    # Detect machine index base (0-based vs 1-based)
    all_mach_indices = [
        mach_idx
        for job in raw_jobs
        for op in job
        for mach_idx, _ in op
    ]
    min_mach_idx = min(all_mach_indices) if all_mach_indices else 1
    is_zero_based = (min_mach_idx == 0)

    # PRNG for deterministic job weights
    rng = random.Random(seed)

    # 3. Construct Enriched Jobs
    enriched_jobs: List[Dict[str, Any]] = []
    total_ops_count = 0
    total_eligible_mach_count = 0

    for j_idx, raw_ops in enumerate(raw_jobs):
        job_id = f"J{j_idx + 1}"
        num_ops = len(raw_ops)
        total_ops_count += num_ops

        # Priority weight w_i in {1, 2, 3}
        weight = rng.choice([1, 2, 3])

        enriched_ops: List[Dict[str, Any]] = []
        total_expected_work_min = 0.0

        for op_idx, alternatives in enumerate(raw_ops):
            op_id = f"{job_id}_O{op_idx + 1}"
            total_eligible_mach_count += len(alternatives)

            # Map machine index to configuration machine_id (M1..M6)
            mapped_alternatives: List[Dict[str, Any]] = []
            alt_times: List[float] = []

            for mach_idx, t_base in alternatives:
                normalized_idx = (mach_idx + 1) if is_zero_based else mach_idx
                m_id = f"M{normalized_idx}"

                # Ensure machine exists in physical system config
                if m_id not in system_config.machines:
                    raise KeyError(
                        f"Machine '{m_id}' referenced in {fjs_path} does not exist in {yaml_config_path}"
                    )

                mapped_alternatives.append({
                    "machine_id": m_id,
                    "t_base_min": float(t_base),
                })
                alt_times.append(float(t_base))

            mean_proc_time = sum(alt_times) / len(alt_times) if alt_times else 0.0
            total_expected_work_min += mean_proc_time

            enriched_ops.append({
                "op_id": op_id,
                "op_index": op_idx,
                "num_eligible_machines": len(mapped_alternatives),
                "mean_processing_time_min": round(mean_proc_time, 4),
                "eligible_machines": mapped_alternatives,
            })

        # Calculate Due Date: D_i = r_i + tightness_factor * W_i
        due_date_min = default_arrival_time + (tightness_factor * total_expected_work_min)

        enriched_jobs.append({
            "job_id": job_id,
            "job_index": j_idx,
            "arrival_time_min": float(default_arrival_time),
            "weight": weight,
            "total_expected_work_min": round(total_expected_work_min, 4),
            "due_date_min": round(due_date_min, 4),
            "num_operations": num_ops,
            "operations": enriched_ops,
        })

    avg_mach_per_op = (
        total_eligible_mach_count / total_ops_count if total_ops_count > 0 else 0.0
    )

    # 4. Serialize Machine Configs with Break-Even Time (Machines M1 .. M_{num_machines})
    break_even_times = system_config.get_break_even_times()
    machines_dict: Dict[str, Any] = {}
    for m_idx in range(1, num_machines + 1):
        m_id = f"M{m_idx}"
        m_cfg = system_config.get_machine(m_id)
        m_dump = m_cfg.model_dump()
        m_dump["break_even_time_min"] = round(break_even_times[m_id], 4)
        machines_dict[m_id] = m_dump

    # Determine peak power limit
    fjs_file_stem = Path(fjs_path).stem
    if p_limit_override is not None:
        p_limit_val = float(p_limit_override)
    elif fjs_file_stem in BENCHMARK_CONFIGS and "p_limit_kw" in BENCHMARK_CONFIGS[fjs_file_stem]:
        p_limit_val = float(BENCHMARK_CONFIGS[fjs_file_stem]["p_limit_kw"])
    else:
        p_limit_val = float(system_config.p_limit_kw)

    scale_cat = BENCHMARK_CONFIGS.get(fjs_file_stem, {}).get("scale_category", "Standard")

    # 5. Assemble complete Green-FJSP benchmark dataset
    enriched_dataset: Dict[str, Any] = {
        "instance_name": f"{fjs_file_stem}_green",
        "source_benchmark": fjs_file_stem,
        "scale_category": scale_cat,
        "unit_conventions": system_config.unit_conventions,
        "global_parameters": {
            "num_jobs": num_jobs,
            "num_machines": num_machines,
            "total_operations": total_ops_count,
            "avg_machines_per_op": round(avg_mach_per_op, 4),
            "tightness_factor": tightness_factor,
            "random_seed": seed,
            "p_limit_kw": p_limit_val,
            "speed_scaling": system_config.speed_scaling.model_dump(),
            "tou_pricing": system_config.tou_pricing.model_dump(),
        },
        "machines": machines_dict,
        "jobs": enriched_jobs,
    }

    # 6. Save to JSON if requested
    if output_json_path is not None:
        out_p = Path(output_json_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(enriched_dataset, f, indent=2)

    return enriched_dataset


def generate_all_green_benchmarks(
    benchmark_dir: Union[str, Path] = "data/benchmarks",
    yaml_config_path: Union[str, Path] = "config/system_params.yaml",
    instances: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Generate and standardize all Brandimarte green benchmarks (Mk01 to Mk05).

    Args:
        benchmark_dir: Directory containing .fjs files and destination for _green.json.
        yaml_config_path: Configuration file system_params.yaml.
        instances: List of benchmark stems to generate (default: Mk01 to Mk05).

    Returns:
        Dict mapping instance name to parsed dataset dictionary.
    """
    if instances is None:
        instances = ["Mk01", "Mk02", "Mk03", "Mk04", "Mk05"]

    b_dir = Path(benchmark_dir)
    b_dir.mkdir(parents=True, exist_ok=True)
    generated: Dict[str, Dict[str, Any]] = {}

    for inst in instances:
        fjs_file = b_dir / f"{inst}.fjs"
        json_file = b_dir / f"{inst}_green.json"

        cfg = BENCHMARK_CONFIGS.get(inst, {})
        tf = cfg.get("tightness_factor", 1.5)
        p_lim = cfg.get("p_limit_kw", None)

        ds = parse_and_enrich_fjs(
            fjs_path=fjs_file,
            yaml_config_path=yaml_config_path,
            output_json_path=json_file,
            tightness_factor=tf,
            p_limit_override=p_lim,
            seed=42,
        )
        generated[inst] = ds

    return generated


def print_benchmark_summary(dataset: Dict[str, Any]) -> None:
    """Print a clean, professional summary table of the Green-FJSP instance."""
    gp = dataset["global_parameters"]
    jobs = dataset["jobs"]

    print("=" * 80)
    print(f"Green-FJSP Benchmark Summary: {dataset['instance_name']}")
    print(f"Source Benchmark: {dataset['source_benchmark']} | Units: Time (min), Power (kW), Energy (kWh)")
    print("=" * 80)
    print(f"Total Jobs (N)                     : {gp['num_jobs']}")
    print(f"Total Machines (M)                 : {gp['num_machines']}")
    print(f"Total Operations (\\sum O_i)        : {gp['total_operations']}")
    print(f"Average Machines per Operation     : {gp['avg_machines_per_op']:.2f}")
    print(f"Due Date Tightness Factor (TF)     : {gp['tightness_factor']:.2f}")
    print(f"Peak Power Ceiling (P_limit)       : {gp['p_limit_kw']:.2f} kW")
    print(f"Priority Weight Seed               : {gp['random_seed']}")
    print("-" * 80)
    print(f"{'Job ID':<8} | {'Ops':<5} | {'Work W_i (min)':<16} | {'Weight w_i':<10} | {'Due Date D_i (min)':<18}")
    print("-" * 80)

    for job in jobs:
        print(
            f"{job['job_id']:<8} | "
            f"{job['num_operations']:<5} | "
            f"{job['total_expected_work_min']:<16.2f} | "
            f"{job['weight']:<10} | "
            f"{job['due_date_min']:<18.2f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    import sys

    fjs_file = "data/benchmarks/Mk01.fjs"
    cfg_file = "config/system_params.yaml"
    out_file = "data/benchmarks/Mk01_green.json"

    print(f"Parsing and enriching benchmark: {fjs_file} -> {out_file}...")
    dataset = parse_and_enrich_fjs(
        fjs_path=fjs_file,
        yaml_config_path=cfg_file,
        output_json_path=out_file,
        tightness_factor=1.5,
        seed=42,
    )

    print_benchmark_summary(dataset)
    print(f"\n[OK] Enriched Green-FJSP benchmark successfully saved to: {out_file}")

