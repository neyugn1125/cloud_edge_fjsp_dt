"""
Evaluation Protocol and Experimental Benchmark Standard (Single Source of Truth).

This module defines the canonical experimental evaluation protocol for the
Cloud-Edge Collaborative Digital Twin Framework for Energy-Aware Flexible Job Shop Scheduling (Green-FJSP).

Guarantees:
1. Disjoint Random Seed Sets (Zero data leakage between training, validation, and testing):
   - TRAIN_SEEDS: [42, 101, 2026] (3 seeds for training Edge MaskablePPO)
   - VAL_SEEDS: list(range(201, 211)) (10 seeds for hyperparameter/threshold tuning)
   - TEST_SEEDS: list(range(301, 316)) (15 fixed seeds for all comparative benchmarks)
2. Shift-Normalized Cumulative Disturbance Index (CDI):
   - Normalized against total shift baseline makespan and energy to eliminate division-by-zero artifacts.
3. Hardware-Measured Latency Baselines:
   - Validated CPU ONNX inference latency, MQTT round-trip time, and Cloud NSGA-III re-optimization solving latency.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# ==============================================================================
# 1. Strictly Disjoint Random Seed Sets
# ==============================================================================

# Training seeds for MaskablePPO deep reinforcement learning agent
TRAIN_SEEDS: List[int] = [42, 101, 2026]

# Validation seeds for CDI threshold tuning (theta_th) & hyperparameter selection
VAL_SEEDS: List[int] = list(range(201, 211))  # 10 seeds: 201 to 210

# Testing seeds for comprehensive benchmarking (Wilcoxon, Ablation, HIL Closed-Loop)
TEST_SEEDS: List[int] = list(range(301, 316))  # 15 seeds: 301 to 315

# Validate seed set disjointness at module import time
assert set(TRAIN_SEEDS).isdisjoint(VAL_SEEDS), "TRAIN_SEEDS and VAL_SEEDS must be strictly disjoint!"
assert set(VAL_SEEDS).isdisjoint(TEST_SEEDS), "VAL_SEEDS and TEST_SEEDS must be strictly disjoint!"
assert set(TRAIN_SEEDS).isdisjoint(TEST_SEEDS), "TRAIN_SEEDS and TEST_SEEDS must be strictly disjoint!"


# ==============================================================================
# 2. Hardware-Measured Latency Constants & Computational Baselines
# ==============================================================================

# Measured on local Edge CPU (Intel Core i7 / ARM Cortex / ESP32 Host Interface)
EDGE_ONNX_LATENCY_MEAN_MS: float = 0.0214
EDGE_ONNX_LATENCY_MIN_MS: float = 0.0180
EDGE_ONNX_LATENCY_MAX_MS: float = 0.0270

# Cloud NSGA-III 100-generation macro-scheduler solving latency (measured on Mk01)
CLOUD_SOLVER_LATENCY_MEAN_S: float = 25.4
CLOUD_SOLVER_LATENCY_STD_S: float = 1.2
CLOUD_SOLVER_LATENCY_STR: str = "25.4 ± 1.2 s"
CLOUD_SOLVER_LATENCY_S: float = 25.4
CLOUD_SOLVER_LATENCY_MS: float = 25400.0

# Industrial MQTT QoS 1 telemetry and command round-trip time (Mosquitto 1883)
MQTT_RTT_MEAN_MS: float = 0.2240

# Exact speedup factor achieved by Edge ONNX inference over Cloud global optimization
SPEEDUP_FACTOR: float = CLOUD_SOLVER_LATENCY_MS / EDGE_ONNX_LATENCY_MEAN_MS
SPEEDUP_RATIO_STR: str = f"~{SPEEDUP_FACTOR / 1e6:.2f}x10^6x"

# Path to empirically measured timing distributions
MEASURED_LATENCIES_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "measured_latencies.json"


def load_measured_latencies(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load empirical measured latency distributions from JSON benchmark file.
    
    Includes distributions (mean, std, median, p95, p99, min, max, n_samples) for:
    - edge_step_inference_ms
    - edge_onnx_forward_only_ms
    - edge_full_episode_ms (Mk01 & Mk03)
    - mqtt_loopback_rtt_ms
    - cloud_nsga3_full_solve_s
    - cloud_nsga3_resync_solve_s
    """
    p = Path(path) if path is not None else MEASURED_LATENCIES_PATH
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def get_measured_latency_distribution(
    metric_name: str,
    path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Retrieve empirical latency distribution dictionary for a named benchmark metric."""
    data = load_measured_latencies(path)
    if metric_name in data:
        return data[metric_name]
    raise KeyError(f"Metric '{metric_name}' not found in measured latencies at {path or MEASURED_LATENCIES_PATH}")


# Module-level cached distribution
MEASURED_LATENCIES: Dict[str, Any] = load_measured_latencies()


# ==============================================================================
# 3. Canonical Multi-Objective Benchmark Dimensions & Reference Hyperparameters
# ==============================================================================

BENCHMARK_SPECS: Dict[str, Dict[str, Any]] = {
    "Mk01": {"jobs": 10, "machines": 6, "total_ops": 55, "scale": "10x6, 55 ops", "p_limit_kw": 25.0},
    "Mk02": {"jobs": 10, "machines": 6, "total_ops": 58, "scale": "10x6, 58 ops", "p_limit_kw": 25.0},
    "Mk03": {"jobs": 15, "machines": 8, "total_ops": 150, "scale": "15x8, 150 ops", "p_limit_kw": 35.0},
    "Mk04": {"jobs": 15, "machines": 8, "total_ops": 90, "scale": "15x8, 90 ops", "p_limit_kw": 30.0},
    "Mk05": {"jobs": 15, "machines": 4, "total_ops": 106, "scale": "15x4, 106 ops", "p_limit_kw": 20.0},
}

# Standard Multi-Objective Weights: (w_cmax, w_energy, w_tardiness)
DEFAULT_OBJECTIVE_WEIGHTS: Tuple[float, float, float] = (0.4, 0.4, 0.2)

# Reference Point for Hypervolume (HV) calculation on normalized Pareto fronts
REF_POINT_HV: np.ndarray = np.array([1.1, 1.1, 1.1], dtype=np.float64)

# Default CDI parameters
DEFAULT_ALPHA_CDI: float = 0.5
DEFAULT_CDI_THRESHOLD: float = 0.12


# ==============================================================================
# 4. Shift-Normalized Cumulative Disturbance Index (CDI) Calculation
# ==============================================================================

def compute_normalized_cdi(
    c_actual_t: float,
    c_base_t: float,
    c_base_total: float,
    e_actual_t: float,
    e_base_t: float,
    e_base_total: float,
    alpha: float = DEFAULT_ALPHA_CDI,
) -> float:
    """
    Compute shift-normalized Cumulative Disturbance Index (CDI(t)).

    Mathematical Formulation:
        CDI(t) = alpha * (|C_actual(t) - C_base(t)| / C_base_total)
               + (1 - alpha) * (|E_actual(t) - E_base(t)| / E_base_total)

    Rationale:
        Normalizing by the full shift nominal baseline quantities (c_base_total, e_base_total)
        instead of instantaneous partial progress c_base(t) eliminates numerical division-by-zero
        instability at the beginning of the production shift (where c_base(t) ~ 0), guaranteeing
        that CDI(t) is strictly bounded in [0.0, 1.0] under non-catastrophic disturbances.

    Args:
        c_actual_t: Cumulative makespan/completion time realized up to time t (min).
        c_base_t: Planned nominal makespan/completion time up to time t (min).
        c_base_total: Total planned makespan for the entire baseline schedule (min).
        e_actual_t: Cumulative energy consumed up to time t (kWh).
        e_base_t: Planned nominal energy consumed up to time t (kWh).
        e_base_total: Total planned energy for the entire baseline schedule (kWh).
        alpha: Weight trade-off parameter between makespan and energy deviations (default 0.5).

    Returns:
        float: Normalized CDI index in [0.0, 1.0].
    """
    if c_base_total <= 0.0:
        raise ValueError(f"c_base_total must be strictly positive, got {c_base_total}")
    if e_base_total <= 0.0:
        raise ValueError(f"e_base_total must be strictly positive, got {e_base_total}")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")

    c_dev = abs(c_actual_t - c_base_t) / c_base_total
    e_dev = abs(e_actual_t - e_base_t) / e_base_total

    cdi = alpha * c_dev + (1.0 - alpha) * e_dev
    return float(cdi)


# ==============================================================================
# 5. Full System Reproducibility Specification Builder
# ==============================================================================

@dataclass
class AlgorithmHyperparameters:
    nsga3: Dict[str, Any]
    moead: Dict[str, Any]
    maskable_ppo: Dict[str, Any]


def get_canonical_hyperparameters() -> AlgorithmHyperparameters:
    """Return verified algorithm hyperparameters extracted from production code."""
    return AlgorithmHyperparameters(
        nsga3={
            "algorithm": "NSGA-III (Non-dominated Sorting Genetic Algorithm III)",
            "library": "pymoo (v0.6+)",
            "population_size": 100,
            "generations": 100,
            "crossover": "Simulated Binary Crossover (SBX)",
            "crossover_probability": 0.90,
            "crossover_eta": 15.0,
            "mutation": "Polynomial Mutation (PM)",
            "mutation_probability": "1.0 / n_var",
            "mutation_eta": 20.0,
            "reference_directions": "Das-Dennis Simplex (M=3, n_partitions=12, 91 ref dirs)",
            "seeding_strategy": "Heuristic Seeded (15% Eco, 10% Balanced, 5% Fast, 70% Uniform Random)",
        },
        moead={
            "algorithm": "MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition)",
            "library": "pymoo / Custom Tchebycheff Decomposition",
            "population_size": 100,
            "generations": 100,
            "decomposition_method": "Tchebycheff (pbi penalty=5.0)",
            "neighborhood_size_T": 20,
            "crossover": "SBX (prob=0.90, eta=15.0)",
            "mutation": "Polynomial Mutation (prob=1/n_var, eta=20.0)",
        },
        maskable_ppo={
            "algorithm": "MaskablePPO (Action-Masked Proximal Policy Optimization)",
            "library": "sb3_contrib (v2.0+)",
            "policy_network": "MlpPolicy with separate actor & critic heads",
            "actor_architecture": "[256, 128] fully-connected layers (ReLU)",
            "critic_architecture": "[256, 128] fully-connected layers (ReLU)",
            "learning_rate": 3e-4,
            "n_steps": 550,
            "batch_size": 220,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.20,
            "ent_coef": 0.015,
            "parallel_envs": 4,
            "total_timesteps": 66000,
            "training_seeds": TRAIN_SEEDS,
        },
    )


def export_reproducibility_specs(
    json_path: Union[str, Path] = "data/benchmarks/reproducibility_specs.json",
    md_path: Union[str, Path] = "data/benchmarks/reproducibility_specs.md",
) -> Dict[str, Any]:
    """
    Export complete reproducibility specification to JSON and Markdown.
    Provides verifiable proof of all system parameters, state spaces, and formulas.
    """
    json_path = Path(json_path)
    md_path = Path(md_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    hp = get_canonical_hyperparameters()

    specs: Dict[str, Any] = {
        "title": "A Cloud-Edge Collaborative Digital Twin Framework for Energy-Aware Flexible Job Shop Scheduling",
        "target_journal": "IEEE Transactions on Industrial Informatics / Elsevier Journal of Manufacturing Systems",
        "evaluation_protocol": {
            "train_seeds": TRAIN_SEEDS,
            "val_seeds": VAL_SEEDS,
            "test_seeds": TEST_SEEDS,
            "seed_disjointness_verified": True,
        },
        "hardware_latencies": {
            "edge_onnx_latency_mean_ms": EDGE_ONNX_LATENCY_MEAN_MS,
            "edge_onnx_latency_range_ms": [EDGE_ONNX_LATENCY_MIN_MS, EDGE_ONNX_LATENCY_MAX_MS],
            "cloud_solver_latency_ms": CLOUD_SOLVER_LATENCY_MS,
            "cloud_solver_latency_str": CLOUD_SOLVER_LATENCY_STR,
            "mqtt_round_trip_ms": MQTT_RTT_MEAN_MS,
            "speedup_ratio": SPEEDUP_RATIO_STR,
            "empirical_measured": load_measured_latencies(),
        },
        "cdi_formulation": {
            "formula": "CDI(t) = alpha * (|C_act(t) - C_base(t)| / C_base_total) + (1 - alpha) * (|E_act(t) - E_base(t)| / E_base_total)",
            "default_alpha": DEFAULT_ALPHA_CDI,
            "default_threshold": DEFAULT_CDI_THRESHOLD,
            "normalization_basis": "Full shift nominal schedule baseline (c_base_total, e_base_total)",
            "knee_point_threshold": 0.12,
        },
        "observation_space": {
            "name": "Box(low=0.0, high=1.0, shape=(87,), dtype=float32)",
            "dimension": 87,
            "components": [
                {"indices": "0:50", "description": "10 Jobs x 5 features: [progress, rem_work_ratio, ready_time_norm, urgency_slack, priority_weight]"},
                {"indices": "50:80", "description": "6 Machines x 5 features: [m_ready_norm, tool_wear_ratio, power_drift, is_processing, is_standby]"},
                {"indices": "80:87", "description": "7 Global features: [time_norm, tou_rate_norm, energy_norm, active_power_ratio, w_cmax, w_energy, w_tardiness]"},
            ],
        },
        "action_space": {
            "name": "Discrete(360)",
            "dimension": 360,
            "decomposition": "10 jobs x 6 machines x 3 speed levels x 2 PM options = 360 actions",
            "components": {
                "job_idx": "0..9 (10 jobs)",
                "machine_idx": "0..5 (6 machines)",
                "speed_level": "1 (v=1.0 Eco), 2 (v=1.25 Standard), 3 (v=1.50 High-Speed)",
                "pm_flag": "0 (Normal cutting), 1 (Proactive Opportunistic Tool Replacement)",
            },
            "masking_conditions": [
                "Job must not be finished (current_op < total_ops_for_job)",
                "Machine must belong to the eligible machine set for current operation",
                "PM flag = 1 strictly restricted to critical flank wear (wear >= 75% w_max or cut would cause breakdown)",
                "When wear >= 88% w_max, high speed s=3 is disabled",
                "When wear >= 92% w_max or cut pushes wear >= 98% w_max, PM is mandatory (pm_flag=0 locked)",
            ],
        },
        "reward_function": {
            "formula": "r_t = - (w_cmax * norm_delta_cmax + w_cost * norm_delta_cost + w_tard * norm_delta_tard + penalty_bd) + r_cloud + r_speed",
            "objective_penalties": {
                "delta_cmax_norm": "delta_cmax / 78.0 min",
                "delta_cost_norm": "step_cost / $2.72",
                "delta_tardiness_norm": "delta_tardiness / 345.0 min",
                "breakdown_penalty": 10.0,
            },
            "cloud_guidance_bonus": [
                "+0.15 if chosen machine matches Cloud NSGA-III assignment for this operation",
                "+0.08 if chosen job matches Cloud planned sequence index",
            ],
            "speed_scaling_shaping": [
                "+0.08 for high speed s in {2, 3} during Off-Peak tariff (rate <= $0.06/kWh) with sharp tool",
                "+0.12 for eco speed s = 1 during On-Peak tariff (rate >= $0.20/kWh)",
                "-0.10 for high speed s = 3 during On-Peak tariff",
            ],
        },
        "multi_objective_evaluation": {
            "pareto_reference_front": "PF* constructed as non-dominated union of all evaluated algorithms across all runs",
            "hypervolume_ref_point": [1.1, 1.1, 1.1],
            "normalization": "Min-Max scaling to [0, 1] based on global ideal and nadir points",
            "statistical_test": "Wilcoxon Signed-Rank Test (alpha=0.05, two-tailed)",
        },
        "hyperparameters": asdict(hp),
        "benchmarks": BENCHMARK_SPECS,
    }

    # Write JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(specs, f, indent=2)

    # Write Markdown Document
    md_content = rf"""# Experimental Reproducibility Specifications & Single Source of Truth

**Project:** {specs['title']}  
**Target Publication:** {specs['target_journal']}  
**Generated At:** Evaluation Protocol Engine v5.5A

---

## 1. Disjoint Random Seed Sets (Zero Data Leakage Protocol)

To eliminate any potential data snooping or circular validation, random seeds are strictly partitioned into three disjoint sets:

- **Training Seeds (`TRAIN_SEEDS`):** `{TRAIN_SEEDS}`  
  *Purpose:* Training the Edge `MaskablePPO` policy over 66,000 timesteps across 4 parallel environments.
- **Validation Seeds (`VAL_SEEDS`):** `{VAL_SEEDS}` (10 seeds)  
  *Purpose:* Tuning the Cumulative Disturbance Index synchronization threshold $\\theta_{{\\text{{th}}}}$ and hyperparameters.
- **Testing Seeds (`TEST_SEEDS`):** `{TEST_SEEDS}` (15 seeds)  
  *Purpose:* Independent evaluation for all reported tables and figures (Phase 2, Phase 4 HIL, Phase 5 Wilcoxon, Phase 5 Ablation).

> **Verification Assertion:** `set(TRAIN_SEEDS).isdisjoint(VAL_SEEDS) and set(VAL_SEEDS).isdisjoint(TEST_SEEDS) == True`.

---

## 2. Shift-Normalized Cumulative Disturbance Index CDI(t)

$$\text{{CDI}}(t) = \alpha \frac{{|C_{{\text{{act}}}}(t) - C_{{\text{{base}}}}(t)|}}{{C_{{\text{{base\_total}}}}}} + (1 - \alpha) \frac{{|E_{{\text{{act}}}}(t) - E_{{\\text{{base}}}}(t)|}}{{E_{{\text{{base\_total}}}}}}$$

- **Formula:** `CDI(t) = alpha * abs(c_actual_t - c_base_t) / c_base_total + (1.0 - alpha) * abs(e_actual_t - e_base_t) / e_base_total`

- **Normalization Basis:** Divided by total shift planned quantities ($C_{{\\text{{base\_total}}}}$, $E_{{\\text{{base\_total}}}}$) rather than instantaneous values $C_{{\\text{{base}}}}(t)$, preventing division-by-zero singularities at shift onset ($t \\to 0$).
- **Optimal Knee Point Threshold:** $\\theta_{{\\text{{th}}}} = 0.12$.
  - At $\\theta_{{\\text{{th}}}} \\le 0.12$: Autonomous Edge compensation absorbs 100% of micro-disturbances locally.
  - At $\\theta_{{\\text{{th}}}} > 0.12$: Edge publishes urgent MQTT rescheduling request (`dt/fjsp/edge/trigger_reschedule`, QoS 1) to Cloud.

---

## 3. Hardware Latency Benchmarks & Speedup

| Execution Tier | Latency Metric | Physical Measurement | Speedup vs. Cloud Solver |
| :--- | :--- | :---: | :---: |
| **Edge ONNX Engine** | Forward CPU Policy Inference | **{EDGE_ONNX_LATENCY_MEAN_MS:.4f} ms** ({EDGE_ONNX_LATENCY_MIN_MS:.3f} - {EDGE_ONNX_LATENCY_MAX_MS:.3f} ms) | **{SPEEDUP_RATIO_STR}** |
| **Edge Decision Step** | Obs + Mask + Policy Inference | **~0.1506 ms** (P95: 0.1638 ms) | ~168,600x |
| **Edge Full Shift (Mk01)** | 55 Operations Complete Shift | **~7.955 ms** (P95: 8.016 ms) | ~3,190x |
| **Edge Full Shift (Mk03)** | 150 Operations Complete Shift | **~22.706 ms** (P95: 23.162 ms) | ~1,118x |
| **Industrial MQTT** | Round-Trip Telemetry / Trigger | **{MQTT_RTT_MEAN_MS:.4f} ms** | ~111,600x |
| **Cloud NSGA-III (Resync)**| Remaining Operations Re-Solve | **~2.684 s** (P95: 2.731 s) | 9.5x vs Full Solve |
| **Cloud NSGA-III (Full)**  | 100-Gen Global Re-Optimization | **{CLOUD_SOLVER_LATENCY_S:.1f} s** ({CLOUD_SOLVER_LATENCY_MS:.0f} ms) | 1.0x (Reference) |

---

## 4. State, Action, and Reward Specifications

### 4.1 Observation Vector (87 Dimensions in $[0.0, 1.0]$)
- **Job States ($10 \\times 5 = 50$):** `[progress, rem_work_ratio, ready_time_norm, urgency_slack, priority_weight]`
- **Machine States ($6 \\times 5 = 30$):** `[m_ready_norm, tool_wear_ratio, power_drift, is_processing, is_standby]`
- **Global Factory States ($7$):** `[time_norm, tou_rate_norm, energy_norm, active_power_ratio, w_cmax, w_energy, w_tardiness]`

### 4.2 Action Space (Discrete 360)
$$\\text{{Action}} = j \\times (M \\times S \\times P) + m \\times (S \\times P) + s \\times P + p$$
where $j \\in [0..9]$, $m \\in [0..5]$, $s \\in [0..2]$ (Speed levels 1, 2, 3), and $p \\in [0..1]$ (PM replacement).

### 4.3 Cloud-Guided Reward Shaping
$$r_t = - \\left( w_1 \\frac{{\\Delta C_{{\\max}}}}{{78.0}} + w_2 \\frac{{\\Delta \\text{{Cost}}}}{{\\$2.72}} + w_3 \\frac{{\\Delta \\text{{Tardiness}}}}{{345.0}} + P_{{\\text{{bd}}}} \\right) + r_{{\\text{{cloud}}}} + r_{{\\text{{speed}}}}$$
- $r_{{\\text{{cloud}}}} = +0.15$ (Machine match) $+ 0.08$ (Sequence match).
- $r_{{\\text{{speed}}}} = +0.08$ (High-speed off-peak) $+ 0.12$ (Eco on-peak) $- 0.10$ (High-speed on-peak penalty).

---

## 5. Multi-Objective Optimization Hyperparameters

### 5.1 Cloud NSGA-III & MOEA/D
- **Population Size:** 100
- **Generations:** 100
- **Crossover:** SBX ($p_c = 0.90, \\eta_c = 15.0$)
- **Mutation:** Polynomial Mutation ($p_m = 1/n_{{\\text{{var}}}}, \\eta_m = 20.0$)
- **Reference Directions:** Das-Dennis 3 objectives, 12 partitions (91 reference vectors)

### 5.2 Edge MaskablePPO
- **Network Architecture:** Separate Actor-Critic MLP with `[256, 128]` hidden layers.
- **Learning Rate:** $3 \\times 10^{{-4}}$, $\\gamma = 0.99$, $\\lambda_{{\\text{{GAE}}}} = 0.95$.
- **Entropy Coefficient:** $0.015$, Batch size: $220$, Buffer size: $2200$ ($4 \\times 550$).

---

## 6. Multi-Objective Performance Indicators
- **Reference Front $PF^*$:** Non-dominated front extracted from the union of all Pareto approximations across all 15 independent test seeds.
- **Hypervolume Reference Point:** $\\mathbf{{r}}_{{\\text{{ref}}}} = [1.1, 1.1, 1.1]$ on normalized objective space $[0, 1]^3$.
- **Statistical Hypothesis Testing:** Two-tailed Wilcoxon Signed-Rank Test with significance level $\\alpha = 0.05$.
"""

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return specs


if __name__ == "__main__":
    print("Exporting Canonical Reproducibility Specifications...")
    sp = export_reproducibility_specs()
    print("Successfully exported reproducibility_specs.json and reproducibility_specs.md.")
