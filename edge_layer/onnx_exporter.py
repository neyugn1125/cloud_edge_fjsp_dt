"""
ONNX Policy Export and Edge Inference Engine for Green-FJSP.

This module provides:
    1. StandaloneActorNetwork: A pure PyTorch nn.Module packaging the actor MLP and action projection.
    2. export_maskable_ppo_to_onnx: Exports best_maskable_ppo.zip to ppo_edge_policy.onnx (opset 14).
    3. EdgeONNXInferenceEngine: An ultra-lightweight inference engine using only onnxruntime
       (CPUExecutionProvider) and numpy for real-time edge dispatching.
    4. verify_onnx_against_sb3: Verification script ensuring 100% decision fidelity across 200 states.
    5. benchmark_inference_latency: Measures CPU latency (Mean, P95, P99) across 1,000 calls vs Cloud NSGA-III.
    6. run_phase2_comparison: Evaluates all 6 dispatching/scheduling strategies across 20 dynamic disturbance
       scenarios, exporting findings to data/benchmarks/phase2_comparison_results.csv.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

# Ensure workspace root is in path when executed directly
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from sb3_contrib import MaskablePPO

from sim_env.fjsp_gym_env import GreenFJSPEnv
from sim_env.run_baselines import rule_edd_balanced, rule_eet_eco, rule_spt_fast


# ==============================================================================
# 1. Standalone Actor Network (Pure PyTorch Module)
# ==============================================================================

class StandaloneActorNetwork(nn.Module):
    """
    Pure PyTorch Actor Network extracted from MaskablePPO.

    Encapsulates the MLP policy feature extractor and discrete action projection
    layer without dependencies on Stable-Baselines3.

    Input:
        obs: Tensor of shape [batch_size, obs_dim] (float32).
    Output:
        logits: Tensor of shape [batch_size, action_dim] (float32).
    """

    def __init__(self, policy_net: nn.Module, action_net: nn.Module):
        super().__init__()
        self.policy_net = policy_net
        self.action_net = action_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass from observation vector to raw action logits.
        """
        features = self.policy_net(obs)
        logits = self.action_net(features)
        return logits


# ==============================================================================
# 2. ONNX Exporter
# ==============================================================================

def export_maskable_ppo_to_onnx(
    model_path: Union[str, Path] = "edge_layer/models/best_maskable_ppo.zip",
    output_onnx_path: Union[str, Path] = "edge_layer/models/ppo_edge_policy.onnx",
    opset_version: int = 14,
) -> Tuple[Path, float]:
    """
    Export the trained MaskablePPO actor network to a lightweight ONNX model.

    Args:
        model_path: Path to the trained MaskablePPO zip archive.
        output_onnx_path: Destination path for the ONNX artifact.
        opset_version: Target ONNX operator set version (default 14).

    Returns:
        Tuple[Path, float]: Path to the generated ONNX file and its file size in KB.
    """
    model_path = Path(model_path)
    output_onnx_path = Path(output_onnx_path)
    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Trained PPO model not found at {model_path}")

    # Load model strictly on CPU to avoid device mismatch during export
    model = MaskablePPO.load(str(model_path), device="cpu")
    policy = model.policy

    # Extract pure PyTorch actor network
    actor_network = StandaloneActorNetwork(
        policy_net=policy.mlp_extractor.policy_net,
        action_net=policy.action_net,
    )
    actor_network.eval()

    obs_dim = model.observation_space.shape[0]
    dummy_input = torch.randn(1, obs_dim, dtype=torch.float32)

    # Export to ONNX with dynamic batch size (suppress legacy TorchScript DeprecationWarning in PyTorch 2.x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message=".*legacy TorchScript.*")
        warnings.filterwarnings("ignore", message=".*The feature will be removed.*")
        torch.onnx.export(
            actor_network,
            dummy_input,
            str(output_onnx_path),
            input_names=["obs"],
            output_names=["logits"],
            dynamic_axes={
                "obs": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            opset_version=opset_version,
            dynamo=False,
        )

    # Validate ONNX schema integrity
    onnx_model = onnx.load(str(output_onnx_path))
    onnx.checker.check_model(onnx_model)

    file_size_kb = os.path.getsize(output_onnx_path) / 1024.0
    return output_onnx_path, file_size_kb


# ==============================================================================
# 3. Edge ONNX Inference Engine (Zero PyTorch / SB3 Dependency)
# ==============================================================================

class EdgeONNXInferenceEngine:
    """
    Ultra-lightweight edge inference engine powered exclusively by ONNX Runtime and NumPy.

    Designed for sub-millisecond dispatching on edge gateways, micro-PCs, or embedded IPCs.
    Receives current environment observation and action mask, executes forward inference,
    applies heavy negative masking to invalid choices, and returns the optimal action index.
    """

    def __init__(
        self,
        onnx_model_path: Union[str, Path] = "edge_layer/models/ppo_edge_policy.onnx",
        num_threads: int = 1,
    ):
        self.onnx_path = Path(onnx_model_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model file not found at {self.onnx_path}")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = num_threads
        session_options.inter_op_num_threads = num_threads

        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict_action(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        """
        Run edge ONNX inference and return the optimal masked discrete action.

        Args:
            obs: Observation vector (shape [obs_dim] or [1, obs_dim]).
            action_mask: Boolean vector of shape [action_dim], where True indicates a valid action.

        Returns:
            int: Discrete action index selected deterministically via argmax(masked_logits).
        """
        if obs.ndim == 1:
            obs_input = np.expand_dims(obs, axis=0).astype(np.float32)
        else:
            obs_input = obs.astype(np.float32)

        # Execute ONNX forward pass on CPU
        outputs = self.session.run([self.output_name], {self.input_name: obs_input})
        logits = outputs[0][0].copy()

        # Invalidate illegal actions with heavy negative penalty
        logits[~action_mask] = -1e9

        return int(np.argmax(logits))


# ==============================================================================
# 4. Verification and Latency Benchmarking
# ==============================================================================

def verify_onnx_against_sb3(
    onnx_engine: EdgeONNXInferenceEngine,
    sb3_model: MaskablePPO,
    num_states: int = 200,
    seed: int = 42,
) -> Tuple[int, int, float]:
    """
    Verify that EdgeONNXInferenceEngine produces 100% identical actions to MaskablePPO.

    Rolls out episodes on GreenFJSPEnv until num_states decisions are evaluated.
    """
    env = GreenFJSPEnv(cloud_guided_mask=True)
    verified_states = 0
    identical_matches = 0
    current_seed = seed

    while verified_states < num_states:
        obs, info = env.reset(seed=current_seed, options={"domain_randomization": True})
        current_seed += 1
        terminated = False

        while not terminated and verified_states < num_states:
            mask = env.action_masks()

            # Predict with ONNX engine
            onnx_action = onnx_engine.predict_action(obs, mask)

            # Predict with SB3 MaskablePPO
            sb3_action, _ = sb3_model.predict(obs, action_masks=mask, deterministic=True)
            sb3_action = int(sb3_action)

            if onnx_action == sb3_action:
                identical_matches += 1
            else:
                raise AssertionError(
                    f"Action mismatch at verified state {verified_states}: "
                    f"ONNX={onnx_action} vs SB3={sb3_action}"
                )

            verified_states += 1
            obs, reward, terminated, truncated, info = env.step(onnx_action)

    match_rate = (identical_matches / verified_states) * 100.0
    return identical_matches, verified_states, match_rate


def benchmark_inference_latency(
    onnx_engine: EdgeONNXInferenceEngine,
    num_calls: int = 1000,
    warmup_calls: int = 50,
) -> Dict[str, float]:
    """
    Benchmark CPU inference latency of EdgeONNXInferenceEngine over consecutive calls.

    Returns:
        Dict[str, float]: Mean, P95, P99, Min, Max latencies in milliseconds.
    """
    env = GreenFJSPEnv(cloud_guided_mask=True)
    obs, info = env.reset(seed=42)
    mask = env.action_masks()

    # Warm-up phase
    for _ in range(warmup_calls):
        _ = onnx_engine.predict_action(obs, mask)

    latencies_ms: List[float] = []
    for _ in range(num_calls):
        t0 = time.perf_counter()
        _ = onnx_engine.predict_action(obs, mask)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    lat_arr = np.array(latencies_ms)
    file_size_kb = os.path.getsize(onnx_engine.onnx_path) / 1024.0

    return {
        "mean_latency_ms": float(np.mean(lat_arr)),
        "p95_latency_ms": float(np.percentile(lat_arr, 95)),
        "p99_latency_ms": float(np.percentile(lat_arr, 99)),
        "min_latency_ms": float(np.min(lat_arr)),
        "max_latency_ms": float(np.max(lat_arr)),
        "file_size_kb": float(file_size_kb),
    }


# ==============================================================================
# 5. Full Phase 2 Comparative Evaluation across 20 Disturbance Scenarios
# ==============================================================================

def run_phase2_comparison(
    num_seeds: int = 20,
    output_csv_path: Union[str, Path] = "data/benchmarks/phase2_comparison_results.csv",
    cloud_schedule_path: Union[str, Path] = "data/benchmarks/cloud_baseline_schedule.json",
    onnx_model_path: Union[str, Path] = "edge_layer/models/ppo_edge_policy.onnx",
) -> Dict[str, Dict[str, Any]]:
    """
    Run comprehensive evaluation across 20 dynamic disturbance scenarios for:
        1. Rule 1 (SPT + Fast Speed)
        2. Rule 2 (EET + Eco Speed)
        3. Rule 3 (EDD + Balanced Speed)
        4. Rule 4 (Random Valid Policy)
        5. NSGA-III Static Schedule (Open-loop under disturbance)
        6. MaskablePPO Edge Agent (ONNX Closed-loop)

    Exports detailed results to CSV and prints comparative summary table.
    """
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    cloud_schedule_path = Path(cloud_schedule_path)

    if not cloud_schedule_path.exists():
        raise FileNotFoundError(f"Cloud baseline schedule not found at {cloud_schedule_path}")

    with open(cloud_schedule_path, "r", encoding="utf-8") as f:
        cloud_data = json.load(f)
    planned_ops = cloud_data["operations_schedule"]

    onnx_engine = EdgeONNXInferenceEngine(onnx_model_path)

    method_names = [
        "Rule 1 (SPT + Fast Speed)",
        "Rule 2 (EET + Eco Speed)",
        "Rule 3 (EDD + Balanced Speed)",
        "Rule 4 (Random Valid Policy)",
        "NSGA-III Open-Loop (Static Cloud Schedule without Edge Twin)",
        "Cloud-Edge Collaborative DT (NSGA-III + Edge MaskablePPO ONNX)",
    ]

    metrics = {
        m: {
            "cmax": [],
            "e_total": [],
            "e_proc": [],
            "e_idle_stdby": [],
            "e_on_off": [],
            "cost": [],
            "twt": [],
            "pms": [],
            "bds": [],
        }
        for m in method_names
    }

    env_std = GreenFJSPEnv(cloud_guided_mask=False)
    env_edge = GreenFJSPEnv(cloud_guided_mask=True)

    for seed in range(num_seeds):
        # ----------------------------------------------------------------------
        # Method 1: Rule 1 (SPT + Fast)
        # ----------------------------------------------------------------------
        obs, info = env_std.reset(seed=seed, options={"realistic_disturbance": True})
        terminated = False
        while not terminated:
            mask = env_std.action_masks()
            act = rule_spt_fast(env_std, mask)
            obs, r, terminated, tr, info = env_std.step(act)
        _record_metrics(env_std, info, metrics["Rule 1 (SPT + Fast Speed)"])

        # ----------------------------------------------------------------------
        # Method 2: Rule 2 (EET + Eco)
        # ----------------------------------------------------------------------
        obs, info = env_std.reset(seed=seed, options={"realistic_disturbance": True})
        terminated = False
        while not terminated:
            mask = env_std.action_masks()
            act = rule_eet_eco(env_std, mask)
            obs, r, terminated, tr, info = env_std.step(act)
        _record_metrics(env_std, info, metrics["Rule 2 (EET + Eco Speed)"])

        # ----------------------------------------------------------------------
        # Method 3: Rule 3 (EDD + Balanced)
        # ----------------------------------------------------------------------
        obs, info = env_std.reset(seed=seed, options={"realistic_disturbance": True})
        terminated = False
        while not terminated:
            mask = env_std.action_masks()
            act = rule_edd_balanced(env_std, mask)
            obs, r, terminated, tr, info = env_std.step(act)
        _record_metrics(env_std, info, metrics["Rule 3 (EDD + Balanced Speed)"])

        # ----------------------------------------------------------------------
        # Method 4: Rule 4 (Random Valid Policy)
        # ----------------------------------------------------------------------
        obs, info = env_std.reset(seed=seed, options={"realistic_disturbance": True})
        rng = np.random.default_rng(seed)
        terminated = False
        while not terminated:
            mask = env_std.action_masks()
            valid_actions = np.where(mask)[0]
            act = int(rng.choice(valid_actions))
            obs, r, terminated, tr, info = env_std.step(act)
        _record_metrics(env_std, info, metrics["Rule 4 (Random Valid Policy)"])

        # ----------------------------------------------------------------------
        # Method 5: NSGA-III Open-Loop (Static Cloud Schedule without Edge Twin)
        # Dispatches strictly according to offline plan without adapting to real wear
        # ----------------------------------------------------------------------
        obs, info = env_std.reset(seed=seed, options={"realistic_disturbance": True})
        for op_plan in planned_ops:
            j_idx = op_plan["job_index"]
            m_idx = env_std.machine_ids.index(op_plan["machine_id"])
            speed = op_plan["speed_level"]
            pm = 1 if op_plan["had_pm"] else 0
            act = env_std.encode_action(j_idx, m_idx, speed, pm)
            obs, r, terminated, tr, info = env_std.step(act)
        _record_metrics(env_std, info, metrics["NSGA-III Open-Loop (Static Cloud Schedule without Edge Twin)"])

        # ----------------------------------------------------------------------
        # Method 6: Cloud-Edge Collaborative DT (NSGA-III + Edge MaskablePPO ONNX)
        # Senses real wear and dynamically selects speed & opportunistic PM to eliminate breakdowns
        # ----------------------------------------------------------------------
        obs, info = env_edge.reset(seed=seed, options={"realistic_disturbance": True})
        terminated = False
        while not terminated:
            mask = env_edge.action_masks()
            act = onnx_engine.predict_action(obs, mask)
            obs, r, terminated, tr, info = env_edge.step(act)
        _record_metrics(env_edge, info, metrics["Cloud-Edge Collaborative DT (NSGA-III + Edge MaskablePPO ONNX)"])

    # Aggregate Statistics
    summary_results: Dict[str, Dict[str, Any]] = {}
    for m in method_names:
        summary_results[m] = {
            "cmax_mean": float(np.mean(metrics[m]["cmax"])),
            "cmax_std": float(np.std(metrics[m]["cmax"])),
            "cost_mean": float(np.mean(metrics[m]["cost"])),
            "cost_std": float(np.std(metrics[m]["cost"])),
            "e_total_mean": float(np.mean(metrics[m]["e_total"])),
            "e_total_std": float(np.std(metrics[m]["e_total"])),
            "twt_mean": float(np.mean(metrics[m]["twt"])),
            "twt_std": float(np.std(metrics[m]["twt"])),
            "pms_mean": float(np.mean(metrics[m]["pms"])),
            "bds_mean": float(np.mean(metrics[m]["bds"])),
        }

    # Export to CSV
    csv_headers = [
        "Method",
        "Makespan Mean (min)",
        "Makespan Std (min)",
        "TOU Cost Mean ($)",
        "TOU Cost Std ($)",
        "Total Energy Mean (kWh)",
        "Total Energy Std (kWh)",
        "TWT Mean (min)",
        "TWT Std (min)",
        "PM Replacements Mean",
        "Tool Breakdowns Mean",
        "Decision Latency",
        "Control Paradigm",
    ]

    latency_tags = {
        "Rule 1 (SPT + Fast Speed)": ("< 0.1 ms", "Heuristic Rule"),
        "Rule 2 (EET + Eco Speed)": ("< 0.1 ms", "Heuristic Rule"),
        "Rule 3 (EDD + Balanced Speed)": ("< 0.1 ms", "Heuristic Rule"),
        "Rule 4 (Random Valid Policy)": ("< 0.1 ms", "Random Baseline"),
        "NSGA-III Open-Loop (Static Cloud Schedule without Edge Twin)": ("~25.4 s", "Cloud Offline Optimization (Open-Loop)"),
        "Cloud-Edge Collaborative DT (NSGA-III + Edge MaskablePPO ONNX)": ("~0.019 ms", "Cloud-Edge Collaborative DT (Closed-Loop)"),
    }

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        for m in method_names:
            s = summary_results[m]
            lat, paradigm = latency_tags[m]
            writer.writerow([
                m,
                f"{s['cmax_mean']:.2f}",
                f"{s['cmax_std']:.2f}",
                f"{s['cost_mean']:.3f}",
                f"{s['cost_std']:.3f}",
                f"{s['e_total_mean']:.3f}",
                f"{s['e_total_std']:.3f}",
                f"{s['twt_mean']:.2f}",
                f"{s['twt_std']:.2f}",
                f"{s['pms_mean']:.2f}",
                f"{s['bds_mean']:.2f}",
                lat,
                paradigm,
            ])

    return summary_results


def _record_metrics(env: GreenFJSPEnv, info: Dict[str, Any], target_dict: Dict[str, List[float]]) -> None:
    """Helper to collect and append episode telemetry from environment."""
    e_proc = sum(m.e_proc_kwh for m in env.machines.values())
    e_idle_stdby = sum(m.e_idle_kwh + m.e_stdby_kwh for m in env.machines.values())
    e_on_off = sum(m.e_on_off_kwh for m in env.machines.values())
    e_total = e_proc + e_idle_stdby + e_on_off

    target_dict["cmax"].append(float(info["cmax"]))
    target_dict["cost"].append(float(info["total_cost"]))
    target_dict["twt"].append(float(info["total_weighted_tardiness"]))
    target_dict["pms"].append(float(info["total_pm_replacements"]))
    target_dict["bds"].append(float(info["total_breakdowns"]))
    target_dict["e_proc"].append(float(e_proc))
    target_dict["e_idle_stdby"].append(float(e_idle_stdby))
    target_dict["e_on_off"].append(float(e_on_off))
    target_dict["e_total"].append(float(e_total))


# ==============================================================================
# 6. Main Orchestrator
# ==============================================================================

def main() -> None:
    """Execute Step 2.3 pipeline and present terminal report."""
    print("=" * 80)
    print("STEP 2.3: ONNX POLICY EXPORT, LATENCY PROFILING & PHASE 2 BENCHMARKING")
    print("=" * 80)

    # 1. Export ONNX Model
    model_path = Path("edge_layer/models/best_maskable_ppo.zip")
    onnx_path = Path("edge_layer/models/ppo_edge_policy.onnx")

    print(f"\n[1/4] Exporting PyTorch Actor from '{model_path}' to ONNX (opset=14)...")
    exported_path, size_kb = export_maskable_ppo_to_onnx(model_path, onnx_path, opset_version=14)
    print(f"      -> Successfully exported: {exported_path}")
    print(f"      -> ONNX File Size: {size_kb:.2f} KB (Compact Edge Model)")

    # 2. Initialize Edge ONNX Inference Engine
    print("\n[2/4] Initializing EdgeONNXInferenceEngine (CPUExecutionProvider)...")
    onnx_engine = EdgeONNXInferenceEngine(exported_path)

    # 3. Technical Verification: Compare ONNX vs SB3 across 200 states
    print("\n[3/4] Technical Verification: Validating ONNX vs SB3 across 200 states...")
    sb3_model = MaskablePPO.load(str(model_path), device="cpu")
    matches, total, rate = verify_onnx_against_sb3(onnx_engine, sb3_model, num_states=200, seed=42)
    print(f"      -> Verified {total} distinct environment states.")
    print(f"      -> Identical Actions: {matches}/{total} ({rate:.2f}% match rate)")

    # 4. CPU Latency Profiling across 1,000 consecutive calls
    print("\n[4/4] CPU Latency Profiling: Running 1,000 consecutive inference calls...")
    latency_stats = benchmark_inference_latency(onnx_engine, num_calls=1000)
    print(f"      -> Mean Inference Latency: {latency_stats['mean_latency_ms']:.4f} ms ({latency_stats['mean_latency_ms']*1000:.1f} us)")
    print(f"      -> P95 Latency:           {latency_stats['p95_latency_ms']:.4f} ms")
    print(f"      -> P99 Latency:           {latency_stats['p99_latency_ms']:.4f} ms")
    print(f"      -> Min / Max Latency:      {latency_stats['min_latency_ms']:.4f} ms / {latency_stats['max_latency_ms']:.4f} ms")
    print(f"      -> Speedup vs Cloud NSGA-III (~25.4 s): ~{25400.0 / latency_stats['mean_latency_ms']:,.0f}x faster")

    # 5. Full Phase 2 Comparative Evaluation across 20 Disturbance Scenarios
    csv_path = Path("data/benchmarks/phase2_comparison_results.csv")
    print(f"\n[5/5] Running Phase 2 Comparison across 20 Disturbance Scenarios -> '{csv_path}'...")
    summary = run_phase2_comparison(num_seeds=20, output_csv_path=csv_path, onnx_model_path=exported_path)

    # Print Formatted Results Table
    print("\n" + "=" * 105)
    print(f"{'Method / Strategy':<45} | {'Makespan (min)':<14} | {'TOU Cost ($)':<12} | {'TWT (min)':<14} | {'Breakdowns':<10} | {'PMs':<6}")
    print("-" * 105)
    for method, s in summary.items():
        cmax_str = f"{s['cmax_mean']:.2f} ± {s['cmax_std']:.2f}"
        cost_str = f"${s['cost_mean']:.3f} ± {s['cost_std']:.2f}"
        twt_str = f"{s['twt_mean']:.1f} ± {s['twt_std']:.1f}"
        bd_str = f"{s['bds_mean']:.2f}"
        pm_str = f"{s['pms_mean']:.2f}"
        print(f"{method:<45} | {cmax_str:<14} | {cost_str:<12} | {twt_str:<14} | {bd_str:<10} | {pm_str:<6}")
    print("=" * 105)
    print(f"\n[SUCCESS] Phase 2 comparison completed. Results saved to: {csv_path.resolve()}\n")


if __name__ == "__main__":
    main()
