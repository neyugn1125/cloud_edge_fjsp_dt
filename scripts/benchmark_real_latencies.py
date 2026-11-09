"""
Empirical Timing Benchmark Script for Cloud-Edge Digital Twin Framework.

Measures real execution latencies using time.perf_counter_ns() on the host machine:
1. edge_step_inference_ms: Full decision step at Edge (_get_obs + action_masks + predict_action).
2. edge_onnx_forward_only_ms: Pure forward pass time of sess.run() in ONNX Runtime.
3. edge_full_episode_ms: Cumulative Edge computation time for full production shift (Mk01 & Mk03).
4. mqtt_loopback_rtt_ms: End-to-end JSON serialization, broker publish/receive, deserialization.
5. cloud_nsga3_full_solve_s: Real execution time of NSGA-III solver for full problem (Mk01 & Mk03).
6. cloud_nsga3_resync_solve_s: Real execution time of NSGA-III solver for rescheduling remaining ops.

Exports empirical distributions to data/benchmarks/measured_latencies.json and results/measured_latencies.json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from cloud_layer.nsga3_optimizer import GreenFJSPProblem, run_nsga3_optimization
from edge_layer.onnx_exporter import EdgeONNXInferenceEngine
from sim_env.fjsp_gym_env import GreenFJSPEnv
from tests.test_task4_3_full_hil import fast_cloud_reschedule_active
from utils.mqtt_broker import InMemoryMQTTBroker
from utils.mqtt_schemas import RescheduleTriggerPayload, TOPIC_EDGE_TRIGGER_RESCHEDULE


def compute_distribution_stats(values: Union[List[float], np.ndarray], unit: str) -> Dict[str, Any]:
    """Calculate comprehensive distribution statistics for a latency sample set."""
    arr = np.array(values, dtype=np.float64)
    precision = 6 if unit == "ms" else 4
    return {
        "mean": float(round(float(np.mean(arr)), precision)),
        "std": float(round(float(np.std(arr)), precision)),
        "median": float(round(float(np.median(arr)), precision)),
        "p95": float(round(float(np.percentile(arr, 95)), precision)),
        "p99": float(round(float(np.percentile(arr, 99)), precision)),
        "min": float(round(float(np.min(arr)), precision)),
        "max": float(round(float(np.max(arr)), precision)),
        "n_samples": int(len(arr)),
        "unit": unit,
    }


def benchmark_edge_onnx_forward_only(
    engine: EdgeONNXInferenceEngine,
    n_samples: int = 1000,
    warmup: int = 50,
) -> Dict[str, Any]:
    """Measure pure forward pass time of sess.run() in ONNX Runtime."""
    obs_dim = engine.session.get_inputs()[0].shape[1]
    dummy_obs = np.random.uniform(0.0, 1.0, size=(1, obs_dim)).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        engine.session.run(None, {engine.input_name: dummy_obs})

    # Benchmark
    latencies_ms: List[float] = []
    for _ in range(n_samples):
        t0 = time.perf_counter_ns()
        engine.session.run(None, {engine.input_name: dummy_obs})
        t1 = time.perf_counter_ns()
        latencies_ms.append((t1 - t0) / 1e6)

    return compute_distribution_stats(latencies_ms, "ms")


def benchmark_edge_step_inference(
    env: GreenFJSPEnv,
    engine: EdgeONNXInferenceEngine,
    n_samples: int = 1000,
    warmup: int = 50,
) -> Dict[str, Any]:
    """
    Measure real execution time of 1 complete decision step at Edge:
    _get_obs() + action_masks() + engine.predict_action(obs, mask).
    """
    env.reset(seed=42)

    # Warmup
    for _ in range(warmup):
        obs = env._get_obs()
        mask = env.action_masks()
        _ = engine.predict_action(obs, mask)

    # Benchmark
    latencies_ms: List[float] = []
    for _ in range(n_samples):
        t0 = time.perf_counter_ns()
        obs = env._get_obs()
        mask = env.action_masks()
        _ = engine.predict_action(obs, mask)
        t1 = time.perf_counter_ns()
        latencies_ms.append((t1 - t0) / 1e6)

    return compute_distribution_stats(latencies_ms, "ms")


def benchmark_edge_full_episode(
    engine: EdgeONNXInferenceEngine,
    n_episodes_mk01: int = 100,
    n_episodes_mk03: int = 50,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Measure total Edge computation time for full production shift:
    - Mk01 (10 jobs x 6 machines, 55 operations)
    - Mk03 (15 jobs x 8 machines, 150 operations)
    """
    # 1. Mk01 Benchmark
    env_mk01 = GreenFJSPEnv("data/benchmarks/Mk01_green.json", cloud_guided_mask=False)
    # Warmup
    for _ in range(3):
        obs, _ = env_mk01.reset(seed=42)
        done = False
        while not done:
            mask = env_mk01.action_masks()
            act = engine.predict_action(obs, mask)
            obs, _, term, trunc, _ = env_mk01.step(act)
            done = term or trunc

    mk01_ep_times_ms: List[float] = []
    for ep in range(n_episodes_mk01):
        obs, _ = env_mk01.reset(seed=100 + ep)
        done = False
        t_edge_total_ns = 0
        while not done:
            t0 = time.perf_counter_ns()
            o = env_mk01._get_obs()
            m = env_mk01.action_masks()
            act = engine.predict_action(o, m)
            t1 = time.perf_counter_ns()
            t_edge_total_ns += (t1 - t0)

            obs, _, term, trunc, _ = env_mk01.step(act)
            done = term or trunc
        mk01_ep_times_ms.append(t_edge_total_ns / 1e6)

    stats_mk01 = compute_distribution_stats(mk01_ep_times_ms, "ms")

    # 2. Mk03 Benchmark
    env_mk03 = GreenFJSPEnv("data/benchmarks/Mk03_green.json", cloud_guided_mask=False)
    # Warmup
    for _ in range(2):
        obs, _ = env_mk03.reset(seed=42)
        done = False
        while not done:
            mask = env_mk03.action_masks()
            act = engine.predict_action(obs, mask)
            obs, _, term, trunc, _ = env_mk03.step(act)
            done = term or trunc

    mk03_ep_times_ms: List[float] = []
    for ep in range(n_episodes_mk03):
        obs, _ = env_mk03.reset(seed=200 + ep)
        done = False
        t_edge_total_ns = 0
        while not done:
            t0 = time.perf_counter_ns()
            o = env_mk03._get_obs()
            m = env_mk03.action_masks()
            act = engine.predict_action(o, m)
            t1 = time.perf_counter_ns()
            t_edge_total_ns += (t1 - t0)

            obs, _, term, trunc, _ = env_mk03.step(act)
            done = term or trunc
        mk03_ep_times_ms.append(t_edge_total_ns / 1e6)

    stats_mk03 = compute_distribution_stats(mk03_ep_times_ms, "ms")
    return stats_mk01, stats_mk03


def benchmark_mqtt_loopback_rtt(
    n_samples: int = 1000,
    warmup: int = 50,
) -> Dict[str, Any]:
    """Measure round-trip latency of MQTT serialization, publish, dispatch, and deserialization."""
    broker = InMemoryMQTTBroker.get_instance()
    received_msgs: List[Any] = []
    topic = TOPIC_EDGE_TRIGGER_RESCHEDULE

    def on_msg(client, userdata, message):
        received_msgs.append(message)

    broker.subscribe(topic, on_msg)

    payload = RescheduleTriggerPayload(
        timestamp=time.time(),
        machine_id="M1",
        current_wear_um=42.5,
        cdi=0.145,
        reason="cdi_exceeded_threshold",
        severity="CRITICAL",
        uncompleted_ops=35,
    )

    # Warmup
    for _ in range(warmup):
        raw = payload.model_dump_json()
        broker.publish(topic, raw)
        msg = received_msgs.pop()
        _ = json.loads(msg.payload.decode("utf-8"))

    # Benchmark
    latencies_ms: List[float] = []
    for _ in range(n_samples):
        t0 = time.perf_counter_ns()
        raw = payload.model_dump_json()
        broker.publish(topic, raw)
        msg = received_msgs.pop()
        _ = json.loads(msg.payload.decode("utf-8"))
        t1 = time.perf_counter_ns()
        latencies_ms.append((t1 - t0) / 1e6)

    broker.unsubscribe(topic, on_msg)
    return compute_distribution_stats(latencies_ms, "ms")


def benchmark_cloud_nsga3_full_solve(
    n_runs: int = 5,
) -> Tuple[Dict[str, Any], float]:
    """Measure real execution time of full NSGA-III 100-gen optimization on Mk01 (and 1 run on Mk03)."""
    mk01_times_s: List[float] = []
    print(f"  [Cloud Benchmark] Running NSGA-III full solve on Mk01 ({n_runs} runs)...")
    for r in range(n_runs):
        t0 = time.perf_counter()
        _ = run_nsga3_optimization(
            benchmark_path="data/benchmarks/Mk01_green.json",
            pop_size=100,
            n_gen=100,
            seed=42 + r,
            verbose=False,
        )
        t1 = time.perf_counter()
        dur = t1 - t0
        mk01_times_s.append(dur)
        print(f"    Run {r + 1}/{n_runs}: {dur:.3f} s")

    stats_mk01 = compute_distribution_stats(mk01_times_s, "s")

    # Run single validation solve on Mk03 (150 ops)
    print("  [Cloud Benchmark] Running NSGA-III full solve on Mk03 (1 run)...")
    t0 = time.perf_counter()
    _ = run_nsga3_optimization(
        benchmark_path="data/benchmarks/Mk03_green.json",
        pop_size=100,
        n_gen=100,
        seed=42,
        verbose=False,
    )
    t1 = time.perf_counter()
    mk03_time_s = t1 - t0
    print(f"    Mk03 solve time: {mk03_time_s:.3f} s")

    return stats_mk01, mk03_time_s


def benchmark_cloud_nsga3_resync_solve(
    n_runs: int = 5,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Measure real execution time of NSGA-III solver and fast heuristic for rescheduling remaining ops.
    Constructs the problem state at Op 20 (where 35 ops remain out of 55 on Mk01).
    """
    with open("data/benchmarks/cloud_baseline_schedule.json", "r", encoding="utf-8") as f:
        sched = json.load(f)["operations_schedule"]

    env = GreenFJSPEnv("data/benchmarks/Mk01_green.json")
    env.reset(seed=42)
    for i in range(20):
        op = sched[i]
        j = int(op["job_index"])
        m_id = op["machine_id"]
        m_idx = env.machine_ids.index(m_id)
        s = int(op["speed_level"])
        act = env.encode_action(j, m_idx, s, 0)
        env.step(act)

    # 1. Benchmark fast heuristic active reschedule
    print("  [Cloud Benchmark] Running fast active reschedule benchmark (1,000 calls)...")
    fast_times_ms: List[float] = []
    # Warmup
    for _ in range(50):
        _ = fast_cloud_reschedule_active(env, sched)
    for _ in range(1000):
        t0 = time.perf_counter_ns()
        _ = fast_cloud_reschedule_active(env, sched)
        t1 = time.perf_counter_ns()
        fast_times_ms.append((t1 - t0) / 1e6)
    stats_fast = compute_distribution_stats(fast_times_ms, "ms")

    # 2. Benchmark NSGA-III re-optimization on remaining 35 ops
    rem_jobs = []
    for j in range(env.num_jobs):
        cur_o = env.current_op_idx[j]
        job_ops = env.jobs_data[j]["operations"][cur_o:]
        if job_ops:
            rem_jobs.append({
                "job_id": j,
                "weight": env.jobs_data[j]["weight"],
                "due_date_min": max(1.0, env.jobs_data[j]["due_date_min"] - env.job_ready_time[j]),
                "operations": job_ops,
            })

    rem_bench = {
        "parameters": {"num_jobs": len(rem_jobs), "num_machines": 6},
        "jobs": rem_jobs,
        "machines": {
            m_id: {"p_cut_base_kw": 4.0, "p_idle_kw": 1.2, "p_stdby_kw": 0.4}
            for m_id in env.machine_ids
        },
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(rem_bench, tf)
        tf_path = tf.name

    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.optimize import minimize
    from pymoo.util.ref_dirs import get_reference_directions

    prob = GreenFJSPProblem(benchmark_path=tf_path)
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

    print(f"  [Cloud Benchmark] Running NSGA-III re-solve on remaining 35 ops ({n_runs} runs)...")
    resync_times_s: List[float] = []
    for r in range(n_runs):
        algo = NSGA3(
            ref_dirs=ref_dirs,
            pop_size=100,
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(prob=1.0 / prob.n_var, eta=20),
        )
        t0 = time.perf_counter()
        _ = minimize(prob, algo, ("n_gen", 100), seed=42 + r, verbose=False)
        t1 = time.perf_counter()
        dur = t1 - t0
        resync_times_s.append(dur)
        print(f"    Re-solve run {r + 1}/{n_runs}: {dur:.3f} s")

    stats_resync = compute_distribution_stats(resync_times_s, "s")
    try:
        os.remove(tf_path)
    except OSError:
        pass

    return stats_resync, stats_fast


def run_full_benchmark(
    n_micro: int = 1000,
    warmup: int = 50,
    n_episodes_mk01: int = 100,
    n_episodes_mk03: int = 50,
    n_cloud_runs: int = 5,
    output_path: Union[str, Path] = "data/benchmarks/measured_latencies.json",
    mirror_path: Union[str, Path] = "results/measured_latencies.json",
) -> Dict[str, Any]:
    """Execute all benchmark components and export empirical timing distributions."""
    print("=" * 80)
    print("EMPIRICAL TIMING BENCHMARK: MEASURING REAL HARDWARE EXECUTION LATENCIES")
    print("=" * 80)

    engine = EdgeONNXInferenceEngine("edge_layer/models/ppo_edge_policy.onnx")
    env = GreenFJSPEnv("data/benchmarks/Mk01_green.json")

    # 1. Edge ONNX forward only
    print(f"\n[1/6] Benchmarking Edge ONNX pure forward pass ({n_micro} iterations)...")
    edge_fwd_stats = benchmark_edge_onnx_forward_only(engine, n_samples=n_micro, warmup=warmup)
    print(f"      Mean: {edge_fwd_stats['mean']:.4f} ms | Median: {edge_fwd_stats['median']:.4f} ms | P95: {edge_fwd_stats['p95']:.4f} ms")

    # 2. Edge step inference (Obs + Mask + ONNX)
    print(f"\n[2/6] Benchmarking Edge complete decision step ({n_micro} iterations)...")
    edge_step_stats = benchmark_edge_step_inference(env, engine, n_samples=n_micro, warmup=warmup)
    print(f"      Mean: {edge_step_stats['mean']:.4f} ms | Median: {edge_step_stats['median']:.4f} ms | P95: {edge_step_stats['p95']:.4f} ms")

    # 3. Edge full episode computation time
    print(f"\n[3/6] Benchmarking Edge full episode computation time (Mk01: {n_episodes_mk01} eps, Mk03: {n_episodes_mk03} eps)...")
    mk01_ep_stats, mk03_ep_stats = benchmark_edge_full_episode(
        engine, n_episodes_mk01=n_episodes_mk01, n_episodes_mk03=n_episodes_mk03
    )
    print(f"      Mk01 Full Shift (55 ops) : Mean: {mk01_ep_stats['mean']:.3f} ms | P95: {mk01_ep_stats['p95']:.3f} ms")
    print(f"      Mk03 Full Shift (150 ops): Mean: {mk03_ep_stats['mean']:.3f} ms | P95: {mk03_ep_stats['p95']:.3f} ms")

    # 4. MQTT round-trip time
    print(f"\n[4/6] Benchmarking Industrial MQTT loopback RTT ({n_micro} iterations)...")
    mqtt_stats = benchmark_mqtt_loopback_rtt(n_samples=n_micro, warmup=warmup)
    print(f"      Mean: {mqtt_stats['mean']:.4f} ms | Median: {mqtt_stats['median']:.4f} ms | P95: {mqtt_stats['p95']:.4f} ms")

    # 5. Cloud NSGA-III full solve
    print(f"\n[5/6] Benchmarking Cloud NSGA-III full optimization ({n_cloud_runs} runs on Mk01)...")
    cloud_full_stats, mk03_full_time = benchmark_cloud_nsga3_full_solve(n_runs=n_cloud_runs)
    print(f"      Mk01 Full Solve: Mean: {cloud_full_stats['mean']:.3f} s ± {cloud_full_stats['std']:.3f} s")
    print(f"      Mk03 Full Solve: {mk03_full_time:.3f} s")

    # 6. Cloud NSGA-III resync solve
    print(f"\n[6/6] Benchmarking Cloud NSGA-III rescheduling solve on remaining operations ({n_cloud_runs} runs)...")
    cloud_resync_stats, fast_resched_stats = benchmark_cloud_nsga3_resync_solve(n_runs=n_cloud_runs)
    print(f"      NSGA-III Resync Solve (35 ops): Mean: {cloud_resync_stats['mean']:.3f} s ± {cloud_resync_stats['std']:.3f} s")
    print(f"      Fast Active Reschedule       : Mean: {fast_resched_stats['mean']:.4f} ms")

    output_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "platform": platform.platform(),
            "processor": platform.processor() or "x86_64",
            "python_version": platform.python_version(),
            "onnx_model_path": "edge_layer/models/ppo_edge_policy.onnx",
            "protocol_reference": "Evaluation Protocol v5.5A",
        },
        "edge_step_inference_ms": edge_step_stats,
        "edge_onnx_forward_only_ms": edge_fwd_stats,
        "edge_full_episode_ms": mk01_ep_stats,
        "edge_full_episode_mk01_ms": mk01_ep_stats,
        "edge_full_episode_mk03_ms": mk03_ep_stats,
        "mqtt_loopback_rtt_ms": mqtt_stats,
        "cloud_nsga3_full_solve_s": cloud_full_stats,
        "cloud_nsga3_full_solve_mk03_s": mk03_full_time,
        "cloud_nsga3_resync_solve_s": cloud_resync_stats,
        "cloud_fast_reschedule_active_ms": fast_resched_stats,
    }

    # Save output JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n[Export] Saved measured latencies to {output_path}")

    # Mirror to results/
    mirror_path = Path(mirror_path)
    mirror_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mirror_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"[Export] Mirrored measured latencies to {mirror_path}")

    print("=" * 80)
    print("BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 80)
    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark empirical latencies on real machine.")
    parser.add_argument("--n_micro", type=int, default=1000, help="Microbenchmark sample iterations (default: 1000)")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations (default: 50)")
    parser.add_argument("--n_episodes_mk01", type=int, default=100, help="Mk01 full episodes (default: 100)")
    parser.add_argument("--n_episodes_mk03", type=int, default=50, help="Mk03 full episodes (default: 50)")
    parser.add_argument("--n_cloud_runs", type=int, default=5, help="NSGA-III repetition runs (default: 5)")
    parser.add_argument("--output", type=str, default="data/benchmarks/measured_latencies.json")
    parser.add_argument("--mirror", type=str, default="results/measured_latencies.json")

    args = parser.parse_args()
    run_full_benchmark(
        n_micro=args.n_micro,
        warmup=args.warmup,
        n_episodes_mk01=args.n_episodes_mk01,
        n_episodes_mk03=args.n_episodes_mk03,
        n_cloud_runs=args.n_cloud_runs,
        output_path=args.output,
        mirror_path=args.mirror,
    )
