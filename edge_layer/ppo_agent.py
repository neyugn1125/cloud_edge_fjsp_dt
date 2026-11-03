"""
Deep Reinforcement Learning Agent for Edge Layer Reactive Scheduling
using Action-Masked Proximal Policy Optimization (MaskablePPO).

This module implements:
    1. Training of MaskablePPO agent with DummyVecEnv (4 parallel environments).
    2. Exact alignment with normalized TOU Energy Cost and Peak Demand Surcharge ($0.30/kWh).
    3. Multi-seed training (seeds: 42, 101, 2026) with Sim-to-Real Domain Randomization
       (+-10% tool wear, +-5% wear rate eta).
    4. Reward design combining Makespan, TOU cost, and Tardiness normalization (Rule 2 baselines),
       heavy penalties for catastrophic tool breakdowns, and shaping bonuses for adaptive speed/PM.
    5. Evaluation and export of best model to `edge_layer/models/best_maskable_ppo.zip`.
    6. High-resolution Q1-standard publication curve generation (`tests/phase2_ppo_training_curve.png`).

Strict Unit Conventions:
    - Time: Minutes (min)
    - Power: Kilowatts (kW)
    - Energy: Kilowatt-hours (kWh)
    - Tool Wear VB: Micrometers (um)
    - Cost: US Dollars ($)
"""

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Ensure workspace root is in path when executed directly
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from sim_env.fjsp_gym_env import GreenFJSPEnv


def mask_fn(env: gym.Env) -> np.ndarray:
    """Action masking function querying valid actions from GreenFJSPEnv."""
    return env.action_masks()


def make_env(
    seed: int,
    domain_rand: bool = True,
    cloud_guided_mask: bool = True,
    benchmark_path: Union[str, Path] = "data/benchmarks/Mk01_green.json",
    config_path: Union[str, Path] = "config/system_params.yaml",
) -> Callable[[], gym.Env]:
    """Factory creating an action-masked GreenFJSPEnv instance with specified seed."""
    def _init() -> gym.Env:
        env = GreenFJSPEnv(
            benchmark_path=benchmark_path,
            config_path=config_path,
            domain_randomization=domain_rand,
            cloud_guided_mask=cloud_guided_mask,
            seed=seed,
        )
        env = ActionMasker(env, mask_fn)
        env.reset(seed=seed)
        return env

    return _init


class TrainingTelemetryCallback(BaseCallback):
    """
    Callback recording episode metrics across parallel environments during PPO training.
    Logs: Return, Makespan (min), TOU Cost ($), Tardiness (min), Breakdowns, and PMs.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.episode_returns: List[float] = []
        self.episode_makespans: List[float] = []
        self.episode_costs: List[float] = []
        self.episode_tardiness: List[float] = []
        self.episode_breakdowns: List[int] = []
        self.episode_pms: List[int] = []

        self._curr_returns: Optional[np.ndarray] = None

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs
        self._curr_returns = np.zeros(n_envs, dtype=np.float64)

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        self._curr_returns += rewards

        for i, done in enumerate(dones):
            if done:
                info = infos[i]
                self.episode_returns.append(float(self._curr_returns[i]))
                self._curr_returns[i] = 0.0

                self.episode_makespans.append(float(info.get("cmax", 0.0)))
                self.episode_costs.append(float(info.get("total_cost", 0.0)))
                self.episode_tardiness.append(float(info.get("total_weighted_tardiness", 0.0)))
                self.episode_breakdowns.append(int(info.get("total_breakdowns", 0)))
                self.episode_pms.append(int(info.get("total_pm_replacements", 0)))

        return True


def evaluate_agent(
    model: MaskablePPO,
    n_eval_episodes: int = 15,
    seed: int = 42,
    domain_rand: bool = False,
    cloud_guided_mask: bool = True,
    benchmark_path: Union[str, Path] = "data/benchmarks/Mk01_green.json",
    config_path: Union[str, Path] = "config/system_params.yaml",
) -> Dict[str, float]:
    """
    Deterministically evaluate a trained policy over N episodes.

    Returns:
        Dict containing mean and std for makespan, cost, tardiness, breakdowns, and PM.
    """
    raw_env = GreenFJSPEnv(
        benchmark_path=benchmark_path,
        config_path=config_path,
        domain_randomization=domain_rand,
        cloud_guided_mask=cloud_guided_mask,
        seed=seed,
    )
    env = ActionMasker(raw_env, mask_fn)

    cmax_list: List[float] = []
    cost_list: List[float] = []
    twt_list: List[float] = []
    bd_list: List[int] = []
    pm_list: List[int] = []
    return_list: List[float] = []

    for ep in range(n_eval_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_ret = 0.0
        while not done:
            masks = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ep_ret += r
            done = term or trunc

        cmax_list.append(float(info["cmax"]))
        cost_list.append(float(info["total_cost"]))
        twt_list.append(float(info["total_weighted_tardiness"]))
        bd_list.append(int(info["total_breakdowns"]))
        pm_list.append(int(info["total_pm_replacements"]))
        return_list.append(ep_ret)

    return {
        "cmax_mean": float(np.mean(cmax_list)),
        "cmax_std": float(np.std(cmax_list)),
        "cost_mean": float(np.mean(cost_list)),
        "cost_std": float(np.std(cost_list)),
        "twt_mean": float(np.mean(twt_list)),
        "twt_std": float(np.std(twt_list)),
        "bd_mean": float(np.mean(bd_list)),
        "pm_mean": float(np.mean(pm_list)),
        "return_mean": float(np.mean(return_list)),
    }


def train_single_seed_ppo(
    seed: int,
    total_timesteps: int = 66000,
    n_envs: int = 4,
    device: str = "auto",
    cloud_guided_mask: bool = True,
    benchmark_path: Union[str, Path] = "data/benchmarks/Mk01_green.json",
    config_path: Union[str, Path] = "config/system_params.yaml",
    verbose: int = 0,
) -> Tuple[MaskablePPO, TrainingTelemetryCallback]:
    """Train a MaskablePPO model on a single seed using parallel DummyVecEnv."""
    env_fns = [
        make_env(
            seed=seed + i * 100,
            domain_rand=True,
            cloud_guided_mask=cloud_guided_mask,
            benchmark_path=benchmark_path,
            config_path=config_path,
        )
        for i in range(n_envs)
    ]
    venv = DummyVecEnv(env_fns)

    policy_kwargs = dict(net_arch=dict(pi=[256, 128], vf=[256, 128]))

    model = MaskablePPO(
        "MlpPolicy",
        venv,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=550,
        batch_size=220,
        gamma=0.99,
        ent_coef=0.015,
        device=device,
        verbose=verbose,
        seed=seed,
    )

    callback = TrainingTelemetryCallback(verbose=0)
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    elapsed = time.time() - t0

    fps = total_timesteps / elapsed if elapsed > 0 else 0
    print(
        f"[Seed {seed}] Training completed in {elapsed:.1f} s ({fps:.1f} FPS, "
        f"{len(callback.episode_returns)} episodes recorded)."
    )

    venv.close()
    return model, callback


def train_multi_seed_ppo(
    seeds: Optional[List[int]] = None,
    total_timesteps: int = 66000,
    n_envs: int = 4,
    device: str = "auto",
    cloud_guided_mask: bool = True,
    save_dir: Union[str, Path] = "edge_layer/models",
    benchmark_path: Union[str, Path] = "data/benchmarks/Mk01_green.json",
    config_path: Union[str, Path] = "config/system_params.yaml",
) -> Tuple[MaskablePPO, Dict[str, Any]]:
    """
    Train MaskablePPO across multiple seeds, evaluate deterministically,
    and save the best model to disk.
    """
    if seeds is None:
        seeds = [42, 101, 2026]

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    best_model_path = save_path / "best_maskable_ppo.zip"

    all_seed_data: Dict[str, List[List[float]]] = {
        "returns": [],
        "makespans": [],
        "costs": [],
        "tardiness": [],
        "breakdowns": [],
        "pms": [],
    }

    best_model: Optional[MaskablePPO] = None
    best_score: float = float("inf")
    best_eval_stats: Dict[str, float] = {}
    best_seed: int = seeds[0]

    for seed in seeds:
        print(f"\n---> Starting MaskablePPO Training on Seed {seed} (timesteps={total_timesteps})...")
        model, cb = train_single_seed_ppo(
            seed=seed,
            total_timesteps=total_timesteps,
            n_envs=n_envs,
            device=device,
            cloud_guided_mask=cloud_guided_mask,
            benchmark_path=benchmark_path,
            config_path=config_path,
        )

        all_seed_data["returns"].append(cb.episode_returns)
        all_seed_data["makespans"].append(cb.episode_makespans)
        all_seed_data["costs"].append(cb.episode_costs)
        all_seed_data["tardiness"].append(cb.episode_tardiness)
        all_seed_data["breakdowns"].append([float(x) for x in cb.episode_breakdowns])
        all_seed_data["pms"].append([float(x) for x in cb.episode_pms])

        # Evaluate model on clean deterministic benchmark
        eval_stats = evaluate_agent(
            model=model,
            n_eval_episodes=15,
            seed=42,
            domain_rand=False,
            cloud_guided_mask=cloud_guided_mask,
            benchmark_path=benchmark_path,
            config_path=config_path,
        )
        print(
            f"     Eval (Seed {seed}): Makespan={eval_stats['cmax_mean']:.2f}m, "
            f"Cost=${eval_stats['cost_mean']:.3f}, TWT={eval_stats['twt_mean']:.2f}m, "
            f"BD={eval_stats['bd_mean']:.2f}, PM={eval_stats['pm_mean']:.2f}"
        )

        # Composite score: weighted normalized sum (lower is better) + heavy breakdown penalty
        score = (
            0.4 * (eval_stats["cmax_mean"] / 78.0)
            + 0.4 * (eval_stats["cost_mean"] / 2.72)
            + 0.2 * (eval_stats["twt_mean"] / 345.0)
            + (5.0 * eval_stats["bd_mean"])
        )

        if score < best_score:
            best_score = score
            best_model = model
            best_eval_stats = eval_stats
            best_seed = seed
            model.save(str(best_model_path.with_suffix("")))
            print(f"     [+] New Best Model saved to {best_model_path} (Seed {seed})")

    assert best_model is not None

    results_summary = {
        "best_seed": best_seed,
        "best_model_path": str(best_model_path),
        "best_eval_stats": best_eval_stats,
        "all_seed_data": all_seed_data,
    }

    return best_model, results_summary


def moving_average(data: np.ndarray, window_size: int = 25) -> np.ndarray:
    """Compute rolling moving average with boundary reflection."""
    if len(data) < window_size:
        return data
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode="valid")


def plot_ppo_training_curves(
    histories: Dict[str, Any],
    output_path: Union[str, Path] = "tests/phase2_ppo_training_curve.png",
    baselines_csv: Union[str, Path] = "data/benchmarks/phase1_baseline_results.csv",
    cloud_json: Union[str, Path] = "data/benchmarks/cloud_baseline_schedule.json",
    window_size: int = 25,
) -> None:
    """
    Generate publication-standard 3-panel training curves comparing:
        Panel 1: Episode Return (Mean +- Std)
        Panel 2: Makespan & TOU Cost reduction vs Rule 2 & NSGA-III
        Panel 3: Elimination of Tool Breakdowns & Opportunistic PM rate
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Read Reference Baselines
    rule2_cmax = 78.0
    rule2_cost = 2.722
    if Path(baselines_csv).exists():
        with open(baselines_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row["Rule Name"]).startswith("Rule 2"):
                    rule2_cmax = float(row["Makespan (min)"])
                    rule2_cost = float(row["TOU Energy Cost ($)"])

    nsga3_cmax = 57.80
    nsga3_cost = 2.451
    if Path(cloud_json).exists():
        with open(cloud_json, mode="r", encoding="utf-8") as f:
            cdata = json.load(f)
            nsga3_cmax = float(cdata["summary"]["makespan_min"])
            nsga3_cost = float(cdata["summary"]["tou_energy_cost_usd"])

    seed_data = histories["all_seed_data"]
    # Harmonize lengths across seeds
    min_len = min(len(s) for s in seed_data["returns"])
    episodes = np.arange(1, min_len + 1)

    # Smooth curves across each seed
    smoothed_returns = np.array([moving_average(np.array(s[:min_len]), window_size) for s in seed_data["returns"]])
    smoothed_cmax = np.array([moving_average(np.array(s[:min_len]), window_size) for s in seed_data["makespans"]])
    smoothed_costs = np.array([moving_average(np.array(s[:min_len]), window_size) for s in seed_data["costs"]])
    smoothed_bds = np.array([moving_average(np.array(s[:min_len]), window_size) for s in seed_data["breakdowns"]])
    smoothed_pms = np.array([moving_average(np.array(s[:min_len]), window_size) for s in seed_data["pms"]])

    smooth_len = smoothed_returns.shape[1]
    ep_axis = episodes[:smooth_len]

    # Calculate Mean +- Std
    ret_mean, ret_std = np.mean(smoothed_returns, axis=0), np.std(smoothed_returns, axis=0)
    cmax_mean, cmax_std = np.mean(smoothed_cmax, axis=0), np.std(smoothed_cmax, axis=0)
    cost_mean, cost_std = np.mean(smoothed_costs, axis=0), np.std(smoothed_costs, axis=0)
    bd_mean, bd_std = np.mean(smoothed_bds, axis=0), np.std(smoothed_bds, axis=0)
    pm_mean, pm_std = np.mean(smoothed_pms, axis=0), np.std(smoothed_pms, axis=0)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)

    # -------------------------------------------------------------------------
    # Subplot 1: Episode Return
    # -------------------------------------------------------------------------
    ax1.plot(ep_axis, ret_mean, color="#1f77b4", linewidth=2.2, label=r"PPO Return ($\mu$)")
    ax1.fill_between(
        ep_axis,
        ret_mean - ret_std,
        ret_mean + ret_std,
        color="#1f77b4",
        alpha=0.25,
        label=r"$\pm 1\sigma$ (3 Seeds)",
    )
    ax1.set_xlabel("Episode", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cumulative Episode Return", fontsize=11, fontweight="bold")
    ax1.set_title("Edge PPO: Episode Return Convergence", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=9)

    # -------------------------------------------------------------------------
    # Subplot 2: Makespan & TOU Cost Reduction
    # -------------------------------------------------------------------------
    color_cmax = "#6a3d9a"  # Purple
    color_cost = "#33a02c"  # Green

    ax2.plot(ep_axis, cmax_mean, color=color_cmax, linewidth=2.0, label="Makespan (min)")
    ax2.fill_between(
        ep_axis,
        cmax_mean - cmax_std,
        cmax_mean + cmax_std,
        color=color_cmax,
        alpha=0.20,
    )
    # References for Makespan
    ax2.axhline(
        rule2_cmax,
        color=color_cmax,
        linestyle="--",
        linewidth=1.4,
        label=f"Rule 2 Base ({rule2_cmax:.1f}m)",
    )
    ax2.axhline(
        nsga3_cmax,
        color=color_cmax,
        linestyle=":",
        linewidth=1.6,
        label=f"NSGA-III Target ({nsga3_cmax:.1f}m)",
    )

    ax2.set_xlabel("Episode", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Makespan (min)", color=color_cmax, fontsize=11, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_cmax)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Twin axis for TOU Cost
    ax2_twin = ax2.twinx()
    ax2_twin.plot(ep_axis, cost_mean, color=color_cost, linewidth=2.0, label="TOU Cost ($)")
    ax2_twin.fill_between(
        ep_axis,
        cost_mean - cost_std,
        cost_mean + cost_std,
        color=color_cost,
        alpha=0.20,
    )
    ax2_twin.axhline(
        rule2_cost,
        color=color_cost,
        linestyle="--",
        linewidth=1.4,
        label=f"Rule 2 Cost (${rule2_cost:.2f})",
    )
    ax2_twin.axhline(
        nsga3_cost,
        color=color_cost,
        linestyle=":",
        linewidth=1.6,
        label=f"NSGA-III Cost (${nsga3_cost:.2f})",
    )

    ax2_twin.set_ylabel("TOU Energy Cost ($)", color=color_cost, fontsize=11, fontweight="bold")
    ax2_twin.tick_params(axis="y", labelcolor=color_cost)
    ax2.set_title("Makespan & Energy Cost Optimization", fontsize=12, fontweight="bold")

    # Combine legends
    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=8)

    # -------------------------------------------------------------------------
    # Subplot 3: Tool Breakdown Elimination & PM Policy
    # -------------------------------------------------------------------------
    color_bd = "#e31a1c"  # Red
    color_pm = "#1f78b4"  # Blue

    ax3.plot(ep_axis, bd_mean, color=color_bd, linewidth=2.0, label="Breakdowns / Ep")
    ax3.fill_between(
        ep_axis,
        np.maximum(0, bd_mean - bd_std),
        bd_mean + bd_std,
        color=color_bd,
        alpha=0.20,
    )
    ax3.set_xlabel("Episode", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Emergency Breakdowns / Episode", color=color_bd, fontsize=11, fontweight="bold")
    ax3.tick_params(axis="y", labelcolor=color_bd)
    ax3.grid(True, linestyle="--", alpha=0.5)

    ax3_twin = ax3.twinx()
    ax3_twin.plot(ep_axis, pm_mean, color=color_pm, linewidth=2.0, linestyle="-.", label="Opportunistic PM / Ep")
    ax3_twin.fill_between(
        ep_axis,
        np.maximum(0, pm_mean - pm_std),
        pm_mean + pm_std,
        color=color_pm,
        alpha=0.20,
    )
    ax3_twin.set_ylabel("Proactive PM Replacements / Episode", color=color_pm, fontsize=11, fontweight="bold")
    ax3_twin.tick_params(axis="y", labelcolor=color_pm)
    ax3.set_title("Reliability: Tool Breakdown Elimination", fontsize=12, fontweight="bold")

    lines_3, labels_3 = ax3.get_legend_handles_labels()
    lines_4, labels_4 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines_3 + lines_4, labels_3 + labels_4, loc="center right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Training curves successfully saved to {out_file}")


def main() -> None:
    """Main execution pipeline for Phase 2 Step 2.2."""
    print("=" * 80)
    print("PHASE 2 - STEP 2.2: EDGE LAYER ACTION-MASKED PPO AGENT TRAINING")
    print("=" * 80)

    benchmark_path = "data/benchmarks/Mk01_green.json"
    config_path = "config/system_params.yaml"
    baselines_csv = "data/benchmarks/phase1_baseline_results.csv"
    cloud_json = "data/benchmarks/cloud_baseline_schedule.json"
    curve_output = "tests/phase2_ppo_training_curve.png"

    # Train across 3 random seeds (66,000 steps per seed ~ 1,200 episodes)
    seeds = [42, 101, 2026]
    best_model, summary = train_multi_seed_ppo(
        seeds=seeds,
        total_timesteps=66000,
        n_envs=4,
        device="auto",
        save_dir="edge_layer/models",
        benchmark_path=benchmark_path,
        config_path=config_path,
    )

    # Plot convergence curves
    plot_ppo_training_curves(
        histories=summary,
        output_path=curve_output,
        baselines_csv=baselines_csv,
        cloud_json=cloud_json,
        window_size=25,
    )

    # Read Cloud baseline and heuristic rules for final comparison table
    b_stats = summary["best_eval_stats"]

    nsga3_cmax = 57.80
    nsga3_cost = 2.451
    nsga3_twt = 75.15
    if Path(cloud_json).exists():
        with open(cloud_json, "r", encoding="utf-8") as f:
            cdata = json.load(f)
            nsga3_cmax = float(cdata["summary"]["makespan_min"])
            nsga3_cost = float(cdata["summary"]["tou_energy_cost_usd"])
            nsga3_twt = float(cdata["summary"]["total_weighted_tardiness_min"])

    baseline_rows: List[Dict[str, Any]] = []
    if Path(baselines_csv).exists():
        with open(baselines_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                baseline_rows.append(row)

    print("\n" + "=" * 115)
    print("CROSS-LAYER PERFORMANCE BENCHMARK: EDGE PPO VS. CLOUD NSGA-III VS. HEURISTICS")
    print("=" * 115)
    header = (
        f"{'Strategy / Layer':<36} | {'C_max (min)':<14} | {'Cost ($)':<12} | "
        f"{'TWT (min)':<14} | {'PM/BD':<7} | {'Operational Profile'}"
    )
    print(header)
    print("-" * 115)

    # Edge PPO Agent
    edge_cmax_str = f"{b_stats['cmax_mean']:.2f} ± {b_stats['cmax_std']:.2f}"
    edge_cost_str = f"{b_stats['cost_mean']:.3f} ± {b_stats['cost_std']:.3f}"
    edge_twt_str = f"{b_stats['twt_mean']:.2f} ± {b_stats['twt_std']:.2f}"
    edge_pm_bd = f"{b_stats['pm_mean']:.1f}/{b_stats['bd_mean']:.1f}"
    print(
        f"{'Edge MaskablePPO (Trained Policy)':<36} | {edge_cmax_str:<14} | "
        f"{edge_cost_str:<12} | {edge_twt_str:<14} | {edge_pm_bd:<7} | ★ Real-time Edge Dispatcher"
    )

    # Cloud NSGA-III
    print(
        f"{'Cloud NSGA-III (Best Compromise)':<36} | {nsga3_cmax:<14.2f} | "
        f"{nsga3_cost:<12.3f} | {nsga3_twt:<14.2f} | 3.0/0.0 | Global Offline Blueprint"
    )

    print("-" * 115)
    for b_row in baseline_rows:
        b_name = str(b_row["Rule Name"])
        b_cmax = float(b_row["Makespan (min)"])
        b_cost = float(b_row["TOU Energy Cost ($)"])
        b_twt = float(b_row["Total Weighted Tardiness (min)"])
        pm_bd = f"{b_row['PM Replacements']}/{b_row['Tool Breakdowns']}"
        status = "Static Heuristic Baseline"
        print(
            f"{b_name:<36} | {b_cmax:<14.2f} | {b_cost:<12.3f} | "
            f"{b_twt:<14.2f} | {pm_bd:<7} | {status}"
        )
    print("=" * 115)


if __name__ == "__main__":
    main()
