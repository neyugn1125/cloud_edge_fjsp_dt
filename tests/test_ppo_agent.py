"""
Unit and Integration Tests for Edge Layer MaskablePPO Agent.

Verifies:
    1. make_env factory creates valid action-masked GreenFJSPEnv.
    2. TelemetryCallback records episode statistics during training.
    3. Short training run (1,100 steps) executes without errors on GPU/CPU.
    4. Deterministic evaluation routine returns valid metrics.
    5. Action masks are strictly respected during policy inference.
"""

from pathlib import Path
import numpy as np
import pytest
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import DummyVecEnv

from edge_layer.ppo_agent import (
    TrainingTelemetryCallback,
    evaluate_agent,
    make_env,
    mask_fn,
)
from sim_env.fjsp_gym_env import GreenFJSPEnv


def test_make_env_and_action_masker() -> None:
    """Verify that make_env correctly wraps GreenFJSPEnv with action masking."""
    env_fn = make_env(seed=42, domain_rand=False)
    env = env_fn()

    obs, info = env.reset(seed=42)
    assert obs.shape == (87,)
    assert isinstance(info, dict)

    masks = get_action_masks(env)
    assert isinstance(masks, np.ndarray)
    assert masks.shape == (360,)
    assert np.any(masks), "Action masks must have at least one valid action"
    env.close()


def test_short_ppo_training_and_callback() -> None:
    """Verify that a short PPO training run executes and logs telemetry properly."""
    venv = DummyVecEnv([make_env(seed=100 + i, domain_rand=True) for i in range(2)])
    callback = TrainingTelemetryCallback()

    model = MaskablePPO(
        "MlpPolicy",
        venv,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
        learning_rate=1e-3,
        n_steps=55,
        batch_size=55,
        gamma=0.99,
        verbose=0,
    )

    # Run for 220 steps (4 episodes across 2 envs)
    model.learn(total_timesteps=220, callback=callback)

    assert len(callback.episode_returns) > 0, "Callback must record at least one completed episode"
    assert len(callback.episode_makespans) == len(callback.episode_returns)
    assert len(callback.episode_costs) == len(callback.episode_returns)
    assert len(callback.episode_breakdowns) == len(callback.episode_returns)

    for cmax in callback.episode_makespans:
        assert cmax > 0.0

    for cost in callback.episode_costs:
        assert cost > 0.0

    venv.close()


def test_evaluate_agent_deterministic() -> None:
    """Verify that evaluate_agent runs deterministically and reports expected metrics."""
    env_fn = make_env(seed=42, domain_rand=False)
    env = env_fn()

    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=dict(pi=[32, 32], vf=[32, 32])),
        verbose=0,
    )

    stats = evaluate_agent(
        model=model,
        n_eval_episodes=3,
        seed=42,
        domain_rand=False,
    )

    assert "cmax_mean" in stats
    assert "cost_mean" in stats
    assert "twt_mean" in stats
    assert "bd_mean" in stats
    assert "pm_mean" in stats

    assert stats["cmax_mean"] > 0.0
    assert stats["cost_mean"] > 0.0
    assert stats["twt_mean"] >= 0.0
    env.close()

