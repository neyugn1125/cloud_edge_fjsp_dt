"""
Unit tests for GreenFJSPEnv Gymnasium discrete-event simulation environment.
"""

import numpy as np
import pytest

from sim_env.fjsp_gym_env import GreenFJSPEnv


@pytest.fixture
def env() -> GreenFJSPEnv:
    """Fixture initializing GreenFJSPEnv with default benchmark and config."""
    return GreenFJSPEnv(
        benchmark_path="data/benchmarks/Mk01_green.json",
        config_path="config/system_params.yaml",
        weights=(0.4, 0.4, 0.2),
        seed=42,
    )


def test_env_spaces_and_reset(env: GreenFJSPEnv):
    """Verify observation space, action space, and initial reset output."""
    assert env.action_space.n == 360  # 10 jobs * 6 machines * 3 speeds * 2 PMs
    assert env.observation_space.shape == (87,)
    assert env.observation_space.dtype == np.float32

    obs, info = env.reset(seed=123)
    assert obs.shape == (87,)
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)

    assert "cmax" in info
    assert "total_energy_kwh" in info
    assert "total_cost" in info
    assert "total_weighted_tardiness" in info
    assert "completed_operations" in info
    assert info["completed_operations"] == 0
    assert info["cmax"] == 0.0


def test_action_encoding_decoding_roundtrip(env: GreenFJSPEnv):
    """Verify bijective consistency of action encoding and decoding across all 360 actions."""
    for action in range(env.total_actions):
        job_idx, machine_idx, speed_level, pm_flag = env.decode_action(action)

        assert 0 <= job_idx < env.num_jobs
        assert 0 <= machine_idx < env.num_machines
        assert speed_level in (1, 2, 3)
        assert pm_flag in (0, 1)

        encoded = env.encode_action(job_idx, machine_idx, speed_level, pm_flag)
        assert encoded == action


def test_action_masks_logic(env: GreenFJSPEnv):
    """Test action masking rules (eligibility, completed jobs, and PM wear threshold)."""
    env.reset(seed=42)
    masks = env.action_masks()

    assert masks.shape == (360,)
    assert np.any(masks)  # Must have valid actions available

    # M3 has initial wear = 45.0 um (< 78% w_max = 78.0 um)
    # Therefore, actions on M3 (machine_idx=2) with pm_flag == 1 must be FALSE
    for act in range(360):
        _, m_idx, _, pm_flag = env.decode_action(act)
        if m_idx == 2 and pm_flag == 1:
            assert bool(masks[act]) is False

    # M1 has initial wear = 55.0 um (< 78% w_max = 78.0 um)
    # In Mk01, J1 O1 is eligible on M1 (machine_idx=0)
    act_j0_m0_s1_no_pm = env.encode_action(job_idx=0, machine_idx=0, speed_level=1, pm_flag=0)
    act_j0_m0_s1_pm = env.encode_action(job_idx=0, machine_idx=0, speed_level=1, pm_flag=1)
    assert bool(masks[act_j0_m0_s1_no_pm]) is True
    # At wear = 55 um (< 78 um), PM is NOT allowed (prevents PM spamming)
    assert bool(masks[act_j0_m0_s1_pm]) is False

    # When M3 wear is increased to 80 um (>= 78% w_max), PM on M3 becomes valid
    env.machines["M3"].current_wear_um = 80.0
    masks_after_wear = env.action_masks()
    act_j0_m2_s1_pm = env.encode_action(job_idx=0, machine_idx=2, speed_level=1, pm_flag=1)
    assert bool(masks_after_wear[act_j0_m2_s1_pm]) is True


def test_rollout_until_termination(env: GreenFJSPEnv):
    """
    Execute a full episode using a random masked policy and verify that
    all 55 operations complete, terminal state is reached, and metrics are physically valid.
    """
    obs, info = env.reset(seed=99)
    rng = np.random.default_rng(99)

    step_count = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        mask = env.action_masks()
        valid_actions = np.where(mask)[0]
        assert len(valid_actions) > 0, f"No valid actions at step {step_count}"

        action = rng.choice(valid_actions)
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (87,)
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
        assert isinstance(reward, float)
        step_count += 1

    assert terminated is True
    assert step_count == 55  # Exactly 55 operations in Mk01
    assert env.completed_operations == 55
    assert info["cmax"] > 0.0
    assert info["total_energy_kwh"] > 0.0
    assert info["total_cost"] > 0.0
    assert info["progress_ratio"] == 1.0


def test_cloud_weights_update(env: GreenFJSPEnv):
    """Verify dynamic objective weight updating from Cloud synchronization."""
    env.set_objective_weights((0.7, 0.2, 0.1))
    assert pytest.approx(env.w_cmax, rel=1e-4) == 0.7
    assert pytest.approx(env.w_energy, rel=1e-4) == 0.2
    assert pytest.approx(env.w_tardiness, rel=1e-4) == 0.1

    # Update via reset options
    obs, _ = env.reset(options={"weights": (0.333, 0.333, 0.334)})
    assert pytest.approx(env.w_cmax, rel=1e-3) == 0.333
    # Check observation vector reflects the new weights in the last 3 elements
    assert pytest.approx(obs[-3], rel=1e-3) == 0.333
    assert pytest.approx(obs[-2], rel=1e-3) == 0.333
    assert pytest.approx(obs[-1], rel=1e-3) == 0.334


def test_action_masks_always_non_empty_until_completion(env: GreenFJSPEnv):
    """
    Stress-test the action_masks() invariant across 10 random seeds:
    At every single step t in [0, 54], action_masks() MUST have at least 1 True action.
    """
    for seed in range(10):
        obs, info = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        step = 0
        terminated = False

        while not terminated:
            mask = env.action_masks()
            num_valid = int(np.sum(mask))
            assert num_valid >= 1, f"Deadlock! Step {step} has 0 valid actions for seed {seed}"

            valid_actions = np.where(mask)[0]
            action = int(rng.choice(valid_actions))
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1

        assert step == 55
        assert env.completed_operations == 55
        # Once completely terminated, all actions must be False
        assert int(np.sum(env.action_masks())) == 0


def test_domain_randomization_reset(env: GreenFJSPEnv):
    """Verify that domain randomization perturbs initial wear and eta within specified bounds."""
    # Deterministic reset (dr = False)
    env.domain_randomization = False
    env.reset(seed=42)
    m1 = env.machines["M1"]
    assert m1.current_wear_um == m1.base_initial_wear_um
    assert m1.eta == m1.base_eta

    # Domain randomization reset (dr = True)
    w_samples = []
    eta_samples = []
    for s in range(20):
        env.reset(seed=s, options={"domain_randomization": True})
        w_samples.append(m1.current_wear_um)
        eta_samples.append(m1.eta)

    # All samples must be within +-10% of base initial wear and +-5% of base eta
    for w in w_samples:
        assert 0.90 * m1.base_initial_wear_um <= w <= 1.10 * m1.base_initial_wear_um
    for eta in eta_samples:
        assert 0.95 * m1.base_eta <= eta <= 1.05 * m1.base_eta

    # Verify variability
    assert np.std(w_samples) > 0.1
    assert np.std(eta_samples) > 0.005

