"""
Unit Tests for ONNX Policy Export and Edge Inference Engine (Step 2.3).
"""

from pathlib import Path
import numpy as np
import onnx
import pytest
import torch
from sb3_contrib import MaskablePPO

from edge_layer.onnx_exporter import (
    EdgeONNXInferenceEngine,
    StandaloneActorNetwork,
    benchmark_inference_latency,
    export_maskable_ppo_to_onnx,
    verify_onnx_against_sb3,
)
from sim_env.fjsp_gym_env import GreenFJSPEnv


@pytest.fixture(scope="module")
def ppo_model_path() -> Path:
    path = Path("edge_layer/models/best_maskable_ppo.zip")
    assert path.exists(), f"Trained PPO model not found at {path}"
    return path


@pytest.fixture(scope="module")
def onnx_model_path() -> Path:
    path = Path("edge_layer/models/ppo_edge_policy.onnx")
    assert path.exists(), f"Exported ONNX model not found at {path}"
    return path


def test_standalone_actor_network_forward(ppo_model_path: Path):
    """Verify StandaloneActorNetwork produces correct logits shape."""
    model = MaskablePPO.load(str(ppo_model_path), device="cpu")
    actor = StandaloneActorNetwork(
        policy_net=model.policy.mlp_extractor.policy_net,
        action_net=model.policy.action_net,
    )
    actor.eval()

    batch_size = 4
    obs_dim = 87
    dummy_obs = torch.randn(batch_size, obs_dim, dtype=torch.float32)

    with torch.no_grad():
        logits = actor(dummy_obs)

    assert logits.shape == (batch_size, 360)
    assert not torch.isnan(logits).any()


def test_export_maskable_ppo_to_onnx(ppo_model_path: Path, tmp_path: Path):
    """Verify export to ONNX creates a valid ONNX model file."""
    temp_onnx = tmp_path / "temp_policy.onnx"
    exported_path, file_size_kb = export_maskable_ppo_to_onnx(
        model_path=ppo_model_path,
        output_onnx_path=temp_onnx,
        opset_version=14,
    )

    assert exported_path.exists()
    assert file_size_kb > 0.0
    assert file_size_kb < 1000.0  # Ultra-compact model < 1MB

    # Verify model validity using onnx checker
    onnx_model = onnx.load(str(exported_path))
    onnx.checker.check_model(onnx_model)


def test_edge_onnx_inference_engine_prediction(onnx_model_path: Path):
    """Verify EdgeONNXInferenceEngine selects a valid masked action."""
    engine = EdgeONNXInferenceEngine(onnx_model_path)
    env = GreenFJSPEnv()
    obs, info = env.reset(seed=42)
    mask = env.action_masks()

    action = engine.predict_action(obs, mask)

    assert isinstance(action, int)
    assert 0 <= action < 360
    assert mask[action] is True or mask[action] == 1


def test_verify_onnx_against_sb3(ppo_model_path: Path, onnx_model_path: Path):
    """Verify 100% decision match between ONNX engine and SB3 model across states."""
    engine = EdgeONNXInferenceEngine(onnx_model_path)
    sb3_model = MaskablePPO.load(str(ppo_model_path), device="cpu")

    matches, total, rate = verify_onnx_against_sb3(
        onnx_engine=engine,
        sb3_model=sb3_model,
        num_states=25,
        seed=101,
    )

    assert total == 25
    assert matches == 25
    assert rate == 100.0


def test_benchmark_inference_latency(onnx_model_path: Path):
    """Verify inference latency benchmarking returns positive metrics in sub-ms range."""
    engine = EdgeONNXInferenceEngine(onnx_model_path)
    stats = benchmark_inference_latency(engine, num_calls=50, warmup_calls=5)

    assert "mean_latency_ms" in stats
    assert "p95_latency_ms" in stats
    assert "p99_latency_ms" in stats
    assert "file_size_kb" in stats
    assert stats["mean_latency_ms"] > 0.0
    assert stats["mean_latency_ms"] < 1.0  # Must be sub-millisecond on CPU
    assert stats["file_size_kb"] < 1000.0


def test_phase2_comparison_csv_integrity():
    """Verify that phase2_comparison_results.csv exists and has complete valid rows."""
    csv_path = Path("data/benchmarks/phase2_comparison_results.csv")
    assert csv_path.exists(), f"Missing comparison results file at {csv_path}"

    with open(csv_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # Header + 6 methods = 7 lines
    assert len(lines) == 7

    methods_expected = [
        "Rule 1 (SPT + Fast Speed)",
        "Rule 2 (EET + Eco Speed)",
        "Rule 3 (EDD + Balanced Speed)",
        "Rule 4 (Random Valid Policy)",
        "NSGA-III Open-Loop (Static Cloud Schedule without Edge Twin)",
        "Cloud-Edge Collaborative DT (NSGA-III + Edge MaskablePPO ONNX)",
    ]

    for m in methods_expected:
        assert any(m in line for line in lines[1:]), f"Missing method {m} in CSV"

