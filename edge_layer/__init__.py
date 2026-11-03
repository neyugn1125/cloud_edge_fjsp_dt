"""Edge Layer package: Real-time monitoring, lightweight surrogate, and local reactive dispatching."""

from edge_layer.breadboard_daq import (
    BreadboardDAQInterface,
    RealTimeToolWearEstimator,
)
from edge_layer.edge_twin_node import EdgeDigitalTwinNode
from edge_layer.grbl_controller import GRBLHardwareController
from edge_layer.mqtt_client import EdgeMQTTClient
from edge_layer.onnx_exporter import EdgeONNXInferenceEngine
from edge_layer.ppo_agent import (
    TrainingTelemetryCallback,
    evaluate_agent,
    make_env,
    plot_ppo_training_curves,
    train_multi_seed_ppo,
    train_single_seed_ppo,
)

__all__ = [
    "make_env",
    "evaluate_agent",
    "TrainingTelemetryCallback",
    "train_single_seed_ppo",
    "train_multi_seed_ppo",
    "plot_ppo_training_curves",
    "BreadboardDAQInterface",
    "RealTimeToolWearEstimator",
    "EdgeMQTTClient",
    "GRBLHardwareController",
    "EdgeDigitalTwinNode",
    "EdgeONNXInferenceEngine",
]
