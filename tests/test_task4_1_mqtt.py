"""
Test Suite for Task 4.1: Industrial MQTT Cloud-Edge Communication Layer.
Verifies:
1. Bidirectional industrial MQTT communication between CloudServer and EdgeNode.
2. High-throughput telemetry streaming (200 packets at QoS 0) with 100% delivery and RTT < 5.0 ms.
3. Global schedule synchronization (20 packets at QoS 1) validated with CloudSyncPayload.
4. Urgent reschedule triggering when CDI exceeds threshold (QoS 1) validated with RescheduleTriggerPayload.
5. Zero-network failure resilience via automatic InMemoryMQTTBroker fallback.
"""

import json
import threading
import time
from typing import List

import numpy as np
import pytest

from cloud_layer.cloud_mqtt_server import CloudMQTTServer
from edge_layer.mqtt_client import EdgeMQTTClient
from utils.mqtt_broker import InMemoryMQTTBroker
from utils.mqtt_schemas import (
    TOPIC_CLOUD_SCHEDULE_SYNC,
    TOPIC_EDGE_TELEMETRY,
    TOPIC_EDGE_TRIGGER_RESCHEDULE,
    CloudSyncPayload,
    MachineState,
    MachineTelemetryPayload,
    RescheduleTriggerPayload,
)


def test_mqtt_telemetry_200_packets_roundtrip_latency():
    """
    Transmits 200 real-time telemetry packets from Edge to Cloud (QoS 0) with echo response.
    Asserts:
    - 100% packet delivery (200 / 200 received without any drop).
    - 100% valid Pydantic schema validation.
    - Mean Round-Trip Latency (RTT) < 5.0 ms.
    Prints Mean, P95, and P99 RTT statistics to Terminal.
    """
    event = threading.Event()
    rtt_measurements: List[float] = []
    t_sent_container = [0.0]

    # Cloud handler: receives telemetry and echoes back a fast sync/ack
    def on_cloud_telemetry(tel: MachineTelemetryPayload):
        assert tel.machine_id == "M1"
        assert tel.power_kw >= 0.0
        # Fast acknowledgment back to edge
        ack_payload = {"echo_id": int(tel.timestamp), "t_sent": t_sent_container[0]}
        cloud_server.publish_schedule_sync(
            CloudSyncPayload(
                weights={"w_cmax": 0.4, "w_energy": 0.4, "w_tardiness": 0.2},
                baseline_cmax=78.5,
                baseline_energy_kwh=32.4,
                p_limit_kw=22.0,
                timestamp=tel.timestamp,
                iteration=int(tel.timestamp),
            ),
            qos=0,  # Fast echo for latency measurement
        )

    # Edge handler: receives echo and computes RTT
    def on_edge_sync(sync: CloudSyncPayload):
        t_recv = time.perf_counter()
        rtt_ms = (t_recv - t_sent_container[0]) * 1000.0
        rtt_measurements.append(rtt_ms)
        event.set()

    cloud_server = CloudMQTTServer(
        client_id="cloud_bench_server",
        on_telemetry_callback=on_cloud_telemetry,
    )
    edge_client = EdgeMQTTClient(
        machine_id="M1",
        client_id="edge_bench_node",
        on_schedule_sync_callback=on_edge_sync,
    )

    try:
        assert cloud_server.start(timeout=3.0) is True
        assert edge_client.start(timeout=3.0) is True
        time.sleep(0.15)  # Allow subscriptions to establish

        broker_type = "InMemoryMQTTBroker" if cloud_server.using_mock_broker else "Local Mosquitto (1883)"

        n_packets = 200
        for i in range(n_packets):
            event.clear()
            t_sent_container[0] = time.perf_counter()

            telemetry = MachineTelemetryPayload(
                machine_id="M1",
                power_kw=4.50 + (i * 0.005),
                vibration_rms=0.85 + (i * 0.001),
                estimated_wear_um=30.0 + (i * 0.1),
                state=MachineState.PROCESSING,
                timestamp=float(i),
                cdi=0.08,
                v_bus=24.02,
                i_amp=2.40,
                p_raw_w=58.2,
                p_filt_w=57.6,
            )

            success = edge_client.publish_telemetry(telemetry, qos=0)
            assert success is True, f"Publishing failed at packet {i}"

            received = event.wait(timeout=2.0)
            assert received is True, f"Packet {i} timed out waiting for RTT response"

        # Statistical Calculations
        rtt_arr = np.array(rtt_measurements)
        mean_rtt = float(np.mean(rtt_arr))
        p95_rtt = float(np.percentile(rtt_arr, 95))
        p99_rtt = float(np.percentile(rtt_arr, 99))
        min_rtt = float(np.min(rtt_arr))
        max_rtt = float(np.max(rtt_arr))
        delivery_rate = (len(rtt_measurements) / n_packets) * 100.0

        # Formatted Terminal Output
        print("\n" + "=" * 76)
        print("TASK 4.1: MQTT INDUSTRIAL CLOUD-EDGE LATENCY BENCHMARK (200 PACKETS)")
        print("=" * 76)
        print(f"Active MQTT Broker  : {broker_type}")
        print(f"Total Sent Packets  : {n_packets} packets")
        print(f"Delivered Packets   : {len(rtt_measurements)} packets ({delivery_rate:.1f}%)")
        print(f"Packet Loss Rate    : {100.0 - delivery_rate:.2f}%")
        print("-" * 76)
        print(f"{'Metric':<25} | {'Latency (ms)':<20} | {'Status'}")
        print("-" * 76)
        print(f"{'Mean RTT':<25} | {mean_rtt:<20.3f} | {'PASSED (< 5.0 ms)' if mean_rtt < 5.0 else 'FAILED'}")
        print(f"{'P95 RTT (95th percentile)':<25} | {p95_rtt:<20.3f} | {'OPTIMAL' if p95_rtt < 5.0 else 'WARN'}")
        print(f"{'P99 RTT (99th percentile)':<25} | {p99_rtt:<20.3f} | {'OPTIMAL' if p99_rtt < 10.0 else 'WARN'}")
        print(f"{'Min / Max RTT':<25} | {min_rtt:.3f} / {max_rtt:.3f} ms")
        print("=" * 76)

        # Core Assertions
        assert delivery_rate == 100.0, f"Expected 100% delivery rate, got {delivery_rate:.1f}%"
        assert mean_rtt < 5.0, f"Mean RTT target failed: {mean_rtt:.3f} ms >= 5.0 ms"

    finally:
        cloud_server.stop()
        edge_client.stop()


def test_cloud_to_edge_schedule_sync_20_packets():
    """
    Transmits 20 schedule synchronization packets from Cloud down to Edge (QoS 1).
    Validates delivery, operation schedule payload integrity, and Pydantic compliance.
    """
    event = threading.Event()
    received_syncs: List[CloudSyncPayload] = []

    def on_sync(sync: CloudSyncPayload):
        received_syncs.append(sync)
        event.set()

    cloud_server = CloudMQTTServer(client_id="cloud_sync_tester")
    edge_client = EdgeMQTTClient(
        machine_id="M1",
        client_id="edge_sync_tester",
        on_schedule_sync_callback=on_sync,
    )

    try:
        assert cloud_server.start(timeout=3.0) is True
        assert edge_client.start(timeout=3.0) is True
        time.sleep(0.15)

        n_sync = 20
        for i in range(n_sync):
            event.clear()
            sync_payload = CloudSyncPayload(
                weights={"w_cmax": 0.40, "w_energy": 0.35, "w_tardiness": 0.25},
                baseline_cmax=78.50 - (i * 0.1),
                baseline_energy_kwh=32.40 - (i * 0.05),
                p_limit_kw=22.0,
                timestamp=float(i * 10),
                iteration=i,
                baseline_schedule=[
                    {"job_id": 0, "op_idx": 0, "machine_id": "M1", "speed_level": 2, "pm_flag": 0},
                    {"job_id": 1, "op_idx": 0, "machine_id": "M2", "speed_level": 3, "pm_flag": 0},
                ],
            )

            pub_ok = cloud_server.publish_schedule_sync(sync_payload, qos=1)
            assert pub_ok is True, f"Publishing sync {i} failed"

            got_it = event.wait(timeout=2.0)
            assert got_it is True, f"Timed out waiting for sync packet {i}"

        assert len(received_syncs) == n_sync
        assert edge_client.latest_schedule_sync is not None
        assert edge_client.latest_schedule_sync.iteration == n_sync - 1
        assert edge_client.latest_schedule_sync.p_limit_kw == 22.0
        assert len(edge_client.latest_schedule_sync.baseline_schedule) == 2

    finally:
        cloud_server.stop()
        edge_client.stop()


def test_edge_reschedule_trigger_when_cdi_exceeds_threshold():
    """
    Simulates shop-floor disturbance: Edge triggers urgent reschedule when CDI > threshold (QoS 1).
    Validates receipt and trigger payload structure on Cloud.
    """
    event = threading.Event()
    received_triggers: List[RescheduleTriggerPayload] = []

    def on_reschedule(trigger: RescheduleTriggerPayload):
        received_triggers.append(trigger)
        event.set()

    cloud_server = CloudMQTTServer(
        client_id="cloud_reschedule_tester",
        on_reschedule_callback=on_reschedule,
    )
    edge_client = EdgeMQTTClient(
        machine_id="M3",
        client_id="edge_reschedule_tester",
    )

    try:
        assert cloud_server.start(timeout=3.0) is True
        assert edge_client.start(timeout=3.0) is True
        time.sleep(0.15)

        trigger_payload = RescheduleTriggerPayload(
            machine_id="M3",
            reason="CDI_THRESHOLD_EXCEEDED",
            cdi=0.342,
            cdi_threshold=0.250,
            current_wear_um=82.5,
            timestamp=25.4,
            details={"drift_pct": 36.8, "power_kw": 4.15},
        )

        pub_ok = edge_client.publish_reschedule_trigger(trigger_payload, qos=1)
        assert pub_ok is True

        got_it = event.wait(timeout=2.0)
        assert got_it is True
        assert len(received_triggers) == 1
        assert len(cloud_server.reschedule_requests) == 1

        rec = received_triggers[0]
        assert rec.machine_id == "M3"
        assert rec.reason == "CDI_THRESHOLD_EXCEEDED"
        assert rec.cdi == pytest.approx(0.342, rel=1e-3)
        assert rec.cdi > rec.cdi_threshold

    finally:
        cloud_server.stop()
        edge_client.stop()


def test_in_memory_broker_fallback_simulation():
    """
    Validates that force_in_memory=True operates with 100% feature parity,
    guaranteeing seamless CI/offline execution without Mosquitto.
    """
    event = threading.Event()
    received_telemetry: List[MachineTelemetryPayload] = []

    def on_telemetry(tel: MachineTelemetryPayload):
        received_telemetry.append(tel)
        event.set()

    cloud_mock = CloudMQTTServer(
        client_id="cloud_in_memory",
        force_in_memory=True,
        on_telemetry_callback=on_telemetry,
    )
    edge_mock = EdgeMQTTClient(
        machine_id="M2",
        client_id="edge_in_memory",
        force_in_memory=True,
    )

    try:
        assert cloud_mock.start() is True
        assert edge_mock.start() is True
        assert cloud_mock.using_mock_broker is True
        assert edge_mock.using_mock_broker is True

        tel = MachineTelemetryPayload(
            machine_id="M2",
            power_kw=6.10,
            vibration_rms=1.35,
            estimated_wear_um=45.2,
            state=MachineState.PROCESSING,
            timestamp=15.0,
            cdi=0.12,
        )

        edge_mock.publish_telemetry(tel, qos=0)
        assert event.wait(timeout=1.0) is True
        assert len(received_telemetry) == 1
        assert received_telemetry[0].machine_id == "M2"

    finally:
        cloud_mock.stop()
        edge_mock.stop()

