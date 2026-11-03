"""
Edge Industrial MQTT Client.
Responsible for:
1. Publishing real-time machine telemetry ('dt/fjsp/edge/telemetry', QoS 0).
2. Publishing urgent rescheduling triggers ('dt/fjsp/edge/trigger_reschedule', QoS 1) when CDI > threshold.
3. Subscribing to global baseline schedule and parameter sync ('dt/fjsp/cloud/schedule_sync', QoS 1).
4. Automatic fallback to InMemoryMQTTBroker if Mosquitto is unavailable.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Union

import paho.mqtt.client as mqtt

from utils.mqtt_broker import InMemoryMQTTBroker, MQTTMessageMock
from utils.mqtt_schemas import (
    TOPIC_CLOUD_SCHEDULE_SYNC,
    TOPIC_EDGE_TELEMETRY,
    TOPIC_EDGE_TRIGGER_RESCHEDULE,
    CloudSyncPayload,
    MachineTelemetryPayload,
    RescheduleTriggerPayload,
)

logger = logging.getLogger(__name__)


class EdgeMQTTClient:
    """
    Edge-side Industrial MQTT Client.
    Deploys on the shop floor to bridge physical CNC twins / breadboards with Cloud Optimization.
    """

    def __init__(
        self,
        machine_id: str = "M1",
        host: str = "localhost",
        port: int = 1883,
        client_id: Optional[str] = None,
        keepalive: int = 60,
        force_in_memory: bool = False,
        on_schedule_sync_callback: Optional[Callable[[CloudSyncPayload], None]] = None,
    ):
        self.machine_id = machine_id.upper()
        self.host = host
        self.port = port
        self.client_id = client_id or f"fjsp_edge_node_{self.machine_id}"
        self.keepalive = keepalive
        self.force_in_memory = force_in_memory

        self.on_schedule_sync_callback = on_schedule_sync_callback

        self.is_connected = False
        self.using_mock_broker = False

        # State storage
        self.latest_schedule_sync: Optional[CloudSyncPayload] = None
        self.sync_history: List[CloudSyncPayload] = []
        self._sync_lock = threading.Lock()

        self._paho_client: Optional[mqtt.Client] = None
        self._in_memory_broker: Optional[InMemoryMQTTBroker] = None
        self._ready_event = threading.Event()

    def start(self, timeout: float = 3.0) -> bool:
        """Connects to the MQTT broker and starts message loop."""
        if self.force_in_memory:
            return self._start_in_memory()

        try:
            self._paho_client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2,
                client_id=self.client_id,
            )
            self._paho_client.on_connect = self._on_paho_connect
            self._paho_client.on_message = self._on_paho_message
            self._paho_client.on_disconnect = self._on_paho_disconnect

            self._paho_client.connect(self.host, self.port, keepalive=self.keepalive)
            self._paho_client.loop_start()

            if self._ready_event.wait(timeout=timeout):
                logger.info(f"[EdgeMQTTClient:{self.machine_id}] Connected to Mosquitto at {self.host}:{self.port}")
                self.is_connected = True
                self.using_mock_broker = False
                return True
            else:
                logger.warning(f"[EdgeMQTTClient:{self.machine_id}] Connection timed out. Falling back to InMemoryMQTTBroker.")
                if self._paho_client:
                    self._paho_client.loop_stop()
                return self._start_in_memory()
        except Exception as e:
            logger.warning(f"[EdgeMQTTClient:{self.machine_id}] Connection to {self.host}:{self.port} failed ({e}). Falling back to InMemoryMQTTBroker.")
            return self._start_in_memory()

    def _start_in_memory(self) -> bool:
        """Starts in-memory mock broker."""
        self._in_memory_broker = InMemoryMQTTBroker.get_instance()
        self._in_memory_broker.subscribe(TOPIC_CLOUD_SCHEDULE_SYNC, self._on_in_memory_message)
        self.is_connected = True
        self.using_mock_broker = True
        self._ready_event.set()
        logger.info(f"[EdgeMQTTClient:{self.machine_id}] Started on InMemoryMQTTBroker.")
        return True

    def _on_paho_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            client.subscribe(TOPIC_CLOUD_SCHEDULE_SYNC, qos=1)
            self.is_connected = True
            self._ready_event.set()
            logger.info(f"[EdgeMQTTClient:{self.machine_id}] Subscribed to {TOPIC_CLOUD_SCHEDULE_SYNC} (QoS 1)")

    def _on_paho_disconnect(self, client, userdata, disconnect_flags, rc, properties=None):
        self.is_connected = False
        logger.info(f"[EdgeMQTTClient:{self.machine_id}] Disconnected from broker.")

    def _on_paho_message(self, client, userdata, msg):
        self._handle_raw_message(msg.topic, msg.payload, msg.qos)

    def _on_in_memory_message(self, client, userdata, msg: MQTTMessageMock):
        self._handle_raw_message(msg.topic, msg.payload, msg.qos)

    def _handle_raw_message(self, topic: str, payload_bytes: bytes, qos: int):
        try:
            payload_str = payload_bytes.decode("utf-8")
            data = json.loads(payload_str)

            if topic == TOPIC_CLOUD_SCHEDULE_SYNC:
                sync_payload = CloudSyncPayload.model_validate(data)
                with self._sync_lock:
                    self.latest_schedule_sync = sync_payload
                    self.sync_history.append(sync_payload)
                logger.info(
                    f"[EdgeMQTTClient:{self.machine_id}] Received Schedule Sync: "
                    f"cmax={sync_payload.baseline_cmax:.1f}m, P_limit={sync_payload.p_limit_kw:.1f}kW, "
                    f"weights={sync_payload.weights}"
                )
                if self.on_schedule_sync_callback:
                    self.on_schedule_sync_callback(sync_payload)
        except Exception as e:
            logger.error(f"[EdgeMQTTClient:{self.machine_id}] Error parsing message on {topic}: {e}")

    def publish_telemetry(self, payload: Union[MachineTelemetryPayload, Dict[str, Any]], qos: int = 0) -> bool:
        """
        Publishes physical telemetry up to Cloud Optimization Layer
        (Topic: 'dt/fjsp/edge/telemetry', QoS 0).
        """
        if isinstance(payload, MachineTelemetryPayload):
            payload_dict = payload.model_dump()
        elif isinstance(payload, dict):
            validated = MachineTelemetryPayload.model_validate(payload)
            payload_dict = validated.model_dump()
        else:
            raise TypeError(f"Payload must be MachineTelemetryPayload or dict, got {type(payload)}")

        payload_json = json.dumps(payload_dict)

        if self.using_mock_broker and self._in_memory_broker:
            self._in_memory_broker.publish(
                TOPIC_EDGE_TELEMETRY, payload_json, qos=qos, sender_id=self.client_id
            )
            return True

        if self._paho_client and self.is_connected:
            info = self._paho_client.publish(TOPIC_EDGE_TELEMETRY, payload_json, qos=qos)
            return info.rc == mqtt.MQTT_ERR_SUCCESS

        return False

    def publish_reschedule_trigger(self, payload: Union[RescheduleTriggerPayload, Dict[str, Any]], qos: int = 1) -> bool:
        """
        Publishes urgent rescheduling trigger up to Cloud
        (Topic: 'dt/fjsp/edge/trigger_reschedule', QoS 1).
        """
        if isinstance(payload, RescheduleTriggerPayload):
            payload_dict = payload.model_dump()
        elif isinstance(payload, dict):
            validated = RescheduleTriggerPayload.model_validate(payload)
            payload_dict = validated.model_dump()
        else:
            raise TypeError(f"Payload must be RescheduleTriggerPayload or dict, got {type(payload)}")

        payload_json = json.dumps(payload_dict)

        if self.using_mock_broker and self._in_memory_broker:
            self._in_memory_broker.publish(
                TOPIC_EDGE_TRIGGER_RESCHEDULE, payload_json, qos=qos, sender_id=self.client_id
            )
            return True

        if self._paho_client and self.is_connected:
            info = self._paho_client.publish(TOPIC_EDGE_TRIGGER_RESCHEDULE, payload_json, qos=qos)
            return info.rc == mqtt.MQTT_ERR_SUCCESS

        return False

    def stop(self):
        """Stops the MQTT client and cleanly disconnects."""
        self.is_connected = False
        if self._paho_client:
            self._paho_client.loop_stop()
            self._paho_client.disconnect()
            self._paho_client = None

        if self._in_memory_broker:
            self._in_memory_broker.unsubscribe(TOPIC_CLOUD_SCHEDULE_SYNC, self._on_in_memory_message)
            self._in_memory_broker = None

    def wait_until_ready(self, timeout: float = 3.0) -> bool:
        """Blocks until the client is connected and subscribed."""
        return self._ready_event.wait(timeout=timeout)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

