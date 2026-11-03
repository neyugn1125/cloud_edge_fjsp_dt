"""
Cloud MQTT Server for Industrial Cloud-Edge Digital Twin.
Responsible for:
1. Subscribing to real-time machine telemetry ('dt/fjsp/edge/telemetry', QoS 0).
2. Subscribing to urgent rescheduling triggers ('dt/fjsp/edge/trigger_reschedule', QoS 1).
3. Publishing global baseline schedule and optimization updates ('dt/fjsp/cloud/schedule_sync', QoS 1).
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


class CloudMQTTServer:
    """
    Cloud-side Industrial MQTT Server.
    Acts as the central coordination endpoint for all Edge Digital Twin nodes.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 1883,
        client_id: str = "fjsp_cloud_server",
        keepalive: int = 60,
        force_in_memory: bool = False,
        on_telemetry_callback: Optional[Callable[[MachineTelemetryPayload], None]] = None,
        on_reschedule_callback: Optional[Callable[[RescheduleTriggerPayload], None]] = None,
    ):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.keepalive = keepalive
        self.force_in_memory = force_in_memory

        self.on_telemetry_callback = on_telemetry_callback
        self.on_reschedule_callback = on_reschedule_callback

        self.is_connected = False
        self.using_mock_broker = False

        # State storage
        self.latest_telemetry: Dict[str, MachineTelemetryPayload] = {}
        self.telemetry_history: List[MachineTelemetryPayload] = []
        self.reschedule_requests: List[RescheduleTriggerPayload] = []
        self._history_lock = threading.Lock()

        self._paho_client: Optional[mqtt.Client] = None
        self._in_memory_broker: Optional[InMemoryMQTTBroker] = None
        self._ready_event = threading.Event()

    def start(self, timeout: float = 3.0) -> bool:
        """Connects to the MQTT broker and starts message processing."""
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
                logger.info(f"[CloudMQTTServer] Connected to Mosquitto at {self.host}:{self.port}")
                self.is_connected = True
                self.using_mock_broker = False
                return True
            else:
                logger.warning(f"[CloudMQTTServer] Mosquitto connection timed out. Falling back to InMemoryMQTTBroker.")
                if self._paho_client:
                    self._paho_client.loop_stop()
                return self._start_in_memory()
        except Exception as e:
            logger.warning(f"[CloudMQTTServer] Connection to {self.host}:{self.port} failed ({e}). Falling back to InMemoryMQTTBroker.")
            return self._start_in_memory()

    def _start_in_memory(self) -> bool:
        """Starts in-memory mock broker."""
        self._in_memory_broker = InMemoryMQTTBroker.get_instance()
        self._in_memory_broker.subscribe(TOPIC_EDGE_TELEMETRY, self._on_in_memory_message)
        self._in_memory_broker.subscribe(TOPIC_EDGE_TRIGGER_RESCHEDULE, self._on_in_memory_message)
        self.is_connected = True
        self.using_mock_broker = True
        self._ready_event.set()
        logger.info("[CloudMQTTServer] Started on InMemoryMQTTBroker.")
        return True

    def _on_paho_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            client.subscribe(TOPIC_EDGE_TELEMETRY, qos=0)
            client.subscribe(TOPIC_EDGE_TRIGGER_RESCHEDULE, qos=1)
            self.is_connected = True
            self._ready_event.set()
            logger.info(f"[CloudMQTTServer] Subscribed to {TOPIC_EDGE_TELEMETRY} (QoS 0) and {TOPIC_EDGE_TRIGGER_RESCHEDULE} (QoS 1)")

    def _on_paho_disconnect(self, client, userdata, disconnect_flags, rc, properties=None):
        self.is_connected = False
        logger.info("[CloudMQTTServer] Disconnected from broker.")

    def _on_paho_message(self, client, userdata, msg):
        self._handle_raw_message(msg.topic, msg.payload, msg.qos)

    def _on_in_memory_message(self, client, userdata, msg: MQTTMessageMock):
        self._handle_raw_message(msg.topic, msg.payload, msg.qos)

    def _handle_raw_message(self, topic: str, payload_bytes: bytes, qos: int):
        try:
            payload_str = payload_bytes.decode("utf-8")
            data = json.loads(payload_str)

            if topic == TOPIC_EDGE_TELEMETRY:
                telemetry = MachineTelemetryPayload.model_validate(data)
                with self._history_lock:
                    self.latest_telemetry[telemetry.machine_id] = telemetry
                    self.telemetry_history.append(telemetry)
                if self.on_telemetry_callback:
                    self.on_telemetry_callback(telemetry)

            elif topic == TOPIC_EDGE_TRIGGER_RESCHEDULE:
                trigger = RescheduleTriggerPayload.model_validate(data)
                with self._history_lock:
                    self.reschedule_requests.append(trigger)
                logger.warning(
                    f"[CloudMQTTServer] Reschedule trigger received from {trigger.machine_id}: "
                    f"reason='{trigger.reason}', CDI={trigger.cdi:.3f}"
                )
                if self.on_reschedule_callback:
                    self.on_reschedule_callback(trigger)
        except Exception as e:
            logger.error(f"[CloudMQTTServer] Error parsing message on {topic}: {e}")

    def publish_schedule_sync(self, payload: Union[CloudSyncPayload, Dict[str, Any]], qos: int = 1) -> bool:
        """
        Publishes schedule synchronization payload down to edge nodes
        (Topic: 'dt/fjsp/cloud/schedule_sync').
        """
        if isinstance(payload, CloudSyncPayload):
            payload_dict = payload.model_dump()
        elif isinstance(payload, dict):
            # Validate through Pydantic
            validated = CloudSyncPayload.model_validate(payload)
            payload_dict = validated.model_dump()
        else:
            raise TypeError(f"Payload must be CloudSyncPayload or dict, got {type(payload)}")

        payload_json = json.dumps(payload_dict)

        if self.using_mock_broker and self._in_memory_broker:
            self._in_memory_broker.publish(
                TOPIC_CLOUD_SCHEDULE_SYNC, payload_json, qos=qos, sender_id=self.client_id
            )
            return True

        if self._paho_client and self.is_connected:
            info = self._paho_client.publish(TOPIC_CLOUD_SCHEDULE_SYNC, payload_json, qos=qos)
            return info.rc == mqtt.MQTT_ERR_SUCCESS

        return False

    def stop(self):
        """Stops the MQTT server and cleanly disconnects."""
        self.is_connected = False
        if self._paho_client:
            self._paho_client.loop_stop()
            self._paho_client.disconnect()
            self._paho_client = None

        if self._in_memory_broker:
            self._in_memory_broker.unsubscribe(TOPIC_EDGE_TELEMETRY, self._on_in_memory_message)
            self._in_memory_broker.unsubscribe(TOPIC_EDGE_TRIGGER_RESCHEDULE, self._on_in_memory_message)
            self._in_memory_broker = None

    def wait_until_ready(self, timeout: float = 3.0) -> bool:
        """Blocks until the server is connected and ready."""
        return self._ready_event.wait(timeout=timeout)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

