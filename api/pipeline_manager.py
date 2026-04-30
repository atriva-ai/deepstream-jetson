import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import zmq

from .camera_registry import CameraRegistry, CORE_ENGINE_DIR, RUNTIME_CONFIG
from .detection_store import DetectionStore

log = logging.getLogger(__name__)

_BINARY = CORE_ENGINE_DIR / "build" / "edge_retail_core_engine"
_ZMQ_ENDPOINT = os.getenv("ZMQ_ENDPOINT", "tcp://localhost:5555")


class PipelineManager:
    def __init__(self, store: DetectionStore, registry: CameraRegistry):
        self._store = store
        self._registry = registry
        self._proc: Optional[subprocess.Popen] = None
        self._proc_lock = threading.Lock()
        self._running = False
        registry.set_restart_fn(self.restart)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        zmq_thread = threading.Thread(target=self._zmq_loop, daemon=True, name="zmq-sub")
        zmq_thread.start()
        if self._registry.has_cameras():
            self._registry._write_config()
            self._launch()

    def stop(self) -> None:
        self._running = False
        self._kill()

    def restart(self) -> None:
        log.info("Restarting DeepStream pipeline")
        self._kill()
        time.sleep(1)
        if self._registry.has_cameras():
            self._launch()

    def is_alive(self) -> bool:
        with self._proc_lock:
            return self._proc is not None and self._proc.poll() is None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _launch(self) -> None:
        if not _BINARY.exists():
            log.error("core_engine binary not found: %s — skip launch", _BINARY)
            return
        with self._proc_lock:
            self._proc = subprocess.Popen(
                [str(_BINARY), "--config", str(RUNTIME_CONFIG)],
                cwd=str(CORE_ENGINE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            log.info("Launched core_engine PID %d", self._proc.pid)

    def _kill(self) -> None:
        with self._proc_lock:
            proc = self._proc
            self._proc = None
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            log.info("core_engine stopped")

    def _zmq_loop(self) -> None:
        ctx = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.setsockopt(zmq.RCVTIMEO, 1000)
        sock.connect(_ZMQ_ENDPOINT)
        log.info("ZMQ subscriber connected to %s", _ZMQ_ENDPOINT)

        while self._running:
            try:
                raw = sock.recv_string()
            except zmq.Again:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if event.get("event") != "frame":
                continue
            source_id = event.get("source_id", 0)
            camera_id = self._registry.resolve_source(source_id)
            if camera_id:
                self._store.update(camera_id, event)

        sock.close()
        ctx.term()
        log.info("ZMQ subscriber stopped")
