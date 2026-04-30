import json
import logging
import os
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional

import yaml

log = logging.getLogger(__name__)

CORE_ENGINE_DIR = Path(os.getenv("CORE_ENGINE_DIR", "/workspace/core_engine"))
RUNTIME_CONFIG = CORE_ENGINE_DIR / "configs" / "runtime.yaml"
DEFAULT_CONFIG = CORE_ENGINE_DIR / "configs" / "default.yaml"

_STATIC_MODELS = {
    "detector": "configs/config_infer_primary_peoplenet.txt",
    "tracker": "configs/config_tracker_NvDCF.yml",
    "reid": "configs/config_infer_secondary_reid.txt",
}
_STATIC_OUTPUT = {"mode": "zmq", "endpoint": "tcp://*:5555"}


class CameraRegistry:
    """
    Maps camera_id (str, backend DB key) <-> source_id (int, DeepStream stream index).
    Writes runtime.yaml and triggers a pipeline restart whenever cameras change.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._cameras: Dict[str, str] = {}   # camera_id -> rtsp_url (insertion-ordered)
        self._active: set = set()             # camera_ids with active inference
        self._restart_fn: Optional[Callable] = None
        self._load_env()

    def set_restart_fn(self, fn: Callable) -> None:
        self._restart_fn = fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, camera_id: str, rtsp_url: str) -> None:
        with self._lock:
            self._cameras[camera_id] = rtsp_url
        self._write_config()
        log.info("Registered camera %s -> %s", camera_id, rtsp_url)
        if self._restart_fn:
            self._restart_fn()

    def unregister(self, camera_id: str) -> bool:
        with self._lock:
            if camera_id not in self._cameras:
                return False
            del self._cameras[camera_id]
            self._active.discard(camera_id)
        self._write_config()
        log.info("Unregistered camera %s", camera_id)
        if self._restart_fn:
            self._restart_fn()
        return True

    def activate(self, camera_id: str) -> None:
        with self._lock:
            self._active.add(camera_id)

    def deactivate(self, camera_id: str) -> None:
        with self._lock:
            self._active.discard(camera_id)

    def is_active(self, camera_id: str) -> bool:
        with self._lock:
            return camera_id in self._active

    def get_rtsp_url(self, camera_id: str) -> Optional[str]:
        with self._lock:
            return self._cameras.get(camera_id)

    def resolve_source(self, source_id: int) -> Optional[str]:
        """Return the camera_id for a DeepStream source_id (0-based insertion order)."""
        with self._lock:
            keys = list(self._cameras.keys())
        return keys[source_id] if source_id < len(keys) else None

    def has_cameras(self) -> bool:
        with self._lock:
            return bool(self._cameras)

    def list(self) -> List[dict]:
        with self._lock:
            return [{"camera_id": k, "rtsp_url": v} for k, v in self._cameras.items()]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_env(self) -> None:
        """Pre-seed cameras from CAMERAS_JSON env var.

        Format: '{"cam1": "rtsp://...", "cam2": "rtsp://..."}'
        """
        raw = os.getenv("CAMERAS_JSON", "")
        if not raw:
            return
        try:
            mapping = json.loads(raw)
            for cam_id, rtsp_url in mapping.items():
                self._cameras[str(cam_id)] = rtsp_url
            log.info("Loaded %d camera(s) from CAMERAS_JSON", len(mapping))
        except json.JSONDecodeError as exc:
            log.warning("Invalid CAMERAS_JSON: %s", exc)

    def _write_config(self) -> None:
        with self._lock:
            cameras = dict(self._cameras)

        # Inherit models/output from default.yaml if present
        base: dict = {}
        if DEFAULT_CONFIG.exists():
            with open(DEFAULT_CONFIG) as f:
                base = yaml.safe_load(f) or {}

        config = {
            "version": 1,
            "sources": [
                {"uri": url, "enabled": True} for url in cameras.values()
            ],
            "models": base.get("models", _STATIC_MODELS),
            "output": base.get("output", _STATIC_OUTPUT),
        }

        RUNTIME_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        with open(RUNTIME_CONFIG, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        log.debug("Wrote runtime config with %d source(s)", len(cameras))
