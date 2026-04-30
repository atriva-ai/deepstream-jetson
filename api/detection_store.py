import threading
import time
from typing import Dict, List, Optional


def _translate(d: dict, camera_id: str, ts: int) -> dict:
    bbox = d.get("bbox", {})
    l = bbox.get("l", 0.0)
    t = bbox.get("t", 0.0)
    w = bbox.get("w", 0.0)
    h = bbox.get("h", 0.0)
    return {
        "camera_id": camera_id,
        "track_id": d.get("tid"),
        "bbox": [l, t, l + w, t + h],
        "timestamp": ts,
        "class_id": d.get("class", 0),
        "class_name": d.get("label", "person"),
        "confidence": d.get("conf", 0.0),
    }


class DetectionFrame:
    __slots__ = ("camera_id", "detections", "detection_count")

    def __init__(self, camera_id: str, detections: list):
        self.camera_id = camera_id
        self.detections = detections
        self.detection_count = len(detections)

    def to_response(self, object_filter: Optional[str] = None) -> dict:
        dets = self.detections
        if object_filter:
            dets = [d for d in dets if d.get("class_name") == object_filter]
        return {
            "camera_id": self.camera_id,
            "frame_path": "",
            "detections": dets,
            "detection_count": len(dets),
        }


class DetectionStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._latest: Dict[str, DetectionFrame] = {}

    def update(self, camera_id: str, zmq_event: dict) -> None:
        ts = int(zmq_event.get("ts_ms", time.time() * 1000) / 1000)
        detections = [
            _translate(d, camera_id, ts)
            for d in zmq_event.get("detections", [])
        ]
        frame = DetectionFrame(camera_id, detections)
        with self._lock:
            self._latest[camera_id] = frame

    def get(self, camera_id: str) -> Optional[DetectionFrame]:
        with self._lock:
            return self._latest.get(camera_id)

    def cameras(self) -> List[str]:
        with self._lock:
            return list(self._latest.keys())
