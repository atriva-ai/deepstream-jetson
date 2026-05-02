"""
Unit tests for deepstream-jetson FastAPI service.
No Docker, no GPU, no C++ binary required — all heavy I/O is mocked.
"""
import json
import os
import sys
import time
import threading
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the service package is importable from repo root
# ---------------------------------------------------------------------------
SERVICE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SERVICE_ROOT))

# Stub out zmq before importing pipeline_manager (avoids zmq import error if
# pyzmq is not installed in the test environment)
import types

zmq_stub = types.ModuleType("zmq")
zmq_stub.Context = MagicMock
zmq_stub.SUB = 2
zmq_stub.SUBSCRIBE = 6
zmq_stub.RCVTIMEO = 27
zmq_stub.Again = Exception
sys.modules.setdefault("zmq", zmq_stub)

import yaml  # must be importable (pyyaml)
import api.camera_registry as cam_reg_mod
from api.camera_registry import CameraRegistry
from api.detection_store import DetectionStore, _translate


@contextmanager
def _tmp_registry(subdir: str):
    """Create a CameraRegistry with RUNTIME_CONFIG redirected to a safe tmp dir."""
    tmp = Path(f"/tmp/ds_test_{subdir}")
    tmp.mkdir(parents=True, exist_ok=True)
    runtime_cfg = tmp / "configs" / "runtime.yaml"
    # CAMERAS_JSON env is read in __init__, patch it too
    with patch.object(cam_reg_mod, "RUNTIME_CONFIG", runtime_cfg), \
         patch.dict(os.environ, {"CAMERAS_JSON": ""}):
        reg = CameraRegistry()
    yield reg, tmp


# ===========================================================================
# CameraRegistry
# ===========================================================================
class TestCameraRegistry(unittest.TestCase):

    def test_register_and_list(self):
        with _tmp_registry("reg") as (reg, tmp):
            runtime_cfg = tmp / "configs" / "runtime.yaml"
            with patch.object(cam_reg_mod, "RUNTIME_CONFIG", runtime_cfg):
                reg.register("cam1", "rtsp://host/s1")
            cams = reg.list()
        self.assertEqual(len(cams), 1)
        self.assertEqual(cams[0]["camera_id"], "cam1")
        self.assertEqual(cams[0]["rtsp_url"], "rtsp://host/s1")

    def test_register_writes_yaml(self):
        with _tmp_registry("yaml") as (reg, tmp):
            cfg_path = tmp / "configs" / "runtime.yaml"
            with patch.object(cam_reg_mod, "RUNTIME_CONFIG", cfg_path):
                reg.register("cam1", "rtsp://host/s1")
            self.assertTrue(cfg_path.exists(), "runtime.yaml not written")
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
        self.assertEqual(len(cfg["sources"]), 1)
        self.assertEqual(cfg["sources"][0]["uri"], "rtsp://host/s1")

    def test_unregister_removes_camera(self):
        with _tmp_registry("unreg") as (reg, tmp):
            cfg_path = tmp / "configs" / "runtime.yaml"
            with patch.object(cam_reg_mod, "RUNTIME_CONFIG", cfg_path):
                reg.register("cam1", "rtsp://host/s1")
                result = reg.unregister("cam1")
        self.assertTrue(result)
        self.assertEqual(reg.list(), [])

    def test_unregister_missing_returns_false(self):
        with _tmp_registry("missing") as (reg, _):
            self.assertFalse(reg.unregister("nonexistent"))

    def test_activate_deactivate(self):
        with _tmp_registry("active") as (reg, tmp):
            cfg_path = tmp / "configs" / "runtime.yaml"
            with patch.object(cam_reg_mod, "RUNTIME_CONFIG", cfg_path):
                reg.register("cam1", "rtsp://host/s1")
            self.assertFalse(reg.is_active("cam1"))
            reg.activate("cam1")
            self.assertTrue(reg.is_active("cam1"))
            reg.deactivate("cam1")
            self.assertFalse(reg.is_active("cam1"))

    def test_resolve_source_order(self):
        with _tmp_registry("order") as (reg, tmp):
            cfg_path = tmp / "configs" / "runtime.yaml"
            with patch.object(cam_reg_mod, "RUNTIME_CONFIG", cfg_path):
                reg.register("cam1", "rtsp://host/s1")
                reg.register("cam2", "rtsp://host/s2")
                reg.register("cam3", "rtsp://host/s3")
        self.assertEqual(reg.resolve_source(0), "cam1")
        self.assertEqual(reg.resolve_source(1), "cam2")
        self.assertEqual(reg.resolve_source(2), "cam3")
        self.assertIsNone(reg.resolve_source(99))

    def test_has_cameras(self):
        with _tmp_registry("has") as (reg, tmp):
            self.assertFalse(reg.has_cameras())
            cfg_path = tmp / "configs" / "runtime.yaml"
            with patch.object(cam_reg_mod, "RUNTIME_CONFIG", cfg_path):
                reg.register("cam1", "rtsp://host/s1")
            self.assertTrue(reg.has_cameras())

    def test_cameras_json_env_preload(self):
        tmp = Path("/tmp/ds_test_env")
        tmp.mkdir(parents=True, exist_ok=True)
        mapping = json.dumps({"1": "rtsp://host/s1", "2": "rtsp://host/s2"})
        with patch.dict(os.environ, {"CAMERAS_JSON": mapping}):
            reg = CameraRegistry()
        cams = reg.list()
        self.assertEqual(len(cams), 2)
        self.assertEqual(reg.get_rtsp_url("1"), "rtsp://host/s1")
        self.assertEqual(reg.get_rtsp_url("2"), "rtsp://host/s2")

    def test_cameras_json_invalid_does_not_crash(self):
        with patch.dict(os.environ, {"CAMERAS_JSON": "bad{json"}):
            reg = CameraRegistry()
        self.assertEqual(reg.list(), [])

    def test_restart_fn_called_on_register(self):
        with _tmp_registry("restart1") as (reg, tmp):
            cfg_path = tmp / "configs" / "runtime.yaml"
            fn = MagicMock()
            reg.set_restart_fn(fn)
            with patch.object(cam_reg_mod, "RUNTIME_CONFIG", cfg_path):
                reg.register("cam1", "rtsp://host/s1")
        fn.assert_called_once()

    def test_restart_fn_called_on_unregister(self):
        with _tmp_registry("restart2") as (reg, tmp):
            cfg_path = tmp / "configs" / "runtime.yaml"
            fn = MagicMock()
            reg.set_restart_fn(fn)
            with patch.object(cam_reg_mod, "RUNTIME_CONFIG", cfg_path):
                reg.register("cam1", "rtsp://host/s1")
                fn.reset_mock()
                reg.unregister("cam1")
        fn.assert_called_once()

    def test_multiple_cameras_in_yaml(self):
        with _tmp_registry("multicam") as (reg, tmp):
            cfg_path = tmp / "configs" / "runtime.yaml"
            with patch.object(cam_reg_mod, "RUNTIME_CONFIG", cfg_path):
                reg.register("cam1", "rtsp://host/s1")
                reg.register("cam2", "rtsp://host/s2")
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
        uris = [s["uri"] for s in cfg["sources"]]
        self.assertIn("rtsp://host/s1", uris)
        self.assertIn("rtsp://host/s2", uris)

    def test_thread_safety_concurrent_register(self):
        """Register from 10 threads simultaneously — no crashes, correct count."""
        with _tmp_registry("thread") as (reg, tmp):
            cfg_path = tmp / "configs" / "runtime.yaml"
            errors = []

            def do_register(i):
                try:
                    reg.register(f"cam{i}", f"rtsp://host/s{i}")
                except Exception as e:
                    errors.append(e)

            # Patch once for all threads (patch.object is not thread-safe as a per-thread ctx)
            with patch.object(cam_reg_mod, "RUNTIME_CONFIG", cfg_path):
                threads = [threading.Thread(target=do_register, args=(i,)) for i in range(10)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

        self.assertEqual(errors, [], f"Thread errors: {errors}")
        self.assertEqual(len(reg.list()), 10)


# ===========================================================================
# DetectionStore
# ===========================================================================
class TestDetectionStore(unittest.TestCase):
    def _frame_event(self, source_id=0, n_dets=2):
        return {
            "event": "frame",
            "ts_ms": int(time.time() * 1000),
            "source_id": source_id,
            "frame": 42,
            "detections": [
                {
                    "class": 0,
                    "label": "person",
                    "conf": 0.85 + i * 0.01,
                    "bbox": {"l": 10.0 * i, "t": 20.0, "w": 80.0, "h": 200.0},
                    "tid": i + 1,
                }
                for i in range(n_dets)
            ],
        }

    def test_update_and_get(self):
        store = DetectionStore()
        event = self._frame_event()
        store.update("cam1", event)
        frame = store.get("cam1")
        self.assertIsNotNone(frame)
        self.assertEqual(frame.detection_count, 2)

    def test_get_missing_returns_none(self):
        store = DetectionStore()
        self.assertIsNone(store.get("nope"))

    def test_cameras_list(self):
        store = DetectionStore()
        store.update("cam1", self._frame_event())
        store.update("cam2", self._frame_event())
        self.assertIn("cam1", store.cameras())
        self.assertIn("cam2", store.cameras())

    def test_to_response_no_filter(self):
        store = DetectionStore()
        store.update("cam1", self._frame_event(n_dets=3))
        resp = store.get("cam1").to_response()
        self.assertEqual(resp["detection_count"], 3)
        self.assertEqual(len(resp["detections"]), 3)
        self.assertEqual(resp["camera_id"], "cam1")

    def test_to_response_with_filter(self):
        store = DetectionStore()
        event = self._frame_event(n_dets=2)
        # Manually add a non-person detection
        event["detections"].append({
            "class": 1, "label": "bag", "conf": 0.7,
            "bbox": {"l": 0, "t": 0, "w": 10, "h": 10}, "tid": 99,
        })
        store.update("cam1", event)
        resp = store.get("cam1").to_response(object_filter="person")
        self.assertEqual(resp["detection_count"], 2)
        for d in resp["detections"]:
            self.assertEqual(d["class_name"], "person")

    def test_latest_overwritten_on_update(self):
        store = DetectionStore()
        store.update("cam1", self._frame_event(n_dets=1))
        store.update("cam1", self._frame_event(n_dets=5))
        frame = store.get("cam1")
        self.assertEqual(frame.detection_count, 5)

    def test_translate_bbox_conversion(self):
        det = {"class": 0, "label": "person", "conf": 0.9,
               "bbox": {"l": 10.0, "t": 20.0, "w": 30.0, "h": 40.0}, "tid": 7}
        result = _translate(det, "cam1", 12345)
        self.assertEqual(result["bbox"], [10.0, 20.0, 40.0, 60.0])  # [l,t,l+w,t+h]
        self.assertEqual(result["track_id"], 7)
        self.assertEqual(result["class_name"], "person")
        self.assertEqual(result["confidence"], 0.9)

    def test_empty_detections(self):
        store = DetectionStore()
        event = {"event": "frame", "ts_ms": 0, "source_id": 0, "detections": []}
        store.update("cam1", event)
        frame = store.get("cam1")
        self.assertEqual(frame.detection_count, 0)


# ===========================================================================
# Routes (FastAPI TestClient)
# ===========================================================================
try:
    from fastapi.testclient import TestClient
    from api.main import app
    HAS_FASTAPI_TEST = True
except ImportError:
    HAS_FASTAPI_TEST = False


_ROUTE_TMP = Path("/tmp/ds_test_routes")
_ROUTE_TMP.mkdir(parents=True, exist_ok=True)
_ROUTE_CFG = _ROUTE_TMP / "configs" / "runtime.yaml"


@unittest.skipUnless(HAS_FASTAPI_TEST, "httpx/fastapi[test] not installed")
class TestRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from api.main import manager, registry, store
        cls.manager = manager
        cls.registry = registry
        cls.store = store
        cls.client = TestClient(app, raise_server_exceptions=False)
        # Redirect all registry writes to a safe tmp dir for this test class
        cls._cfg_patcher = patch.object(cam_reg_mod, "RUNTIME_CONFIG", _ROUTE_CFG)
        cls._cfg_patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls._cfg_patcher.stop()

    def test_root(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["architecture"], "deepstream")

    def test_model_info(self):
        r = self.client.get("/model/info")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["architecture"], "deepstream")
        self.assertIn("tensorrt", data["accelerators"])

    def test_models(self):
        r = self.client.get("/models")
        self.assertEqual(r.status_code, 200)
        self.assertIn("available_models", r.json())

    def test_list_cameras_empty(self):
        r = self.client.get("/cameras")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["cameras"], [])

    def test_video_pipeline_health(self):
        r = self.client.get("/api/v1/video-pipeline/health/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_video_pipeline_debug(self):
        r = self.client.get("/api/v1/video-pipeline/debug/")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("pipeline_alive", data)
        self.assertIn("cameras", data)

    def test_inference_status_no_camera(self):
        r = self.client.get("/inference/continuous/status?camera_id=1")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertFalse(data["running"])

    def test_register_and_unregister_camera(self):
        r = self.client.post("/cameras/register?camera_id=test99&rtsp_url=rtsp://fake/stream")
        self.assertEqual(r.status_code, 200)
        r2 = self.client.get("/cameras")
        ids = [c["camera_id"] for c in r2.json()["cameras"]]
        self.assertIn("test99", ids)
        r3 = self.client.delete("/cameras/test99")
        self.assertEqual(r3.status_code, 200)
        r4 = self.client.get("/cameras")
        ids_after = [c["camera_id"] for c in r4.json()["cameras"]]
        self.assertNotIn("test99", ids_after)

    def test_unregister_missing_camera_404(self):
        r = self.client.delete("/cameras/nonexistent_xyz")
        self.assertEqual(r.status_code, 404)

    def test_vp_decode_start_no_url_no_registered_400(self):
        r = self.client.post("/api/v1/video-pipeline/decode/",
                             data={"camera_id": "cam_new_xyz"})
        self.assertEqual(r.status_code, 400)

    def test_vp_decode_start_with_url(self):
        r = self.client.post("/api/v1/video-pipeline/decode/",
                             data={"camera_id": "cam_url_test",
                                   "url": "rtsp://fake/stream"})
        self.assertIn(r.status_code, [200, 400])  # 400 only if pipeline fails to start

    def test_vp_decode_status_unknown_camera(self):
        r = self.client.get("/api/v1/video-pipeline/decode/status/?camera_id=unknown_cam")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["status"], "stopped")
        self.assertEqual(data["frame_count"], 0)

    def test_vp_decode_stop(self):
        r = self.client.post("/api/v1/video-pipeline/decode/stop/",
                             data={"camera_id": "cam_stop_test"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "stopped")

    def test_vp_latest_frame_503(self):
        r = self.client.get("/api/v1/video-pipeline/latest-frame/?camera_id=1")
        self.assertEqual(r.status_code, 503)

    def test_vp_video_info_url(self):
        r = self.client.post("/api/v1/video-pipeline/video-info-url/",
                             data={"url": "rtsp://fake/stream"})
        self.assertEqual(r.status_code, 200)
        info = r.json()["info"]
        self.assertEqual(info["codec"], "h264")
        self.assertEqual(info["width"], 1920)

    def test_detections_no_pipeline_503(self):
        r = self.client.get("/shared/cameras/cam1/detections/latest")
        self.assertEqual(r.status_code, 503)

    def test_shared_cameras_empty(self):
        r = self.client.get("/shared/cameras")
        self.assertEqual(r.status_code, 200)
        self.assertIn("cameras", r.json())

    def test_stop_inference(self):
        r = self.client.post("/inference/continuous/stop?camera_id=cam1")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "stopped")


if __name__ == "__main__":
    unittest.main(verbosity=2)
