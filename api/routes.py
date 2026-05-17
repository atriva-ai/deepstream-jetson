import asyncio
import glob
import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse, Response

from .camera_registry import CameraRegistry
from .detection_store import DetectionStore
from .pipeline_manager import PipelineManager

ARCHITECTURE = "deepstream"


def make_router(
    manager: PipelineManager,
    registry: CameraRegistry,
    store: DetectionStore,
) -> APIRouter:
    router = APIRouter()

    # ------------------------------------------------------------------
    # Health / info
    # ------------------------------------------------------------------

    @router.get("/")
    def root():
        return {"message": "DeepStream retail service", "architecture": ARCHITECTURE}

    @router.get("/model/info")
    def model_info():
        return {
            "architecture": ARCHITECTURE,
            "models": [{"name": "peoplenet+reid", "accelerator": "tensorrt"}],
            "accelerators": ["tensorrt"],
        }

    @router.get("/models")
    def list_models():
        return {"available_models": ["peoplenet+reid"]}

    # ------------------------------------------------------------------
    # Camera management
    # ------------------------------------------------------------------

    @router.get("/cameras")
    def list_cameras():
        return {"cameras": registry.list()}

    @router.post("/cameras/register")
    def register_camera(camera_id: str, rtsp_url: str):
        registry.register(camera_id, rtsp_url)
        return {"camera_id": camera_id, "status": "registered"}

    @router.delete("/cameras/{camera_id}")
    def unregister_camera(camera_id: str):
        removed = registry.unregister(camera_id)
        if not removed:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        return {"camera_id": camera_id, "status": "unregistered"}

    # ------------------------------------------------------------------
    # Inference control (matches backend polling contract)
    # ------------------------------------------------------------------

    @router.post("/inference/continuous/start")
    def start_inference(
        camera_id: str,
        rtsp_url: Optional[str] = None,
        model_name: Optional[str] = None,      # ignored — DeepStream uses PeopleNet
        accelerator: Optional[str] = None,      # ignored — TensorRT always
        object_filter: Optional[str] = "person",
        inference_fps: Optional[float] = None,  # ignored — DeepStream runs at stream FPS
    ):
        if rtsp_url:
            registry.register(camera_id, rtsp_url)
        elif not registry.get_rtsp_url(camera_id):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Camera {camera_id} has no registered RTSP URL. "
                    "Pass rtsp_url= or call POST /cameras/register first, "
                    "or set CAMERAS_JSON env var."
                ),
            )

        registry.activate(camera_id)

        if not manager.is_alive():
            manager.restart()

        return {
            "camera_id": camera_id,
            "status": "started",
            "architecture": ARCHITECTURE,
        }

    @router.post("/inference/continuous/stop")
    def stop_inference(camera_id: str):
        registry.deactivate(camera_id)
        return {"camera_id": camera_id, "status": "stopped"}

    @router.get("/inference/continuous/status")
    def inference_status(camera_id: str):
        return {
            "camera_id": camera_id,
            "running": manager.is_alive() and registry.is_active(camera_id),
            "architecture": ARCHITECTURE,
        }

    # ------------------------------------------------------------------
    # Detection results (backend polls this every second)
    # ------------------------------------------------------------------

    @router.get("/shared/cameras")
    def list_active_cameras():
        return {"cameras": store.cameras()}

    @router.get("/shared/cameras/{camera_id}/detections/latest")
    def latest_detections(camera_id: str, object_filter: Optional[str] = None):
        if not manager.is_alive():
            raise HTTPException(status_code=503, detail="Pipeline not running")
        frame = store.get(camera_id)
        if frame is None:
            raise HTTPException(
                status_code=404,
                detail=f"No detections available for camera {camera_id}",
            )
        return frame.to_response(object_filter)

    # ------------------------------------------------------------------
    # Video pipeline shim — DeepStream acts as both video pipeline and
    # AI inference service on Jetson. The backend calls VIDEO_PIPELINE_URL
    # to start/stop RTSP decode and check frame availability before it
    # calls AI_SERVICE_URL to start inference. These shim endpoints satisfy
    # that contract so the backend never needs a platform-specific code path.
    # ------------------------------------------------------------------

    @router.get("/api/v1/video-pipeline/health/")
    def vp_health():
        return {"status": "ok", "architecture": ARCHITECTURE}

    @router.get("/api/v1/video-pipeline/debug/")
    def vp_debug():
        return {
            "architecture": ARCHITECTURE,
            "pipeline_alive": manager.is_alive(),
            "cameras": registry.list(),
        }

    @router.post("/api/v1/video-pipeline/decode/")
    def vp_decode_start(
        camera_id: str = Form(...),
        url: Optional[str] = Form(None),
        fps: Optional[int] = Form(None),         # ignored — DeepStream runs at stream FPS
        force_format: Optional[str] = Form(None), # ignored — DeepStream uses NVDEC
    ):
        """Register the RTSP URL and start the DeepStream pipeline.

        The backend calls this when activating a camera. On Jetson, DeepStream
        owns RTSP decode natively so we register the URL and launch the pipeline
        instead of starting a separate ffmpeg process.
        """
        if url:
            registry.register(camera_id, url)
        elif not registry.get_rtsp_url(camera_id):
            raise HTTPException(
                status_code=400,
                detail=f"Camera {camera_id} has no RTSP URL. Pass url= in the request.",
            )

        registry.activate(camera_id)

        if not manager.is_alive():
            manager.restart()

        return {
            "status": "started",
            "camera_id": camera_id,
            "architecture": ARCHITECTURE,
            "message": "DeepStream pipeline started (NVDEC + TensorRT)",
        }

    @router.post("/api/v1/video-pipeline/decode/stop/")
    def vp_decode_stop(camera_id: str = Form(...)):
        """Deactivate a camera in the DeepStream pipeline."""
        registry.deactivate(camera_id)
        return {"status": "stopped", "camera_id": camera_id}

    @router.get("/api/v1/video-pipeline/decode/status/")
    def vp_decode_status(camera_id: str):
        """Return a status compatible with the backend's frame-count guard.

        The backend only starts AI inference after confirming frame_count > 0.
        We report frame_count=1 whenever the pipeline is alive and the camera
        is registered, so the backend proceeds to call /inference/continuous/start.
        """
        registered = registry.get_rtsp_url(camera_id) is not None
        alive = manager.is_alive() and registered
        return {
            "status": "running" if alive else "stopped",
            "camera_id": camera_id,
            "frame_count": 1 if alive else 0,
            "architecture": ARCHITECTURE,
        }

    _OSD_MKV = Path("/tmp/reid_annotated.mkv")

    @router.get("/api/v1/video-pipeline/latest-frame/")
    async def vp_latest_frame(camera_id: str):
        """Extract a JPEG thumbnail from the running DeepStream OSD recording.

        Reads /tmp/reid_annotated.mkv (written continuously by the pipeline)
        via gst-launch-1.0 + nvv4l2decoder → jpegenc → multifilesink.
        Returns 404 if camera not registered or pipeline not yet running.
        """
        if not registry.get_rtsp_url(camera_id):
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not registered")

        if not manager.is_alive() or not _OSD_MKV.exists():
            raise HTTPException(
                status_code=404,
                detail="Frame snapshot not available for this camera backend",
            )

        tmp_dir = tempfile.mkdtemp(prefix="ds_snap_")
        frame_pattern = os.path.join(tmp_dir, "%05d.jpg")

        cmd = [
            "gst-launch-1.0", "-e",
            "filesrc", f"location={_OSD_MKV}",
            "!", "matroskademux",
            "!", "h264parse",
            "!", "nvv4l2decoder",
            "!", "nvvideoconvert",
            "!", "video/x-raw,format=I420",
            "!", "jpegenc", "quality=85",
            "!", "multifilesink", f"location={frame_pattern}", "max-files=1",
        ]
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.communicate(), timeout=5.0)
        except asyncio.TimeoutError:
            if proc:
                try:
                    proc.kill()
                except Exception:
                    pass

        frames = sorted(glob.glob(os.path.join(tmp_dir, "*.jpg")))
        if not frames:
            try:
                os.rmdir(tmp_dir)
            except Exception:
                pass
            raise HTTPException(status_code=503, detail="Frame capture failed")

        with open(frames[-1], "rb") as f:
            jpeg_bytes = f.read()

        for fp in frames:
            try:
                os.unlink(fp)
            except Exception:
                pass
        try:
            os.rmdir(tmp_dir)
        except Exception:
            pass

        return Response(content=jpeg_bytes, media_type="image/jpeg")

    @router.post("/api/v1/video-pipeline/video-info-url/")
    def vp_video_info_url(url: str = Form(...)):
        """Return mock stream metadata so the backend can save video_info to the DB.

        DeepStream probes the stream internally; we return a plausible H.264 stub
        so the backend marks the camera as validated and continues.
        """
        return {
            "message": "Video info retrieved (DeepStream stub)",
            "info": {
                "codec": "h264",
                "width": 1920,
                "height": 1080,
                "fps": 30,
                "url": url,
                "source": "deepstream",
            },
        }

    return router
