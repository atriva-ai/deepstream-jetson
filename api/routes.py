from typing import Optional

from fastapi import APIRouter, HTTPException

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
        # If rtsp_url provided, register or update the camera
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

        # Start pipeline if it isn't running yet (first camera activation)
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

    return router
