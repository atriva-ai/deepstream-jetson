import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .camera_registry import CameraRegistry
from .detection_store import DetectionStore
from .pipeline_manager import PipelineManager
from .routes import make_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

store = DetectionStore()
registry = CameraRegistry()
manager = PipelineManager(store, registry)


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager.start()
    yield
    manager.stop()


app = FastAPI(
    title="Edge Retail Intelligence — DeepStream Service",
    description="Unified video decode + AI inference service for NVIDIA Jetson",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(make_router(manager, registry, store))
