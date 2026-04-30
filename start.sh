#!/bin/bash
set -e
# Python FastAPI manages the C++ core_engine subprocess lifecycle.
# Single worker required — PipelineManager holds in-process ZMQ + detection state.
exec uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8001 \
    --workers 1 \
    --log-level info
