#!/usr/bin/env bash
# Download NVIDIA TAO ReIdentificationNet ONNX (ResNet-50, Market-1501 + AI City 156).
# Uses deployable_v1.2 — plain ONNX, no ETLT decryption needed.
# Requires NGC CLI authenticated, or NGC_API_KEY env var.
#
# Run this ONCE on the host before building the Docker image.
# The Docker build copies models/ into the image; it does not re-download them.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST="${SCRIPT_DIR}/../models/reid"
mkdir -p "$DEST"
DEST="$(cd "$DEST" && pwd)"
MODEL_VERSION="deployable_v1.2"
ONNX_FILE="resnet50_market1501_aicity156.onnx"
NGC_MODEL="nvidia/tao/reidentificationnet:${MODEL_VERSION}"
NGC_API="https://api.ngc.nvidia.com/v2/models/nvidia/tao/reidentificationnet/versions/${MODEL_VERSION}/files"

if [ -f "${DEST}/${ONNX_FILE}" ]; then
  echo "[info] ${ONNX_FILE} already present in ${DEST}, skipping."
  exit 0
fi

if command -v ngc &>/dev/null; then
  echo "[info] Using NGC CLI to download ${NGC_MODEL}"
  ngc registry model download-version "${NGC_MODEL}" --dest "${DEST}"
  FOUND="$(find "${DEST}" -name "${ONNX_FILE}" -type f 2>/dev/null | head -n1)"
  if [ -z "${FOUND}" ]; then
    echo "[error] ${ONNX_FILE} not found after download."
    exit 1
  fi
  [ "${FOUND}" != "${DEST}/${ONNX_FILE}" ] && mv -f "${FOUND}" "${DEST}/"
  find "${DEST}" -mindepth 1 -type d -empty -delete 2>/dev/null || true
  echo "[ok] Model saved to ${DEST}/${ONNX_FILE}"
  exit 0
fi

if [ -z "${NGC_API_KEY:-}" ]; then
  echo "[error] NGC CLI not found and NGC_API_KEY is not set."
  echo "        Install NGC CLI or: export NGC_API_KEY=<key>"
  exit 1
fi

echo "[info] Downloading ${ONNX_FILE} via NGC REST API..."
wget --header="Authorization: ApiKey ${NGC_API_KEY}" \
     --progress=bar:force \
     -O "${DEST}/${ONNX_FILE}" \
     "${NGC_API}/${ONNX_FILE}"

echo "[ok] Model saved to ${DEST}/${ONNX_FILE}"
echo "NOTE: TRT engine auto-generated on first container start (~3 min on Orin NX), cached alongside the ONNX."
