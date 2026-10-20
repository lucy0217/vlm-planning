#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/data/home/x_long32/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac}"
LMDEPLOY_BIN="${LMDEPLOY_BIN:-/data/home/x_long32/.conda/envs/lmdeploy/bin/lmdeploy}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="$NO_PROXY"

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "Missing local model: $MODEL_PATH" >&2
    exit 1
fi

exec "$LMDEPLOY_BIN" serve api_server "$MODEL_PATH" \
    --model-name Qwen/Qwen2-VL-7B-Instruct \
    --backend pytorch \
    --dtype float16 \
    --tp 1 \
    --session-len 12000 \
    --cache-max-entry-count 0.4 \
    --max-batch-size 1 \
    --server-name 127.0.0.1 \
    --server-port "${PORT:-23333}" "$@"
