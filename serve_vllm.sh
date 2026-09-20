#!/usr/bin/env bash
# Launch a vLLM OpenAI-compatible server for one model, so grade_essays.py can
# grade against it with --backend vllm.
#
# MODEL can be a local snapshot dir (e.g. from src/pull_model.py) or a bare HF
# repo id (vLLM downloads it itself, using HF_TOKEN below, on first launch).
#
# Usage:
#   ./serve_vllm.sh Qwen/Qwen3-8B
#   ./serve_vllm.sh models_cache/gpt-oss-20b --port 8001
#   VLLM_PORT=8001 ./serve_vllm.sh openai/gpt-oss-20b -- --tensor-parallel-size 2
#
# Requires vLLM installed on this host (`pip install vllm` — GPU + CUDA build
# specific to your hardware; not a dependency of this project's uv env).
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model-path-or-hf-repo-id> [--port PORT] [-- extra vllm args...]" >&2
  exit 1
fi

MODEL="$1"; shift

PORT="${VLLM_PORT:-8000}"
if [[ "${1:-}" == "--port" ]]; then
  PORT="$2"; shift 2
fi

if [[ "${1:-}" == "--" ]]; then
  shift
fi

if ! command -v vllm >/dev/null 2>&1; then
  echo "error: 'vllm' not found on PATH. Install it in a GPU-capable env, e.g.:" >&2
  echo "  pip install vllm   # or: uv pip install vllm" >&2
  exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$REPO_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_DIR/.env"
  set +a
fi

echo "→ Serving $MODEL on http://0.0.0.0:${PORT}/v1"
exec vllm serve "$MODEL" --port "$PORT" "$@"
