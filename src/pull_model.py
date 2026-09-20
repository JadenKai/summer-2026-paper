"""Download a model's weights from Hugging Face for local vLLM serving.

Pre-staging a download is optional — `vllm serve <repo_id>` will pull weights
itself on first launch (using HF_TOKEN the same way). Use this script when you
want the download to happen ahead of time (large models, gated repos, slow/
shared connections) or want a pinned local path independent of vLLM's cache.

Reads HF_TOKEN from .env / the environment for gated or rate-limited repos;
public repos work without it.

Usage:
    uv run python src/pull_model.py openai/gpt-oss-20b
    uv run python src/pull_model.py Qwen/Qwen3-8B --revision main
    uv run python src/pull_model.py google/gemma-3-4b-it --out models_cache/gemma-3-4b-it
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = REPO / "models_cache"

# Skip original (fp32/bf16 .bin or .pt) checkpoint formats when safetensors are
# present, and skip non-inference artifacts — cuts download size substantially.
IGNORE_PATTERNS = ["*.bin", "*.pt", "*.h5", "*.msgpack", "*.ot", "original/*"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a model snapshot from Hugging Face.")
    parser.add_argument("repo_id", help="HF repo id, e.g. Qwen/Qwen3-8B")
    parser.add_argument("--revision", default=None, help="Git revision/commit to pin (default: main)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Destination dir (default: models_cache/<repo-name>)")
    parser.add_argument("--allow-bin", action="store_true",
                        help="Also fetch .bin/.pt checkpoints (default: safetensors only)")
    args = parser.parse_args()

    load_dotenv()
    token = os.getenv("HF_TOKEN") or None

    out_dir = args.out or (DEFAULT_CACHE_DIR / args.repo_id.split("/")[-1])
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"→ Downloading {args.repo_id}" + (f" @ {args.revision}" if args.revision else ""))
    print(f"→ Destination: {out_dir}")
    print(f"→ Auth: {'HF_TOKEN found' if token else 'no HF_TOKEN (public repos only)'}")

    try:
        path = snapshot_download(
            repo_id=args.repo_id,
            revision=args.revision,
            local_dir=out_dir,
            token=token,
            ignore_patterns=None if args.allow_bin else IGNORE_PATTERNS,
        )
    except Exception as exc:  # noqa: BLE001 - surface HF errors (auth/gating/network) directly
        sys.exit(f"Download failed: {exc}")

    print(f"\nDone. Model snapshot at: {path}")
    print(f"Serve it with:\n  ./serve_vllm.sh {path}")


if __name__ == "__main__":
    main()
