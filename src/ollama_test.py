"""Smoke test for the Ollama API.

Sends a single prompt to a model and prints the reply, so we can confirm the
Ollama connection works before building the grading pipeline.

Targets Ollama Cloud (https://ollama.com) when an OLLAMA_KEY is present in the
environment / .env, otherwise falls back to a local Ollama server on
http://localhost:11434.

Usage:
    uv run python src/ollama_test.py                  # auto-pick a model
    uv run python src/ollama_test.py --model gpt-oss:20b
    uv run python src/ollama_test.py --prompt "Say hello in one sentence."
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv
from ollama import Client

# Default cloud model — small, fast, and a ★ required model for this study.
DEFAULT_CLOUD_MODEL = "gpt-oss:20b"
DEFAULT_PROMPT = "In one short sentence, confirm you are reachable and name your model."


def build_client() -> tuple[Client, bool]:
    """Return (client, is_cloud). Uses Ollama Cloud if OLLAMA_KEY is set."""
    load_dotenv()  # pulls OLLAMA_KEY from .env if present
    key = os.getenv("OLLAMA_KEY")

    if key:
        client = Client(
            host="https://ollama.com",
            headers={"Authorization": f"Bearer {key}"},
        )
        return client, True

    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    return Client(host=host), False


def pick_model(client: Client, is_cloud: bool, requested: str | None) -> str:
    """Resolve which model to use, preferring an explicit --model."""
    if requested:
        return requested
    if is_cloud:
        return DEFAULT_CLOUD_MODEL

    # Local: pick the first installed model.
    try:
        models = client.list().get("models", [])
    except Exception as exc:  # noqa: BLE001 - surface connection problems clearly
        sys.exit(f"Could not reach local Ollama server: {exc}")
    if not models:
        sys.exit(
            "No local models installed. Run e.g. `ollama pull gpt-oss:20b`, "
            "or set OLLAMA_KEY to use Ollama Cloud."
        )
    return models[0].get("model") or models[0]["name"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a test prompt to Ollama.")
    parser.add_argument("--model", help="Model name, e.g. gpt-oss:20b")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to send")
    args = parser.parse_args()

    client, is_cloud = build_client()
    model = pick_model(client, is_cloud, args.model)

    target = "Ollama Cloud" if is_cloud else "local Ollama"
    print(f"→ Using {target} · model={model}")
    print(f"→ Prompt: {args.prompt}\n")

    try:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": args.prompt}],
        )
    except Exception as exc:  # noqa: BLE001 - report the failure to the user
        sys.exit(f"Request failed: {exc}")

    print("← Response:")
    print(response["message"]["content"].strip())


if __name__ == "__main__":
    main()
