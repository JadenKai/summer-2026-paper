"""Score ASAP 2.0 essays with one LLM and report agreement with human scores.

Pipeline (single-model, no schema/parser ablation yet — this is the Q1 baseline):
  1. Sample N essays from dataset/ASAP2_0/train.csv (reproducible seed).
  2. For each essay, send the holistic rubric + essay text to one model and
     ask for an integer holistic score 1-6 via constrained JSON output.
  3. Compare predicted vs. human scores and report accuracy, adjacent accuracy,
     QWK (the ASAP standard metric), Cohen's kappa, MAE, and a confusion matrix.

Backend selection (first match wins unless --backend forces one):
  - ANTHROPIC_KEY set → Anthropic API (default model claude-opus-4-8)
  - OLLAMA_KEY set    → Ollama Cloud (default model gpt-oss:20b)
  - otherwise        → local Ollama server (OLLAMA_HOST, default localhost:11434)

  --backend vllm always opts in to a self-hosted vLLM OpenAI-compatible server
  (VLLM_HOST, default http://localhost:8000) — see src/pull_model.py to fetch
  weights from Hugging Face (HF_TOKEN) and serve_vllm.sh to launch the server.

Usage:
    uv run python src/grade_essays.py
    uv run python src/grade_essays.py --n 20 --model gpt-oss:20b --seed 42
    uv run python src/grade_essays.py --n 20 --model claude-opus-4-8 --seed 42
    uv run python src/grade_essays.py --n 20 --backend vllm --model Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from transformers import AutoProcessor, AutoModelForMultimodalLM, AutoModel, PreTrainedModel

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    mean_absolute_error,
)

MODEL_ID = "google/gemma-4-E2B-it"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto"
)

REPO = Path(__file__).resolve().parents[1]
TRAIN_CSV = REPO / "dataset" / "ASAP2_0" / "train.csv"
RUBRIC_MD = REPO / "dataset" / "ASAP2_0" / "rubric_holistic.md"
RUNS_CSV = REPO / "results" / "runs.csv"

DEFAULT_CLOUD_MODEL = "gpt-oss:20b"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-8"
SCORE_MIN, SCORE_MAX = 1, 6

# JSON schema for Ollama's `format` arg (supports min/max constraints).
OLLAMA_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": SCORE_MIN, "maximum": SCORE_MAX},
        "rationale": {"type": "string"},
    },
    "required": ["score"],
}

# JSON schema for Anthropic structured output (min/max constraints not supported).
ANTHROPIC_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer"},
        "rationale": {"type": "string"},
    },
    "required": ["score", "rationale"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are an expert essay rater. Score the student essay holistically on an "
    "integer scale from 1 (minimum) to 6 (maximum) using ONLY the rubric provided. "
    "Return your answer as JSON with an integer 'score' and a one-sentence 'rationale'."
)

def build_prompt(rubric: str, essay: str) -> str:
    return (
        f"# Holistic Scoring Rubric\n\n{rubric}\n\n"
        f"---\n\n# Essay to score\n\n{essay}\n\n"
        f"---\n\nAssign one integer holistic score from {SCORE_MIN} to {SCORE_MAX}."
    )


def grade_one(model: PreTrainedModel, rubric: str, essay: str) -> tuple[int | None, str]:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a short joke about saving RAM."},
    ]

    # Process input
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    # Generate output
    outputs = model.generate(**inputs, max_new_tokens=1024)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    print(response)

    # Parse output
    processor.parse_response(response)
    print(model, rubric, essay)
    return (0, "")


def _parse_score(content: str) -> int | None:
    """Extract a valid score from raw model output using progressively looser strategies."""
    text = content.strip()

    def valid(s: int) -> bool:
        return SCORE_MIN <= s <= SCORE_MAX

    def extract_from_obj(obj: object) -> int | None:
        if isinstance(obj, (int, float)):
            s = int(obj)
            return s if valid(s) else None
        if isinstance(obj, dict):
            raw = obj.get("score") or obj.get("Score") or obj.get("SCORE")
            if raw is not None:
                try:
                    s = int(float(raw))
                    return s if valid(s) else None
                except (ValueError, TypeError):
                    pass
        return None

    # 1. Strip markdown fences, then try full JSON parse (case-insensitive key, float-safe).
    cleaned = re.sub(r"^```json\s*|^```\s*|```\s*$", "", text, flags=re.MULTILINE).strip()
    try:
        result = extract_from_obj(json.loads(cleaned))
        if result is not None:
            return result
    except json.JSONDecodeError:
        pass

    # 2. Find any JSON object anywhere in the response and parse it.
    for m in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
        try:
            result = extract_from_obj(json.loads(m.group()))
            if result is not None:
                return result
        except json.JSONDecodeError:
            continue

    # 3. Find "score": <number> anywhere in the text.
    m = re.search(r'"[Ss][Cc][Oo][Rr][Ee]"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
    if m:
        s = int(float(m.group(1)))
        if valid(s):
            return s

    # 4. Find a standalone digit 1–6 (word boundary) — catches plain "3" or "Score: 4".
    for m in re.finditer(r'(?<!\d)([1-6])(?!\d)', text):
        return int(m.group(1))

    return None

def report(df: pd.DataFrame, model: str, elapsed: float) -> dict:
    """Print agreement metrics and return them as a dict."""
    ok = df.dropna(subset=["pred"]).copy()
    ok["pred"] = ok["pred"].astype(int)
    y_true, y_pred = ok["score"].to_numpy(), ok["pred"].to_numpy()

    n_total, n_ok = len(df), len(ok)
    exact = accuracy_score(y_true, y_pred)
    adjacent = (abs(y_true - y_pred) <= 1).mean()
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=range(SCORE_MIN, SCORE_MAX + 1))
    kappa = cohen_kappa_score(y_true, y_pred, labels=range(SCORE_MIN, SCORE_MAX + 1))
    mae = mean_absolute_error(y_true, y_pred)
    bias = (y_pred - y_true).mean()

    print("\n" + "=" * 60)
    print(f"RESULTS  ·  model={model}  ·  N={n_total} ({n_ok} parsed)")
    print("=" * 60)
    print(f"  Exact accuracy      : {exact:.3f}")
    print(f"  Adjacent acc (±1)   : {adjacent:.3f}")
    print(f"  QWK (quadratic κ)   : {qwk:.3f}   <- ASAP standard metric")
    print(f"  Cohen's κ (unweighted): {kappa:.3f}")
    print(f"  MAE                 : {mae:.3f}")
    print(f"  Bias (mean pred-true): {bias:+.3f}")
    print(f"  Total wall time     : {elapsed:.1f}s  ({elapsed / n_total:.1f}s/essay)")

    labels = list(range(SCORE_MIN, SCORE_MAX + 1))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("\n  Confusion matrix (rows=human, cols=pred):")
    print("        " + "".join(f"{c:>4}" for c in labels))
    for label, row in zip(labels, cm):
        print(f"   h={label} " + "".join(f"{v:>4}" for v in row))
    if n_ok < n_total:
        print(f"\n  WARNING: {n_total - n_ok} essay(s) had unparseable output (excluded).")

    return dict(
        qwk=qwk, exact_acc=exact, adj_acc=adjacent,
        kappa=kappa, mae=mae, bias=bias,
        n_parsed=n_ok, n_total=n_total,
        s_per_essay=elapsed / n_total, elapsed_s=elapsed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade ASAP 2.0 essays with one LLM.")
    parser.add_argument("--n", type=int, default=20, help="Number of essays to sample")
    parser.add_argument("--model", help="Model to use (e.g. claude-opus-4-8, gpt-oss:20b)")
    parser.add_argument("--backend", choices=["anthropic", "ollama", "vllm"],
                        help="Force backend (default: anthropic if ANTHROPIC_KEY set, else ollama; "
                             "vllm targets VLLM_HOST, default http://localhost:8000)")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (reproducibility)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print raw model output for unparseable responses")
    args = parser.parse_args()

    if not TRAIN_CSV.exists():
        sys.exit(f"Missing {TRAIN_CSV} — download ASAP 2.0 first.")
    rubric = RUBRIC_MD.read_text(encoding="utf-8")

    df = pd.read_csv(TRAIN_CSV).sample(n=args.n, random_state=args.seed).reset_index(drop=True)
    print(f"Sampled {len(df)} essays (seed={args.seed}). Human score distribution:")
    print(df["score"].value_counts().sort_index().to_string())

    

    preds: list[int | None] = []
    start = time.perf_counter()
    for i, row in df.iterrows():
        try:
            score, _ = grade_one(model, rubric, row["full_text"])
            break
        except Exception as exc:  # noqa: BLE001 - keep going, mark as missing
            print(f"  [{i + 1:>2}/{len(df)}] {row['essay_id']}  ERROR: {exc}")
            preds.append(None)
            continue
        preds.append(score)
        flag = "" if score is not None else "  (unparseable)"
        print(f"  [{i + 1:>2}/{len(df)}] {row['essay_id']}  human={row['score']}  pred={score}{flag}")
    elapsed = time.perf_counter() - start

    df["pred"] = preds
    metrics = report(df, model, elapsed)

    RUNS_CSV.parent.mkdir(exist_ok=True)
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"),
           "model": model, "n": args.n, "seed": args.seed, **metrics}
    header = not RUNS_CSV.exists()
    pd.DataFrame([row]).to_csv(RUNS_CSV, mode="a", header=header, index=False)
    print(f"\nAppended run to {RUNS_CSV}")

    safe_name = model.replace(":", "-").replace("/", "-")
    essay_csv = REPO / "results" / "csv" / f"{safe_name}.csv"
    essay_csv.parent.mkdir(parents=True, exist_ok=True)
    df[["essay_id", "score", "pred"]].to_csv(essay_csv, index=False)
    print(f"Wrote per-essay predictions to {essay_csv}")


if __name__ == "__main__":
    main()
