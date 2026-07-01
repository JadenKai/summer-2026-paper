"""Score ASAP 2.0 essays with one LLM and report agreement with human scores.

Pipeline (single-model, no schema/parser ablation yet — this is the Q1 baseline):
  1. Sample N essays from dataset/ASAP2_0/train.csv (reproducible seed).
  2. For each essay, send the holistic rubric + essay text to one Ollama model and
     ask for an integer holistic score 1-6 via constrained JSON output.
  3. Compare predicted vs. human scores and report accuracy, adjacent accuracy,
     QWK (the ASAP standard metric), Cohen's kappa, MAE, and a confusion matrix.

Uses Ollama Cloud when OLLAMA_KEY is set (default model gpt-oss:20b), else a local
Ollama server. Mirrors the connection logic in src/ollama_test.py.

Usage:
    uv run python src/grade_essays.py
    uv run python src/grade_essays.py --n 20 --model gpt-oss:20b --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from ollama import Client
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    mean_absolute_error,
)

REPO = Path(__file__).resolve().parents[1]
TRAIN_CSV = REPO / "dataset" / "ASAP2_0" / "train.csv"
RUBRIC_MD = REPO / "dataset" / "ASAP2_0" / "rubric_holistic.md"

DEFAULT_CLOUD_MODEL = "gpt-oss:20b"
SCORE_MIN, SCORE_MAX = 1, 6

# JSON schema handed to Ollama's `format` arg to constrain the output.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": SCORE_MIN, "maximum": SCORE_MAX},
        "rationale": {"type": "string"},
    },
    "required": ["score"],
}

SYSTEM_PROMPT = (
    "You are an expert essay rater. Score the student essay holistically on an "
    "integer scale from 1 (minimum) to 6 (maximum) using ONLY the rubric provided. "
    "Return your answer as JSON with an integer 'score' and a one-sentence 'rationale'."
)


def build_client() -> tuple[Client, bool]:
    """Return (client, is_cloud). Uses Ollama Cloud if OLLAMA_KEY is set."""
    load_dotenv()
    key = os.getenv("OLLAMA_KEY")
    if key:
        client = Client(
            host="https://ollama.com",
            headers={"Authorization": f"Bearer {key}"},
        )
        return client, True
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    return Client(host=host), False


def build_prompt(rubric: str, essay: str) -> str:
    return (
        f"# Holistic Scoring Rubric\n\n{rubric}\n\n"
        f"---\n\n# Essay to score\n\n{essay}\n\n"
        f"---\n\nAssign one integer holistic score from {SCORE_MIN} to {SCORE_MAX}."
    )


def grade_one(client: Client, model: str, rubric: str, essay: str) -> tuple[int | None, str]:
    """Return (predicted_score, raw_content). score is None if unparseable."""
    resp = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(rubric, essay)},
        ],
        format=RESPONSE_SCHEMA,
        options={"temperature": 0},
    )
    content = resp["message"]["content"].strip()
    try:
        score = int(json.loads(content)["score"])
        if SCORE_MIN <= score <= SCORE_MAX:
            return score, content
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        pass
    return None, content


def report(df: pd.DataFrame, model: str, elapsed: float) -> None:
    """Print agreement metrics for rows with a valid prediction."""
    ok = df.dropna(subset=["pred"]).copy()
    ok["pred"] = ok["pred"].astype(int)
    y_true, y_pred = ok["score"].to_numpy(), ok["pred"].to_numpy()

    n_total, n_ok = len(df), len(ok)
    exact = accuracy_score(y_true, y_pred)
    adjacent = (abs(y_true - y_pred) <= 1).mean()
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=range(SCORE_MIN, SCORE_MAX + 1))
    kappa = cohen_kappa_score(y_true, y_pred, labels=range(SCORE_MIN, SCORE_MAX + 1))
    mae = mean_absolute_error(y_true, y_pred)

    print("\n" + "=" * 60)
    print(f"RESULTS  ·  model={model}  ·  N={n_total} ({n_ok} parsed)")
    print("=" * 60)
    print(f"  Exact accuracy      : {exact:.3f}")
    print(f"  Adjacent acc (±1)   : {adjacent:.3f}")
    print(f"  QWK (quadratic κ)   : {qwk:.3f}   <- ASAP standard metric")
    print(f"  Cohen's κ (unweighted): {kappa:.3f}")
    print(f"  MAE                 : {mae:.3f}")
    print(f"  Bias (mean pred-true): {(y_pred - y_true).mean():+.3f}")
    print(f"  Total wall time     : {elapsed:.1f}s  ({elapsed / n_total:.1f}s/essay)")

    labels = list(range(SCORE_MIN, SCORE_MAX + 1))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("\n  Confusion matrix (rows=human, cols=pred):")
    print("        " + "".join(f"{c:>4}" for c in labels))
    for label, row in zip(labels, cm):
        print(f"   h={label} " + "".join(f"{v:>4}" for v in row))
    if n_ok < n_total:
        print(f"\n  WARNING: {n_total - n_ok} essay(s) had unparseable output (excluded).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade ASAP 2.0 essays with one LLM.")
    parser.add_argument("--n", type=int, default=20, help="Number of essays to sample")
    parser.add_argument("--model", help=f"Ollama model (default {DEFAULT_CLOUD_MODEL} on cloud)")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (reproducibility)")
    parser.add_argument("--out", help="Optional CSV path to write per-essay predictions")
    args = parser.parse_args()

    if not TRAIN_CSV.exists():
        sys.exit(f"Missing {TRAIN_CSV} — download ASAP 2.0 first.")
    rubric = RUBRIC_MD.read_text(encoding="utf-8")

    df = pd.read_csv(TRAIN_CSV).sample(n=args.n, random_state=args.seed).reset_index(drop=True)
    print(f"Sampled {len(df)} essays (seed={args.seed}). Human score distribution:")
    print(df["score"].value_counts().sort_index().to_string())

    client, is_cloud = build_client()
    model = args.model or (DEFAULT_CLOUD_MODEL if is_cloud else None)
    if model is None:
        sys.exit("No local model resolved — pass --model or set OLLAMA_KEY for cloud.")
    print(f"\n→ {'Ollama Cloud' if is_cloud else 'local Ollama'} · model={model}\n")

    preds: list[int | None] = []
    start = time.perf_counter()
    for i, row in df.iterrows():
        try:
            score, _ = grade_one(client, model, rubric, row["full_text"])
        except Exception as exc:  # noqa: BLE001 - keep going, mark as missing
            print(f"  [{i + 1:>2}/{len(df)}] {row['essay_id']}  ERROR: {exc}")
            preds.append(None)
            continue
        preds.append(score)
        flag = "" if score is not None else "  (unparseable)"
        print(f"  [{i + 1:>2}/{len(df)}] {row['essay_id']}  human={row['score']}  pred={score}{flag}")
    elapsed = time.perf_counter() - start

    df["pred"] = preds
    report(df, model, elapsed)

    if args.out:
        out = Path(args.out)
        df[["essay_id", "score", "pred"]].to_csv(out, index=False)
        print(f"\nWrote per-essay predictions to {out}")


if __name__ == "__main__":
    main()
