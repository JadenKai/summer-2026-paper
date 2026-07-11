"""Generate seaborn plots from grading run results.

Reads results/runs.csv (written by grade_essays.py) and results/model_params.csv
(created/updated interactively). Produces three charts:
  - QWK by model (horizontal bar, sorted by parameter count)
  - Latency s/essay by model (horizontal bar, sorted by parameter count)
  - Parameter count vs QWK (scatter with model labels)

Usage:
    uv run python src/plot_results.py
    uv run python src/plot_results.py --model gpt-oss:20b --model glm-5.2:cloud
    uv run python src/plot_results.py --out results/plots/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO = Path(__file__).resolve().parents[1]
RUNS_CSV = REPO / "results" / "runs.csv"
PARAMS_CSV = REPO / "results" / "model_params.csv"


def load_params() -> pd.DataFrame:
    if PARAMS_CSV.exists():
        return pd.read_csv(PARAMS_CSV)
    return pd.DataFrame(columns=["model", "params_b"])


def save_params(df: pd.DataFrame) -> None:
    PARAMS_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(PARAMS_CSV, index=False)


def prompt_missing_params(models: list[str], params_df: pd.DataFrame) -> pd.DataFrame:
    known = set(params_df["model"])
    new_rows = []
    for model in models:
        if model not in known:
            while True:
                raw = input(f"Parameter count for '{model}' (billions, e.g. 7.0): ").strip()
                try:
                    val = float(raw)
                    new_rows.append({"model": model, "params_b": val})
                    break
                except ValueError:
                    print("  Enter a number, e.g. 7 or 230.5")
    if new_rows:
        params_df = pd.concat([params_df, pd.DataFrame(new_rows)], ignore_index=True)
        save_params(params_df)
    return params_df


def build_plot_df(runs: pd.DataFrame, params: pd.DataFrame) -> pd.DataFrame:
    # Average metrics across multiple runs of the same model.
    agg = (
        runs.groupby("model", as_index=False)
        .agg(qwk=("qwk", "mean"), s_per_essay=("s_per_essay", "mean"))
    )
    merged = agg.merge(params, on="model", how="left")
    return merged.sort_values("params_b").reset_index(drop=True)


def plot_qwk(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(df))))
    sns.barplot(data=df, x="qwk", y="model", orient="h", ax=ax, color="#4C72B0")
    ax.axvline(0.70, color="crimson", linewidth=1.2, linestyle="--", label="QWK = 0.70 threshold")
    ax.set_xlim(0, 1)
    ax.set_xlabel("QWK (quadratic weighted κ)")
    ax.set_ylabel("Model")
    ax.set_title("Eval Accuracy by Model")
    ax.legend(fontsize=8)
    sns.despine(left=True)
    fig.tight_layout()
    path = out_dir / "qwk_by_model.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_latency(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(df))))
    sns.barplot(data=df, x="s_per_essay", y="model", orient="h", ax=ax, color="#55A868")
    ax.set_xlabel("Seconds per essay")
    ax.set_ylabel("Model")
    ax.set_title("Eval Time by Model")
    sns.despine(left=True)
    fig.tight_layout()
    path = out_dir / "latency_by_model.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_params_vs_qwk(df: pd.DataFrame, out_dir: Path) -> None:
    df_valid = df.dropna(subset=["params_b"])
    if df_valid.empty:
        print("  Skipping params_vs_qwk — no param counts recorded yet.")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df_valid, x="params_b", y="qwk", ax=ax, s=80, color="#C44E52")
    for _, row in df_valid.iterrows():
        ax.annotate(row["model"], (row["params_b"], row["qwk"]),
                    textcoords="offset points", xytext=(6, 0), fontsize=7, va="center")
    ax.axhline(0.70, color="crimson", linewidth=1.0, linestyle="--", label="QWK = 0.70 threshold")
    ax.set_xscale("log")
    ax.set_xlabel("Parameter count (billions, log scale)")
    ax.set_ylabel("QWK")
    ax.set_title("Parameter Count vs Eval Accuracy")
    ax.legend(fontsize=8)
    sns.despine()
    fig.tight_layout()
    path = out_dir / "params_vs_qwk.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot grading run results.")
    parser.add_argument("--model", action="append", dest="models",
                        help="Filter to specific model(s) (repeatable); default = all")
    parser.add_argument("--out", default=str(REPO / "results" / "plots"),
                        help="Output directory for PNG files")
    args = parser.parse_args()

    if not RUNS_CSV.exists():
        sys.exit(f"No runs found at {RUNS_CSV} — run grade_essays.py first.")

    runs = pd.read_csv(RUNS_CSV)
    if args.models:
        runs = runs[runs["model"].isin(args.models)]
        if runs.empty:
            sys.exit(f"No runs found for models: {args.models}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = load_params()
    params = prompt_missing_params(runs["model"].unique().tolist(), params)

    sns.set_theme(style="whitegrid", font_scale=0.95)

    df = build_plot_df(runs, params)
    print(f"\nPlotting {len(df)} model(s):")
    print(df[["model", "params_b", "qwk", "s_per_essay"]].to_string(index=False))
    print()

    plot_qwk(df, out_dir)
    plot_latency(df, out_dir)
    plot_params_vs_qwk(df, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
