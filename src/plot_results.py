"""Generate seaborn plots from grading run results.

Reads results/runs.csv (written by grade_essays.py), the per-essay predictions in
results/csv/<model>.csv, and results/model_params.csv (created/updated interactively).
Produces three kinds of charts:
  - Per-model score distribution: for each ground-truth score (1-6), the distribution of
    the model's predicted scores, overlaid and color-coded (seaborn histogram + KDE).
    One PNG per model.
  - Metric table: model x {MAE, adjacent accuracy, parameters, eval time}.
  - MAE vs. parameter-count bubble plot (bubble size = eval time s/essay).

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
import numpy as np
import pandas as pd
import seaborn as sns

REPO = Path(__file__).resolve().parents[1]
RUNS_CSV = REPO / "results" / "runs.csv"
PARAMS_CSV = REPO / "results" / "model_params.csv"
ESSAY_CSV_DIR = REPO / "results" / "csv"

SCORE_MIN, SCORE_MAX = 1, 6
SCORES = list(range(SCORE_MIN, SCORE_MAX + 1))


def safe_model_name(model: str) -> str:
    """Match the sanitization grade_essays.py uses for per-essay CSV filenames."""
    return model.replace(":", "-").replace("/", "-")


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


def load_essay_preds(model: str) -> pd.DataFrame | None:
    """Load per-essay (score, pred) for a model, or None if the file is missing."""
    path = ESSAY_CSV_DIR / f"{safe_model_name(model)}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, usecols=["essay_id", "score", "pred"])
    df = df.dropna(subset=["pred"]).copy()
    df["score"] = df["score"].astype(int)
    df["pred"] = df["pred"].astype(int)
    return df


def build_plot_df(runs: pd.DataFrame, params: pd.DataFrame) -> pd.DataFrame:
    # Average metrics across multiple runs of the same model.
    agg = (
        runs.groupby("model", as_index=False)
        .agg(
            mae=("mae", "mean"),
            adj_acc=("adj_acc", "mean"),
            s_per_essay=("s_per_essay", "mean"),
        )
    )
    merged = agg.merge(params, on="model", how="left")
    return merged.sort_values("params_b").reset_index(drop=True)


def plot_score_distribution(model: str, preds: pd.DataFrame, out_dir: Path) -> None:
    """Grouped bars: one group per ground-truth score, 6 bars = predicted-score counts.

    Within each ground-truth group the model's predicted-score distribution is shown as
    6 side-by-side bars (colored by predicted score).
    """
    # Distinct qualitative colors so adjacent predicted scores are easy to tell apart.
    palette = sns.color_palette("deep", len(SCORES))
    n_pred = len(SCORES)
    group_width = 0.82
    bar_width = group_width / n_pred

    fig, ax = plt.subplots(figsize=(11, 6))
    group_totals = []
    for gi, g in enumerate(SCORES):  # x-axis groups = ground-truth score
        grp = preds[preds["score"] == g]
        group_totals.append(len(grp))
        counts = np.array([(grp["pred"] == p).sum() for p in SCORES], dtype=float)
        offsets = (np.arange(n_pred) - (n_pred - 1) / 2) * bar_width
        xs = gi + offsets
        for pi in range(n_pred):
            ax.bar(xs[pi], counts[pi], width=bar_width * 0.92,
                   color=palette[pi], edgecolor="white", linewidth=0.4,
                   label=str(SCORES[pi]) if gi == 0 else None)

    ax.set_xticks(range(n_pred))
    ax.set_xticklabels([f"{g}\n(n={t})" for g, t in zip(SCORES, group_totals)])
    ax.set_xlabel("Ground-truth score")
    ax.set_ylabel("Essay count")
    ax.set_title(f"Predicted-score distribution per ground-truth score — {model}")
    ax.legend(title="Predicted score", ncol=1, bbox_to_anchor=(1.01, 1), loc="upper left")
    sns.despine()
    fig.tight_layout()
    path = out_dir / f"score_dist_{safe_model_name(model)}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_human_vs_model(model: str, preds: pd.DataFrame, out_dir: Path) -> None:
    """Two bars per score: how many essays humans gave each score vs. how many the model did."""
    human_counts = np.array([(preds["score"] == s).sum() for s in SCORES], dtype=float)
    model_counts = np.array([(preds["pred"] == s).sum() for s in SCORES], dtype=float)
    x = np.arange(len(SCORES))
    bar_width = 0.4

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - bar_width / 2, human_counts, width=bar_width,
           color="#4C72B0", edgecolor="white", linewidth=0.4, label="Human")
    ax.bar(x + bar_width / 2, model_counts, width=bar_width,
           color="#DD8452", edgecolor="white", linewidth=0.4, label="Model")

    ax.set_xticks(x)
    ax.set_xticklabels(SCORES)
    ax.set_xlabel("Score")
    ax.set_ylabel("Essay count")
    ax.set_title(f"Human vs. model score counts — {model}")
    ax.legend(title="Assigned by")
    sns.despine()
    fig.tight_layout()
    path = out_dir / f"human_vs_model_{safe_model_name(model)}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_metric_table(df: pd.DataFrame, out_dir: Path) -> None:
    columns = ["MAE", "Adjacent accuracy", "Parameters (B)", "Eval time (s/essay)"]

    def fmt_params(v: float) -> str:
        return "—" if pd.isna(v) else f"{v:.1f}"

    cell_text = [
        [
            f"{row['mae']:.3f}",
            f"{row['adj_acc']:.3f}",
            fmt_params(row["params_b"]),
            f"{row['s_per_essay']:.2f}",
        ]
        for _, row in df.iterrows()
    ]
    row_labels = df["model"].tolist()

    csv_df = pd.DataFrame(cell_text, columns=columns)
    csv_df.insert(0, "Model", row_labels)
    csv_path = out_dir / "metrics_table.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")

    fig, ax = plt.subplots(figsize=(9, max(1.5, 0.55 * len(df) + 0.8)))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=columns,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    # Bold the header row and row labels.
    for (r, _c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight="bold")
    ax.set_title("Model metrics", pad=12)
    fig.tight_layout()
    path = out_dir / "metrics_table.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_mae_params_bubble(df: pd.DataFrame, out_dir: Path) -> None:
    df_valid = df.dropna(subset=["params_b"])
    if df_valid.empty:
        print("  Skipping mae_vs_params_bubble — no param counts recorded yet.")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df_valid,
        x="mae",
        y="params_b",
        size="s_per_essay",
        sizes=(40, 400),
        color="#4C72B0",
        alpha=0.7,
        legend="brief",
        ax=ax,
    )
    for _, row in df_valid.iterrows():
        ax.annotate(
            row["model"],
            (row["mae"], row["params_b"]),
            textcoords="offset points",
            xytext=(6, 0),
            fontsize=7,
            va="center",
        )
    ax.set_yscale("log")
    ax.set_xlabel("MAE (mean absolute error)")
    ax.set_ylabel("Parameter count (billions, log scale)")
    ax.set_title("MAE vs. Parameter Count (bubble size = eval time s/essay)")
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title("Eval time (s/essay)")
    sns.despine()
    fig.tight_layout()
    path = out_dir / "mae_vs_params_bubble.png"
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
    print(df[["model", "params_b", "mae", "adj_acc", "s_per_essay"]].to_string(index=False))
    print()

    print("Per-model score distributions:")
    for model in df["model"]:
        preds = load_essay_preds(model)
        if preds is None:
            print(f"  Skipping {model} — no per-essay CSV in {ESSAY_CSV_DIR}.")
            continue
        plot_score_distribution(model, preds, out_dir)
        plot_human_vs_model(model, preds, out_dir)

    plot_metric_table(df, out_dir)
    plot_mae_params_bubble(df, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
