"""Analyze misclassified test samples for a completed experiment run.

Usage examples:

- python -m src.analysis.analyze_misclassifications --run-dir models/experiment_runs/20260412_120000/full_mlp
- python -m src.analysis.analyze_misclassifications --suite-dir models/experiment_runs/20260412_120000
- python -m src.analysis.analyze_misclassifications --latest-suite-dir
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze misclassifications for one run folder.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Path to an experiment run folder (for example .../<timestamp>/full_mlp).",
    )
    parser.add_argument(
        "--suite-dir",
        type=str,
        default="",
        help=(
            "Optional convenience input: suite folder path (.../experiment_runs/<timestamp>). "
            "When used without --run-dir, the script analyzes <suite-dir>/<experiment-name>."
        ),
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="full_mlp",
        help="Experiment subfolder name used with --suite-dir (default: full_mlp).",
    )
    parser.add_argument(
        "--latest-suite-dir",
        action="store_true",
        help=(
            "Convenience flag: use the newest timestamped suite folder under models/experiment_runs "
            "and analyze its full_mlp subfolder (or --experiment-name if provided)."
        ),
    )
    return parser.parse_args()


def find_latest_suite_dir(experiment_runs_root: Path) -> Path:
    if not experiment_runs_root.exists():
        raise FileNotFoundError(f"Missing suite root: {experiment_runs_root}")

    candidates = [p for p in experiment_runs_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No suite directories found under: {experiment_runs_root}")

    return sorted(candidates, key=lambda p: p.name)[-1]


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        return Path(args.run_dir).expanduser().resolve()

    if args.latest_suite_dir:
        suite_dir = find_latest_suite_dir(Path("models/experiment_runs"))
        return (suite_dir / args.experiment_name).resolve()

    if args.suite_dir:
        suite_dir = Path(args.suite_dir).expanduser().resolve()
        return (suite_dir / args.experiment_name).resolve()

    raise ValueError("Provide --run-dir, or --suite-dir, or --latest-suite-dir.")


def load_predictions(run_dir: Path) -> pd.DataFrame:
    """Load best available predictions artifact from a run folder."""
    candidate_names = [
        "predictions.csv",
        "mlp_test_predictions.csv",
        "test_predictions.csv",
        "mlp_motion_test_predictions.csv",
        "lstm_motion_test_predictions.csv",
        "gru_motion_test_predictions.csv",
    ]

    for name in candidate_names:
        path = run_dir / name
        if path.exists():
            print(f"[analysis] Using predictions file: {path}")
            return pd.read_csv(path)

    raise FileNotFoundError(
        "No predictions file found in run folder. Expected one of: "
        f"{', '.join(candidate_names)}"
    )


def normalize_prediction_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    """Support both legacy and richer prediction export schemas."""
    df = predictions.copy()

    rename_map = {
        "y_true": "true_label_id",
        "y_pred": "predicted_label_id",
        "true_label": "true_label_name",
        "pred_label": "predicted_label_name",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    if "sample_index" not in df.columns:
        df["sample_index"] = range(len(df))
    if "split" not in df.columns:
        df["split"] = "test"

    required_names = ["true_label_name", "predicted_label_name"]
    missing_name_cols = [c for c in required_names if c not in df.columns]
    if missing_name_cols:
        raise ValueError(
            "Predictions file is missing required label-name columns: "
            f"{missing_name_cols}."
        )

    if "is_correct" not in df.columns:
        df["is_correct"] = df["true_label_name"] == df["predicted_label_name"]

    if "confidence_of_predicted_class" not in df.columns:
        # Legacy predictions files may not include confidence values.
        df["confidence_of_predicted_class"] = pd.NA

    return df


def build_confusions_by_pair(df: pd.DataFrame) -> pd.DataFrame:
    mis = df[~df["is_correct"]].copy()
    if mis.empty:
        return pd.DataFrame(columns=["true_label", "predicted_label", "count"])

    pair_counts = (
        mis.groupby(["true_label_name", "predicted_label_name"], dropna=False)
        .size()
        .reset_index(name="count")
        .rename(columns={"true_label_name": "true_label", "predicted_label_name": "predicted_label"})
        .sort_values("count", ascending=False)
    )
    return pair_counts


def summarize_by_class(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for class_name in sorted(df["true_label_name"].dropna().unique()):
        subset = df[df["true_label_name"] == class_name]
        total = int(len(subset))
        mistakes = int((~subset["is_correct"]).sum())
        rows.append(
            {
                "class_name": class_name,
                "total_samples": total,
                "mistake_count": mistakes,
                "mistake_rate": float(mistakes / total) if total else 0.0,
            }
        )

    return pd.DataFrame(rows).sort_values(["mistake_count", "class_name"], ascending=[False, True])


def generate_plots(analysis_dir: Path, class_summary: pd.DataFrame, pair_summary: pd.DataFrame) -> None:
    if not class_summary.empty:
        plt.figure(figsize=(10, 5))
        plt.bar(class_summary["class_name"], class_summary["mistake_count"])
        plt.title("Per-class misclassification counts")
        plt.xlabel("True class")
        plt.ylabel("Number of mistakes")
        plt.xticks(rotation=40, ha="right")
        plt.tight_layout()
        plt.savefig(analysis_dir / "per_class_error_bar.png", dpi=150)
        plt.close()

    if not pair_summary.empty:
        top_pairs = pair_summary.head(10).copy()
        labels = [f"{t} → {p}" for t, p in zip(top_pairs["true_label"], top_pairs["predicted_label"])]
        plt.figure(figsize=(11, 5))
        plt.bar(labels, top_pairs["count"])
        plt.title("Top confusion pairs")
        plt.xlabel("True → Predicted")
        plt.ylabel("Count")
        plt.xticks(rotation=40, ha="right")
        plt.tight_layout()
        plt.savefig(analysis_dir / "top_confusion_pairs_bar.png", dpi=150)
        plt.close()


def maybe_classification_report(df: pd.DataFrame) -> dict[str, Any]:
    try:
        report = classification_report(
            df["true_label_name"],
            df["predicted_label_name"],
            output_dict=True,
            zero_division=0,
        )
        return report
    except Exception:
        return {}


def write_summary_markdown(
    analysis_dir: Path,
    run_dir: Path,
    df: pd.DataFrame,
    pair_summary: pd.DataFrame,
    class_summary: pd.DataFrame,
    clf_report: dict[str, Any],
) -> None:
    total = len(df)
    incorrect = int((~df["is_correct"]).sum())
    correct = total - incorrect
    accuracy = (correct / total) if total else 0.0

    lines = [
        "# Misclassification Analysis",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Total test samples: **{total}**",
        f"- Correct: **{correct}**",
        f"- Incorrect: **{incorrect}**",
        f"- Test accuracy from predictions: **{accuracy:.4f}**",
        "",
        "## Top confusion pairs",
        "",
    ]

    if pair_summary.empty:
        lines.append("No misclassifications were found in this predictions file.")
    else:
        top_n = pair_summary.head(10)
        for _, row in top_n.iterrows():
            lines.append(f"- `{row['true_label']}` → `{row['predicted_label']}`: {int(row['count'])}")

    lines.extend(["", "## Class-by-class mistake breakdown", ""])
    for _, row in class_summary.iterrows():
        lines.append(
            f"- `{row['class_name']}`: mistakes={int(row['mistake_count'])} / total={int(row['total_samples'])} "
            f"(rate={float(row['mistake_rate']):.3f})"
        )

    lines.extend(["", "## Short interpretation notes", ""])
    lines.append("- Use `highest_confidence_errors.csv` to inspect likely label/representation mismatches.")
    lines.append("- Use `hardest_correct_samples.csv` to inspect borderline-but-correct examples.")
    lines.append(
        "- Use metadata fields (`person`, `session`, `take`, `sample_path`) to check if confusions cluster by take."
    )

    if clf_report:
        lines.extend(["", "## Precision / recall snapshot", ""])
        for class_name, metrics in clf_report.items():
            if not isinstance(metrics, dict):
                continue
            if class_name in {"macro avg", "weighted avg"}:
                continue
            precision = float(metrics.get("precision", 0.0))
            recall = float(metrics.get("recall", 0.0))
            support = int(metrics.get("support", 0))
            lines.append(
                f"- `{class_name}`: precision={precision:.3f}, recall={recall:.3f}, support={support}"
            )

    # Helpful trace examples for top confusion pairs when metadata exists.
    metadata_cols = [c for c in ["person", "session", "take", "sample_path"] if c in df.columns]
    if metadata_cols and not pair_summary.empty:
        lines.extend(["", "## Example take references for top confusion pairs", ""])
        for _, pair in pair_summary.head(5).iterrows():
            subset = df[
                (~df["is_correct"])
                & (df["true_label_name"] == pair["true_label"])
                & (df["predicted_label_name"] == pair["predicted_label"])
            ].head(3)
            lines.append(f"- `{pair['true_label']}` → `{pair['predicted_label']}`:")
            if subset.empty:
                lines.append("  - (no rows)")
                continue
            for _, row in subset.iterrows():
                reference_bits = []
                for col in metadata_cols:
                    value = row.get(col)
                    if pd.notna(value):
                        reference_bits.append(f"{col}={value}")
                if reference_bits:
                    lines.append(f"  - {', '.join(reference_bits)}")

    (analysis_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    print(f"[analysis] Run directory: {run_dir}")
    predictions = load_predictions(run_dir)
    df = normalize_prediction_columns(predictions)

    analysis_dir = run_dir / "misclassification_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    total = len(df)
    incorrect_mask = ~df["is_correct"]
    incorrect = int(incorrect_mask.sum())
    correct = total - incorrect
    accuracy = float(correct / total) if total else 0.0

    print(f"[analysis] Total test rows: {total}")
    print(f"[analysis] Incorrect rows: {incorrect}")

    pair_summary = build_confusions_by_pair(df)
    class_summary = summarize_by_class(df)

    misclassified = df[incorrect_mask].copy().sort_values(
        by=["confidence_of_predicted_class", "sample_index"],
        ascending=[False, True],
    )
    misclassified.to_csv(analysis_dir / "misclassified_samples.csv", index=False)

    highest_conf_errors = misclassified.head(50)
    highest_conf_errors.to_csv(analysis_dir / "highest_confidence_errors.csv", index=False)

    hardest_correct = (
        df[df["is_correct"]]
        .copy()
        .sort_values(by=["confidence_of_predicted_class", "sample_index"], ascending=[True, True])
        .head(50)
    )
    hardest_correct.to_csv(analysis_dir / "hardest_correct_samples.csv", index=False)

    pair_summary.to_csv(analysis_dir / "confusions_by_pair.csv", index=False)

    clf_report = maybe_classification_report(df)
    per_class_precision_recall: dict[str, dict[str, float]] = {}
    for label_name, metrics in clf_report.items():
        if not isinstance(metrics, dict):
            continue
        if label_name in {"macro avg", "weighted avg"}:
            continue
        per_class_precision_recall[label_name] = {
            "precision": float(metrics.get("precision", 0.0)),
            "recall": float(metrics.get("recall", 0.0)),
            "support": float(metrics.get("support", 0.0)),
        }

    summary_payload = {
        "run_dir": str(run_dir),
        "analysis_dir": str(analysis_dir),
        "total_test_samples": int(total),
        "total_correct": int(correct),
        "total_incorrect": int(incorrect),
        "overall_test_accuracy": accuracy,
        "most_common_confusion_pairs": pair_summary.head(10).to_dict(orient="records"),
        "per_class_mistake_counts": class_summary[["class_name", "mistake_count"]].to_dict(orient="records"),
        "per_class_precision_recall": per_class_precision_recall,
        "source_predictions_columns": list(df.columns),
    }
    (analysis_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    write_summary_markdown(
        analysis_dir=analysis_dir,
        run_dir=run_dir,
        df=df,
        pair_summary=pair_summary,
        class_summary=class_summary,
        clf_report=clf_report,
    )

    generate_plots(analysis_dir=analysis_dir, class_summary=class_summary, pair_summary=pair_summary)

    print(f"[analysis] Wrote outputs to: {analysis_dir}")


if __name__ == "__main__":
    main()
