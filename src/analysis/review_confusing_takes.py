"""Generate visual take-review artifacts for confusing gesture pairs.

Usage:
    python -m src.analysis.review_confusing_takes --run-dir models/experiment_runs/<timestamp>/full_mlp
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec

from src.analysis.analyze_misclassifications import (
    ensure_traceability_columns,
    load_metadata_candidates,
    load_predictions,
    normalize_prediction_columns,
)

SKELETON_EDGES: list[tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (8, 12),
    (12, 13),
    (13, 14),
    (0, 15),
    (15, 17),
    (0, 16),
    (16, 18),
]


@dataclass
class SampleChoice:
    sample_type: str
    true_label: str
    predicted_label: str
    confidence_of_predicted_class: float | None
    confidence_of_true_class: float | None
    top2_predicted_label_name: str | None
    person: str | None
    session: str | None
    take: str | None
    short_sample_path: str
    sample_index: int | None
    source_row: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review confusing takes with visual side-by-side summaries.")
    parser.add_argument("--run-dir", required=True, type=str, help="Path to one experiment run folder.")
    parser.add_argument(
        "--top-pairs",
        type=int,
        default=6,
        help="Maximum number of top confusion pairs to build visual cases for.",
    )
    parser.add_argument(
        "--cases-per-pair",
        type=int,
        default=2,
        help="How many misclassified samples to review per confusion pair.",
    )
    parser.add_argument(
        "--frames-per-sample",
        type=int,
        default=7,
        help="Number of evenly spaced frames rendered per take.",
    )
    return parser.parse_args()


def _safe_float(value: Any) -> float | None:
    num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(num):
        return None
    return float(num)


def _as_str_or_none(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def _short_path(path_value: str | None) -> str:
    if not path_value:
        return "(missing)"
    p = Path(path_value)
    if p.is_absolute():
        try:
            return str(p.relative_to(Path.cwd()))
        except Exception:
            return str(p)
    return str(p)


def _resolve_take_path(row: pd.Series) -> Path | None:
    candidates = [_as_str_or_none(row.get("original_sample_path")), _as_str_or_none(row.get("sample_path"))]
    for raw in candidates:
        if not raw:
            continue
        normalized = raw.replace("\\", "/")
        path = Path(normalized)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if path.is_dir():
            return path
        if path.is_file() and path.suffix.lower() == ".json":
            return path.parent
    return None


def _load_pose_frames(sample_dir: Path, frame_count: int) -> list[np.ndarray]:
    json_files = sorted(sample_dir.glob("*.json"))
    if not json_files:
        return []

    sample_ix = np.linspace(0, len(json_files) - 1, num=min(frame_count, len(json_files)), dtype=int)
    selected = [json_files[i] for i in sample_ix]

    frames: list[np.ndarray] = []
    for json_path in selected:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        people = payload.get("people", [])
        if not people:
            frames.append(np.full((25, 3), np.nan, dtype=float))
            continue
        keypoints = people[0].get("pose_keypoints_2d", [])
        if len(keypoints) < 75:
            frames.append(np.full((25, 3), np.nan, dtype=float))
            continue
        arr = np.array(keypoints[:75], dtype=float).reshape(25, 3)
        frames.append(arr)
    return frames


def _collect_bounds(all_frames: list[list[np.ndarray]]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for sample_frames in all_frames:
        for frame in sample_frames:
            valid = np.isfinite(frame[:, 0]) & np.isfinite(frame[:, 1]) & (frame[:, 2] > 0.0)
            if not np.any(valid):
                continue
            xs.extend(frame[valid, 0].tolist())
            ys.extend(frame[valid, 1].tolist())

    if not xs or not ys:
        return (0.0, 1.0, 0.0, 1.0)

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad_x = max((xmax - xmin) * 0.12, 1.0)
    pad_y = max((ymax - ymin) * 0.12, 1.0)
    return (xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y)


def _draw_frame(ax: plt.Axes, frame: np.ndarray, bounds: tuple[float, float, float, float]) -> None:
    xmin, xmax, ymin, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#f7f7f7")

    valid = np.isfinite(frame[:, 0]) & np.isfinite(frame[:, 1]) & (frame[:, 2] > 0.0)
    for a, b in SKELETON_EDGES:
        if a < len(frame) and b < len(frame) and valid[a] and valid[b]:
            ax.plot(
                [frame[a, 0], frame[b, 0]],
                [frame[a, 1], frame[b, 1]],
                color="#1f77b4",
                linewidth=1.2,
            )

    if np.any(valid):
        ax.scatter(frame[valid, 0], frame[valid, 1], s=9, color="#222222")


def _pick_first_row(df: pd.DataFrame, filter_expr: pd.Series) -> pd.Series | None:
    subset = df[filter_expr]
    if subset.empty:
        return None
    return subset.iloc[0]


def _choose_references(df: pd.DataFrame, hardest_correct_df: pd.DataFrame, row: pd.Series) -> list[SampleChoice]:
    chosen: list[SampleChoice] = []

    true_label = str(row["true_label_name"])
    pred_label = str(row["predicted_label_name"])

    # Strong correct from true class.
    true_ref = _pick_first_row(
        df,
        (df["is_correct"]) & (df["true_label_name"] == true_label),
    )
    if true_ref is not None:
        chosen.append(_row_to_choice(true_ref, sample_type="strong_correct_true_class"))

    # Strong correct from confusing predicted class.
    pred_ref = _pick_first_row(
        df,
        (df["is_correct"]) & (df["true_label_name"] == pred_label),
    )
    if pred_ref is not None:
        chosen.append(_row_to_choice(pred_ref, sample_type="strong_correct_predicted_class"))

    # Borderline correct from either class.
    borderline = _pick_first_row(
        hardest_correct_df,
        hardest_correct_df["true_label_name"].isin([true_label, pred_label]),
    )
    if borderline is not None:
        chosen.append(_row_to_choice(borderline, sample_type="borderline_correct_reference"))

    return chosen


def _row_to_choice(row: pd.Series, sample_type: str) -> SampleChoice:
    return SampleChoice(
        sample_type=sample_type,
        true_label=str(row.get("true_label_name", "")),
        predicted_label=str(row.get("predicted_label_name", "")),
        confidence_of_predicted_class=_safe_float(row.get("confidence_of_predicted_class")),
        confidence_of_true_class=_safe_float(row.get("confidence_of_true_class")),
        top2_predicted_label_name=_as_str_or_none(row.get("top2_predicted_label_name")),
        person=_as_str_or_none(row.get("person")),
        session=_as_str_or_none(row.get("session")),
        take=_as_str_or_none(row.get("take")),
        short_sample_path=_short_path(_as_str_or_none(row.get("original_sample_path")) or _as_str_or_none(row.get("sample_path"))),
        sample_index=int(row["sample_index"]) if pd.notna(row.get("sample_index")) else None,
        source_row={k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()},
    )


def _render_case_image(image_path: Path, sample_choices: list[SampleChoice], frame_count: int) -> list[str]:
    loaded_frames: list[list[np.ndarray]] = []
    skipped_messages: list[str] = []

    for choice in sample_choices:
        sample_dir = _resolve_take_path(pd.Series(choice.source_row))
        if sample_dir is None:
            skipped_messages.append(f"Missing sample path for sample_index={choice.sample_index} ({choice.sample_type}).")
            loaded_frames.append([])
            continue
        frames = _load_pose_frames(sample_dir, frame_count)
        if not frames:
            skipped_messages.append(f"No JSON frames found in: {sample_dir}")
        loaded_frames.append(frames)

    bounds = _collect_bounds(loaded_frames)

    cols = len(sample_choices)
    fig = plt.figure(figsize=(min(5 * cols, 24), 7.5), constrained_layout=True)
    outer = gridspec.GridSpec(nrows=2, ncols=cols, height_ratios=[1.2, 3.6], figure=fig)

    for col_ix, choice in enumerate(sample_choices):
        text_ax = fig.add_subplot(outer[0, col_ix])
        text_ax.axis("off")

        meta_lines = [
            f"type: {choice.sample_type}",
            f"true: {choice.true_label}",
            f"pred: {choice.predicted_label}",
            f"pred_conf: {choice.confidence_of_predicted_class if choice.confidence_of_predicted_class is not None else 'n/a'}",
            f"true_conf: {choice.confidence_of_true_class if choice.confidence_of_true_class is not None else 'n/a'}",
            f"runner_up: {choice.top2_predicted_label_name or 'n/a'}",
            f"person/session/take: {choice.person or '-'} / {choice.session or '-'} / {choice.take or '-'}",
            f"path: {choice.short_sample_path}",
        ]
        text_ax.text(0.0, 1.0, "\n".join(meta_lines), va="top", ha="left", fontsize=8.6, family="monospace")

        frame_grid = outer[1, col_ix].subgridspec(1, max(frame_count, 1), wspace=0.05)
        frames = loaded_frames[col_ix]
        for frame_ix in range(frame_count):
            ax = fig.add_subplot(frame_grid[0, frame_ix])
            if frame_ix < len(frames):
                _draw_frame(ax, frames[frame_ix], bounds)
            else:
                ax.axis("off")

    fig.suptitle(image_path.stem.replace("_", " "), fontsize=12)
    fig.savefig(image_path, dpi=180)
    plt.close(fig)
    return skipped_messages


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text.lower()).strip("_")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    analysis_dir = run_dir / "misclassification_analysis"
    review_dir = run_dir / "take_review"
    review_dir.mkdir(parents=True, exist_ok=True)

    print(f"[take-review] Run directory: {run_dir}")
    print("[take-review] Loading prediction artifacts...")

    predictions_df = normalize_prediction_columns(load_predictions(run_dir))
    metadata_df = load_metadata_candidates()
    predictions_df, _ = ensure_traceability_columns(predictions_df, metadata_df)

    misclassified_path = analysis_dir / "misclassified_samples.csv"
    hardest_correct_path = analysis_dir / "hardest_correct_samples.csv"

    if misclassified_path.exists():
        misclassified_df = pd.read_csv(misclassified_path)
        print(f"[take-review] Using {misclassified_path}")
    else:
        misclassified_df = predictions_df[~predictions_df["is_correct"]].copy()
        print("[take-review][warn] misclassified_samples.csv missing; computed from predictions.csv")

    if hardest_correct_path.exists():
        hardest_correct_df = pd.read_csv(hardest_correct_path)
        print(f"[take-review] Using {hardest_correct_path}")
    else:
        hardest_correct_df = (
            predictions_df[predictions_df["is_correct"]]
            .copy()
            .sort_values(by=["confidence_of_predicted_class", "sample_index"], ascending=[True, True])
        )
        print("[take-review][warn] hardest_correct_samples.csv missing; computed from predictions.csv")

    predictions_df = predictions_df.sort_values(
        by=["is_correct", "confidence_of_predicted_class", "sample_index"],
        ascending=[True, False, True],
    )

    pair_counts = (
        misclassified_df.groupby(["true_label_name", "predicted_label_name"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    reviewed_cases: list[dict[str, Any]] = []
    review_rows_md: list[str] = []
    skipped_warnings: list[str] = []

    case_id = 1
    top_pairs = pair_counts.head(max(args.top_pairs, 1))
    print(f"[take-review] Reviewing top {len(top_pairs)} confusion pairs...")

    for _, pair in top_pairs.iterrows():
        true_label = str(pair["true_label_name"])
        pred_label = str(pair["predicted_label_name"])

        pair_mis = misclassified_df[
            (misclassified_df["true_label_name"] == true_label)
            & (misclassified_df["predicted_label_name"] == pred_label)
        ].sort_values(by=["confidence_of_predicted_class", "sample_index"], ascending=[False, True])

        for _, mis_row in pair_mis.head(max(args.cases_per_pair, 1)).iterrows():
            mis_choice = _row_to_choice(mis_row, sample_type="misclassified")
            refs = _choose_references(predictions_df, hardest_correct_df, mis_row)
            sample_choices = [mis_choice, *refs]

            case_name = f"review_case_{case_id:03d}_{_slug(true_label)}_vs_{_slug(pred_label)}.png"
            image_path = review_dir / case_name
            warnings = _render_case_image(image_path, sample_choices, frame_count=max(args.frames_per_sample, 1))
            skipped_warnings.extend(warnings)

            case_payload = {
                "case_id": case_id,
                "confusion_pair": {
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "pair_count": int(pair["count"]),
                },
                "image": str(image_path.relative_to(run_dir)),
                "samples": [asdict(choice) for choice in sample_choices],
                "warnings": warnings,
            }
            reviewed_cases.append(case_payload)
            review_rows_md.append(
                f"- Case {case_id:03d}: `{true_label}` → `{pred_label}` | image: `{image_path.name}` | "
                f"misclassified take: `{mis_choice.short_sample_path}`"
            )
            case_id += 1

    summary_json = {
        "run_dir": str(run_dir),
        "review_dir": str(review_dir),
        "total_misclassified_samples": int(len(misclassified_df)),
        "top_confusion_pairs": pair_counts.to_dict(orient="records"),
        "reviewed_cases": reviewed_cases,
        "warnings": skipped_warnings,
    }
    (review_dir / "review_summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    summary_md_lines = [
        "# Take Review Summary",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Total misclassified samples found: **{len(misclassified_df)}**",
        f"- Reviewed confusion pairs: **{len(top_pairs)}**",
        f"- Review images generated: **{len(reviewed_cases)}**",
        "",
        "## Top confusion pairs (by count)",
        "",
    ]

    if pair_counts.empty:
        summary_md_lines.append("No misclassified samples found.")
    else:
        for _, row in pair_counts.head(10).iterrows():
            summary_md_lines.append(
                f"- `{row['true_label_name']}` → `{row['predicted_label_name']}`: {int(row['count'])}"
            )

    summary_md_lines.extend([
        "",
        "## Reviewed cases",
        "",
    ])
    summary_md_lines.extend(review_rows_md or ["- No review cases generated."])

    summary_md_lines.extend([
        "",
        "## Short interpretation notes",
        "",
        "- Compare each misclassified take against strong correct examples from both the true and confusing classes.",
        "- Borderline correct references help identify whether the error looks like an ambiguous/weak execution.",
        "- Use person/session/take/path metadata shown in image headers for traceable follow-up.",
    ])

    if skipped_warnings:
        summary_md_lines.extend(["", "## Warnings", ""])
        summary_md_lines.extend([f"- {w}" for w in skipped_warnings])

    (review_dir / "review_summary.md").write_text("\n".join(summary_md_lines) + "\n", encoding="utf-8")

    # Optional contact-sheet dashboard.
    if reviewed_cases:
        thumb_paths = [run_dir / case["image"] for case in reviewed_cases[: min(12, len(reviewed_cases))]]
        cols = 3
        rows = int(np.ceil(len(thumb_paths) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows))
        axes_arr = np.array(axes).reshape(-1)
        for ax in axes_arr:
            ax.axis("off")
        for ix, image_path in enumerate(thumb_paths):
            try:
                img = plt.imread(image_path)
                axes_arr[ix].imshow(img)
                axes_arr[ix].set_title(image_path.name, fontsize=8)
            except Exception:
                axes_arr[ix].text(0.5, 0.5, f"Could not load\n{image_path.name}", ha="center", va="center")
        fig.suptitle("Top confusion visual dashboard", fontsize=13)
        fig.tight_layout()
        fig.savefig(review_dir / "top_confusions_contact_sheet.png", dpi=160)
        plt.close(fig)

    print(f"[take-review] Wrote visual review outputs to: {review_dir}")


if __name__ == "__main__":
    main()
