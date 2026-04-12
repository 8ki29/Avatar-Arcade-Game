"""Build a targeted dataset recollection + curation plan from run analysis artifacts.

Usage:
    python -m src.analysis.plan_recollection --run-dir models/experiment_runs/<timestamp>/full_mlp
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.analyze_misclassifications import (
    ensure_traceability_columns,
    load_metadata_candidates,
    load_predictions,
    normalize_prediction_columns,
)


@dataclass
class BoundaryStats:
    boundary: str
    class_a: str
    class_b: str
    confusion_count: int = 0
    high_conf_error_count: int = 0
    avg_wrong_confidence: float = 0.0
    borderline_correct_count: int = 0
    visual_review_case_count: int = 0
    visual_top_pair_count: int = 0
    review_md_mentions: int = 0
    score: float = 0.0


PRIORITY_COLUMNS = [
    "sample_index",
    "true_label_name",
    "predicted_label_name",
    "confidence_of_predicted_class",
    "confidence_of_true_class",
    "top2_predicted_label_name",
    "person",
    "session",
    "take",
    "original_sample_path",
    "reason_flagged",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a targeted recollection/curation plan from one run directory.")
    parser.add_argument("--run-dir", required=True, type=str, help="Path to one run folder (for example .../<timestamp>/full_mlp).")
    parser.add_argument(
        "--top-boundaries",
        type=int,
        default=5,
        help="Number of top-ranked boundaries included in prioritized recommendations (default: 5).",
    )
    return parser.parse_args()


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _boundary_pair(label_a: str, label_b: str) -> tuple[str, str]:
    a = _normalize_text(label_a)
    b = _normalize_text(label_b)
    return tuple(sorted([a, b]))


def _boundary_name(pair: tuple[str, str]) -> str:
    return f"{pair[0]} vs {pair[1]}"


def _coerce_confidences(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["confidence_of_predicted_class", "confidence_of_true_class", "top2_predicted_confidence"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _build_boundary_stats(
    misclassified_df: pd.DataFrame,
    high_conf_errors_df: pd.DataFrame,
    hardest_correct_df: pd.DataFrame,
    review_summary_json: dict[str, Any],
    review_summary_md: str,
) -> list[BoundaryStats]:
    stats: dict[tuple[str, str], BoundaryStats] = {}

    def ensure(pair: tuple[str, str]) -> BoundaryStats:
        if pair not in stats:
            stats[pair] = BoundaryStats(boundary=_boundary_name(pair), class_a=pair[0], class_b=pair[1])
        return stats[pair]

    if not misclassified_df.empty:
        grouped = misclassified_df.groupby(["true_label_name", "predicted_label_name"], dropna=False).size()
        for (true_label, pred_label), count in grouped.items():
            pair = _boundary_pair(str(true_label), str(pred_label))
            rec = ensure(pair)
            rec.confusion_count += int(count)

    if not high_conf_errors_df.empty:
        grouped = high_conf_errors_df.groupby(["true_label_name", "predicted_label_name"], dropna=False)
        for (true_label, pred_label), group in grouped:
            pair = _boundary_pair(str(true_label), str(pred_label))
            rec = ensure(pair)
            rec.high_conf_error_count += int(len(group))
            rec.avg_wrong_confidence = max(
                rec.avg_wrong_confidence,
                float(pd.to_numeric(group.get("confidence_of_predicted_class"), errors="coerce").mean() or 0.0),
            )

    if not hardest_correct_df.empty:
        hc = hardest_correct_df.copy()
        if "top2_predicted_label_name" in hc.columns:
            mask = hc["top2_predicted_label_name"].notna() & (hc["top2_predicted_label_name"].astype(str).str.len() > 0)
            borderline = hc[mask]
            for _, row in borderline.iterrows():
                pair = _boundary_pair(row.get("true_label_name"), row.get("top2_predicted_label_name"))
                rec = ensure(pair)
                rec.borderline_correct_count += 1

    for pair_row in review_summary_json.get("top_confusion_pairs", []):
        pair = _boundary_pair(pair_row.get("true_label_name"), pair_row.get("predicted_label_name"))
        rec = ensure(pair)
        rec.visual_top_pair_count += int(pair_row.get("count", 0) or 0)

    for case in review_summary_json.get("reviewed_cases", []):
        pair_data = case.get("confusion_pair", {})
        pair = _boundary_pair(pair_data.get("true_label"), pair_data.get("predicted_label"))
        rec = ensure(pair)
        rec.visual_review_case_count += 1

    for pair, rec in stats.items():
        md_hits = 0
        if review_summary_md:
            md_hits = review_summary_md.lower().count(pair[0].lower()) + review_summary_md.lower().count(pair[1].lower())
        rec.review_md_mentions = int(md_hits)

    for rec in stats.values():
        rec.score = (
            rec.confusion_count * 1.00
            + rec.high_conf_error_count * 1.35
            + rec.borderline_correct_count * 0.85
            + rec.visual_review_case_count * 1.25
            + rec.visual_top_pair_count * 0.55
            + rec.avg_wrong_confidence * 2.00
        )

    return sorted(stats.values(), key=lambda x: (-x.score, -x.confusion_count, -x.high_conf_error_count, x.boundary))


def _default_row() -> dict[str, Any]:
    return {c: "" for c in PRIORITY_COLUMNS}


def _row_key(row: dict[str, Any]) -> str:
    sample_index = _normalize_text(row.get("sample_index"))
    if sample_index:
        return f"idx:{sample_index}"
    return f"path:{_normalize_text(row.get('original_sample_path'))}:{_normalize_text(row.get('take'))}"


def _collect_priority_review_rows(
    predictions_df: pd.DataFrame,
    high_conf_errors_df: pd.DataFrame,
    hardest_correct_df: pd.DataFrame,
    review_summary_json: dict[str, Any],
    top_boundary_pairs: set[tuple[str, str]],
) -> pd.DataFrame:
    rows_by_key: dict[str, dict[str, Any]] = {}

    def upsert(source: dict[str, Any], reason: str) -> None:
        new_row = _default_row()
        for col in PRIORITY_COLUMNS:
            if col == "reason_flagged":
                continue
            if col in source:
                value = source.get(col)
                new_row[col] = "" if pd.isna(value) else value
        key = _row_key(new_row)
        if key not in rows_by_key:
            new_row["reason_flagged"] = reason
            rows_by_key[key] = new_row
        else:
            existing = set(filter(None, str(rows_by_key[key].get("reason_flagged", "")).split(";")))
            existing.add(reason)
            rows_by_key[key]["reason_flagged"] = ";".join(sorted(existing))

    if not high_conf_errors_df.empty:
        for _, row in high_conf_errors_df.iterrows():
            pair = _boundary_pair(row.get("true_label_name"), row.get("predicted_label_name"))
            if pair in top_boundary_pairs:
                upsert(row.to_dict(), "high_confidence_error")

    if not hardest_correct_df.empty:
        for _, row in hardest_correct_df.iterrows():
            top2 = _normalize_text(row.get("top2_predicted_label_name"))
            if not top2:
                continue
            pair = _boundary_pair(row.get("true_label_name"), top2)
            if pair in top_boundary_pairs:
                margin = None
                pred_conf = pd.to_numeric(pd.Series([row.get("confidence_of_predicted_class")]), errors="coerce").iloc[0]
                top2_conf = pd.to_numeric(pd.Series([row.get("top2_predicted_confidence")]), errors="coerce").iloc[0]
                if pd.notna(pred_conf) and pd.notna(top2_conf):
                    margin = float(pred_conf - top2_conf)
                if margin is None or margin <= 0.12:
                    upsert(row.to_dict(), "borderline_correct_near_boundary")

    for case in review_summary_json.get("reviewed_cases", []):
        samples = case.get("samples", [])
        for sample in samples:
            src = sample.get("source_row") or {}
            pair = _boundary_pair(sample.get("true_label"), sample.get("predicted_label"))
            if pair in top_boundary_pairs:
                upsert(src, "reviewed_visual_ambiguity")
                if sample.get("sample_type") == "borderline_correct_reference":
                    upsert(src, "likely_definition_overlap")

    # If join keys were sparse in artifacts, backfill from predictions metadata where possible.
    out = pd.DataFrame(rows_by_key.values())
    if out.empty:
        return pd.DataFrame(columns=PRIORITY_COLUMNS)

    if "sample_index" in out.columns and "sample_index" in predictions_df.columns:
        lookup_cols = [
            c
            for c in ["sample_index", "person", "session", "take", "original_sample_path", "top2_predicted_label_name"]
            if c in predictions_df.columns
        ]
        lookup = predictions_df[lookup_cols].drop_duplicates(subset=["sample_index"], keep="first")
        out = out.merge(lookup, on="sample_index", how="left", suffixes=("", "_pred"))
        for col in ["person", "session", "take", "original_sample_path", "top2_predicted_label_name"]:
            pred_col = f"{col}_pred"
            if pred_col in out.columns:
                out[col] = out[col].where(out[col].astype(str).str.len() > 0, out[pred_col])
                out = out.drop(columns=[pred_col])

    out = out.reindex(columns=PRIORITY_COLUMNS)
    out = out.sort_values(by=["reason_flagged", "confidence_of_predicted_class"], ascending=[True, False], na_position="last")
    return out.reset_index(drop=True)


def _instructions_for_boundary(class_a: str, class_b: str) -> list[str]:
    pair = {class_a, class_b}
    if {"attack_fire", "defense_fire"} == pair:
        return [
            "attack_fire: use a committed, directional strike shape (clear forward intent, visible extension).",
            "defense_fire: use a guarded, held blocking shape (stable posture, reduced forward reach).",
            "Record short instruction reminders before each take so performers keep class intent distinct.",
        ]
    if {"defense_earth", "idle"} == pair:
        return [
            "idle: stay neutral, relaxed, and non-elemental (avoid defensive arm structure).",
            "defense_earth: emphasize a grounded defensive frame (clear brace/guard, not relaxed standing).",
            "Use a 1-2 second neutral reset between takes to reduce drift from previous gestures.",
        ]
    return [
        f"Clarify visual intent between {class_a} and {class_b} with one class-specific cue per take.",
        "Coach performers to exaggerate the discriminative pose in the first second of execution.",
    ]


def _suggest_take_count(priority_rank: int) -> int:
    if priority_rank == 1:
        return 10
    if priority_rank == 2:
        return 8
    return 6


def _build_recollect_targets(boundaries: list[BoundaryStats], top_n: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rank, boundary in enumerate(boundaries[: max(top_n, 1)], start=1):
        notes = " ".join(_instructions_for_boundary(boundary.class_a, boundary.class_b))
        for true_class, contrast in [(boundary.class_a, boundary.class_b), (boundary.class_b, boundary.class_a)]:
            rows.append(
                {
                    "priority_rank": rank,
                    "focus_boundary": boundary.boundary,
                    "recommended_true_class": true_class,
                    "recommended_contrast_class": contrast,
                    "suggested_number_of_new_takes": _suggest_take_count(rank),
                    "rationale": (
                        f"High-impact boundary from confusion/error review. "
                        f"Signals: confusion_count={boundary.confusion_count}, "
                        f"high_conf_errors={boundary.high_conf_error_count}, "
                        f"borderline_correct={boundary.borderline_correct_count}."
                    ),
                    "performer_instruction_notes": notes,
                }
            )
    return pd.DataFrame(rows)


def _write_priority_plot(plan_dir: Path, boundaries: list[BoundaryStats]) -> None:
    if not boundaries:
        return
    labels = [b.boundary for b in boundaries[:8]]
    scores = [b.score for b in boundaries[:8]]
    plt.figure(figsize=(10, 4.5))
    bars = plt.barh(labels[::-1], scores[::-1], color="#4C78A8")
    plt.xlabel("Priority score")
    plt.title("Boundary priority for recollection")
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.08, bar.get_y() + bar.get_height() / 2, f"{width:.1f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(plan_dir / "boundary_priority_bar.png", dpi=160)
    plt.close()


def _write_markdown_plan(
    run_dir: Path,
    plan_dir: Path,
    boundaries: list[BoundaryStats],
    review_takes_df: pd.DataFrame,
    recollect_targets_df: pd.DataFrame,
) -> None:
    top = boundaries[:5]
    lines: list[str] = [
        "# Recollection & Curation Plan",
        "",
        "## Current conclusion",
        "",
        "- The current best model conclusion is unchanged: **pose-only full_mlp remains strong**.",
        "- Remaining errors are concentrated into a small number of boundaries, not broad random failure.",
        "- This phase is intentionally a **dataset-refinement planning step** (not a model-training step).",
        "",
        "## Priority boundaries",
        "",
        "Boundary prioritization score combines: confusion count, confidence of wrong predictions, borderline-correct proximity, and visual take-review signals.",
        "",
        "| Rank | Boundary | Score | Confusions | High-conf errors | Borderline correct | Visual cases |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]

    for rank, row in enumerate(top, start=1):
        lines.append(
            f"| {rank} | `{row.boundary}` | {row.score:.2f} | {row.confusion_count} | {row.high_conf_error_count} | {row.borderline_correct_count} | {row.visual_review_case_count} |"
        )

    lines.extend([
        "",
        "## Flagged takes to review",
        "",
        f"- Total priority review rows: **{len(review_takes_df)}** (see `priority_review_takes.csv`).",
        "- Rows include high-confidence errors, borderline-correct near-boundary samples, and visual-review-emphasized takes.",
        "",
        "## Proposed recollection targets",
        "",
        f"- Targets are intentionally modest and focused (see `priority_recollect_targets.csv`).",
        "- Start with top-ranked boundaries before collecting broader new data.",
        "",
        "| Priority | Focus boundary | Class to recollect | Contrast class | New takes |",
        "|---:|---|---|---|---:|",
    ])

    for _, row in recollect_targets_df.head(10).iterrows():
        lines.append(
            f"| {int(row['priority_rank'])} | `{row['focus_boundary']}` | `{row['recommended_true_class']}` | `{row['recommended_contrast_class']}` | {int(row['suggested_number_of_new_takes'])} |"
        )

    lines.extend([
        "",
        "## Performer instruction notes",
        "",
    ])

    for boundary in boundaries[:3]:
        lines.append(f"### {boundary.boundary}")
        for note in _instructions_for_boundary(boundary.class_a, boundary.class_b):
            lines.append(f"- {note}")
        lines.append("")

    lines.extend([
        "## Suggested curation workflow",
        "",
        "1. Manually inspect the top flagged rows in `priority_review_takes.csv`.",
        "2. For each flagged take, decide: keep as-is, relabel candidate, or exclude candidate for future training sets.",
        "3. Track any repeated performer/session clusters and prioritize those for corrective recollection.",
        "4. Keep a simple curation log (decision + reason) before modifying any dataset files.",
        "5. Re-run analysis after curation/recollection to verify boundary separation improved.",
        "",
        "## Next recommended action",
        "",
        "- Review top-ranked flagged takes first, then recollect a small batch for the top 1-2 boundaries.",
        f"- Output directory: `{plan_dir.relative_to(run_dir)}` under run `{run_dir}`.",
    ])

    (plan_dir / "recollection_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    analysis_dir = run_dir / "misclassification_analysis"
    review_dir = run_dir / "take_review"
    plan_dir = run_dir / "recollection_plan"
    plan_dir.mkdir(parents=True, exist_ok=True)

    predictions_df = normalize_prediction_columns(load_predictions(run_dir))
    metadata_df = load_metadata_candidates()
    predictions_df, _ = ensure_traceability_columns(predictions_df, metadata_df)
    predictions_df = _coerce_confidences(predictions_df)

    misclassified_df = _coerce_confidences(_safe_read_csv(analysis_dir / "misclassified_samples.csv"))
    if misclassified_df.empty:
        misclassified_df = predictions_df[~predictions_df["is_correct"]].copy()

    high_conf_errors_df = _coerce_confidences(_safe_read_csv(analysis_dir / "highest_confidence_errors.csv"))
    if high_conf_errors_df.empty:
        high_conf_errors_df = misclassified_df.sort_values(
            by=["confidence_of_predicted_class", "sample_index"],
            ascending=[False, True],
        ).head(min(50, len(misclassified_df)))

    hardest_correct_df = _coerce_confidences(_safe_read_csv(analysis_dir / "hardest_correct_samples.csv"))
    if hardest_correct_df.empty:
        hardest_correct_df = predictions_df[predictions_df["is_correct"]].copy().sort_values(
            by=["confidence_of_predicted_class", "sample_index"],
            ascending=[True, True],
        ).head(min(120, len(predictions_df)))

    review_summary_json = _safe_read_json(review_dir / "review_summary.json")
    review_summary_md = ""
    review_summary_md_path = review_dir / "review_summary.md"
    if review_summary_md_path.exists():
        review_summary_md = review_summary_md_path.read_text(encoding="utf-8")

    boundaries = _build_boundary_stats(
        misclassified_df=misclassified_df,
        high_conf_errors_df=high_conf_errors_df,
        hardest_correct_df=hardest_correct_df,
        review_summary_json=review_summary_json,
        review_summary_md=review_summary_md,
    )

    top_boundaries = boundaries[: max(args.top_boundaries, 1)]
    top_pairs = {_boundary_pair(b.class_a, b.class_b) for b in top_boundaries}

    review_takes_df = _collect_priority_review_rows(
        predictions_df=predictions_df,
        high_conf_errors_df=high_conf_errors_df,
        hardest_correct_df=hardest_correct_df,
        review_summary_json=review_summary_json,
        top_boundary_pairs=top_pairs,
    )
    recollect_targets_df = _build_recollect_targets(boundaries, top_n=max(args.top_boundaries, 1))

    review_takes_df.to_csv(plan_dir / "priority_review_takes.csv", index=False)
    recollect_targets_df.to_csv(plan_dir / "priority_recollect_targets.csv", index=False)

    _write_priority_plot(plan_dir, boundaries)
    _write_markdown_plan(run_dir, plan_dir, boundaries, review_takes_df, recollect_targets_df)

    plan_json = {
        "run_dir": str(run_dir),
        "plan_dir": str(plan_dir),
        "current_conclusion": {
            "best_model_status": "unchanged",
            "best_model": "pose-only full_mlp",
            "interpretation": "Model is strong; remaining errors are concentrated at specific class boundaries.",
        },
        "priority_boundaries": [boundary.__dict__ for boundary in boundaries],
        "priority_review_takes_file": str((plan_dir / "priority_review_takes.csv").relative_to(run_dir)),
        "priority_recollect_targets_file": str((plan_dir / "priority_recollect_targets.csv").relative_to(run_dir)),
        "plot_files": ["boundary_priority_bar.png"],
    }
    (plan_dir / "recollection_plan.json").write_text(json.dumps(plan_json, indent=2), encoding="utf-8")

    print(f"[recollection-plan] Wrote plan artifacts to: {plan_dir}")


if __name__ == "__main__":
    main()
