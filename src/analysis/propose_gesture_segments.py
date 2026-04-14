"""Semi-automatic active gesture span proposal/review for OpenPose take folders.

This tool proposes `(active_start_frame, active_end_frame)` per take from motion energy,
then optionally lets the user review/override before writing a central manifest.

Examples:
    python -m src.analysis.propose_gesture_segments --take-dir data/raw/openpose_json/attack_air/luis/s01/take_001 --accept-auto
    python -m src.analysis.propose_gesture_segments --gesture attack_air --interactive
    python -m src.analysis.propose_gesture_segments --all --save-plots --plot-dir logs/analysis/gesture_segment_plots
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing.preprocess_constants import SELECTED_BODY25_INDICES, SEQUENCE_LENGTH
from src.utils.paths import load_paths_config, resolve_path

DEFAULT_PROPOSAL_METHOD = "motion_energy_v1"
DEFAULT_MANIFEST_NAME = "active_gesture_ranges.csv"


@dataclass
class TakeInfo:
    gesture: str
    person: str
    session: str
    take: str
    take_dir: Path


@dataclass
class SegmentProposal:
    active_start_frame: int
    active_end_frame: int
    threshold: float
    method: str = DEFAULT_PROPOSAL_METHOD


@dataclass
class SegmentRecord:
    gesture: str
    person: str
    session: str
    take: str
    take_path: str
    active_start_frame: int
    active_end_frame: int
    proposal_method: str
    label_status: str


MANIFEST_COLUMNS = [
    "gesture",
    "person",
    "session",
    "take",
    "take_path",
    "active_start_frame",
    "active_end_frame",
    "proposal_method",
    "label_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Propose and review active gesture ranges for OpenPose takes. "
            "Outputs one central CSV manifest with start/end labels per take."
        )
    )
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--take-dir", type=str, help="One take folder to process.")
    selector.add_argument("--gesture", type=str, help="Process all takes under one gesture folder.")
    selector.add_argument("--all", action="store_true", help="Process all takes under the OpenPose root.")

    parser.add_argument(
        "--openpose-root",
        type=str,
        default=None,
        help="Override OpenPose root (defaults to configs/paths.yaml -> openpose_raw_dir).",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help=(
            "Output CSV manifest path. Default: <openpose_root>/active_gesture_ranges.csv."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing manifest entries; otherwise existing takes are skipped.",
    )

    parser.add_argument("--start-frame", type=int, default=None, help="Manually set start frame.")
    parser.add_argument("--end-frame", type=int, default=None, help="Manually set end frame.")
    parser.add_argument(
        "--accept-auto",
        action="store_true",
        help="Accept automatic proposal without interactive prompt.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for accept/correct per take (ignored when --accept-auto is set).",
    )

    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Moving-average smoothing window for motion energy.",
    )
    parser.add_argument(
        "--threshold-multiplier",
        type=float,
        default=0.60,
        help="Threshold = median + multiplier * std over smoothed motion.",
    )
    parser.add_argument(
        "--min-active-run",
        type=int,
        default=3,
        help="Minimum consecutive active frames required for an active segment.",
    )

    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save per-take motion plot with proposal lines.",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Directory for saved plots (default: logs/analysis/gesture_segment_plots).",
    )

    return parser.parse_args()


def _resolve_openpose_root(openpose_root_arg: str | None) -> Path:
    if openpose_root_arg:
        return resolve_path(openpose_root_arg)
    paths_cfg = load_paths_config("configs/paths.yaml")
    return resolve_path(paths_cfg["openpose_raw_dir"])


def _discover_take_infos(openpose_root: Path, args: argparse.Namespace) -> list[TakeInfo]:
    if args.take_dir:
        return [_take_info_from_path(openpose_root, resolve_path(args.take_dir))]

    take_infos: list[TakeInfo] = []
    gesture_dirs = [openpose_root / args.gesture] if args.gesture else sorted(p for p in openpose_root.iterdir() if p.is_dir())

    for gesture_dir in gesture_dirs:
        if not gesture_dir.exists() or not gesture_dir.is_dir():
            continue
        gesture = gesture_dir.name
        for person_dir in sorted(p for p in gesture_dir.iterdir() if p.is_dir()):
            for session_dir in sorted(p for p in person_dir.iterdir() if p.is_dir()):
                for take_dir in sorted(p for p in session_dir.iterdir() if p.is_dir()):
                    take_infos.append(
                        TakeInfo(
                            gesture=gesture,
                            person=person_dir.name,
                            session=session_dir.name,
                            take=take_dir.name,
                            take_dir=take_dir,
                        )
                    )
    return take_infos


def _take_info_from_path(openpose_root: Path, take_dir: Path) -> TakeInfo:
    rel = take_dir.resolve().relative_to(openpose_root.resolve())
    parts = rel.parts
    if len(parts) < 4:
        raise ValueError(
            f"Take directory must follow gesture/person/session/take under {openpose_root}. Got: {take_dir}"
        )
    return TakeInfo(
        gesture=parts[0],
        person=parts[1],
        session=parts[2],
        take=parts[3],
        take_dir=take_dir,
    )


def _load_selected_joint_xy(frame_path: Path) -> np.ndarray:
    payload = json.loads(frame_path.read_text(encoding="utf-8"))
    people = payload.get("people", [])
    if not people:
        return np.full((len(SELECTED_BODY25_INDICES), 2), np.nan, dtype=np.float32)

    keypoints = people[0].get("pose_keypoints_2d", [])
    if len(keypoints) < 75:
        return np.full((len(SELECTED_BODY25_INDICES), 2), np.nan, dtype=np.float32)

    body25 = np.array(keypoints[:75], dtype=np.float32).reshape(25, 3)
    selected_xy = body25[SELECTED_BODY25_INDICES, :2]
    selected_conf = body25[SELECTED_BODY25_INDICES, 2]
    selected_xy[selected_conf <= 0.0] = np.nan
    return selected_xy


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    w = max(1, int(window))
    if w <= 1:
        return values.copy()
    kernel = np.ones(w, dtype=np.float32) / float(w)
    return np.convolve(values, kernel, mode="same")


def _fill_nans_1d(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    idx = np.arange(len(out))
    valid = np.isfinite(out)
    if not np.any(valid):
        return np.zeros_like(out)
    out[~valid] = np.interp(idx[~valid], idx[valid], out[valid])
    return out


def _enforce_min_run(mask: np.ndarray, min_run: int) -> np.ndarray:
    if min_run <= 1:
        return mask
    out = np.zeros_like(mask, dtype=bool)
    start: int | None = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        if (not value or i == len(mask) - 1) and start is not None:
            end = i if value and i == len(mask) - 1 else i - 1
            if end - start + 1 >= min_run:
                out[start : end + 1] = True
            start = None
    return out


def propose_segment(
    take_dir: Path,
    smooth_window: int,
    threshold_multiplier: float,
    min_active_run: int,
) -> tuple[SegmentProposal, np.ndarray, np.ndarray]:
    frame_paths = sorted(p for p in take_dir.iterdir() if p.is_file() and p.suffix == ".json")[:SEQUENCE_LENGTH]
    if not frame_paths:
        raise ValueError(f"No JSON frames found in take: {take_dir}")

    joints = [_load_selected_joint_xy(path) for path in frame_paths]
    n = len(joints)
    motion = np.zeros(n, dtype=np.float32)

    for idx in range(1, n):
        prev = joints[idx - 1]
        cur = joints[idx]
        valid = np.isfinite(prev[:, 0]) & np.isfinite(prev[:, 1]) & np.isfinite(cur[:, 0]) & np.isfinite(cur[:, 1])
        if not np.any(valid):
            motion[idx] = motion[idx - 1]
            continue
        deltas = np.linalg.norm(cur[valid] - prev[valid], axis=1)
        motion[idx] = float(np.mean(deltas))

    motion = _fill_nans_1d(motion)
    smoothed = _moving_average(motion, smooth_window)

    threshold = float(np.median(smoothed) + threshold_multiplier * np.std(smoothed))
    active_mask = smoothed >= threshold
    active_mask = _enforce_min_run(active_mask, min_active_run)

    if not np.any(active_mask):
        peak = int(np.argmax(smoothed))
        active_start = max(0, peak - 2)
        active_end = min(n - 1, peak + 2)
    else:
        active_indices = np.where(active_mask)[0]
        active_start = int(active_indices[0])
        active_end = int(active_indices[-1])

    proposal = SegmentProposal(
        active_start_frame=active_start,
        active_end_frame=active_end,
        threshold=threshold,
    )
    return proposal, motion, smoothed


def _make_plot(
    take_info: TakeInfo,
    raw_motion: np.ndarray,
    smoothed_motion: np.ndarray,
    proposal: SegmentProposal,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    x = np.arange(len(smoothed_motion))
    plt.plot(x, raw_motion, label="raw motion energy", alpha=0.45)
    plt.plot(x, smoothed_motion, label="smoothed motion energy", linewidth=2.0)
    plt.axhline(proposal.threshold, color="tab:orange", linestyle="--", label="threshold")
    plt.axvline(proposal.active_start_frame, color="tab:green", linestyle="--", label="active start")
    plt.axvline(proposal.active_end_frame, color="tab:red", linestyle="--", label="active end")

    plt.title(f"{take_info.gesture}/{take_info.person}/{take_info.session}/{take_info.take}")
    plt.xlabel("frame index")
    plt.ylabel("motion energy")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def _clamp_segment(start: int, end: int, frame_count: int) -> tuple[int, int]:
    start = int(np.clip(start, 0, frame_count - 1))
    end = int(np.clip(end, 0, frame_count - 1))
    if start > end:
        start, end = end, start
    return start, end


def _resolve_user_segment(
    proposal: SegmentProposal,
    frame_count: int,
    args: argparse.Namespace,
) -> tuple[int, int, str]:
    if args.start_frame is not None or args.end_frame is not None:
        if args.start_frame is None or args.end_frame is None:
            raise ValueError("When overriding frames manually, provide both --start-frame and --end-frame.")
        start, end = _clamp_segment(args.start_frame, args.end_frame, frame_count)
        return start, end, "manual_override"

    if args.accept_auto or not args.interactive:
        return proposal.active_start_frame, proposal.active_end_frame, "auto_accepted"

    print(
        f"  Auto proposal: start={proposal.active_start_frame}, end={proposal.active_end_frame}. "
        "Accept? [Y/n]"
    )
    reply = input().strip().lower()
    if reply in {"", "y", "yes"}:
        return proposal.active_start_frame, proposal.active_end_frame, "auto_accepted"

    print("  Enter corrected start frame:")
    start = int(input().strip())
    print("  Enter corrected end frame:")
    end = int(input().strip())
    start, end = _clamp_segment(start, end, frame_count)
    return start, end, "manual_adjusted"


def _record_key(take: TakeInfo) -> tuple[str, str, str, str]:
    return (take.gesture, take.person, take.session, take.take)


def _load_existing_records(manifest_path: Path) -> dict[tuple[str, str, str, str], SegmentRecord]:
    if not manifest_path.exists() or manifest_path.stat().st_size == 0:
        return {}

    existing: dict[tuple[str, str, str, str], SegmentRecord] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = SegmentRecord(
                gesture=row["gesture"],
                person=row["person"],
                session=row["session"],
                take=row["take"],
                take_path=row["take_path"],
                active_start_frame=int(row["active_start_frame"]),
                active_end_frame=int(row["active_end_frame"]),
                proposal_method=row["proposal_method"],
                label_status=row["label_status"],
            )
            existing[(record.gesture, record.person, record.session, record.take)] = record
    return existing


def _write_manifest(manifest_path: Path, records: list[SegmentRecord]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "gesture": record.gesture,
                "person": record.person,
                "session": record.session,
                "take": record.take,
                "take_path": record.take_path,
                "active_start_frame": record.active_start_frame,
                "active_end_frame": record.active_end_frame,
                "proposal_method": record.proposal_method,
                "label_status": record.label_status,
            })


def main() -> None:
    args = parse_args()
    openpose_root = _resolve_openpose_root(args.openpose_root)
    manifest_path = resolve_path(args.manifest_path) if args.manifest_path else (openpose_root / DEFAULT_MANIFEST_NAME)
    plot_dir = resolve_path(args.plot_dir) if args.plot_dir else resolve_path("logs/analysis/gesture_segment_plots")

    take_infos = _discover_take_infos(openpose_root, args)
    if not take_infos:
        print("No take folders found for selection.")
        return

    existing_records = _load_existing_records(manifest_path)
    merged_records = dict(existing_records)

    processed = 0
    skipped = 0
    for take in take_infos:
        key = _record_key(take)
        if key in merged_records and not args.overwrite:
            skipped += 1
            continue

        proposal, raw_motion, smoothed_motion = propose_segment(
            take_dir=take.take_dir,
            smooth_window=args.smooth_window,
            threshold_multiplier=args.threshold_multiplier,
            min_active_run=args.min_active_run,
        )
        frame_count = len(smoothed_motion)

        if args.save_plots:
            plot_name = f"{take.gesture}__{take.person}__{take.session}__{take.take}.png"
            _make_plot(
                take_info=take,
                raw_motion=raw_motion,
                smoothed_motion=smoothed_motion,
                proposal=proposal,
                output_path=plot_dir / plot_name,
            )

        start, end, label_status = _resolve_user_segment(proposal, frame_count, args)

        take_abs = take.take_dir.resolve()
        try:
            rel_take_path = str(take_abs.relative_to(Path.cwd().resolve()))
        except ValueError:
            rel_take_path = str(take_abs)
        merged_records[key] = SegmentRecord(
            gesture=take.gesture,
            person=take.person,
            session=take.session,
            take=take.take,
            take_path=rel_take_path,
            active_start_frame=start,
            active_end_frame=end,
            proposal_method=proposal.method,
            label_status=label_status,
        )

        processed += 1
        print(
            f"[{processed}] {take.gesture}/{take.person}/{take.session}/{take.take} "
            f"-> start={start}, end={end}, status={label_status}"
        )

    ordered_records = sorted(
        merged_records.values(),
        key=lambda r: (r.gesture, r.person, r.session, r.take),
    )
    _write_manifest(manifest_path, ordered_records)

    print(f"Wrote manifest: {manifest_path}")
    print(f"Processed: {processed} | Skipped(existing): {skipped} | Total tracked: {len(ordered_records)}")


if __name__ == "__main__":
    main()
