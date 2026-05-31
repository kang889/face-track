from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

REQUIRED_KEYS = {
    "session_label",
    "trial_index",
    "pressure_label",
    "source_local_index",
    "source_mp_id",
    "visible_mask",
    "blocked_mask",
    "neutral_cheek_points_px",
    "corrected_cheek_points_px",
    "cheek_displacement_xy",
    "visible_cheek_displacement_xy",
    "neutral_cheek_xyz",
    "current_cheek_xyz",
    "cheek_displacement_xyz",
    "visible_cheek_displacement_xyz",
    "point_motion_magnitude_visible_xy",
    "hand_present",
    "hand_landmarks_px",
    "hand_landmarks_xyz",
    "fingertip_px",
    "fingertip_xyz",
    "patch_edges",
    "left_cheek_ids",
    "anchor_alignment_error",
}


@dataclass
class SampleInfo:
    file: Path
    source_local_index: int
    source_mp_id: int
    trial_index: int
    pressure_label: str
    session_label: str
    visible_mask: np.ndarray
    blocked_mask: np.ndarray
    cheek_displacement_xy: np.ndarray
    cheek_displacement_xyz: np.ndarray
    visible_cheek_displacement_xy: np.ndarray
    visible_cheek_displacement_xyz: np.ndarray
    point_motion_magnitude_visible_xy: np.ndarray
    hand_present: bool
    fingertip_px: np.ndarray
    corrected_cheek_points_px: np.ndarray
    neutral_cheek_points_px: np.ndarray
    left_cheek_ids: np.ndarray
    anchor_alignment_error: float



def _scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        if value.size == 1:
            return value.reshape(()).item()
    return value



def load_sample(path: Path) -> SampleInfo:
    data = np.load(path, allow_pickle=True)
    missing = REQUIRED_KEYS - set(data.files)
    if missing:
        raise KeyError(f"{path.name} missing keys: {sorted(missing)}")

    return SampleInfo(
        file=path,
        source_local_index=int(_scalar(data["source_local_index"])),
        source_mp_id=int(_scalar(data["source_mp_id"])),
        trial_index=int(_scalar(data["trial_index"])),
        pressure_label=str(_scalar(data["pressure_label"])),
        session_label=str(_scalar(data["session_label"])),
        visible_mask=data["visible_mask"].astype(bool),
        blocked_mask=data["blocked_mask"].astype(bool),
        cheek_displacement_xy=data["cheek_displacement_xy"].astype(np.float32),
        cheek_displacement_xyz=data["cheek_displacement_xyz"].astype(np.float32),
        visible_cheek_displacement_xy=data["visible_cheek_displacement_xy"].astype(np.float32),
        visible_cheek_displacement_xyz=data["visible_cheek_displacement_xyz"].astype(np.float32),
        point_motion_magnitude_visible_xy=data["point_motion_magnitude_visible_xy"].astype(np.float32),
        hand_present=bool(int(_scalar(data["hand_present"]))),
        fingertip_px=data["fingertip_px"].astype(np.float32),
        corrected_cheek_points_px=data["corrected_cheek_points_px"].astype(np.float32),
        neutral_cheek_points_px=data["neutral_cheek_points_px"].astype(np.float32),
        left_cheek_ids=data["left_cheek_ids"].astype(np.int32),
        anchor_alignment_error=float(_scalar(data["anchor_alignment_error"])),
    )



def summarize_float(values: list[float]) -> dict[str, float | None]:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v)) and not math.isinf(float(v))]
    if not clean:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": float(mean(clean)),
        "median": float(median(clean)),
        "min": float(min(clean)),
        "max": float(max(clean)),
    }



def maybe_plot(out_dir: Path, landmark_ids: np.ndarray, vis_counts: np.ndarray, total_samples: int,
               visible_counts: list[int], alignment_errors: list[float]) -> None:
    if not HAS_MPL:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    coverage_pct = 100.0 * vis_counts / max(total_samples, 1)
    ax.bar(range(len(landmark_ids)), coverage_pct)
    ax.set_xlabel("Local cheek index")
    ax.set_ylabel("Visibility coverage (%)")
    ax.set_title("Per-landmark visibility coverage")
    fig.tight_layout()
    fig.savefig(out_dir / "landmark_visibility_coverage.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.hist(visible_counts, bins=min(15, max(5, len(set(visible_counts)))))
    ax.set_xlabel("Visible cheek points per sample")
    ax.set_ylabel("Count")
    ax.set_title("Visible cheek-point count distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "visible_point_histogram.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.hist(alignment_errors, bins=20)
    ax.set_xlabel("Anchor alignment error (px)")
    ax.set_ylabel("Count")
    ax.set_title("Anchor alignment error distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "alignment_error_histogram.png", dpi=150)
    plt.close(fig)



def build_recommendation(total_samples: int,
                         dominant_source_fraction: float,
                         mean_visible_ratio: float,
                         min_landmark_coverage_ratio: float,
                         mean_alignment_error: float | None,
                         fingertip_dist_mean: float | None) -> tuple[str, list[str]]:
    notes: list[str] = []

    if total_samples < 200:
        notes.append("Dataset is still small; record more before training.")
    else:
        notes.append("Sample count is large enough for a first prototype model.")

    if dominant_source_fraction >= 0.9:
        notes.append("Source point is consistent enough for a single-source prototype.")
    else:
        notes.append("Multiple source points were mixed; filter or re-record for a cleaner single-source experiment.")

    if mean_visible_ratio >= 0.55:
        notes.append("Visible landmark coverage per frame looks acceptable for a visible-region model.")
    else:
        notes.append("Too many cheek points are blocked per frame; try gentler presses or smaller angle changes.")

    if min_landmark_coverage_ratio >= 0.1:
        notes.append("Every cheek landmark has at least some usable coverage across the dataset.")
    else:
        notes.append("Some cheek landmarks are rarely or never visible; more multi-pass coverage is needed.")

    if mean_alignment_error is not None and mean_alignment_error <= 5.0:
        notes.append("Head-motion compensation looks reasonably stable.")
    elif mean_alignment_error is not None:
        notes.append("Alignment error is a bit high; inspect whether rigid-face compensation is strong enough.")

    if fingertip_dist_mean is not None and fingertip_dist_mean <= 80.0:
        notes.append("Finger contact stays reasonably close to the chosen source region.")
    elif fingertip_dist_mean is not None:
        notes.append("Finger contact may be drifting too far from the chosen source point.")

    ready = (
        total_samples >= 200
        and dominant_source_fraction >= 0.9
        and mean_visible_ratio >= 0.55
        and min_landmark_coverage_ratio >= 0.1
        and (mean_alignment_error is None or mean_alignment_error <= 6.5)
    )

    overall = (
        "Good enough to move forward to a first prototype model and renderer-facing sparse deformation export."
        if ready else
        "Not clean enough yet for the next stage without filtering or another recording pass."
    )
    return overall, notes



def analyze_dataset(dataset_dir: Path, output_dir: Path) -> dict[str, Any]:
    sample_files = sorted(dataset_dir.rglob("sample_*.npz"))
    if not sample_files:
        raise FileNotFoundError(f"No sample_*.npz files found under {dataset_dir}")

    samples: list[SampleInfo] = []
    bad_files: list[str] = []
    for path in sample_files:
        try:
            samples.append(load_sample(path))
        except Exception as exc:  # noqa: BLE001
            bad_files.append(f"{path.name}: {exc}")

    if not samples:
        raise RuntimeError("No valid samples could be loaded.")

    landmark_ids = samples[0].left_cheek_ids
    n_landmarks = len(landmark_ids)

    source_counter = Counter(s.source_local_index for s in samples)
    source_mp_counter = Counter(s.source_mp_id for s in samples)
    pressure_counter = Counter(s.pressure_label for s in samples)
    trial_counter = Counter(s.trial_index for s in samples)
    session_counter = Counter(s.session_label for s in samples)

    visible_counts: list[int] = []
    blocked_counts: list[int] = []
    alignment_errors: list[float] = []
    fingertip_source_dists: list[float] = []
    per_landmark_visibility = np.zeros(n_landmarks, dtype=np.int32)
    per_landmark_blocked = np.zeros(n_landmarks, dtype=np.int32)
    per_landmark_visible_disp_xy: dict[int, list[float]] = defaultdict(list)
    per_landmark_visible_disp_xyz: dict[int, list[float]] = defaultdict(list)

    trial_rows: dict[tuple[int, str, int], dict[str, Any]] = {}

    for s in samples:
        visible_counts.append(int(np.sum(s.visible_mask)))
        blocked_counts.append(int(np.sum(s.blocked_mask)))
        alignment_errors.append(float(s.anchor_alignment_error))
        per_landmark_visibility += s.visible_mask.astype(np.int32)
        per_landmark_blocked += s.blocked_mask.astype(np.int32)

        visible_xy_mag = np.linalg.norm(np.nan_to_num(s.visible_cheek_displacement_xy, nan=0.0), axis=1)
        visible_xyz_mag = np.linalg.norm(np.nan_to_num(s.visible_cheek_displacement_xyz, nan=0.0), axis=1)
        for idx in range(n_landmarks):
            if s.visible_mask[idx]:
                per_landmark_visible_disp_xy[idx].append(float(visible_xy_mag[idx]))
                per_landmark_visible_disp_xyz[idx].append(float(visible_xyz_mag[idx]))

        if s.hand_present and np.isfinite(s.fingertip_px).all() and 0 <= s.source_local_index < len(s.corrected_cheek_points_px):
            source_pt = s.corrected_cheek_points_px[s.source_local_index]
            fingertip_source_dists.append(float(np.linalg.norm(s.fingertip_px - source_pt)))

        key = (s.trial_index, s.pressure_label, s.source_local_index)
        row = trial_rows.setdefault(key, {
            "trial_index": s.trial_index,
            "pressure_label": s.pressure_label,
            "source_local_index": s.source_local_index,
            "source_mp_id": s.source_mp_id,
            "sample_count": 0,
            "mean_visible_points_values": [],
            "mean_alignment_error_values": [],
            "mean_fingertip_dist_values": [],
        })
        row["sample_count"] += 1
        row["mean_visible_points_values"].append(int(np.sum(s.visible_mask)))
        row["mean_alignment_error_values"].append(float(s.anchor_alignment_error))
        if s.hand_present and np.isfinite(s.fingertip_px).all() and 0 <= s.source_local_index < len(s.corrected_cheek_points_px):
            row["mean_fingertip_dist_values"].append(float(np.linalg.norm(s.fingertip_px - s.corrected_cheek_points_px[s.source_local_index])))

    dominant_source_index, dominant_source_count = source_counter.most_common(1)[0]
    dominant_source_fraction = dominant_source_count / len(samples)
    mean_visible_ratio = float(mean(visible_counts) / n_landmarks)
    landmark_coverage_ratio = per_landmark_visibility / len(samples)
    min_landmark_coverage_ratio = float(np.min(landmark_coverage_ratio))
    mean_alignment_error = summarize_float(alignment_errors)["mean"]
    fingertip_dist_mean = summarize_float(fingertip_source_dists)["mean"]

    overall, notes = build_recommendation(
        total_samples=len(samples),
        dominant_source_fraction=dominant_source_fraction,
        mean_visible_ratio=mean_visible_ratio,
        min_landmark_coverage_ratio=min_landmark_coverage_ratio,
        mean_alignment_error=mean_alignment_error,
        fingertip_dist_mean=fingertip_dist_mean,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    landmark_csv = output_dir / "landmark_coverage.csv"
    with landmark_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "local_index", "mp_id", "visible_count", "blocked_count",
            "visible_ratio", "blocked_ratio", "mean_visible_disp_xy", "mean_visible_disp_xyz",
        ])
        for idx, mp_id in enumerate(landmark_ids):
            writer.writerow([
                idx,
                int(mp_id),
                int(per_landmark_visibility[idx]),
                int(per_landmark_blocked[idx]),
                float(per_landmark_visibility[idx] / len(samples)),
                float(per_landmark_blocked[idx] / len(samples)),
                summarize_float(per_landmark_visible_disp_xy[idx])["mean"],
                summarize_float(per_landmark_visible_disp_xyz[idx])["mean"],
            ])

    trial_csv = output_dir / "trial_summary.csv"
    with trial_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trial_index", "pressure_label", "source_local_index", "source_mp_id",
            "sample_count", "mean_visible_points", "mean_alignment_error", "mean_fingertip_source_distance_px",
        ])
        for key in sorted(trial_rows):
            row = trial_rows[key]
            writer.writerow([
                row["trial_index"],
                row["pressure_label"],
                row["source_local_index"],
                row["source_mp_id"],
                row["sample_count"],
                summarize_float(row["mean_visible_points_values"])["mean"],
                summarize_float(row["mean_alignment_error_values"])["mean"],
                summarize_float(row["mean_fingertip_dist_values"])["mean"],
            ])

    report = {
        "dataset_dir": str(dataset_dir),
        "valid_sample_count": len(samples),
        "bad_file_count": len(bad_files),
        "bad_files": bad_files,
        "unique_sessions": dict(session_counter),
        "unique_source_local_indices": dict(source_counter),
        "unique_source_mp_ids": dict(source_mp_counter),
        "pressure_distribution": dict(pressure_counter),
        "trial_distribution": dict(trial_counter),
        "visible_points_per_sample": summarize_float(visible_counts),
        "blocked_points_per_sample": summarize_float(blocked_counts),
        "anchor_alignment_error_px": summarize_float(alignment_errors),
        "fingertip_to_source_distance_px": summarize_float(fingertip_source_dists),
        "dominant_source_local_index": int(dominant_source_index),
        "dominant_source_fraction": float(dominant_source_fraction),
        "mean_visible_ratio": float(mean_visible_ratio),
        "min_landmark_coverage_ratio": float(min_landmark_coverage_ratio),
        "overall_recommendation": overall,
        "next_step_notes": notes,
        "output_files": {
            "landmark_coverage_csv": str(landmark_csv),
            "trial_summary_csv": str(trial_csv),
        },
    }

    report_path = output_dir / "analysis_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    maybe_plot(output_dir, landmark_ids, per_landmark_visibility, len(samples), visible_counts, alignment_errors)
    return report



def print_console_summary(report: dict[str, Any]) -> None:
    print("=" * 72)
    print("FACE PRESS DATASET ANALYSIS")
    print("=" * 72)
    print(f"Dataset: {report['dataset_dir']}")
    print(f"Valid samples: {report['valid_sample_count']}")
    if report["bad_file_count"]:
        print(f"Bad files: {report['bad_file_count']}")
    print(f"Source distribution: {report['unique_source_local_indices']}")
    print(f"Pressure distribution: {report['pressure_distribution']}")
    print(f"Trials: {report['trial_distribution']}")
    print(f"Visible points/sample: {report['visible_points_per_sample']}")
    print(f"Blocked points/sample: {report['blocked_points_per_sample']}")
    print(f"Anchor alignment error (px): {report['anchor_alignment_error_px']}")
    print(f"Fingertip->source distance (px): {report['fingertip_to_source_distance_px']}")
    print(f"Dominant source fraction: {report['dominant_source_fraction']:.3f}")
    print(f"Mean visible ratio: {report['mean_visible_ratio']:.3f}")
    print(f"Min landmark coverage ratio: {report['min_landmark_coverage_ratio']:.3f}")
    print("-" * 72)
    print(report["overall_recommendation"])
    for note in report["next_step_notes"]:
        print(f"- {note}")
    print("-" * 72)
    print("Saved:")
    for k, v in report["output_files"].items():
        print(f"  {k}: {v}")
    print("=" * 72)



def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a multi-pass face-press dataset for prototype readiness.")
    parser.add_argument("dataset_dir", type=Path, help="Folder containing sample_*.npz files or the session root folder.")
    parser.add_argument("--out", type=Path, default=None, help="Output folder for report/csv/plots. Defaults to <dataset_dir>/analysis.")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    output_dir = args.out if args.out is not None else (dataset_dir / "analysis")

    report = analyze_dataset(dataset_dir, output_dir)
    print_console_summary(report)


if __name__ == "__main__":
    main()
