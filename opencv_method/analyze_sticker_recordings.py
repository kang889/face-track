from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(values: np.ndarray) -> dict:
    values = np.asarray(values)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze sticker deformation recordings and judge whether they are reliable enough to move forward."
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        nargs="?",
        default=Path("sticker_deformation_dataset_occlusion_aware"),
        help="Folder containing sticker sample_XXXXXX.npz files",
    )
    parser.add_argument(
        "--max-align-error",
        type=float,
        default=12.0,
        help="Frames above this alignment error are considered weak",
    )
    parser.add_argument(
        "--min-visible-points",
        type=int,
        default=6,
        help="Minimum visible points for a frame to be considered usable",
    )
    parser.add_argument(
        "--good-coverage-threshold",
        type=float,
        default=0.50,
        help="Per-point visibility ratio threshold considered good coverage",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    files = sorted(dataset_dir.glob("sample_*.npz"))
    if not files:
        raise RuntimeError(f"No sample_*.npz files found in {dataset_dir}")

    visible_counts = []
    blocked_counts = []
    visible_ratios = []
    alignment_errors = []
    source_indices = []

    all_visible_masks = []
    all_blocked_masks = []
    all_motion_magnitudes = []

    total_points = None
    point_ids = None

    for f in files:
        with np.load(f, allow_pickle=True) as data:
            required = ["visible_mask", "blocked_mask", "neutral_points", "corrected_points", "alignment_error"]
            for key in required:
                if key not in data:
                    raise RuntimeError(f"{f} is missing required field: {key}")

            visible_mask = np.array(data["visible_mask"]).astype(bool)
            blocked_mask = np.array(data["blocked_mask"]).astype(bool)
            neutral_points = np.array(data["neutral_points"]).astype(np.float32)
            corrected_points = np.array(data["corrected_points"]).astype(np.float32)
            alignment_error = float(np.array(data["alignment_error"]).item())
            source_idx = int(np.array(data["source_local_index"]).item()) if "source_local_index" in data else -1

            if total_points is None:
                total_points = int(len(visible_mask))
                point_ids = np.arange(total_points, dtype=np.int32)
            elif len(visible_mask) != total_points:
                raise RuntimeError(f"{f} has inconsistent point count. Expected {total_points}, got {len(visible_mask)}")

            displacement = corrected_points - neutral_points
            motion_mag = np.linalg.norm(displacement, axis=1)

            visible_counts.append(int(np.sum(visible_mask)))
            blocked_counts.append(int(np.sum(blocked_mask)))
            visible_ratios.append(float(np.sum(visible_mask)) / float(total_points))
            alignment_errors.append(alignment_error)
            source_indices.append(source_idx)

            all_visible_masks.append(visible_mask.astype(np.uint8))
            all_blocked_masks.append(blocked_mask.astype(np.uint8))
            all_motion_magnitudes.append(motion_mag.astype(np.float32))

    visible_counts = np.array(visible_counts, dtype=np.float32)
    blocked_counts = np.array(blocked_counts, dtype=np.float32)
    visible_ratios = np.array(visible_ratios, dtype=np.float32)
    alignment_errors = np.array(alignment_errors, dtype=np.float32)
    source_indices = np.array(source_indices, dtype=np.int32)

    all_visible_masks = np.stack(all_visible_masks, axis=0).astype(np.uint8)
    all_blocked_masks = np.stack(all_blocked_masks, axis=0).astype(np.uint8)
    all_motion_magnitudes = np.stack(all_motion_magnitudes, axis=0).astype(np.float32)

    point_visibility_ratio = all_visible_masks.mean(axis=0).astype(np.float32)
    point_block_ratio = all_blocked_masks.mean(axis=0).astype(np.float32)
    point_motion_mean = all_motion_magnitudes.mean(axis=0).astype(np.float32)

    good_frame_mask = (visible_counts >= args.min_visible_points) & (alignment_errors <= args.max_align_error)
    good_frame_fraction = float(np.mean(good_frame_mask.astype(np.float32)))

    dominant_source_fraction = 0.0
    dominant_source = None
    if np.any(source_indices >= 0):
        values, counts = np.unique(source_indices[source_indices >= 0], return_counts=True)
        best = int(np.argmax(counts))
        dominant_source = int(values[best])
        dominant_source_fraction = float(counts[best] / len(source_indices[source_indices >= 0]))

    min_point_coverage = float(np.min(point_visibility_ratio))
    mean_point_coverage = float(np.mean(point_visibility_ratio))
    num_good_points = int(np.sum(point_visibility_ratio >= args.good_coverage_threshold))

    reliability_notes = []
    reliable = True

    if len(files) < 50:
        reliability_notes.append("Sample count is still small. More recording would strengthen the dataset.")
        reliable = False
    else:
        reliability_notes.append("Sample count is large enough for a first prototype analysis.")

    if good_frame_fraction < 0.60:
        reliability_notes.append("Too many frames fail the visibility/alignment requirements.")
        reliable = False
    else:
        reliability_notes.append("A good fraction of frames meet visibility and alignment requirements.")

    if mean_point_coverage < 0.50:
        reliability_notes.append("Overall point visibility coverage is weak.")
        reliable = False
    else:
        reliability_notes.append("Overall point visibility coverage is acceptable.")

    if min_point_coverage < 0.20:
        reliability_notes.append("Some sticker points are rarely visible and may be unreliable.")
    else:
        reliability_notes.append("Every sticker point has at least some usable visibility coverage.")

    if np.any(source_indices >= 0):
        if dominant_source_fraction < 0.90:
            reliability_notes.append("Source point selection is inconsistent across the dataset.")
            reliable = False
        else:
            reliability_notes.append("Source point selection is consistent enough.")

    summary = {
        "dataset_dir": str(dataset_dir),
        "sample_count": int(len(files)),
        "points_per_sample": int(total_points),
        "visible_points_per_sample": summarize(visible_counts),
        "blocked_points_per_sample": summarize(blocked_counts),
        "visible_ratio_per_sample": summarize(visible_ratios),
        "alignment_error_px": summarize(alignment_errors),
        "good_frame_fraction": good_frame_fraction,
        "dominant_source_index": dominant_source,
        "dominant_source_fraction": dominant_source_fraction,
        "per_point_summary": {
            "min_visibility_ratio": min_point_coverage,
            "mean_visibility_ratio": mean_point_coverage,
            "good_visibility_threshold": float(args.good_coverage_threshold),
            "num_points_above_good_threshold": num_good_points,
        },
        "reliable_enough_to_move_forward": reliable,
        "notes": reliability_notes,
    }

    out_dir = dataset_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "sticker_recording_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    coverage_csv = out_dir / "point_coverage.csv"
    with coverage_csv.open("w", encoding="utf-8") as f:
        f.write("local_index,visibility_ratio,block_ratio,mean_motion_magnitude\n")
        for i in range(total_points):
            f.write(
                f"{int(point_ids[i])},"
                f"{float(point_visibility_ratio[i]):.6f},"
                f"{float(point_block_ratio[i]):.6f},"
                f"{float(point_motion_mean[i]):.6f}\n"
            )

    good_idx_txt = out_dir / "good_frame_indices.txt"
    good_indices = np.nonzero(good_frame_mask)[0]
    good_idx_txt.write_text("\n".join(str(int(i)) for i in good_indices))

    print("=" * 72)
    print(f"Sample count: {len(files)}")
    print(f"Points/sample: {total_points}")
    print(f"Visible points/sample: {summary['visible_points_per_sample']}")
    print(f"Blocked points/sample: {summary['blocked_points_per_sample']}")
    print(f"Alignment error (px): {summary['alignment_error_px']}")
    print(f"Good frame fraction: {good_frame_fraction:.3f}")
    if dominant_source is not None:
        print(f"Dominant source index: {dominant_source}")
        print(f"Dominant source fraction: {dominant_source_fraction:.3f}")
    print(f"Mean point visibility ratio: {mean_point_coverage:.3f}")
    print(f"Min point visibility ratio: {min_point_coverage:.3f}")
    print("-" * 72)
    if reliable:
        print("Recording looks reliable enough to move forward to the next stage.")
    else:
        print("Recording is not reliable enough yet. Check the notes and record more if needed.")
    for note in reliability_notes:
        print(f"- {note}")
    print("-" * 72)
    print("Saved:")
    print(f"  {summary_path}")
    print(f"  {coverage_csv}")
    print(f"  {good_idx_txt}")
    print("=" * 72)


if __name__ == "__main__":
    main()
