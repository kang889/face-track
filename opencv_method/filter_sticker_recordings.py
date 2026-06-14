from __future__ import annotations

import argparse
import json
import shutil
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
        description="Filter a raw sticker deformation dataset into a smaller cleaner subset."
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        nargs="?",
        default=Path("sticker_deformation_dataset_occlusion_aware"),
        help="Folder containing raw sample_XXXXXX.npz files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Folder to store the cleaned subset. Default: <dataset_dir>/filtered_subset",
    )
    parser.add_argument(
        "--source-index",
        type=int,
        default=None,
        help="Keep only this source point. Default: auto-pick dominant source point.",
    )
    parser.add_argument(
        "--min-visible-points",
        type=int,
        default=8,
        help="Minimum visible sticker points required to keep a frame.",
    )
    parser.add_argument(
        "--max-align-error",
        type=float,
        default=5.0,
        help="Maximum alignment error in pixels required to keep a frame.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    raw_files = sorted(dataset_dir.glob("sample_*.npz"))
    if not raw_files:
        raise RuntimeError(f"No sample_*.npz files found in {dataset_dir}")

    source_indices = []
    meta_rows = []

    for i, f in enumerate(raw_files):
        with np.load(f, allow_pickle=True) as data:
            visible_mask = np.array(data["visible_mask"]).astype(bool)
            blocked_mask = np.array(data["blocked_mask"]).astype(bool)
            alignment_error = float(np.array(data["alignment_error"]).item())
            source_idx = int(np.array(data["source_local_index"]).item()) if "source_local_index" in data else -1

            meta_rows.append({
                "raw_index": i,
                "file": f,
                "source_index": source_idx,
                "visible_count": int(np.sum(visible_mask)),
                "blocked_count": int(np.sum(blocked_mask)),
                "alignment_error": alignment_error,
            })
            source_indices.append(source_idx)

    source_indices = np.array(source_indices, dtype=np.int32)

    if args.source_index is None:
        valid_sources = source_indices[source_indices >= 0]
        if valid_sources.size == 0:
            source_to_use = -1
        else:
            vals, counts = np.unique(valid_sources, return_counts=True)
            source_to_use = int(vals[np.argmax(counts)])
    else:
        source_to_use = int(args.source_index)

    kept_rows = []
    for row in meta_rows:
        keep = True
        if source_to_use >= 0 and row["source_index"] != source_to_use:
            keep = False
        if row["visible_count"] < args.min_visible_points:
            keep = False
        if row["alignment_error"] > args.max_align_error:
            keep = False
        if keep:
            kept_rows.append(row)

    if not kept_rows:
        raise RuntimeError("No frames remained after filtering. Try loosening the thresholds.")

    output_dir = args.output_dir or (dataset_dir / "filtered_subset")
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    all_visible_masks = []
    all_blocked_masks = []
    all_motion_magnitudes = []
    visible_counts = []
    blocked_counts = []
    alignment_errors = []
    kept_indices = []
    kept_file_paths = []
    point_count = None

    for row in kept_rows:
        src_path = row["file"]
        dst_path = samples_dir / src_path.name
        shutil.copy2(src_path, dst_path)

        kept_indices.append(int(row["raw_index"]))
        kept_file_paths.append(str(dst_path))

        with np.load(src_path, allow_pickle=True) as data:
            visible_mask = np.array(data["visible_mask"]).astype(bool)
            blocked_mask = np.array(data["blocked_mask"]).astype(bool)
            neutral_points = np.array(data["neutral_points"]).astype(np.float32)
            corrected_points = np.array(data["corrected_points"]).astype(np.float32)
            alignment_error = float(np.array(data["alignment_error"]).item())

            if point_count is None:
                point_count = int(len(visible_mask))

            displacement = corrected_points - neutral_points
            motion_mag = np.linalg.norm(displacement, axis=1)

            all_visible_masks.append(visible_mask.astype(np.uint8))
            all_blocked_masks.append(blocked_mask.astype(np.uint8))
            all_motion_magnitudes.append(motion_mag.astype(np.float32))
            visible_counts.append(int(np.sum(visible_mask)))
            blocked_counts.append(int(np.sum(blocked_mask)))
            alignment_errors.append(alignment_error)

    all_visible_masks = np.stack(all_visible_masks, axis=0)
    all_blocked_masks = np.stack(all_blocked_masks, axis=0)
    all_motion_magnitudes = np.stack(all_motion_magnitudes, axis=0)

    visible_counts = np.array(visible_counts, dtype=np.float32)
    blocked_counts = np.array(blocked_counts, dtype=np.float32)
    alignment_errors = np.array(alignment_errors, dtype=np.float32)

    point_visibility_ratio = all_visible_masks.mean(axis=0).astype(np.float32)
    point_block_ratio = all_blocked_masks.mean(axis=0).astype(np.float32)
    point_motion_mean = all_motion_magnitudes.mean(axis=0).astype(np.float32)

    summary = {
        "raw_dataset_dir": str(dataset_dir),
        "filtered_subset_dir": str(output_dir),
        "raw_sample_count": int(len(raw_files)),
        "kept_sample_count": int(len(kept_rows)),
        "kept_fraction": float(len(kept_rows) / len(raw_files)),
        "source_index_used": int(source_to_use),
        "filters": {
            "source_index": int(source_to_use),
            "min_visible_points": int(args.min_visible_points),
            "max_alignment_error_px": float(args.max_align_error),
        },
        "filtered_visible_points_per_sample": summarize(visible_counts),
        "filtered_blocked_points_per_sample": summarize(blocked_counts),
        "filtered_alignment_error_px": summarize(alignment_errors),
        "filtered_point_visibility": {
            "mean_visibility_ratio": float(np.mean(point_visibility_ratio)),
            "min_visibility_ratio": float(np.min(point_visibility_ratio)),
            "max_visibility_ratio": float(np.max(point_visibility_ratio)),
        },
    }

    summary_path = output_dir / "filtered_sticker_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    coverage_csv = output_dir / "filtered_point_coverage.csv"
    with coverage_csv.open("w", encoding="utf-8") as f:
        f.write("local_index,visibility_ratio,block_ratio,mean_motion_magnitude\n")
        for i in range(point_count):
            f.write(
                f"{i},"
                f"{float(point_visibility_ratio[i]):.6f},"
                f"{float(point_block_ratio[i]):.6f},"
                f"{float(point_motion_mean[i]):.6f}\n"
            )

    kept_idx_txt = output_dir / "kept_indices.txt"
    kept_idx_txt.write_text("\n".join(str(i) for i in kept_indices), encoding="utf-8")

    kept_files_txt = output_dir / "kept_files.txt"
    kept_files_txt.write_text("\n".join(kept_file_paths), encoding="utf-8")

    print("=" * 72)
    print(f"Raw sample count: {len(raw_files)}")
    print(f"Kept sample count: {len(kept_rows)}")
    print(f"Kept fraction: {len(kept_rows) / len(raw_files):.3f}")
    print(f"Source index used: {source_to_use}")
    print(f"Visible points >= {args.min_visible_points}")
    print(f"Alignment error <= {args.max_align_error:.2f}px")
    print(f"Filtered visible points/sample: {summary['filtered_visible_points_per_sample']}")
    print(f"Filtered blocked points/sample: {summary['filtered_blocked_points_per_sample']}")
    print(f"Filtered alignment error (px): {summary['filtered_alignment_error_px']}")
    print("Saved:")
    print(f"  {summary_path}")
    print(f"  {coverage_csv}")
    print(f"  {kept_idx_txt}")
    print(f"  {kept_files_txt}")
    print(f"  {samples_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
