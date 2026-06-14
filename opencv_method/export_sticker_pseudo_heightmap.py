from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def idw_heightmap(points_xy: np.ndarray, values: np.ndarray, width: int, height: int, power: float = 2.0) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    px = points_xy[:, 0][None, None, :]
    py = points_xy[:, 1][None, None, :]
    dist2 = (xx[:, :, None] - px) ** 2 + (yy[:, :, None] - py) ** 2
    w = 1.0 / np.maximum(dist2, 1e-6) ** (power / 2.0)
    weighted = (w * values[None, None, :]).sum(axis=2)
    norm = w.sum(axis=2)
    return (weighted / np.maximum(norm, 1e-8)).astype(np.float32)


def build_patch_mask(points_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    hull = cv2.convexHull(np.round(points_xy).astype(np.int32))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def normalize_relative_to_u8(arr: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, float, float]:
    if mask is not None and np.any(mask > 0):
        vals = arr[mask > 0]
    else:
        vals = arr.reshape(-1)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if abs(vmax - vmin) < 1e-8:
        out = np.full(arr.shape, 127, dtype=np.uint8)
        if mask is not None:
            out = out.copy()
            out[mask == 0] = 0
        return out, vmin, vmax
    norm = (arr - vmin) / (vmax - vmin)
    out = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    if mask is not None:
        out = out.copy()
        out[mask == 0] = 0
    return out, vmin, vmax


def normalize_absolute_to_u8(arr: np.ndarray, fixed_min: float, fixed_max: float, mask: np.ndarray | None = None) -> np.ndarray:
    if abs(fixed_max - fixed_min) < 1e-8:
        out = np.full(arr.shape, 127, dtype=np.uint8)
        if mask is not None:
            out = out.copy()
            out[mask == 0] = 0
        return out
    norm = (arr - fixed_min) / (fixed_max - fixed_min)
    out = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    if mask is not None:
        out = out.copy()
        out[mask == 0] = 0
    return out


def save_heightmap_csv(path: Path, arr: np.ndarray) -> None:
    h, w = arr.shape
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["y", "x", "value"])
        for y in range(h):
            for x in range(w):
                writer.writerow([y, x, float(arr[y, x])])


def draw_reference(points_xy: np.ndarray, width: int, height: int, mask: np.ndarray, point_ids: np.ndarray, source_idx: int) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    img[mask > 0] = (60, 60, 60)
    hull = cv2.convexHull(np.round(points_xy).astype(np.int32))
    cv2.polylines(img, [hull], isClosed=True, color=(180, 180, 180), thickness=1, lineType=cv2.LINE_AA)
    for i, pt in enumerate(points_xy):
        xy = tuple(np.round(pt).astype(int))
        color = (0, 255, 255) if i == source_idx else (0, 255, 0)
        cv2.circle(img, xy, 4, color, -1)
        cv2.putText(img, str(int(point_ids[i])), (xy[0] + 5, xy[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def load_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def compute_radial_inward_scalar(neutral_points: np.ndarray, corrected_points: np.ndarray, visible_mask: np.ndarray, source_idx: int) -> np.ndarray:
    """
    Convert 2D sticker displacement into a 1D pseudo-height value.

    For each sticker:
    - displacement = corrected - neutral
    - inward direction = from sticker toward source point in neutral layout
    - scalar = dot(displacement, inward_unit_direction)

    Positive scalar means motion toward the chosen source point.
    Blocked stickers get NaN.
    """
    displacement = corrected_points - neutral_points
    source_pt = neutral_points[source_idx]
    vec_to_source = source_pt[None, :] - neutral_points
    dist = np.linalg.norm(vec_to_source, axis=1, keepdims=True)
    inward_unit = np.divide(
        vec_to_source,
        np.maximum(dist, 1e-8),
        out=np.zeros_like(vec_to_source, dtype=np.float32),
        where=dist > 1e-8,
    )
    scalar = np.sum(displacement * inward_unit, axis=1).astype(np.float32)

    # For the source point itself, use its motion magnitude as a simple local proxy.
    scalar[source_idx] = float(np.linalg.norm(displacement[source_idx]))

    scalar[~visible_mask] = np.nan
    return scalar


def save_sparse_csv(
    path: Path,
    point_ids: np.ndarray,
    neutral_points: np.ndarray,
    roi_points: np.ndarray,
    avg_visibility_ratio: np.ndarray,
    avg_radial_scalar: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "local_index",
            "neutral_x",
            "neutral_y",
            "roi_x",
            "roi_y",
            "avg_visibility_ratio",
            "avg_radial_inward_scalar",
        ])
        for i in range(len(point_ids)):
            writer.writerow([
                int(point_ids[i]),
                float(neutral_points[i, 0]),
                float(neutral_points[i, 1]),
                float(roi_points[i, 0]),
                float(roi_points[i, 1]),
                float(avg_visibility_ratio[i]),
                float(avg_radial_scalar[i]),
            ])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create relative and absolute pseudo-height maps from filtered OpenCV sticker recordings."
    )
    parser.add_argument(
        "filtered_subset_dir",
        type=Path,
        nargs="?",
        default=Path("sticker_deformation_dataset_occlusion_aware") / "filtered_subset",
        help="Filtered sticker subset folder created by filter_sticker_recordings.py",
    )
    parser.add_argument("--size", type=int, default=256, help="Output texture size (square)")
    parser.add_argument("--patch-margin", type=int, default=8, help="Extra margin around the sticker patch")
    parser.add_argument("--idw-power", type=float, default=2.0, help="IDW interpolation power")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output folder")
    args = parser.parse_args()

    filtered_subset_dir = args.filtered_subset_dir
    samples_dir = filtered_subset_dir / "samples"
    if not samples_dir.exists():
        raise FileNotFoundError(f"Filtered samples folder not found: {samples_dir}")

    files = sorted(samples_dir.glob("sample_*.npz"))
    if not files:
        raise RuntimeError(f"No sample_*.npz files found in {samples_dir}")

    summary = load_summary(filtered_subset_dir / "filtered_sticker_summary.json")
    source_idx_from_summary = None
    if summary is not None:
        source_idx_from_summary = int(summary["source_index_used"])

    output_dir = args.output_dir or (filtered_subset_dir / "heightmap_export")
    output_dir.mkdir(parents=True, exist_ok=True)

    radial_scalars = []
    visibility_masks = []
    neutral_template = None
    source_idx = None
    point_count = None
    patch_edges = None

    for f in files:
        with np.load(f, allow_pickle=True) as data:
            visible_mask = np.array(data["visible_mask"]).astype(bool)
            neutral_points = np.array(data["neutral_points"]).astype(np.float32)
            corrected_points = np.array(data["corrected_points"]).astype(np.float32)
            src = int(np.array(data["source_local_index"]).item()) if "source_local_index" in data else -1
            edges = np.array(data["patch_edges"]).astype(np.int32) if "patch_edges" in data else None

            if neutral_template is None:
                neutral_template = neutral_points.copy()
                point_count = len(neutral_points)
                source_idx = src if src >= 0 else source_idx_from_summary
                patch_edges = edges
            scalar = compute_radial_inward_scalar(neutral_points, corrected_points, visible_mask, source_idx)
            radial_scalars.append(scalar)
            visibility_masks.append(visible_mask.astype(np.uint8))

    radial_scalars = np.stack(radial_scalars, axis=0).astype(np.float32)
    visibility_masks = np.stack(visibility_masks, axis=0).astype(np.uint8)
    avg_visibility_ratio = visibility_masks.mean(axis=0).astype(np.float32)
    avg_scalar = np.nanmean(radial_scalars, axis=0).astype(np.float32)

    valid_global = radial_scalars[~np.isnan(radial_scalars)]
    if valid_global.size == 0:
        raise RuntimeError("No valid scalar values found after filtering.")
    global_min = float(np.min(valid_global))
    global_max = float(np.max(valid_global))

    pts = neutral_template.copy()
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    span_x = max(float(x_max - x_min), 1e-6)
    span_y = max(float(y_max - y_min), 1e-6)

    margin = float(args.patch_margin)
    pts_roi = np.empty_like(pts, dtype=np.float32)
    pts_roi[:, 0] = ((pts[:, 0] - x_min + margin) / (span_x + 2 * margin)) * (args.size - 1)
    pts_roi[:, 1] = ((pts[:, 1] - y_min + margin) / (span_y + 2 * margin)) * (args.size - 1)

    mask = build_patch_mask(pts_roi, args.size, args.size)
    height_float = idw_heightmap(pts_roi, avg_scalar, args.size, args.size, power=args.idw_power)
    height_float *= (mask > 0).astype(np.float32)

    relative_png, rel_min, rel_max = normalize_relative_to_u8(height_float, mask)
    absolute_png = normalize_absolute_to_u8(height_float, global_min, global_max, mask)
    ref_img = draw_reference(pts_roi, args.size, args.size, mask, np.arange(point_count), source_idx)

    np.save(output_dir / "opencv_sticker_heightmap_float.npy", height_float.astype(np.float32))
    save_heightmap_csv(output_dir / "opencv_sticker_heightmap_float.csv", height_float.astype(np.float32))
    cv2.imwrite(str(output_dir / "opencv_sticker_heightmap_relative.png"), relative_png)
    cv2.imwrite(str(output_dir / "opencv_sticker_heightmap_absolute.png"), absolute_png)
    cv2.imwrite(str(output_dir / "opencv_sticker_patch_reference.png"), ref_img)

    np.savez_compressed(
        output_dir / "opencv_sticker_average_sparse_points.npz",
        source_local_index=np.int32(source_idx),
        neutral_points=neutral_template.astype(np.float32),
        roi_points_px=pts_roi.astype(np.float32),
        avg_visibility_ratio=avg_visibility_ratio.astype(np.float32),
        avg_radial_inward_scalar=avg_scalar.astype(np.float32),
        patch_edges=patch_edges if patch_edges is not None else np.empty((0, 2), dtype=np.int32),
    )
    save_sparse_csv(
        output_dir / "opencv_sticker_average_sparse_points.csv",
        np.arange(point_count, dtype=np.int32),
        neutral_template.astype(np.float32),
        pts_roi.astype(np.float32),
        avg_visibility_ratio.astype(np.float32),
        avg_scalar.astype(np.float32),
    )

    summary_out = {
        "description": "OpenCV sticker pseudo-height map export",
        "method": "2D sticker displacement projected toward the chosen source point to create a pseudo-height scalar, then interpolated across the patch",
        "filtered_subset_dir": str(filtered_subset_dir),
        "sample_count": int(len(files)),
        "source_index": int(source_idx),
        "point_count": int(point_count),
        "texture_size": int(args.size),
        "relative_scaling": {
            "map_min": rel_min,
            "map_max": rel_max,
        },
        "absolute_scaling": {
            "dataset_min": global_min,
            "dataset_max": global_max,
        },
        "files": {
            "relative_png": "opencv_sticker_heightmap_relative.png",
            "absolute_png": "opencv_sticker_heightmap_absolute.png",
            "float_npy": "opencv_sticker_heightmap_float.npy",
            "float_csv": "opencv_sticker_heightmap_float.csv",
            "patch_reference_png": "opencv_sticker_patch_reference.png",
            "average_sparse_npz": "opencv_sticker_average_sparse_points.npz",
            "average_sparse_csv": "opencv_sticker_average_sparse_points.csv",
        },
        "note": "This is a pseudo-height map from 2D sticker motion, not a true depth map.",
    }
    (output_dir / "opencv_sticker_heightmap_summary.json").write_text(json.dumps(summary_out, indent=2), encoding="utf-8")

    print("=" * 72)
    print("Created OpenCV sticker pseudo-height map export")
    print(f"Sample count: {len(files)}")
    print(f"Source index: {source_idx}")
    print(f"Points: {point_count}")
    print(f"Texture size: {args.size}x{args.size}")
    print("Saved:")
    print(f"  {output_dir / 'opencv_sticker_heightmap_relative.png'}")
    print(f"  {output_dir / 'opencv_sticker_heightmap_absolute.png'}")
    print(f"  {output_dir / 'opencv_sticker_heightmap_float.csv'}")
    print(f"  {output_dir / 'opencv_sticker_patch_reference.png'}")
    print(f"  {output_dir / 'opencv_sticker_average_sparse_points.csv'}")
    print(f"  {output_dir / 'opencv_sticker_heightmap_summary.json'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
