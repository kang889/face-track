from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn


class FacePressMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: tuple[int, ...] = (64, 128, 128, 64)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hid in hidden_dims:
            layers.append(nn.Linear(prev, hid))
            layers.append(nn.ReLU())
            prev = hid
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def idw_heightmap(points_xy: np.ndarray, values: np.ndarray, width: int, height: int, power: float = 2.0) -> np.ndarray:
    """Inverse-distance-weighted interpolation from sparse cheek points to a dense patch map."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    px = points_xy[:, 0][None, None, :]
    py = points_xy[:, 1][None, None, :]
    dist2 = (xx[:, :, None] - px) ** 2 + (yy[:, :, None] - py) ** 2
    w = 1.0 / np.maximum(dist2, 1e-6) ** (power / 2.0)
    weighted = (w * values[None, None, :]).sum(axis=2)
    norm = w.sum(axis=2)
    return (weighted / np.maximum(norm, 1e-8)).astype(np.float32)


def build_patch_mask(points_xy: np.ndarray, width: int, height: int, margin: int = 0) -> np.ndarray:
    hull = cv2.convexHull(np.round(points_xy).astype(np.int32))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    if margin > 0:
        kernel = np.ones((margin * 2 + 1, margin * 2 + 1), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def normalize_to_u16(arr: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, float, float]:
    if mask is not None and np.any(mask > 0):
        vals = arr[mask > 0]
    else:
        vals = arr.reshape(-1)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if abs(vmax - vmin) < 1e-8:
        out = np.full(arr.shape, 32767, dtype=np.uint16)
        return out, vmin, vmax
    norm = (arr - vmin) / (vmax - vmin)
    out = np.clip(norm * 65535.0, 0, 65535).astype(np.uint16)
    return out, vmin, vmax


def draw_reference(points_xy: np.ndarray, width: int, height: int, mask: np.ndarray, point_ids: np.ndarray) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    img[mask > 0] = (60, 60, 60)
    hull = cv2.convexHull(np.round(points_xy).astype(np.int32))
    cv2.polylines(img, [hull], isClosed=True, color=(180, 180, 180), thickness=1, lineType=cv2.LINE_AA)
    for i, pt in enumerate(points_xy):
        xy = tuple(np.round(pt).astype(int))
        cv2.circle(img, xy, 3, (0, 255, 255), -1)
        cv2.putText(img, str(int(point_ids[i])), (xy[0] + 4, xy[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def save_heightmap_csv(path: Path, arr: np.ndarray) -> None:
    """Save dense height map as CSV with y,x,value columns."""
    h, w = arr.shape
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["y", "x", "value"])
        for y in range(h):
            for x in range(w):
                writer.writerow([y, x, float(arr[y, x])])


def save_sparse_points_csv(
    path: Path,
    point_ids: np.ndarray,
    neutral_template_px: np.ndarray,
    roi_points_px: np.ndarray,
    avg_visibility_ratio: np.ndarray,
    avg_pred_dxyz: np.ndarray,
    avg_true_dxyz: np.ndarray,
) -> None:
    """Save the averaged 25-point sparse cheek deformation as CSV."""
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "local_index",
            "mp_id",
            "neutral_x",
            "neutral_y",
            "roi_x",
            "roi_y",
            "avg_visibility_ratio",
            "avg_pred_dx",
            "avg_pred_dy",
            "avg_pred_dz",
            "avg_true_dx",
            "avg_true_dy",
            "avg_true_dz",
        ])
        for i, mp_id in enumerate(point_ids):
            writer.writerow([
                i,
                int(mp_id),
                float(neutral_template_px[i, 0]),
                float(neutral_template_px[i, 1]),
                float(roi_points_px[i, 0]),
                float(roi_points_px[i, 1]),
                float(avg_visibility_ratio[i]),
                float(avg_pred_dxyz[i, 0]),
                float(avg_pred_dxyz[i, 1]),
                float(avg_pred_dxyz[i, 2]),
                float(avg_true_dxyz[i, 0]),
                float(avg_true_dxyz[i, 1]),
                float(avg_true_dxyz[i, 2]),
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Average model-predicted cheek deformation into a regional approximate height map.")
    parser.add_argument("--model", type=Path, required=True, help="Path to best_model.pt")
    parser.add_argument("--filtered", type=Path, required=True, help="Path to face_press_filtered_subset.npz")
    parser.add_argument("--size", type=int, default=256, help="Output texture size (square)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output folder (default: model/export_average)")
    parser.add_argument("--patch-margin", type=int, default=8, help="Extra pixel margin around convex hull in the reference patch")
    parser.add_argument("--idw-power", type=float, default=2.0, help="IDW interpolation power")
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if not args.filtered.exists():
        raise FileNotFoundError(f"Filtered subset file not found: {args.filtered}")

    output_dir = args.output_dir or (args.model.parent / "export_average")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)

    with np.load(args.filtered, allow_pickle=True) as data:
        X = data["X"].astype(np.float32)
        Y = data["Y"].astype(np.float32)
        M = data["M"].astype(np.float32)
        source_local_index = int(np.array(data["source_local_index"]).item())
        source_mp_id = int(np.array(data["source_mp_id"]).item())
        left_cheek_ids = data["left_cheek_ids"].astype(np.int32)
        patch_edges = data["patch_edges"].astype(np.int32)
        neutral_template_px = data["neutral_template_px"].astype(np.float32)
        trials = data["trials"].astype(np.int64)
        pressure_index = data["pressure_index"].astype(np.int64)
        fingertip_dist_px = data["fingertip_dist_px"].astype(np.float32)
        visible_points = data["visible_points"].astype(np.int32)
        alignment_error_px = data["alignment_error_px"].astype(np.float32)

    input_dim = int(checkpoint["input_dim"])
    output_dim = int(checkpoint["output_dim"])
    model = FacePressMLP(input_dim=input_dim, output_dim=output_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x_mean = checkpoint["x_mean"].astype(np.float32)
    x_std = checkpoint["x_std"].astype(np.float32)
    Xn = (X - x_mean) / x_std

    with torch.no_grad():
        pred = model(torch.from_numpy(Xn).float()).cpu().numpy().astype(np.float32)

    pred_3 = pred.reshape(pred.shape[0], 25, 3)
    true_3 = Y.reshape(Y.shape[0], 25, 3)
    mask_3 = M.reshape(M.shape[0], 25, 3)

    avg_pred_dxyz = pred_3.mean(axis=0)
    avg_true_dxyz = true_3.mean(axis=0)
    avg_mask = mask_3.mean(axis=0)[:, 0]  # visibility frequency per point across subset
    avg_pred_dz = avg_pred_dxyz[:, 2].astype(np.float32)
    avg_true_dz = avg_true_dxyz[:, 2].astype(np.float32)

    pts = neutral_template_px.copy()
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    span_x = max(float(x_max - x_min), 1e-6)
    span_y = max(float(y_max - y_min), 1e-6)

    margin = float(args.patch_margin)
    pts_roi = np.empty_like(pts, dtype=np.float32)
    pts_roi[:, 0] = ((pts[:, 0] - x_min + margin) / (span_x + 2 * margin)) * (args.size - 1)
    pts_roi[:, 1] = ((pts[:, 1] - y_min + margin) / (span_y + 2 * margin)) * (args.size - 1)

    mask = build_patch_mask(pts_roi, args.size, args.size, margin=0)
    height_pred = idw_heightmap(pts_roi, avg_pred_dz, args.size, args.size, power=args.idw_power)
    height_true = idw_heightmap(pts_roi, avg_true_dz, args.size, args.size, power=args.idw_power)
    height_pred *= (mask > 0).astype(np.float32)
    height_true *= (mask > 0).astype(np.float32)

    height_pred_u16, pred_min, pred_max = normalize_to_u16(height_pred, mask)
    height_true_u16, true_min, true_max = normalize_to_u16(height_true, mask)

    ref_img = draw_reference(pts_roi, args.size, args.size, mask, left_cheek_ids)

    json_out = {
        "description": "Average regional approximate cheek height map export from filtered face-press subset.",
        "source_local_index": source_local_index,
        "source_mp_id": source_mp_id,
        "subset_count": int(X.shape[0]),
        "texture_size": int(args.size),
        "pressure_distribution": {str(int(k)): int(v) for k, v in zip(*np.unique(pressure_index, return_counts=True))},
        "trial_count": int(len(np.unique(trials))),
        "fingertip_dist_px": {
            "mean": float(np.mean(fingertip_dist_px)),
            "median": float(np.median(fingertip_dist_px)),
            "min": float(np.min(fingertip_dist_px)),
            "max": float(np.max(fingertip_dist_px)),
        },
        "visible_points": {
            "mean": float(np.mean(visible_points)),
            "median": float(np.median(visible_points)),
            "min": int(np.min(visible_points)),
            "max": int(np.max(visible_points)),
        },
        "alignment_error_px": {
            "mean": float(np.mean(alignment_error_px)),
            "median": float(np.median(alignment_error_px)),
            "min": float(np.min(alignment_error_px)),
            "max": float(np.max(alignment_error_px)),
        },
        "height_map_region": "regional cheek patch defined by the 25 neutral template points",
        "note": "This is an interpolated approximate regional height map from sparse landmark deformation, not a true dense depth map.",
        "png_note": "The exported PNG height maps are 16-bit grayscale images normalized over the patch region.",
        "files": {
            "heightmap_pred_png": "regional_heightmap_pred.png",
            "heightmap_pred_png_bit_depth": 16,
            "heightmap_true_png": "regional_heightmap_true_avg.png",
            "heightmap_true_png_bit_depth": 16,
            "heightmap_pred_float_npy": "regional_heightmap_pred_float.npy",
            "heightmap_true_float_npy": "regional_heightmap_true_avg_float.npy",
            "heightmap_pred_csv": "regional_heightmap_pred_float.csv",
            "heightmap_true_csv": "regional_heightmap_true_avg_float.csv",
            "reference_patch_png": "regional_patch_reference.png",
            "average_sparse_npz": "regional_average_sparse_points.npz",
            "average_sparse_csv": "regional_average_sparse_points.csv",
        },
        "pred_height_range": {"min": pred_min, "max": pred_max},
        "true_height_range": {"min": true_min, "max": true_max},
    }

    np.save(output_dir / "regional_heightmap_pred_float.npy", height_pred.astype(np.float32))
    np.save(output_dir / "regional_heightmap_true_avg_float.npy", height_true.astype(np.float32))
    save_heightmap_csv(output_dir / "regional_heightmap_pred_float.csv", height_pred.astype(np.float32))
    save_heightmap_csv(output_dir / "regional_heightmap_true_avg_float.csv", height_true.astype(np.float32))
    cv2.imwrite(str(output_dir / "regional_heightmap_pred.png"), height_pred_u16)
    cv2.imwrite(str(output_dir / "regional_heightmap_true_avg.png"), height_true_u16)
    cv2.imwrite(str(output_dir / "regional_patch_reference.png"), ref_img)

    np.savez_compressed(
        output_dir / "regional_average_sparse_points.npz",
        source_local_index=np.int32(source_local_index),
        source_mp_id=np.int32(source_mp_id),
        left_cheek_ids=left_cheek_ids,
        patch_edges=patch_edges,
        neutral_template_px=neutral_template_px.astype(np.float32),
        roi_points_px=pts_roi.astype(np.float32),
        avg_visibility_ratio=avg_mask.astype(np.float32),
        avg_pred_dxyz=avg_pred_dxyz.astype(np.float32),
        avg_true_dxyz=avg_true_dxyz.astype(np.float32),
        avg_pred_dz=avg_pred_dz.astype(np.float32),
        avg_true_dz=avg_true_dz.astype(np.float32),
    )
    save_sparse_points_csv(
        output_dir / "regional_average_sparse_points.csv",
        left_cheek_ids,
        neutral_template_px.astype(np.float32),
        pts_roi.astype(np.float32),
        avg_mask.astype(np.float32),
        avg_pred_dxyz.astype(np.float32),
        avg_true_dxyz.astype(np.float32),
    )

    (output_dir / "regional_heightmap_summary.json").write_text(json.dumps(json_out, indent=2))

    print("=" * 72)
    print("Created regional average approximate height-map export")
    print(f"Subset count: {X.shape[0]}")
    print(f"Source: local {source_local_index} | MP {source_mp_id}")
    print(f"Texture size: {args.size}x{args.size}")
    print("Saved:")
    print(f"  {output_dir / 'regional_heightmap_pred.png'}")
    print(f"  {output_dir / 'regional_heightmap_pred_float.npy'}")
    print(f"  {output_dir / 'regional_heightmap_pred_float.csv'}")
    print(f"  {output_dir / 'regional_patch_reference.png'}")
    print(f"  {output_dir / 'regional_average_sparse_points.npz'}")
    print(f"  {output_dir / 'regional_average_sparse_points.csv'}")
    print(f"  {output_dir / 'regional_heightmap_summary.json'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
