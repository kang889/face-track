from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy.spatial import cKDTree
import trimesh


def load_obj_points(path: Path) -> np.ndarray:
    """Load OBJ vertices from a mesh or scene."""
    obj = trimesh.load(path, process=False)
    if isinstance(obj, trimesh.Scene):
        pts = []
        for geom in obj.geometry.values():
            if hasattr(geom, "vertices") and len(geom.vertices):
                pts.append(np.asarray(geom.vertices, dtype=np.float64))
        if not pts:
            raise RuntimeError(f"No vertices found in scene: {path}")
        return np.vstack(pts)
    if not hasattr(obj, "vertices") or len(obj.vertices) == 0:
        raise RuntimeError(f"No vertices found in mesh: {path}")
    return np.asarray(obj.vertices, dtype=np.float64)


def percentile_crop(points: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Remove extreme scan outliers using percentile bounds."""
    lo = np.percentile(points, low, axis=0)
    hi = np.percentile(points, high, axis=0)
    keep = np.all((points >= lo) & (points <= hi), axis=1)
    return points[keep]


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    if voxel <= 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    out = np.zeros((inv.max() + 1, 3), dtype=np.float64)
    counts = np.bincount(inv)
    np.add.at(out, inv, points)
    out /= counts[:, None]
    return out


def kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return R,t so (R @ src.T).T + t approximates dst."""
    cs = src.mean(axis=0)
    cd = dst.mean(axis=0)
    xs = src - cs
    xd = dst - cd
    h = xs.T @ xd
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = cd - r @ cs
    return r, t


def icp_point_to_point(
    source: np.ndarray,
    target: np.ndarray,
    max_iters: int = 35,
    trim_fraction: float = 0.70,
    max_corr_dist: float | None = None,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    Robust point-to-point ICP.
    source is transformed to target.
    """
    src = source.copy()
    tgt = target.copy()

    # Initial center alignment.
    t0 = tgt.mean(axis=0) - src.mean(axis=0)
    src = src + t0
    r_total = np.eye(3)
    t_total = t0.copy()
    errors = []

    tree = cKDTree(tgt)
    for _ in range(max_iters):
        dist, idx = tree.query(src, k=1)
        keep = np.isfinite(dist)

        if max_corr_dist is not None:
            keep &= dist < max_corr_dist

        if keep.sum() < 20:
            break

        kept_dist = dist[keep]
        cutoff = np.quantile(kept_dist, trim_fraction)
        keep &= dist <= cutoff

        if keep.sum() < 20:
            break

        matched_src = src[keep]
        matched_tgt = tgt[idx[keep]]
        r, t = kabsch(matched_src, matched_tgt)

        src = (r @ src.T).T + t
        r_total = r @ r_total
        t_total = r @ t_total + t

        mean_err = float(np.mean(np.linalg.norm(src[keep] - matched_tgt, axis=1)))
        errors.append(mean_err)

        if len(errors) > 2 and abs(errors[-2] - errors[-1]) < 1e-6:
            break

    return r_total, t_total, errors


def normalized_coords(points: np.ndarray, ref_points: np.ndarray) -> np.ndarray:
    center = (ref_points.min(axis=0) + ref_points.max(axis=0)) * 0.5
    half = np.maximum((ref_points.max(axis=0) - ref_points.min(axis=0)) * 0.5, 1e-8)
    return (points - center) / half


def crop_by_normalized_roi(
    points: np.ndarray,
    ref_points: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> np.ndarray:
    n = normalized_coords(points, ref_points)
    keep = (
        (n[:, 0] >= x_min) & (n[:, 0] <= x_max) &
        (n[:, 1] >= y_min) & (n[:, 1] <= y_max) &
        (n[:, 2] >= z_min) & (n[:, 2] <= z_max)
    )
    return points[keep]


def estimate_normals_pca(points: np.ndarray, k: int = 24) -> np.ndarray:
    """Estimate local normals using PCA and orient them roughly toward +Z."""
    tree = cKDTree(points)
    _, idxs = tree.query(points, k=min(k, len(points)))
    normals = np.zeros_like(points)
    for i, idx in enumerate(idxs):
        nb = points[idx]
        cov = np.cov((nb - nb.mean(axis=0)).T)
        w, v = np.linalg.eigh(cov)
        n = v[:, np.argmin(w)]
        # Orient toward positive z by default; flip later with --flip-sign if needed.
        if n[2] < 0:
            n = -n
        normals[i] = n
    return normals


def compute_roi_displacement(
    neutral_roi: np.ndarray,
    pressed_aligned: np.ndarray,
    use_normals: bool = True,
    flip_sign: bool = False,
    max_dist: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """For each neutral ROI point, find nearest pressed point and compute signed displacement."""
    tree = cKDTree(pressed_aligned)
    dist, idx = tree.query(neutral_roi, k=1)
    corr = pressed_aligned[idx]
    vec = corr - neutral_roi

    if use_normals:
        normals = estimate_normals_pca(neutral_roi)
        disp = np.sum(vec * normals, axis=1)
    else:
        disp = vec[:, 2]

    if flip_sign:
        disp = -disp

    valid = np.ones(len(disp), dtype=bool)
    if max_dist is not None:
        valid &= dist <= max_dist
    return disp, valid


def rasterize_heightmap(
    roi_points: np.ndarray,
    disp: np.ndarray,
    valid: np.ndarray,
    res: int,
    neutral_value: int = 128,
    strength: float | None = None,
    fill_radius_px: int = 8,
    blur_radius: float = 1.2,
) -> tuple[Image.Image, Image.Image, np.ndarray]:
    """Project local cheek points to XY patch image and convert signed displacement to grayscale."""
    pts = roi_points[valid]
    vals = disp[valid]

    if len(pts) < 20:
        raise RuntimeError("Too few valid ROI points to rasterize height map.")

    # Map local XY bounding box into image coordinates.
    xy = pts[:, :2]
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    span = np.maximum(hi - lo, 1e-8)
    uv = (xy - lo) / span
    px = np.clip((uv[:, 0] * (res - 1)).astype(int), 0, res - 1)
    py = np.clip(((1.0 - uv[:, 1]) * (res - 1)).astype(int), 0, res - 1)

    # Auto strength from robust percentile.
    if strength is None or strength <= 0:
        q = np.percentile(np.abs(vals), 95)
        strength = max(float(q), 1e-6)

    # Convert displacement to 8-bit: negative/inward should be darker if sign is set correctly.
    gray_vals = neutral_value + (vals / strength) * 90.0
    gray_vals = np.clip(gray_vals, 0, 255)

    # Put sparse points into image using averaging.
    acc = np.zeros((res, res), dtype=np.float64)
    cnt = np.zeros((res, res), dtype=np.float64)
    for x, y, g in zip(px, py, gray_vals):
        acc[y, x] += g
        cnt[y, x] += 1.0

    sparse_valid = cnt > 0
    sparse = np.full((res, res), neutral_value, dtype=np.float64)
    sparse[sparse_valid] = acc[sparse_valid] / cnt[sparse_valid]

    # Fill by nearest neighbor in 2D, limited by radius.
    yy, xx = np.mgrid[0:res, 0:res]
    valid_pixels = np.column_stack([py, px])
    all_pixels = np.column_stack([yy.ravel(), xx.ravel()])
    tree2 = cKDTree(valid_pixels)
    d, nearest = tree2.query(all_pixels, k=1)

    filled = np.full(res * res, neutral_value, dtype=np.float64)
    nearest_vals = gray_vals[nearest]
    use = d <= fill_radius_px
    filled[use] = nearest_vals[use]
    filled = filled.reshape(res, res)

    mask = (use.reshape(res, res).astype(np.uint8) * 255)
    img = Image.fromarray(np.clip(filled, 0, 255).astype(np.uint8), mode="L")
    mask_img = Image.fromarray(mask, mode="L")

    if blur_radius > 0:
        # Blur only visually; keep mask separate.
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        # restore outside-mask neutral
        arr = np.asarray(img).copy()
        arr[mask == 0] = neutral_value
        img = Image.fromarray(arr, mode="L")

    meta = {
        "xy_bbox": {"min": lo.tolist(), "max": hi.tolist()},
        "strength_used": float(strength),
        "valid_points": int(len(vals)),
    }
    return img, mask_img, meta


def save_debug_projection(path: Path, neutral: np.ndarray, pressed: np.ndarray, roi: np.ndarray | None = None):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    pairs = [(0, 1, "X-Y"), (0, 2, "X-Z"), (1, 2, "Y-Z")]
    for ax, (a, b, title) in zip(axes, pairs):
        ax.scatter(neutral[:, a], neutral[:, b], s=0.5, alpha=0.25, label="neutral")
        ax.scatter(pressed[:, a], pressed[:, b], s=0.5, alpha=0.25, label="pressed aligned")
        if roi is not None and len(roi):
            ax.scatter(roi[:, a], roi[:, b], s=1.5, alpha=0.8, label="neutral ROI")
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
    axes[0].legend(markerscale=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Prototype pipeline: align iPhone neutral/pressed cheek OBJ scans and export a local cheek deformation height map."
    )
    ap.add_argument("--neutral", type=Path, required=True, help="Neutral OBJ scan")
    ap.add_argument("--pressed", type=Path, required=True, help="Pressed-cheek OBJ scan")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--name", default="scan_pair", help="Name prefix for outputs")
    ap.add_argument("--heightmap-res", type=int, default=256)
    ap.add_argument("--voxel", type=float, default=0.003, help="Downsample voxel size in scan units")
    ap.add_argument("--trim-fraction", type=float, default=0.70)
    ap.add_argument("--icp-max-iters", type=int, default=35)
    ap.add_argument("--icp-max-corr-dist", type=float, default=0.08)
    # ROI in normalized coordinates of neutral scan bbox after outlier crop.
    ap.add_argument("--roi-x-min", type=float, default=-0.85)
    ap.add_argument("--roi-x-max", type=float, default=0.10)
    ap.add_argument("--roi-y-min", type=float, default=-0.35)
    ap.add_argument("--roi-y-max", type=float, default=0.55)
    ap.add_argument("--roi-z-min", type=float, default=-1.0)
    ap.add_argument("--roi-z-max", type=float, default=1.0)
    ap.add_argument("--no-normals", action="store_true", help="Use Z difference instead of PCA normal projection")
    ap.add_argument("--flip-sign", action="store_true", help="Flip displacement sign if indentation appears bright instead of dark")
    ap.add_argument("--max-nearest-dist", type=float, default=0.035, help="Reject neutral ROI points if nearest pressed point is too far")
    ap.add_argument("--neutral-value", type=int, default=128)
    ap.add_argument("--strength", type=float, default=0.0, help="Manual displacement-to-grayscale scale. 0 = auto")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    neutral_raw = load_obj_points(args.neutral)
    pressed_raw = load_obj_points(args.pressed)

    neutral_crop = percentile_crop(neutral_raw, 1, 99)
    pressed_crop = percentile_crop(pressed_raw, 1, 99)

    neutral_ds = voxel_downsample(neutral_crop, args.voxel)
    pressed_ds = voxel_downsample(pressed_crop, args.voxel)

    r, t, errors = icp_point_to_point(
        source=pressed_ds,
        target=neutral_ds,
        max_iters=args.icp_max_iters,
        trim_fraction=args.trim_fraction,
        max_corr_dist=args.icp_max_corr_dist,
    )

    pressed_aligned = (r @ pressed_crop.T).T + t
    pressed_aligned_ds = (r @ pressed_ds.T).T + t

    neutral_roi = crop_by_normalized_roi(
        neutral_crop,
        neutral_crop,
        args.roi_x_min, args.roi_x_max,
        args.roi_y_min, args.roi_y_max,
        args.roi_z_min, args.roi_z_max,
    )

    if len(neutral_roi) < 50:
        raise RuntimeError(
            f"ROI has too few points ({len(neutral_roi)}). Adjust --roi-* parameters."
        )

    disp, valid = compute_roi_displacement(
        neutral_roi,
        pressed_aligned,
        use_normals=not args.no_normals,
        flip_sign=args.flip_sign,
        max_dist=args.max_nearest_dist,
    )

    heightmap, mask, hm_meta = rasterize_heightmap(
        neutral_roi,
        disp,
        valid,
        res=args.heightmap_res,
        neutral_value=args.neutral_value,
        strength=args.strength if args.strength > 0 else None,
    )

    heightmap_path = args.out_dir / f"{args.name}_cheek_heightmap_{args.heightmap_res}.png"
    mask_path = args.out_dir / f"{args.name}_cheek_valid_mask_{args.heightmap_res}.png"
    debug_path = args.out_dir / f"{args.name}_alignment_debug.png"
    meta_path = args.out_dir / f"{args.name}_metadata.json"

    heightmap.save(heightmap_path)
    mask.save(mask_path)
    save_debug_projection(debug_path, neutral_ds, pressed_aligned_ds, neutral_roi)

    metadata = {
        "neutral": str(args.neutral),
        "pressed": str(args.pressed),
        "raw_vertices": {
            "neutral": int(len(neutral_raw)),
            "pressed": int(len(pressed_raw)),
        },
        "after_percentile_crop": {
            "neutral": int(len(neutral_crop)),
            "pressed": int(len(pressed_crop)),
        },
        "downsampled_vertices": {
            "neutral": int(len(neutral_ds)),
            "pressed": int(len(pressed_ds)),
        },
        "icp": {
            "errors": errors,
            "final_mean_error": errors[-1] if errors else None,
            "trim_fraction": args.trim_fraction,
            "max_corr_dist": args.icp_max_corr_dist,
        },
        "roi_parameters": {
            "x_min": args.roi_x_min,
            "x_max": args.roi_x_max,
            "y_min": args.roi_y_min,
            "y_max": args.roi_y_max,
            "z_min": args.roi_z_min,
            "z_max": args.roi_z_max,
        },
        "displacement": {
            "method": "PCA local normal projection" if not args.no_normals else "Z-axis difference",
            "flip_sign": args.flip_sign,
            "valid_points": int(valid.sum()),
            "total_roi_points": int(len(valid)),
            "min": float(np.min(disp[valid])) if valid.any() else None,
            "max": float(np.max(disp[valid])) if valid.any() else None,
            "mean": float(np.mean(disp[valid])) if valid.any() else None,
        },
        "heightmap": hm_meta,
        "outputs": {
            "heightmap": heightmap_path.name,
            "valid_mask": mask_path.name,
            "alignment_debug": debug_path.name,
            "metadata": meta_path.name,
        },
        "note": "This is a prototype. Inspect alignment_debug and heightmap before using the result as the cheek-map input to the UV exporter.",
    }

    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("=" * 72)
    print("Created local cheek scan-derived height map")
    print(f"Heightmap : {heightmap_path}")
    print(f"Mask      : {mask_path}")
    print(f"Debug     : {debug_path}")
    print(f"Metadata  : {meta_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
