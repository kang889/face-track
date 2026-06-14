from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


DEFAULTS = {
    "canvas_size": 1024,
    "base_mode": "prototype",  # flat or prototype
    "side": "right",           # anatomical side
    "neutral_16": 32768,
    "outside_face_16": 0,
    "face_depth_span_16": 5000,
    "indent_range_16": 7000,
    "cheek_center_x": 0.34,
    "cheek_center_y": 0.56,
    "cheek_width": 0.22,
    "cheek_height": 0.24,
    "cheek_rotation_deg": -8.0,
    "feather_px": 18,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export a cheek-only full-face height map by injecting a measured cheek deformation patch into a neutral/prototype full-face canvas."
    )
    p.add_argument("--cheek-map", type=Path, required=True, help="Input cheek height map PNG.")
    p.add_argument("--out-dir", type=Path, default=Path("full_face_heightmap_export"), help="Output folder.")
    p.add_argument("--canvas-size", type=int, default=DEFAULTS["canvas_size"], help="Square output resolution, e.g. 1024.")
    p.add_argument("--base-mode", choices=["flat", "prototype"], default=DEFAULTS["base_mode"], help="flat = neutral face interior only, prototype = synthetic face-like neutral base.")
    p.add_argument("--side", choices=["right", "left"], default=DEFAULTS["side"], help="Anatomical cheek side. For a front-facing image, anatomical right = viewer-left.")
    p.add_argument("--neutral-16", type=int, default=DEFAULTS["neutral_16"], help="Neutral face value in 16-bit output.")
    p.add_argument("--outside-face-16", type=int, default=DEFAULTS["outside_face_16"], help="Value outside the face mask.")
    p.add_argument("--face-depth-span-16", type=int, default=DEFAULTS["face_depth_span_16"], help="Total depth variation for prototype base face.")
    p.add_argument("--indent-range-16", type=int, default=DEFAULTS["indent_range_16"], help="Maximum inward deformation range applied from the cheek map.")
    p.add_argument("--cheek-center-x", type=float, default=DEFAULTS["cheek_center_x"], help="Normalized cheek center x in output canvas.")
    p.add_argument("--cheek-center-y", type=float, default=DEFAULTS["cheek_center_y"], help="Normalized cheek center y in output canvas.")
    p.add_argument("--cheek-width", type=float, default=DEFAULTS["cheek_width"], help="Normalized cheek patch width in output canvas.")
    p.add_argument("--cheek-height", type=float, default=DEFAULTS["cheek_height"], help="Normalized cheek patch height in output canvas.")
    p.add_argument("--cheek-rotation-deg", type=float, default=DEFAULTS["cheek_rotation_deg"], help="Cheek patch rotation in degrees.")
    p.add_argument("--feather-px", type=int, default=DEFAULTS["feather_px"], help="Boundary feather for smooth blending.")
    p.add_argument("--save-config", action="store_true", help="Also save a reusable config.json.")
    return p.parse_args()


def load_cheek_map(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    img = Image.open(path).convert("RGBA")
    arr = np.asarray(img).astype(np.float32)
    rgb = arr[..., :3]
    alpha = arr[..., 3] / 255.0
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255.0

    # If the file carries a meaningful alpha channel, use it.
    # Otherwise, treat near-black as background/no-data.
    if np.min(alpha) < 0.999:
        valid = alpha > 0.01
    else:
        valid = gray > 0.02
    gray = np.where(valid, gray, 0.0)
    return gray, valid.astype(np.uint8)


def make_face_mask(size: int) -> np.ndarray:
    """Procedural full-face silhouette mask, slightly more face-like than a plain oval."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    x = (xx / (size - 1) - 0.5) / 0.30
    y = (yy / (size - 1) - 0.53) / 0.39

    # Base head superellipse.
    head = (np.abs(x) ** 2.1 + np.abs(y) ** 2.5) <= 1.0

    # Chin narrowing / jaw taper.
    lower = y > 0.15
    width_scale = 1.0 - 0.38 * np.clip((y - 0.15) / 0.85, 0.0, 1.0)
    jaw = (np.abs(x) / np.maximum(width_scale, 1e-5) <= 1.0) & (np.abs(y) <= 1.0)
    mask = head & (~lower | jaw)

    # Slight neck cut.
    neck_cut = (yy > size * 0.93) & ((np.abs(xx - size * 0.5) > size * 0.18))
    mask[neck_cut] = False
    return (mask.astype(np.uint8) * 255)



def gaussian2d(x: np.ndarray, y: np.ndarray, cx: float, cy: float, sx: float, sy: float) -> np.ndarray:
    return np.exp(-0.5 * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))



def make_prototype_face_base(size: int, face_mask_u8: np.ndarray, neutral_16: int, span_16: int) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    x = xx / (size - 1)
    y = yy / (size - 1)

    # Relief-like synthetic face features.
    z = np.zeros((size, size), dtype=np.float32)

    # Global frontal convexity.
    z += 0.85 * gaussian2d(x, y, 0.50, 0.52, 0.22, 0.29)
    # Forehead.
    z += 0.28 * gaussian2d(x, y, 0.50, 0.27, 0.16, 0.10)
    # Nose bridge + tip.
    z += 0.40 * gaussian2d(x, y, 0.50, 0.48, 0.035, 0.13)
    z += 0.33 * gaussian2d(x, y, 0.50, 0.57, 0.055, 0.05)
    # Cheeks.
    z += 0.18 * gaussian2d(x, y, 0.37, 0.56, 0.10, 0.10)
    z += 0.18 * gaussian2d(x, y, 0.63, 0.56, 0.10, 0.10)
    # Brow / supraorbital region.
    z += 0.10 * gaussian2d(x, y, 0.40, 0.39, 0.08, 0.035)
    z += 0.10 * gaussian2d(x, y, 0.60, 0.39, 0.08, 0.035)
    # Chin.
    z += 0.16 * gaussian2d(x, y, 0.50, 0.79, 0.10, 0.08)
    # Eye sockets slight inward.
    z -= 0.08 * gaussian2d(x, y, 0.41, 0.46, 0.055, 0.03)
    z -= 0.08 * gaussian2d(x, y, 0.59, 0.46, 0.055, 0.03)
    # Philtrum / mouth region slight inward.
    z -= 0.05 * gaussian2d(x, y, 0.50, 0.67, 0.09, 0.035)

    # Normalize only inside face.
    face = face_mask_u8 > 0
    z_face = z[face]
    z_norm = (z - z_face.min()) / max(z_face.max() - z_face.min(), 1e-6)
    z_centered = (z_norm - 0.5) * span_16

    base = np.full((size, size), float(neutral_16), dtype=np.float32)
    base[face] = neutral_16 + z_centered[face]
    return base



def transform_patch_to_canvas(gray: np.ndarray, valid: np.ndarray, size: int, center_x: float, center_y: float,
                              width_n: float, height_n: float, rotation_deg: float, side: str,
                              feather_px: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns patch_gray_on_canvas [0..1], alpha_on_canvas [0..1], hard_mask_u8 [0/255]."""
    # Mirror horizontally for anatomical left if needed.
    if side == "left":
        gray = np.fliplr(gray)
        valid = np.fliplr(valid)
        center_x = 1.0 - center_x
        rotation_deg = -rotation_deg

    patch_w = max(2, int(round(width_n * size)))
    patch_h = max(2, int(round(height_n * size)))

    gray_img = Image.fromarray(np.clip(gray * 255.0, 0, 255).astype(np.uint8), mode="L")
    valid_img = Image.fromarray((valid > 0).astype(np.uint8) * 255, mode="L")

    gray_resized = gray_img.resize((patch_w, patch_h), resample=Image.Resampling.BICUBIC)
    valid_resized = valid_img.resize((patch_w, patch_h), resample=Image.Resampling.NEAREST)

    gray_rot = gray_resized.rotate(rotation_deg, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=0)
    valid_rot = valid_resized.rotate(rotation_deg, resample=Image.Resampling.NEAREST, expand=False, fillcolor=0)

    alpha_soft = valid_rot.filter(ImageFilter.GaussianBlur(radius=max(feather_px, 0))) if feather_px > 0 else valid_rot

    canvas_gray = Image.new("L", (size, size), color=0)
    canvas_alpha = Image.new("L", (size, size), color=0)
    canvas_mask = Image.new("L", (size, size), color=0)

    x0 = int(round(center_x * size - patch_w / 2))
    y0 = int(round(center_y * size - patch_h / 2))

    canvas_gray.paste(gray_rot, (x0, y0))
    canvas_alpha.paste(alpha_soft, (x0, y0))
    canvas_mask.paste(valid_rot, (x0, y0))

    return (np.asarray(canvas_gray).astype(np.float32) / 255.0,
            np.asarray(canvas_alpha).astype(np.float32) / 255.0,
            np.asarray(canvas_mask).astype(np.uint8))



def make_shaded_preview(height16: np.ndarray) -> np.ndarray:
    """Simple shaded preview from gradients."""
    h = height16.astype(np.float32)
    gx = np.gradient(h, axis=1)
    gy = np.gradient(h, axis=0)
    nx = -gx / 4000.0
    ny = -gy / 4000.0
    nz = np.ones_like(h)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
    nx /= norm
    ny /= norm
    nz /= norm
    light = np.array([-0.35, -0.35, 0.87], dtype=np.float32)
    shade = np.clip(nx * light[0] + ny * light[1] + nz * light[2], 0, 1)
    preview = (shade * 255).astype(np.uint8)
    return preview



def percentile_normalize_u8(arr16: np.ndarray, mask: np.ndarray) -> np.ndarray:
    arr = arr16.astype(np.float32)
    if np.any(mask):
        vals = arr[mask > 0]
        lo = np.percentile(vals, 1)
        hi = np.percentile(vals, 99)
    else:
        lo, hi = float(arr.min()), float(arr.max())
    out = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    return (out * 255).astype(np.uint8)



def overlay_debug(preview_u8: np.ndarray, face_mask_u8: np.ndarray, cheek_mask_u8: np.ndarray) -> Image.Image:
    rgb = np.stack([preview_u8] * 3, axis=-1)
    # Blue face outline and yellow cheek outline.
    img = Image.fromarray(rgb, mode="RGB")
    face_outline = Image.fromarray(face_mask_u8, mode="L").filter(ImageFilter.FIND_EDGES)
    cheek_outline = Image.fromarray(cheek_mask_u8, mode="L").filter(ImageFilter.FIND_EDGES)
    arr = np.asarray(img).copy()
    fo = np.asarray(face_outline) > 0
    co = np.asarray(cheek_outline) > 0
    arr[fo] = [80, 170, 255]
    arr[co] = [255, 220, 0]
    return Image.fromarray(arr, mode="RGB")



def save_u16_png(path: Path, arr16: np.ndarray) -> None:
    img = Image.fromarray(arr16.astype(np.uint16), mode="I;16")
    img.save(path)



def main() -> None:
    args = parse_args()
    if not args.cheek_map.exists():
        raise FileNotFoundError(f"Cheek map not found: {args.cheek_map}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    size = args.canvas_size
    face_mask = make_face_mask(size)
    base = np.full((size, size), float(args.outside_face_16), dtype=np.float32)
    face = face_mask > 0
    if args.base_mode == "flat":
        base[face] = float(args.neutral_16)
    else:
        base[:] = float(args.outside_face_16)
        base_prototype = make_prototype_face_base(size, face_mask, args.neutral_16, args.face_depth_span_16)
        base[face] = base_prototype[face]

    cheek_gray, cheek_valid = load_cheek_map(args.cheek_map)
    patch_gray_canvas, alpha_canvas, cheek_mask_canvas = transform_patch_to_canvas(
        cheek_gray, cheek_valid, size,
        args.cheek_center_x, args.cheek_center_y,
        args.cheek_width, args.cheek_height,
        args.cheek_rotation_deg, args.side,
        args.feather_px
    )

    # Keep only pixels where the patch overlaps the face.
    cheek_mask_canvas = np.where(face_mask > 0, cheek_mask_canvas, 0).astype(np.uint8)
    alpha_canvas = np.where(face_mask > 0, alpha_canvas, 0.0)

    # Compute inward deformation factor from the cheek map values.
    valid_patch = cheek_mask_canvas > 0
    if not np.any(valid_patch):
        raise RuntimeError("Cheek patch did not overlap the face. Adjust cheek placement parameters.")

    valid_vals = patch_gray_canvas[valid_patch]
    pmin = float(valid_vals.min())
    pmax = float(valid_vals.max())
    inward = np.zeros_like(patch_gray_canvas, dtype=np.float32)
    if pmax - pmin < 1e-6:
        inward[valid_patch] = 0.5
    else:
        inward[valid_patch] = 1.0 - ((patch_gray_canvas[valid_patch] - pmin) / (pmax - pmin))

    deformed = base.copy()
    deformed_amount = inward * float(args.indent_range_16)
    target = base - deformed_amount
    deformed = base * (1.0 - alpha_canvas) + target * alpha_canvas
    deformed[~face] = float(args.outside_face_16)
    deformed = np.clip(deformed, 0, 65535).astype(np.uint16)

    # Extra outputs.
    face_mask_u8 = face_mask.astype(np.uint8)
    valid_mask_u8 = (valid_patch.astype(np.uint8) * 255)
    alpha_mask_u8 = np.clip(alpha_canvas * 255.0, 0, 255).astype(np.uint8)
    preview_u8 = percentile_normalize_u8(deformed, face_mask_u8)
    shaded_preview_u8 = make_shaded_preview(deformed)
    debug_overlay_img = overlay_debug(preview_u8, face_mask_u8, valid_mask_u8)

    # Save files.
    save_u16_png(out_dir / "full_face_heightmap_16bit.png", deformed)
    Image.fromarray(preview_u8, mode="L").save(out_dir / "full_face_heightmap_preview_8bit.png")
    Image.fromarray(shaded_preview_u8, mode="L").save(out_dir / "full_face_heightmap_shaded_preview.png")
    Image.fromarray(face_mask_u8, mode="L").save(out_dir / "full_face_face_mask.png")
    Image.fromarray(valid_mask_u8, mode="L").save(out_dir / "full_face_valid_mask.png")
    Image.fromarray(alpha_mask_u8, mode="L").save(out_dir / "full_face_valid_alpha_mask.png")
    debug_overlay_img.save(out_dir / "full_face_debug_overlay.png")

    metadata = {
        "description": "Cheek-only full-face height-map prototype exporter.",
        "input_cheek_map": str(args.cheek_map),
        "canvas_size": size,
        "base_mode": args.base_mode,
        "anatomical_side": args.side,
        "neutral_16": args.neutral_16,
        "outside_face_16": args.outside_face_16,
        "face_depth_span_16": args.face_depth_span_16,
        "indent_range_16": args.indent_range_16,
        "placement": {
            "cheek_center_x": args.cheek_center_x,
            "cheek_center_y": args.cheek_center_y,
            "cheek_width": args.cheek_width,
            "cheek_height": args.cheek_height,
            "cheek_rotation_deg": args.cheek_rotation_deg,
            "feather_px": args.feather_px,
        },
        "interpretation": {
            "valid_region": "Only the cheek region in full_face_valid_mask.png contains measured deformation.",
            "outside_region": "Remaining facial regions are neutral/prototype placeholders.",
            "dark_semantics": "Darker cheek values are interpreted as more inward indentation.",
        },
        "outputs": {
            "heightmap_16bit": "full_face_heightmap_16bit.png",
            "heightmap_preview_8bit": "full_face_heightmap_preview_8bit.png",
            "shaded_preview": "full_face_heightmap_shaded_preview.png",
            "face_mask": "full_face_face_mask.png",
            "valid_mask": "full_face_valid_mask.png",
            "valid_alpha_mask": "full_face_valid_alpha_mask.png",
            "debug_overlay": "full_face_debug_overlay.png",
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if args.save_config:
        (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    print("=" * 72)
    print("Created full-face height-map export")
    print(f"Input cheek map : {args.cheek_map}")
    print(f"Output folder   : {out_dir}")
    print("Saved:")
    for name in [
        "full_face_heightmap_16bit.png",
        "full_face_heightmap_preview_8bit.png",
        "full_face_heightmap_shaded_preview.png",
        "full_face_face_mask.png",
        "full_face_valid_mask.png",
        "full_face_valid_alpha_mask.png",
        "full_face_debug_overlay.png",
        "metadata.json",
    ]:
        print(f"  {out_dir / name}")
    print("=" * 72)


if __name__ == "__main__":
    main()
