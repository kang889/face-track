from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from collections import deque

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


@dataclass
class Face:
    obj: str
    verts: list[int]
    uvs: list[Optional[int]]


def parse_obj(path: Path):
    verts: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    faces: list[Face] = []
    obj_name = "default"

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("o "):
                obj_name = s[2:].strip()
            elif s.startswith("v "):
                parts = s.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif s.startswith("vt "):
                parts = s.split()
                uvs.append((float(parts[1]), float(parts[2])))
            elif s.startswith("f "):
                verts_idx: list[int] = []
                uvs_idx: list[Optional[int]] = []
                for tok in s.split()[1:]:
                    pieces = tok.split("/")
                    vi = int(pieces[0]) - 1
                    ti = int(pieces[1]) - 1 if len(pieces) > 1 and pieces[1] else None
                    verts_idx.append(vi)
                    uvs_idx.append(ti)
                faces.append(Face(obj=obj_name, verts=verts_idx, uvs=uvs_idx))

    return np.asarray(verts, dtype=np.float32), np.asarray(uvs, dtype=np.float32), faces


def uv_to_px(uv: tuple[float, float] | np.ndarray, res: int) -> tuple[float, float]:
    u, v = float(uv[0]), float(uv[1])
    return (u * (res - 1), (1.0 - v) * (res - 1))


def polygon_points(face: Face, uv_array: np.ndarray, res: int) -> Optional[list[tuple[float, float]]]:
    pts: list[tuple[float, float]] = []
    for ti in face.uvs:
        if ti is None or ti < 0 or ti >= len(uv_array):
            return None
        pts.append(uv_to_px(uv_array[ti], res))
    return pts if len(pts) >= 3 else None


def make_uv_wireframe(faces: list[Face], uv_array: np.ndarray, res: int, object_name: str) -> Image.Image:
    img = Image.new("RGB", (res, res), "white")
    d = ImageDraw.Draw(img)
    for face in faces:
        if face.obj != object_name:
            continue
        pts = polygon_points(face, uv_array, res)
        if pts is not None:
            d.line(pts + [pts[0]], fill=(185, 185, 185), width=max(1, res // 1024))
    return img


def select_cheek_faces(
    verts: np.ndarray,
    faces: list[Face],
    object_name: str,
    side: str,
    model_right_sign: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
) -> list[Face]:
    obj_faces = [f for f in faces if f.obj == object_name]
    obj_vert_idx = sorted({vi for f in obj_faces for vi in f.verts})
    if not obj_vert_idx:
        raise RuntimeError(f"No vertices found for object: {object_name}")

    vv = verts[obj_vert_idx]
    bmin = vv.min(axis=0)
    bmax = vv.max(axis=0)
    center = (bmin + bmax) * 0.5
    half = np.maximum((bmax - bmin) * 0.5, 1e-6)

    # By default anatomical right is model_right_sign side.
    # side_sign is the coordinate side to select in model X.
    side_sign = int(model_right_sign) if side == "right" else -int(model_right_sign)

    selected: list[Face] = []
    for face in obj_faces:
        xyz = verts[face.verts].mean(axis=0)
        xn, yn, zn = (xyz - center) / half
        if (side_sign * xn > x_min and side_sign * xn < x_max and
                yn > y_min and yn < y_max and zn > z_min):
            selected.append(face)
    return selected


def rasterize_faces(faces: list[Face], uv_array: np.ndarray, res: int, fill: int = 255) -> Image.Image:
    mask = Image.new("L", (res, res), 0)
    d = ImageDraw.Draw(mask)
    for face in faces:
        pts = polygon_points(face, uv_array, res)
        if pts is not None:
            d.polygon(pts, fill=fill)
    return mask


def keep_largest_component(mask: Image.Image) -> Image.Image:
    arr = np.asarray(mask) > 0
    h, w = arr.shape
    visited = np.zeros((h, w), dtype=bool)
    best_pixels: list[tuple[int, int]] = []
    ys, xs = np.nonzero(arr)
    for sy, sx in zip(ys, xs):
        if visited[sy, sx]:
            continue
        q = deque([(int(sy), int(sx))])
        visited[sy, sx] = True
        pixels: list[tuple[int, int]] = []
        while q:
            y, x = q.popleft()
            pixels.append((y, x))
            for ny in (y - 1, y, y + 1):
                for nx in (x - 1, x, x + 1):
                    if ny == y and nx == x:
                        continue
                    if 0 <= ny < h and 0 <= nx < w and arr[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))
        if len(pixels) > len(best_pixels):
            best_pixels = pixels
    out = np.zeros((h, w), dtype=np.uint8)
    for y, x in best_pixels:
        out[y, x] = 255
    return Image.fromarray(out, mode="L")


def mask_bbox(mask: Image.Image) -> tuple[int, int, int, int]:
    bbox = mask.getbbox()
    if bbox is None:
        raise RuntimeError("Selected cheek mask is empty. Adjust selection parameters.")
    return bbox



def fill_invalid_with_blur(gray: np.ndarray, valid: np.ndarray, iterations: int = 30) -> np.ndarray:
    """Fill invalid source-map pixels so resizing does not treat black background as deep indentation."""
    filled = gray.astype(np.float32).copy()
    mask = valid.astype(np.float32).copy()
    if not np.any(mask > 0):
        return filled
    # Initialize invalid pixels to mean valid value.
    filled[mask <= 0] = float(filled[mask > 0].mean())
    img = Image.fromarray(np.clip(filled, 0, 255).astype(np.uint8), mode="L")
    mask_img = Image.fromarray(np.clip(mask * 255, 0, 255).astype(np.uint8), mode="L")
    for _ in range(iterations):
        blurred = img.filter(ImageFilter.GaussianBlur(radius=3))
        b = np.asarray(blurred).astype(np.float32)
        cur = np.asarray(img).astype(np.float32)
        m = np.asarray(mask_img).astype(np.float32) / 255.0
        cur = cur * m + b * (1.0 - m)
        img = Image.fromarray(np.clip(cur, 0, 255).astype(np.uint8), mode="L")
    return np.asarray(img).astype(np.float32)

def load_cheek_patch(path: Path, bg_threshold: int = 5) -> tuple[Image.Image, Image.Image]:
    rgba = Image.open(path).convert("RGBA")
    arr = np.asarray(rgba).astype(np.float32)
    rgb = arr[..., :3]
    alpha = arr[..., 3]
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)

    if alpha.min() < 250:
        valid = alpha > 8
    else:
        valid = gray > bg_threshold

    gray_img = Image.fromarray(gray, mode="L")
    valid_img = Image.fromarray((valid.astype(np.uint8) * 255), mode="L")
    bbox = valid_img.getbbox()
    if bbox is None:
        raise RuntimeError("Cheek patch has no valid pixels. Check input image or bg threshold.")
    crop_gray = np.asarray(gray_img.crop(bbox)).astype(np.float32)
    crop_valid = np.asarray(valid_img.crop(bbox)) > 0
    filled = fill_invalid_with_blur(crop_gray, crop_valid, iterations=35)
    return Image.fromarray(np.clip(filled, 0, 255).astype(np.uint8), mode="L"), Image.fromarray((crop_valid.astype(np.uint8) * 255), mode="L")


def paste_patch_into_uv(
    cheek_gray: Image.Image,
    cheek_valid: Image.Image,
    uv_cheek_mask: Image.Image,
    res: int,
    neutral_8: int,
    indent_strength_8: int,
    feather_px: int,
    source_mode: str,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    bbox = mask_bbox(uv_cheek_mask)
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0

    patch_gray = cheek_gray.resize((bw, bh), resample=Image.Resampling.BICUBIC)
    patch_valid = cheek_valid.resize((bw, bh), resample=Image.Resampling.BICUBIC)

    patch_canvas = Image.new("L", (res, res), neutral_8)
    patch_mask_canvas = Image.new("L", (res, res), 0)
    patch_canvas.paste(patch_gray, (x0, y0))
    patch_mask_canvas.paste(patch_valid, (x0, y0))

    # The final valid region follows the OBJ UV cheek region.
    # The source patch is pre-filled before resizing so its original black background
    # does not become false indentation.
    uv_arr = np.asarray(uv_cheek_mask).astype(np.float32) / 255.0
    hard_valid = (uv_arr > 0.5).astype(np.uint8) * 255
    hard_valid_img = Image.fromarray(hard_valid, mode="L")

    alpha_img = hard_valid_img.filter(ImageFilter.GaussianBlur(radius=max(0, feather_px))) if feather_px > 0 else hard_valid_img
    alpha = np.asarray(alpha_img).astype(np.float32) / 255.0

    src = np.asarray(patch_canvas).astype(np.float32)
    out = np.full((res, res), float(neutral_8), dtype=np.float32)

    if source_mode == "preserve":
        target = src
    else:
        vals = src[hard_valid > 0]
        lo = float(vals.min())
        hi = float(vals.max())
        if hi - lo < 1e-6:
            inward = np.zeros_like(src)
        else:
            inward = 1.0 - np.clip((src - lo) / (hi - lo), 0.0, 1.0)
        # Neutral background, darker where inward. This is renderer-friendly.
        target = neutral_8 - inward * float(indent_strength_8)

    out = out * (1.0 - alpha) + target * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="L"), hard_valid_img, alpha_img


def make_debug_overlay(wire: Image.Image, uv_cheek_mask: Image.Image, heightmap: Image.Image, valid_mask: Image.Image) -> Image.Image:
    img = wire.convert("RGBA")
    h_arr = np.asarray(heightmap).astype(np.uint8)
    v_arr = np.asarray(valid_mask).astype(np.uint8)
    # Yellow cheek selected area, semi-transparent; grayscale height inside valid.
    overlay = np.zeros((img.height, img.width, 4), dtype=np.uint8)
    uv = np.asarray(uv_cheek_mask) > 0
    overlay[uv] = [255, 210, 0, 70]
    valid = v_arr > 0
    overlay[valid, 0] = h_arr[valid]
    overlay[valid, 1] = h_arr[valid]
    overlay[valid, 2] = h_arr[valid]
    overlay[valid, 3] = 190
    base = Image.alpha_composite(img, Image.fromarray(overlay, mode="RGBA"))
    # Draw valid outline in blue.
    edge = valid_mask.filter(ImageFilter.FIND_EDGES)
    edge_arr = np.asarray(edge) > 0
    out = np.asarray(base.convert("RGB")).copy()
    out[edge_arr] = [0, 90, 255]
    return Image.fromarray(out, mode="RGB")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export a square PNG height map for a specific OBJ's UV layout, with a cheek deformation patch placed on the anatomical cheek region."
    )
    ap.add_argument("--obj", type=Path, required=True, help="OBJ file with UVs, e.g. FemaleBust.obj")
    ap.add_argument("--cheek-map", type=Path, required=True, help="Regional cheek height/deformation PNG")
    ap.add_argument("--out-dir", type=Path, default=Path("uv_heightmap_export"), help="Output folder")
    ap.add_argument("--resolution", type=int, default=1024, help="Square PNG resolution, e.g. 256, 512, 1024, 2048")
    ap.add_argument("--object-name", default="Anna_Head", help="OBJ object to use for face UVs")
    ap.add_argument("--side", choices=["right", "left"], default="right", help="Anatomical cheek side")
    ap.add_argument("--model-right-sign", type=int, choices=[-1, 1], default=-1,
                    help="Which OBJ X side corresponds to anatomical right. Default -1 matched this model's viewer-left cheek in UV debug.")
    ap.add_argument("--neutral-8", type=int, default=128, help="Neutral/no-displacement value in 8-bit PNG")
    ap.add_argument("--indent-strength-8", type=int, default=90, help="How dark maximum inward indentation becomes below neutral")
    ap.add_argument("--feather-px", type=int, default=18, help="Blend feather radius in pixels")
    ap.add_argument("--source-mode", choices=["indent", "preserve"], default="indent",
                    help="indent maps darker source to values below neutral; preserve pastes the source grayscale values directly")
    ap.add_argument("--keep-all-uv-islands", action="store_true",
                    help="Keep all selected UV islands. By default only the largest cheek island is used to avoid seam/island artifacts.")
    # Selection parameters in normalized head-object bounding-box coordinates.
    ap.add_argument("--cheek-x-min", type=float, default=0.22)
    ap.add_argument("--cheek-x-max", type=float, default=0.72)
    ap.add_argument("--cheek-y-min", type=float, default=0.02)
    ap.add_argument("--cheek-y-max", type=float, default=0.48)
    ap.add_argument("--cheek-z-min", type=float, default=0.15)
    args = ap.parse_args()

    if args.resolution <= 0 or (args.resolution & (args.resolution - 1)) != 0:
        raise ValueError("--resolution should be a positive power-of-two square size, e.g. 256, 512, 1024, 2048.")
    if not args.obj.exists():
        raise FileNotFoundError(args.obj)
    if not args.cheek_map.exists():
        raise FileNotFoundError(args.cheek_map)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    verts, uv_array, faces = parse_obj(args.obj)
    if len(uv_array) == 0:
        raise RuntimeError("OBJ has no UV coordinates (no vt lines). Ask for a UV template or unwrap the mesh first.")

    selected = select_cheek_faces(
        verts=verts,
        faces=faces,
        object_name=args.object_name,
        side=args.side,
        model_right_sign=args.model_right_sign,
        x_min=args.cheek_x_min,
        x_max=args.cheek_x_max,
        y_min=args.cheek_y_min,
        y_max=args.cheek_y_max,
        z_min=args.cheek_z_min,
    )
    if not selected:
        raise RuntimeError("No cheek faces selected. Adjust side or cheek selection parameters.")

    res = args.resolution
    wire = make_uv_wireframe(faces, uv_array, res, args.object_name)
    uv_cheek_mask = rasterize_faces(selected, uv_array, res, fill=255)
    if not args.keep_all_uv_islands:
        uv_cheek_mask = keep_largest_component(uv_cheek_mask)
    cheek_gray, cheek_valid = load_cheek_patch(args.cheek_map)

    heightmap, valid_mask, alpha_mask = paste_patch_into_uv(
        cheek_gray=cheek_gray,
        cheek_valid=cheek_valid,
        uv_cheek_mask=uv_cheek_mask,
        res=res,
        neutral_8=args.neutral_8,
        indent_strength_8=args.indent_strength_8,
        feather_px=args.feather_px,
        source_mode=args.source_mode,
    )

    debug = make_debug_overlay(wire, uv_cheek_mask, heightmap, valid_mask)

    heightmap.save(out / f"heightmap_{res}.png")
    valid_mask.save(out / f"heightmap_valid_mask_{res}.png")
    alpha_mask.save(out / f"heightmap_alpha_mask_{res}.png")
    wire.save(out / f"uv_layout_{res}.png")
    uv_cheek_mask.save(out / f"uv_cheek_region_mask_{res}.png")
    debug.save(out / f"uv_heightmap_debug_{res}.png")

    metadata = {
        "description": "UV-based cheek height map exporter for OBJ face model.",
        "obj": str(args.obj),
        "cheek_map": str(args.cheek_map),
        "object_name": args.object_name,
        "resolution": [res, res],
        "file_format": "PNG 8-bit grayscale",
        "side": args.side,
        "neutral_8": args.neutral_8,
        "indent_strength_8": args.indent_strength_8,
        "source_mode": args.source_mode,
        "selected_cheek_faces": len(selected),
        "notes": [
            "The exported height map is aligned to the OBJ UV layout, not a guessed full-face oval.",
            "Only the valid mask region contains measured/inserted cheek deformation.",
            "All other pixels are neutral/no displacement.",
            "If the cheek appears on the wrong side in the renderer, re-run with --model-right-sign 1 or --side left.",
            "By default, only the largest selected UV island is used to avoid small detached seam islands.",
        ],
        "outputs": {
            "heightmap": f"heightmap_{res}.png",
            "valid_mask": f"heightmap_valid_mask_{res}.png",
            "alpha_mask": f"heightmap_alpha_mask_{res}.png",
            "uv_layout": f"uv_layout_{res}.png",
            "uv_cheek_region_mask": f"uv_cheek_region_mask_{res}.png",
            "debug": f"uv_heightmap_debug_{res}.png",
        },
        "selection_parameters": {
            "model_right_sign": args.model_right_sign,
            "cheek_x_min": args.cheek_x_min,
            "cheek_x_max": args.cheek_x_max,
            "cheek_y_min": args.cheek_y_min,
            "cheek_y_max": args.cheek_y_max,
            "cheek_z_min": args.cheek_z_min,
        },
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("=" * 72)
    print("Created UV-based cheek height map export")
    print(f"OBJ        : {args.obj}")
    print(f"Cheek map  : {args.cheek_map}")
    print(f"Resolution : {res}x{res}")
    print(f"Object     : {args.object_name}")
    print(f"Selected cheek faces: {len(selected)}")
    print("Saved:")
    for name in [
        f"heightmap_{res}.png",
        f"heightmap_valid_mask_{res}.png",
        f"heightmap_alpha_mask_{res}.png",
        f"uv_layout_{res}.png",
        f"uv_cheek_region_mask_{res}.png",
        f"uv_heightmap_debug_{res}.png",
        "metadata.json",
    ]:
        print(f"  {out / name}")
    print("=" * 72)


if __name__ == "__main__":
    main()
