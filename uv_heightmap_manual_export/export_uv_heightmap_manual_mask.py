from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


@dataclass
class Face:
    obj: str
    verts: list[int]
    uvs: list[Optional[int]]


# Default polygon is for this FemaleBust/Anna_Head UV layout at 1024x1024.
# It is anatomical right cheek, which appears on viewer-left side of the front UV.
DEFAULT_RIGHT_CHEEK_POLYGON_1024 = [
    [265, 405],  # upper outer cheek, below eye
    [350, 395],
    [426, 425],
    [478, 485],
    [468, 565],
    [425, 635],
    [350, 690],
    [280, 665],
    [235, 585],
    [230, 500],
]


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
                p = s.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif s.startswith("vt "):
                p = s.split()
                uvs.append((float(p[1]), float(p[2])))
            elif s.startswith("f "):
                vi: list[int] = []
                ti: list[Optional[int]] = []
                for tok in s.split()[1:]:
                    pieces = tok.split("/")
                    vi.append(int(pieces[0]) - 1)
                    ti.append(int(pieces[1]) - 1 if len(pieces) > 1 and pieces[1] else None)
                faces.append(Face(obj=obj_name, verts=vi, uvs=ti))
    return np.asarray(verts, dtype=np.float32), np.asarray(uvs, dtype=np.float32), faces


def uv_to_px(uv, res: int) -> tuple[float, float]:
    u, v = float(uv[0]), float(uv[1])
    return u * (res - 1), (1.0 - v) * (res - 1)


def make_uv_wireframe(faces: list[Face], uv_array: np.ndarray, res: int, object_name: str) -> Image.Image:
    img = Image.new("RGB", (res, res), "white")
    d = ImageDraw.Draw(img)
    for face in faces:
        if object_name and face.obj != object_name:
            continue
        pts = []
        ok = True
        for ti in face.uvs:
            if ti is None or ti < 0 or ti >= len(uv_array):
                ok = False
                break
            pts.append(uv_to_px(uv_array[ti], res))
        if ok and len(pts) >= 3:
            d.line(pts + [pts[0]], fill=(185, 185, 185), width=max(1, res // 1024))
    return img


def load_polygon(path: Path | None, res: int, side: str) -> list[tuple[int, int]]:
    if path is None:
        pts = DEFAULT_RIGHT_CHEEK_POLYGON_1024
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        pts = data["polygon_px"] if "polygon_px" in data else data["right_cheek_polygon_px"]

    scale = res / 1024.0
    out = [(int(round(x * scale)), int(round(y * scale))) for x, y in pts]

    if side == "left":
        out = [(res - 1 - x, y) for x, y in out]
    return out


def load_cheek_patch(path: Path, bg_threshold: int = 5) -> tuple[Image.Image, Image.Image]:
    rgba = Image.open(path).convert("RGBA")
    arr = np.asarray(rgba).astype(np.float32)
    rgb = arr[..., :3]
    alpha = arr[..., 3]
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)

    # If alpha is meaningful, use alpha. Otherwise treat black background as no-data.
    if alpha.min() < 250:
        valid = alpha > 8
    else:
        valid = gray > bg_threshold

    gray_img = Image.fromarray(gray, mode="L")
    valid_img = Image.fromarray((valid.astype(np.uint8) * 255), mode="L")
    bbox = valid_img.getbbox()
    if bbox is None:
        raise RuntimeError("Cheek patch has no valid pixels. Check the input cheek map.")

    crop_gray = gray_img.crop(bbox)
    crop_valid = valid_img.crop(bbox)

    # Fill original black invalid background with mean valid value so resize does not create false indentation.
    g = np.asarray(crop_gray).astype(np.float32)
    m = np.asarray(crop_valid) > 0
    if np.any(m):
        g[~m] = float(g[m].mean())
    return Image.fromarray(np.clip(g, 0, 255).astype(np.uint8), mode="L"), crop_valid


def polygon_mask(res: int, pts: list[tuple[int, int]]) -> Image.Image:
    mask = Image.new("L", (res, res), 0)
    d = ImageDraw.Draw(mask)
    d.polygon(pts, fill=255)
    return mask


def bbox_from_polygon(pts: list[tuple[int, int]], pad: int, res: int) -> tuple[int, int, int, int]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0 = max(0, min(xs) - pad)
    y0 = max(0, min(ys) - pad)
    x1 = min(res, max(xs) + pad)
    y1 = min(res, max(ys) + pad)
    return x0, y0, x1, y1


def export_heightmap(
    cheek_gray: Image.Image,
    cheek_valid: Image.Image,
    mask: Image.Image,
    polygon_pts: list[tuple[int, int]],
    res: int,
    neutral_8: int,
    indent_strength_8: int,
    feather_px: int,
    pad_px: int,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    x0, y0, x1, y1 = bbox_from_polygon(polygon_pts, pad_px, res)
    bw, bh = x1 - x0, y1 - y0

    patch = cheek_gray.resize((bw, bh), Image.Resampling.BICUBIC)
    patch_valid = cheek_valid.resize((bw, bh), Image.Resampling.BICUBIC)

    src_canvas = Image.new("L", (res, res), neutral_8)
    src_valid_canvas = Image.new("L", (res, res), 0)
    src_canvas.paste(patch, (x0, y0))
    src_valid_canvas.paste(patch_valid, (x0, y0))

    hard_valid = Image.new("L", (res, res), 0)
    # Final region follows both: manual UV polygon AND source cheek valid shape.
    hard_arr = ((np.asarray(mask) > 0) & (np.asarray(src_valid_canvas) > 10)).astype(np.uint8) * 255
    hard_valid = Image.fromarray(hard_arr, mode="L")

    alpha = hard_valid.filter(ImageFilter.GaussianBlur(radius=max(0, feather_px))) if feather_px > 0 else hard_valid
    alpha_arr = np.asarray(alpha).astype(np.float32) / 255.0

    src = np.asarray(src_canvas).astype(np.float32)
    valid = np.asarray(hard_valid) > 0
    height = np.full((res, res), float(neutral_8), dtype=np.float32)

    if np.any(valid):
        vals = src[valid]
        lo, hi = float(vals.min()), float(vals.max())
        inward = np.zeros_like(src, dtype=np.float32)
        if hi - lo > 1e-6:
            inward[valid] = 1.0 - np.clip((src[valid] - lo) / (hi - lo), 0.0, 1.0)
        target = np.full_like(src, float(neutral_8))
        target[valid] = neutral_8 - inward[valid] * float(indent_strength_8)
        height = height * (1.0 - alpha_arr) + target * alpha_arr

    return (
        Image.fromarray(np.clip(height, 0, 255).astype(np.uint8), mode="L"),
        hard_valid,
        alpha,
    )


def make_debug(wire: Image.Image, polygon_pts: list[tuple[int, int]], height: Image.Image, valid: Image.Image, alpha: Image.Image) -> Image.Image:
    base = wire.convert("RGBA")
    overlay = np.zeros((base.height, base.width, 4), dtype=np.uint8)

    valid_arr = np.asarray(valid) > 0
    alpha_arr = np.asarray(alpha)
    h = np.asarray(height)

    # Height values inside valid region.
    overlay[valid_arr, 0] = h[valid_arr]
    overlay[valid_arr, 1] = h[valid_arr]
    overlay[valid_arr, 2] = h[valid_arr]
    overlay[valid_arr, 3] = 210

    out = Image.alpha_composite(base, Image.fromarray(overlay, mode="RGBA"))
    draw = ImageDraw.Draw(out)
    draw.line(polygon_pts + [polygon_pts[0]], fill=(255, 190, 0, 255), width=max(2, base.width // 512))

    # Blue valid edge.
    edge = valid.filter(ImageFilter.FIND_EDGES)
    arr = np.asarray(out.convert("RGB")).copy()
    edge_arr = np.asarray(edge) > 0
    arr[edge_arr] = [0, 80, 255]
    return Image.fromarray(arr, mode="RGB")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export UV height map using a manually registered cheek UV polygon mask.")
    ap.add_argument("--obj", type=Path, required=True)
    ap.add_argument("--cheek-map", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("uv_heightmap_manual_export"))
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--object-name", default="Anna_Head")
    ap.add_argument("--side", choices=["right", "left"], default="right")
    ap.add_argument("--polygon-json", type=Path, default=None, help="Optional JSON with polygon_px list for 1024x1024 layout.")
    ap.add_argument("--neutral-8", type=int, default=128)
    ap.add_argument("--indent-strength-8", type=int, default=90)
    ap.add_argument("--feather-px", type=int, default=14)
    ap.add_argument("--bbox-pad-px", type=int, default=8)
    args = ap.parse_args()

    if not args.obj.exists():
        raise FileNotFoundError(args.obj)
    if not args.cheek_map.exists():
        raise FileNotFoundError(args.cheek_map)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    verts, uvs, faces = parse_obj(args.obj)
    if len(uvs) == 0:
        raise RuntimeError("OBJ has no vt UV coordinates.")

    res = args.resolution
    wire = make_uv_wireframe(faces, uvs, res, args.object_name)
    poly = load_polygon(args.polygon_json, res, args.side)
    mask = polygon_mask(res, poly)
    cheek_gray, cheek_valid = load_cheek_patch(args.cheek_map)
    height, valid, alpha = export_heightmap(
        cheek_gray, cheek_valid, mask, poly, res,
        args.neutral_8, args.indent_strength_8, args.feather_px, args.bbox_pad_px
    )
    debug = make_debug(wire, poly, height, valid, alpha)

    height.save(args.out_dir / f"heightmap_{res}.png")
    valid.save(args.out_dir / f"heightmap_valid_mask_{res}.png")
    alpha.save(args.out_dir / f"heightmap_alpha_mask_{res}.png")
    wire.save(args.out_dir / f"uv_layout_{res}.png")
    mask.save(args.out_dir / f"manual_uv_cheek_polygon_mask_{res}.png")
    debug.save(args.out_dir / f"uv_heightmap_debug_{res}.png")

    metadata = {
        "description": "Manual UV cheek-mask heightmap export. The cheek region is manually registered to the OBJ UV layout instead of selected by rough 3D bbox.",
        "obj": str(args.obj),
        "cheek_map": str(args.cheek_map),
        "resolution": [res, res],
        "object_name": args.object_name,
        "side": args.side,
        "neutral_8": args.neutral_8,
        "indent_strength_8": args.indent_strength_8,
        "feather_px": args.feather_px,
        "polygon_px_1024_basis": DEFAULT_RIGHT_CHEEK_POLYGON_1024 if args.polygon_json is None else json.loads(args.polygon_json.read_text(encoding="utf-8")),
        "outputs": {
            "heightmap": f"heightmap_{res}.png",
            "valid_mask": f"heightmap_valid_mask_{res}.png",
            "alpha_mask": f"heightmap_alpha_mask_{res}.png",
            "debug": f"uv_heightmap_debug_{res}.png",
        },
        "note": "Check debug output. If the region touches the eye, edit the polygon points and rerun.",
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Save editable polygon config.
    (args.out_dir / "cheek_uv_polygon_config.json").write_text(json.dumps({"polygon_px": DEFAULT_RIGHT_CHEEK_POLYGON_1024}, indent=2), encoding="utf-8")

    print("=" * 72)
    print("Created manual-UV-mask cheek height map export")
    print(f"Saved to: {args.out_dir}")
    for name in [
        f"heightmap_{res}.png",
        f"heightmap_valid_mask_{res}.png",
        f"heightmap_alpha_mask_{res}.png",
        f"manual_uv_cheek_polygon_mask_{res}.png",
        f"uv_heightmap_debug_{res}.png",
        "cheek_uv_polygon_config.json",
        "metadata.json",
    ]:
        print(f"  {args.out_dir / name}")
    print("=" * 72)


if __name__ == "__main__":
    main()
