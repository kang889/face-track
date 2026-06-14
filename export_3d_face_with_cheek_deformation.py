#!/usr/bin/env python3
"""
Export a procedural 3D prototype face mesh with an anatomical-right-cheek
height/deformation map applied.

This is intended as a renderer-prototype exporter, not a true 3D face scan.
The generated full face is a neutral procedural reference surface; only the
cheek patch comes from measured/predicted deformation data.

Coordinate convention:
- OBJ X axis: viewer-left is negative X when looking at the face.
- OBJ Y axis: up.
- OBJ Z axis: forward, out of the face toward the viewer.
- anatomical right cheek on a front-facing face = viewer-left = negative X.
- darker cheek-map values are treated as larger inward indentation by default.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# -----------------------------
# Procedural neutral face model
# -----------------------------

def face_half_width_at_y(y: np.ndarray | float) -> np.ndarray | float:
    """Approximate half-width of a neutral front-facing face at height y."""
    y_arr = np.asarray(y, dtype=np.float32)
    w = np.zeros_like(y_arr)

    # Forehead / temple area
    top = y_arr > 0.35
    w[top] = 0.70 - 0.08 * np.clip((y_arr[top] - 0.35) / 0.90, 0, 1) ** 2

    # Cheek area: widest part of the face
    mid = (y_arr <= 0.35) & (y_arr > -0.45)
    w[mid] = 0.76 - 0.04 * ((y_arr[mid] + 0.05) / 0.70) ** 2

    # Jaw taper
    jaw = (y_arr <= -0.45) & (y_arr > -0.90)
    t = np.clip((-0.45 - y_arr[jaw]) / 0.45, 0, 1)
    w[jaw] = (1.0 - t) * 0.72 + t * 0.50

    # Chin taper
    chin = y_arr <= -0.90
    t = np.clip((-0.90 - y_arr[chin]) / 0.35, 0, 1)
    w[chin] = (1.0 - t) * 0.50 + t * 0.34

    if np.isscalar(y):
        return float(w)
    return w


def face_inside_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Boolean mask for procedural face silhouette, excluding the background."""
    y_min, y_max = -1.25, 1.18
    w = face_half_width_at_y(y)
    return (y >= y_min) & (y <= y_max) & (np.abs(x) <= w)


def neutral_face_z(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Smooth neutral face relief in arbitrary units.
    This gives a prototype face-like surface: rounded cheeks, nose protrusion,
    slight eye-socket, lip, and chin/jaw relief.
    """
    w = np.maximum(face_half_width_at_y(y), 1e-6)
    side = np.clip(np.abs(x) / w, 0.0, 1.0)
    vertical = np.clip((y + 1.25) / (1.18 + 1.25), 0.0, 1.0)

    # Main rounded face surface. Edges recede, center protrudes.
    z = 0.12 + 0.36 * np.sqrt(np.clip(1.0 - side ** 2, 0.0, 1.0))
    z *= 0.92 + 0.08 * np.cos((vertical - 0.5) * math.pi)

    # Nose ridge and tip.
    z += 0.13 * np.exp(-((x / 0.12) ** 2 + ((y - 0.18) / 0.42) ** 2))
    z += 0.10 * np.exp(-((x / 0.18) ** 2 + ((y + 0.05) / 0.16) ** 2))
    z += 0.04 * np.exp(-((x / 0.09) ** 2 + ((y - 0.55) / 0.35) ** 2))

    # Eye socket depressions.
    z -= 0.045 * np.exp(-(((x - 0.28) / 0.18) ** 2 + ((y - 0.37) / 0.13) ** 2))
    z -= 0.045 * np.exp(-(((x + 0.28) / 0.18) ** 2 + ((y - 0.37) / 0.13) ** 2))

    # Mouth / lip and chin relief.
    z += 0.035 * np.exp(-((x / 0.28) ** 2 + ((y + 0.53) / 0.08) ** 2))
    z -= 0.020 * np.exp(-((x / 0.35) ** 2 + ((y + 0.44) / 0.08) ** 2))
    z += 0.040 * np.exp(-((x / 0.35) ** 2 + ((y + 0.86) / 0.18) ** 2))

    # Subtle cheek fullness.
    z += 0.030 * np.exp(-(((x - 0.43) / 0.25) ** 2 + ((y + 0.05) / 0.38) ** 2))
    z += 0.030 * np.exp(-(((x + 0.43) / 0.25) ** 2 + ((y + 0.05) / 0.38) ** 2))
    return z.astype(np.float32)


# -----------------------------
# Cheek map loading and sampling
# -----------------------------

def load_cheek_map(path: Path, valid_threshold: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load cheek grayscale height map.

    Returns:
        values: float image in [0, 1]
        valid: boolean valid data mask. Black outside patch is treated as no data.
        alpha: soft edge alpha in [0, 1]
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read cheek map: {path}")

    valid = img > valid_threshold
    if int(valid.sum()) < 16:
        raise ValueError(
            "Cheek map has too few valid pixels. Check that black means background "
            "and non-black means deformation data."
        )

    # Fill any tiny holes inside the patch, but do not include black background.
    kernel = np.ones((3, 3), dtype=np.uint8)
    valid_u8 = cv2.morphologyEx(valid.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel, iterations=1)
    valid = valid_u8 > 0

    # Feather the edge so the patch blends into the procedural face surface.
    dist_in = cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 5)
    feather = 12.0
    alpha = np.clip(dist_in / feather, 0.0, 1.0)
    alpha = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), sigmaX=2.0, sigmaY=2.0)
    alpha = np.clip(alpha, 0.0, 1.0)

    values = img.astype(np.float32) / 255.0
    return values, valid, alpha.astype(np.float32)


def bilinear_sample(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Sample image at normalized coordinates u,v in [0,1]."""
    h, w = img.shape[:2]
    x = u * (w - 1)
    y = v * (h - 1)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)

    wx = x - x0
    wy = y - y0

    a = img[y0, x0]
    b = img[y0, x1]
    c = img[y1, x0]
    d = img[y1, x1]
    return (a * (1 - wx) * (1 - wy) + b * wx * (1 - wy) + c * (1 - wx) * wy + d * wx * wy)


def cheek_uv_from_xy(
    x: np.ndarray,
    y: np.ndarray,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    rotation_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert face x,y coordinates into local cheek-map u,v coordinates.
    Rotation is applied around the cheek ROI center.
    """
    theta = math.radians(rotation_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    dx = x - center_x
    dy = y - center_y

    # Rotate world coords into patch-local coordinates by inverse rotation.
    lx = cos_t * dx + sin_t * dy
    ly = -sin_t * dx + cos_t * dy

    u = lx / width + 0.5
    v = 0.5 - ly / height
    inside = (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
    return u, v, inside


def compute_cheek_deformation(
    x: np.ndarray,
    y: np.ndarray,
    cheek_values: np.ndarray,
    cheek_valid: np.ndarray,
    cheek_alpha: np.ndarray,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    rotation_deg: float,
    max_indent: float,
    dark_is_inward: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute inward z displacement and valid alpha for each x,y point.
    Returns displacement where positive means inward; apply as z -= displacement.
    """
    u, v, inside_roi = cheek_uv_from_xy(x, y, center_x, center_y, width, height, rotation_deg)

    sampled_val = np.zeros_like(x, dtype=np.float32)
    sampled_alpha = np.zeros_like(x, dtype=np.float32)
    sampled_valid = np.zeros_like(x, dtype=bool)

    if np.any(inside_roi):
        uu = np.clip(u[inside_roi], 0.0, 1.0)
        vv = np.clip(v[inside_roi], 0.0, 1.0)
        sampled_val[inside_roi] = bilinear_sample(cheek_values, uu, vv)
        sampled_alpha[inside_roi] = bilinear_sample(cheek_alpha, uu, vv)
        sampled_valid[inside_roi] = bilinear_sample(cheek_valid.astype(np.float32), uu, vv) > 0.5

    valid_vals = cheek_values[cheek_valid]
    lo = float(np.percentile(valid_vals, 2.0))
    hi = float(np.percentile(valid_vals, 98.0))
    denom = max(hi - lo, 1e-6)

    if dark_is_inward:
        inward01 = np.clip((hi - sampled_val) / denom, 0.0, 1.0)
    else:
        inward01 = np.clip((sampled_val - lo) / denom, 0.0, 1.0)

    alpha = sampled_alpha * sampled_valid.astype(np.float32)
    inward_disp = max_indent * inward01 * alpha
    return inward_disp.astype(np.float32), alpha.astype(np.float32)


# -----------------------------
# Mesh generation / export
# -----------------------------

def build_mesh(
    grid: int,
    cheek_values: np.ndarray,
    cheek_valid: np.ndarray,
    cheek_alpha: np.ndarray,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    rotation_deg: float,
    max_indent: float,
    dark_is_inward: bool,
) -> Dict[str, np.ndarray | List[Tuple[int, int, int]]]:
    """Create vertices, UVs, faces, normals, and per-vertex deformation alpha."""
    y_min, y_max = -1.25, 1.18
    x_min, x_max = -0.85, 0.85

    xs = np.linspace(x_min, x_max, grid, dtype=np.float32)
    ys = np.linspace(y_max, y_min, grid, dtype=np.float32)  # top to bottom for texture V
    xx, yy = np.meshgrid(xs, ys)
    inside = face_inside_mask(xx, yy)

    z = neutral_face_z(xx, yy)
    disp, deform_alpha = compute_cheek_deformation(
        xx,
        yy,
        cheek_values,
        cheek_valid,
        cheek_alpha,
        center_x,
        center_y,
        width,
        height,
        rotation_deg,
        max_indent,
        dark_is_inward,
    )
    z_deformed = z - disp

    index_grid = -np.ones_like(xx, dtype=np.int32)
    vertices: List[Tuple[float, float, float]] = []
    uvs: List[Tuple[float, float]] = []
    alpha_list: List[float] = []

    for r in range(grid):
        for c in range(grid):
            if inside[r, c]:
                idx = len(vertices)
                index_grid[r, c] = idx
                vertices.append((float(xx[r, c]), float(yy[r, c]), float(z_deformed[r, c])))
                u = (float(xx[r, c]) - x_min) / (x_max - x_min)
                v = (float(yy[r, c]) - y_min) / (y_max - y_min)
                uvs.append((u, v))
                alpha_list.append(float(deform_alpha[r, c]))

    faces: List[Tuple[int, int, int]] = []
    for r in range(grid - 1):
        for c in range(grid - 1):
            a = index_grid[r, c]
            b = index_grid[r, c + 1]
            cidx = index_grid[r + 1, c]
            d = index_grid[r + 1, c + 1]
            if a >= 0 and b >= 0 and cidx >= 0:
                faces.append((int(a), int(cidx), int(b)))
            if b >= 0 and cidx >= 0 and d >= 0:
                faces.append((int(b), int(cidx), int(d)))

    vertices_np = np.array(vertices, dtype=np.float32)
    normals_np = compute_vertex_normals(vertices_np, faces)

    return {
        "vertices": vertices_np,
        "uvs": np.array(uvs, dtype=np.float32),
        "normals": normals_np,
        "faces": faces,
        "alpha": np.array(alpha_list, dtype=np.float32),
        "index_grid": index_grid,
        "inside_grid": inside,
        "x_grid": xx,
        "y_grid": yy,
        "z_grid": z_deformed,
        "disp_grid": disp,
        "alpha_grid": deform_alpha,
    }


def compute_vertex_normals(vertices: np.ndarray, faces: Iterable[Tuple[int, int, int]]) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float32)
    for i0, i1, i2 in faces:
        v0 = vertices[i0]
        v1 = vertices[i1]
        v2 = vertices[i2]
        n = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(n)
        if norm > 1e-8:
            n = n / norm
            normals[i0] += n
            normals[i1] += n
            normals[i2] += n
    lens = np.linalg.norm(normals, axis=1)
    ok = lens > 1e-8
    normals[ok] /= lens[ok, None]
    normals[~ok] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return normals


def write_obj(out_obj: Path, vertices: np.ndarray, uvs: np.ndarray, normals: np.ndarray, faces: List[Tuple[int, int, int]], mtl_name: str) -> None:
    with out_obj.open("w", encoding="utf-8") as f:
        f.write("# Procedural face mesh with anatomical-right-cheek deformation applied\n")
        f.write("# Not a true scanned full-face deformation field. Cheek region only is measured/predicted.\n")
        f.write(f"mtllib {mtl_name}\n")
        f.write("usemtl face_material\n")
        for x, y, z in vertices:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for u, v in uvs:
            f.write(f"vt {u:.6f} {v:.6f}\n")
        for nx, ny, nz in normals:
            f.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
        # Same index is used for v/vt/vn because UVs and normals are per vertex.
        for i0, i1, i2 in faces:
            a, b, c = i0 + 1, i1 + 1, i2 + 1
            f.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")


def write_mtl(out_mtl: Path, texture_name: str) -> None:
    out_mtl.write_text(
        "newmtl face_material\n"
        "Ka 0.80 0.80 0.80\n"
        "Kd 0.90 0.90 0.90\n"
        "Ks 0.10 0.10 0.10\n"
        "Ns 16\n"
        f"map_Kd {texture_name}\n",
        encoding="utf-8",
    )


# -----------------------------
# Texture and debug exports
# -----------------------------

def make_texture_and_masks(
    tex_size: int,
    cheek_values: np.ndarray,
    cheek_valid: np.ndarray,
    cheek_alpha: np.ndarray,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    rotation_deg: float,
    max_indent: float,
    dark_is_inward: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create:
        texture: grayscale diffuse preview with cheek deformation visible
        valid_mask: white where cheek data exists
        displacement_map: grayscale full-face displacement, black background
    """
    y_min, y_max = -1.25, 1.18
    x_min, x_max = -0.85, 0.85
    xs = np.linspace(x_min, x_max, tex_size, dtype=np.float32)
    ys = np.linspace(y_max, y_min, tex_size, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    inside = face_inside_mask(xx, yy)
    z = neutral_face_z(xx, yy)

    disp, alpha = compute_cheek_deformation(
        xx,
        yy,
        cheek_values,
        cheek_valid,
        cheek_alpha,
        center_x,
        center_y,
        width,
        height,
        rotation_deg,
        max_indent,
        dark_is_inward,
    )

    z_norm = np.zeros_like(z, dtype=np.float32)
    z_inside = z[inside]
    z_norm[inside] = (z[inside] - float(z_inside.min())) / max(float(z_inside.max() - z_inside.min()), 1e-6)

    texture = np.zeros((tex_size, tex_size), dtype=np.uint8)
    base = 120 + 105 * z_norm
    # Make the cheek deformation visually darker in the diffuse texture.
    base = base - 80 * np.clip(disp / max(max_indent, 1e-6), 0.0, 1.0) * alpha

    texture[inside] = np.clip(base[inside], 0, 255).astype(np.uint8)

    # Draw simple eyes, brows, mouth cues into the texture so it reads as a face.
    texture_rgb = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
    draw_feature_lines(texture_rgb, x_min, x_max, y_min, y_max)
    texture = cv2.cvtColor(texture_rgb, cv2.COLOR_BGR2GRAY)
    texture[~inside] = 0

    valid_mask = np.zeros_like(texture, dtype=np.uint8)
    valid_mask[(alpha > 0.05) & inside] = 255

    disp_map = np.zeros_like(texture, dtype=np.uint8)
    disp01 = np.clip(disp / max(max_indent, 1e-6), 0.0, 1.0)
    disp_map[inside] = np.clip(disp01[inside] * 65535 / 257, 0, 255).astype(np.uint8)
    return texture, valid_mask, disp_map


def xy_to_px(x: float, y: float, x_min: float, x_max: float, y_min: float, y_max: float, size: int) -> Tuple[int, int]:
    px = int(round((x - x_min) / (x_max - x_min) * (size - 1)))
    py = int(round((y_max - y) / (y_max - y_min) * (size - 1)))
    return px, py


def draw_feature_lines(img_bgr: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
    """Draw minimal grayscale facial cues onto the diffuse texture."""
    size = img_bgr.shape[0]
    dark = (45, 45, 45)
    mid = (80, 80, 80)

    # Eyes
    for sx in [-1, 1]:
        cx, cy = xy_to_px(0.28 * sx, 0.36, x_min, x_max, y_min, y_max, size)
        ax = int(0.10 / (x_max - x_min) * size)
        ay = int(0.025 / (y_max - y_min) * size)
        cv2.ellipse(img_bgr, (cx, cy), (ax, ay), 0, 0, 360, dark, 2, cv2.LINE_AA)
        # Brow
        x0, y0 = xy_to_px(0.18 * sx, 0.52, x_min, x_max, y_min, y_max, size)
        x1, y1 = xy_to_px(0.42 * sx, 0.55, x_min, x_max, y_min, y_max, size)
        cv2.line(img_bgr, (x0, y0), (x1, y1), mid, 3, cv2.LINE_AA)

    # Nose ridge and nostril hint
    x0, y0 = xy_to_px(0.0, 0.35, x_min, x_max, y_min, y_max, size)
    x1, y1 = xy_to_px(0.0, -0.10, x_min, x_max, y_min, y_max, size)
    cv2.line(img_bgr, (x0, y0), (x1, y1), (105, 105, 105), 2, cv2.LINE_AA)
    for sx in [-1, 1]:
        cx, cy = xy_to_px(0.06 * sx, -0.10, x_min, x_max, y_min, y_max, size)
        cv2.circle(img_bgr, (cx, cy), 4, dark, -1, cv2.LINE_AA)

    # Mouth
    x0, y0 = xy_to_px(-0.23, -0.54, x_min, x_max, y_min, y_max, size)
    x1, y1 = xy_to_px(0.23, -0.54, x_min, x_max, y_min, y_max, size)
    cv2.line(img_bgr, (x0, y0), (x1, y1), dark, 2, cv2.LINE_AA)


def make_debug_registration(
    texture: np.ndarray,
    mask: np.ndarray,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    rotation_deg: float,
    out_path: Path,
) -> None:
    """Save a texture-space debug image showing where the cheek ROI is placed."""
    tex_size = texture.shape[0]
    x_min, x_max = -0.85, 0.85
    y_min, y_max = -1.25, 1.18
    debug = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
    debug[mask > 0] = (debug[mask > 0] * 0.55 + np.array([0, 200, 255]) * 0.45).astype(np.uint8)

    # ROI rectangle corners in world coordinates, rotated.
    theta = math.radians(rotation_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    local = [(-width / 2, -height / 2), (width / 2, -height / 2), (width / 2, height / 2), (-width / 2, height / 2)]
    pts = []
    for lx, ly in local:
        x = center_x + cos_t * lx - sin_t * ly
        y = center_y + sin_t * lx + cos_t * ly
        pts.append(xy_to_px(x, y, x_min, x_max, y_min, y_max, tex_size))
    cv2.polylines(debug, [np.array(pts, dtype=np.int32)], True, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(debug, "anatomical right cheek ROI", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), debug)


def save_preview_png(vertices: np.ndarray, faces: List[Tuple[int, int, int]], out_path: Path) -> None:
    """Render a simple 3D preview using matplotlib if available."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except Exception:
        return

    # Downsample faces to keep preview rendering quick.
    step = max(1, len(faces) // 9000)
    tri_vertices = [vertices[list(face)] for face in faces[::step]]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    coll = Poly3DCollection(tri_vertices, linewidths=0.02, alpha=1.0)
    coll.set_edgecolor((0.1, 0.1, 0.1, 0.10))
    coll.set_facecolor((0.65, 0.65, 0.65, 1.0))
    ax.add_collection3d(coll)

    ax.set_xlim(-0.9, 0.9)
    ax.set_ylim(-1.3, 1.2)
    ax.set_zlim(0.0, 0.75)
    ax.view_init(elev=8, azim=-90)
    ax.set_axis_off()
    ax.set_box_aspect((1.8, 2.5, 0.8))
    plt.tight_layout(pad=0)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# -----------------------------
# Main entry point
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a procedural 3D face OBJ with anatomical-right-cheek deformation applied."
    )
    parser.add_argument("--cheek-map", type=Path, required=True, help="Input cheek height/deformation map PNG.")
    parser.add_argument("--out-dir", type=Path, default=Path("face_3d_export"), help="Output folder.")
    parser.add_argument("--grid", type=int, default=180, help="Mesh grid resolution. 160-240 is usually enough.")
    parser.add_argument("--texture-size", type=int, default=1024, help="Diffuse/mask texture resolution.")
    parser.add_argument("--max-indent", type=float, default=0.075, help="Maximum inward indentation in OBJ units.")
    parser.add_argument(
        "--side",
        choices=["right", "left"],
        default="right",
        help="Anatomical cheek side. right = viewer-left on a front-facing face.",
    )
    parser.add_argument("--cheek-center-x", type=float, default=None, help="Override cheek ROI center X in face coordinates.")
    parser.add_argument("--cheek-center-y", type=float, default=-0.05, help="Cheek ROI center Y in face coordinates.")
    parser.add_argument("--cheek-width", type=float, default=0.52, help="Cheek ROI width in face coordinates.")
    parser.add_argument("--cheek-height", type=float, default=0.70, help="Cheek ROI height in face coordinates.")
    parser.add_argument("--cheek-rotation-deg", type=float, default=-8.0, help="Cheek ROI rotation in degrees.")
    parser.add_argument(
        "--bright-is-inward",
        action="store_true",
        help="Use this only if brighter cheek-map pixels should mean more inward indentation. Default: darker is inward.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.grid < 32:
        raise ValueError("--grid should be at least 32")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cheek_values, cheek_valid, cheek_alpha = load_cheek_map(args.cheek_map)

    # anatomical right cheek = viewer-left = negative X on a front-facing face.
    default_center_x = -0.36 if args.side == "right" else 0.36
    center_x = args.cheek_center_x if args.cheek_center_x is not None else default_center_x
    rotation = args.cheek_rotation_deg if args.side == "right" else -args.cheek_rotation_deg

    mesh = build_mesh(
        grid=args.grid,
        cheek_values=cheek_values,
        cheek_valid=cheek_valid,
        cheek_alpha=cheek_alpha,
        center_x=center_x,
        center_y=args.cheek_center_y,
        width=args.cheek_width,
        height=args.cheek_height,
        rotation_deg=rotation,
        max_indent=args.max_indent,
        dark_is_inward=not args.bright_is_inward,
    )

    out_obj = args.out_dir / "prototype_face_right_cheek_deformed.obj"
    out_mtl = args.out_dir / "prototype_face_right_cheek_deformed.mtl"
    out_tex = args.out_dir / "prototype_face_texture.png"
    out_mask = args.out_dir / "prototype_face_cheek_valid_mask.png"
    out_disp = args.out_dir / "prototype_face_displacement_map_preview.png"
    out_debug = args.out_dir / "prototype_face_cheek_registration_debug.png"
    out_preview = args.out_dir / "prototype_face_3d_preview.png"
    out_meta = args.out_dir / "prototype_face_export_metadata.json"

    texture, mask, disp_map = make_texture_and_masks(
        tex_size=args.texture_size,
        cheek_values=cheek_values,
        cheek_valid=cheek_valid,
        cheek_alpha=cheek_alpha,
        center_x=center_x,
        center_y=args.cheek_center_y,
        width=args.cheek_width,
        height=args.cheek_height,
        rotation_deg=rotation,
        max_indent=args.max_indent,
        dark_is_inward=not args.bright_is_inward,
    )
    cv2.imwrite(str(out_tex), texture)
    cv2.imwrite(str(out_mask), mask)
    cv2.imwrite(str(out_disp), disp_map)
    make_debug_registration(texture, mask, center_x, args.cheek_center_y, args.cheek_width, args.cheek_height, rotation, out_debug)

    write_obj(
        out_obj,
        mesh["vertices"],  # type: ignore[arg-type]
        mesh["uvs"],  # type: ignore[arg-type]
        mesh["normals"],  # type: ignore[arg-type]
        mesh["faces"],  # type: ignore[arg-type]
        out_mtl.name,
    )
    write_mtl(out_mtl, out_tex.name)
    save_preview_png(mesh["vertices"], mesh["faces"], out_preview)  # type: ignore[arg-type]

    disp_grid = mesh["disp_grid"]  # type: ignore[assignment]
    alpha_grid = mesh["alpha_grid"]  # type: ignore[assignment]
    active = np.asarray(alpha_grid) > 0.05
    max_applied = float(np.max(np.asarray(disp_grid)[active])) if np.any(active) else 0.0

    metadata = {
        "description": "Procedural 3D face mesh with cheek-only deformation applied.",
        "source_cheek_map": str(args.cheek_map),
        "important_limitations": [
            "This is not a true scanned 3D face.",
            "The full face is a procedural neutral prototype surface.",
            "Only the cheek region uses measured/predicted deformation data.",
            "Remaining facial regions should be treated as neutral/unmeasured.",
        ],
        "coordinate_convention": {
            "x": "viewer-left negative, viewer-right positive",
            "y": "up positive",
            "z": "forward/out of face positive",
            "anatomical_right_cheek": "viewer-left on a front-facing face, therefore negative X",
        },
        "deformation_interpretation": {
            "dark_is_inward": not args.bright_is_inward,
            "application": "z_deformed = z_neutral - inward_displacement",
            "max_indent_requested_obj_units": args.max_indent,
            "max_indent_applied_obj_units": max_applied,
        },
        "cheek_registration": {
            "side": args.side,
            "center_x": center_x,
            "center_y": args.cheek_center_y,
            "width": args.cheek_width,
            "height": args.cheek_height,
            "rotation_deg": rotation,
        },
        "outputs": {
            "obj": out_obj.name,
            "mtl": out_mtl.name,
            "texture": out_tex.name,
            "cheek_valid_mask": out_mask.name,
            "displacement_map_preview": out_disp.name,
            "registration_debug": out_debug.name,
            "preview": out_preview.name,
        },
        "mesh_stats": {
            "vertices": int(len(mesh["vertices"])),
            "triangles": int(len(mesh["faces"])),
            "grid_resolution": int(args.grid),
        },
    }
    out_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("=" * 72)
    print("Export complete: procedural 3D face with right-cheek deformation")
    print(f"Output folder: {args.out_dir}")
    print(f"OBJ: {out_obj}")
    print(f"Texture: {out_tex}")
    print(f"Mask: {out_mask}")
    print(f"Debug: {out_debug}")
    print(f"Preview: {out_preview}")
    print(f"Metadata: {out_meta}")
    print(f"Vertices: {len(mesh['vertices'])} | Triangles: {len(mesh['faces'])}")
    print("=" * 72)


if __name__ == "__main__":
    main()
