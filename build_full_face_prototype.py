from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_LAYOUT = {
    "canvas_size": [1024, 1024],
    "background_value": 0,
    "face_oval": {
        "center_xy_norm": [0.50, 0.50],
        "axes_xy_norm": [0.28, 0.38],
        "rotation_deg": 0.0,
        "fill_value": 255
    },
    "cheek_patch": {
        "center_xy_norm": [0.34, 0.56],
        "size_xy_norm": [0.22, 0.24],
        "rotation_deg": -8.0,
        "feather_px": 18
    }
}


def load_or_create_layout(layout_path: Path | None, output_dir: Path) -> tuple[dict, Path]:
    if layout_path is not None and layout_path.exists():
        return json.loads(layout_path.read_text(encoding="utf-8")), layout_path
    final_path = layout_path if layout_path is not None else (output_dir / "full_face_prototype_layout.json")
    if not final_path.exists():
        final_path.write_text(json.dumps(DEFAULT_LAYOUT, indent=2), encoding="utf-8")
    return json.loads(final_path.read_text(encoding="utf-8")), final_path


def make_face_mask(canvas_h: int, canvas_w: int, layout: dict) -> np.ndarray:
    mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    oval = layout["face_oval"]
    cx = int(round(oval["center_xy_norm"][0] * canvas_w))
    cy = int(round(oval["center_xy_norm"][1] * canvas_h))
    ax = int(round(oval["axes_xy_norm"][0] * canvas_w))
    ay = int(round(oval["axes_xy_norm"][1] * canvas_h))
    rot = float(oval["rotation_deg"])
    cv2.ellipse(mask, (cx, cy), (ax, ay), rot, 0, 360, 255, -1, lineType=cv2.LINE_AA)
    return mask


def resize_and_rotate_patch(patch: np.ndarray, target_w: int, target_h: int, rotation_deg: float) -> np.ndarray:
    resized = cv2.resize(patch, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    center = (target_w / 2.0, target_h / 2.0)
    m = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
    rotated = cv2.warpAffine(
        resized,
        m,
        (target_w, target_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rotated


def build_patch_alpha(rotated_patch: np.ndarray, feather_px: int) -> np.ndarray:
    alpha = np.zeros_like(rotated_patch, dtype=np.uint8)
    alpha[rotated_patch > 0] = 255
    if feather_px > 0:
        k = feather_px * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    return alpha.astype(np.float32) / 255.0


def paste_patch(canvas: np.ndarray, face_mask: np.ndarray, patch: np.ndarray, alpha: np.ndarray, x0: int, y0: int) -> np.ndarray:
    out = canvas.copy()
    h, w = patch.shape
    x1 = x0 + w
    y1 = y0 + h

    if x0 < 0 or y0 < 0 or x1 > canvas.shape[1] or y1 > canvas.shape[0]:
        raise ValueError("Cheek patch placement goes outside the canvas. Adjust the layout JSON.")

    roi = out[y0:y1, x0:x1].astype(np.float32)
    face_roi = (face_mask[y0:y1, x0:x1] > 0).astype(np.float32)
    final_alpha = alpha * face_roi

    patch_f = patch.astype(np.float32)
    blended = roi * (1.0 - final_alpha) + patch_f * final_alpha
    out[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def make_debug_overlay(face_img: np.ndarray, face_mask: np.ndarray, box_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    debug = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
    debug[face_mask == 0] = (0, 0, 0)
    x0, y0, x1, y1 = box_xyxy
    cv2.rectangle(debug, (x0, y0), (x1, y1), (0, 255, 255), 2)
    cv2.putText(
        debug,
        "Cheek patch box (anatomical right cheek)",
        (max(10, x0 - 20), max(25, y0 - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return debug


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a simple full-face prototype image by placing a cheek height map onto an anatomical-right-cheek face canvas."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("regional_heightmap_pred_absolute.png"),
        help="Cheek source PNG to place onto the full-face prototype.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("full_face_prototype"),
        help="Folder to save the prototype outputs.",
    )
    parser.add_argument(
        "--layout",
        type=Path,
        default=None,
        help="Optional JSON layout config. If omitted, a default layout JSON is created.",
    )
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Source cheek map not found: {args.source}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    layout, layout_path = load_or_create_layout(args.layout, output_dir)

    cheek_img = cv2.imread(str(args.source), cv2.IMREAD_GRAYSCALE)
    if cheek_img is None:
        raise RuntimeError(f"Could not read cheek source image: {args.source}")

    canvas_w, canvas_h = layout["canvas_size"]
    canvas = np.full((canvas_h, canvas_w), int(layout["background_value"]), dtype=np.uint8)

    face_mask = make_face_mask(canvas_h, canvas_w, layout)
    face_fill = int(layout["face_oval"]["fill_value"])
    canvas[face_mask > 0] = face_fill

    cheek_cfg = layout["cheek_patch"]
    patch_w = int(round(cheek_cfg["size_xy_norm"][0] * canvas_w))
    patch_h = int(round(cheek_cfg["size_xy_norm"][1] * canvas_h))
    rotation_deg = float(cheek_cfg["rotation_deg"])
    feather_px = int(cheek_cfg["feather_px"])

    rotated_patch = resize_and_rotate_patch(cheek_img, patch_w, patch_h, rotation_deg)
    alpha = build_patch_alpha(rotated_patch, feather_px)

    cx = int(round(cheek_cfg["center_xy_norm"][0] * canvas_w))
    cy = int(round(cheek_cfg["center_xy_norm"][1] * canvas_h))
    x0 = cx - patch_w // 2
    y0 = cy - patch_h // 2
    x1 = x0 + patch_w
    y1 = y0 + patch_h

    prototype = paste_patch(canvas, face_mask, rotated_patch, alpha, x0, y0)
    debug = make_debug_overlay(prototype, face_mask, (x0, y0, x1, y1))

    out_main = output_dir / "full_face_prototype.png"
    out_debug = output_dir / "full_face_prototype_debug.png"
    out_summary = output_dir / "full_face_prototype_summary.json"

    cv2.imwrite(str(out_main), prototype)
    cv2.imwrite(str(out_debug), debug)

    summary = {
        "description": "Simple full-face prototype generated from a single cheek height map source.",
        "source_cheek_map": str(args.source),
        "canvas_size": [canvas_w, canvas_h],
        "side": "anatomical right cheek of the face (viewer-left on a front-facing image)",
        "layout_json": str(layout_path),
        "notes": [
            "This is a prototype full-face image, not a measured full-face deformation map.",
            "Only the cheek region comes from recorded deformation.",
            "The rest of the face is left at the neutral fill value.",
            "Later, the layout JSON can be edited to better match the renderer texture/layout."
        ],
        "outputs": {
            "prototype_png": str(out_main),
            "debug_png": str(out_debug),
            "layout_json": str(layout_path),
        },
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 72)
    print("Created simple full-face prototype")
    print(f"Source cheek map: {args.source}")
    print(f"Canvas size: {canvas_w}x{canvas_h}")
    print("Saved:")
    print(f"  {out_main}")
    print(f"  {out_debug}")
    print(f"  {layout_path}")
    print(f"  {out_summary}")
    print("=" * 72)


if __name__ == "__main__":
    main()
