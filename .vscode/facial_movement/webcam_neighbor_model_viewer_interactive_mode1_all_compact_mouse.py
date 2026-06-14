"""
Interactive left-cheek patch explorer and Mode 1 model viewer.

Purpose
-------
This viewer extends the earlier hard-coded one-point neighbor model demo.
It adds:
- selectable active source point
- selectable patch definition / patch size rule
- live visualization of the currently active patch
- automatic Mode 1 checkpoint loading for every source point
- mouse-click source-point selection, including before neutral capture

Project role
------------
This file is meant for the interaction-design and validation stage.
It lets you answer questions such as:
- Which source point is active?
- Which points belong to the active patch?
- Can Mode 1 prediction work for every cheek point when toggling [ and ]?

Important limitation
--------------------
Prediction is only fully supported for:
- Mode 1 (direct-neighbor patch)
- source points that already have a trained checkpoint in MODE1_MODELS_DIR

For Mode 2 and Mode 3, this viewer still works as a patch explorer and
real-motion viewer, but it will display a clear message that no matching
trained model is available for prediction.

Controls
--------
N : capture neutral
C : clear neutral
I : toggle all landmark IDs
O : toggle cheek-only landmark IDs
Mouse click : select active source point by clicking a cheek point
1 : direct-neighbor patch mode
2 : 2-ring patch mode
3 : radius patch mode
- / = : decrease / increase radius (radius mode only)
Q or ESC : quit
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from torch import nn
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MODEL_PATH = "models/face_landmarker.task"
MODE1_MODELS_DIR = Path("mode1_models")
MODE1_FILENAME_TEMPLATE = "point_{:02d}_direct.pt"

# Current left-cheek landmark subset.
LEFT_CHEEK_IDS = [
    36, 50, 101, 111, 116, 117, 118, 123, 135, 137, 138, 147, 177,
    187, 192, 203, 205, 206, 207, 212, 213, 214, 215, 216, 227,
]

# Initial active source point for the interactive viewer.
INITIAL_SOURCE_POINT_INDEX = 13

# Relatively stable landmarks used only for head-motion compensation.
HEAD_ANCHOR_IDS = [33, 133, 362, 263, 6, 168]

# Neutral capture.
NEUTRAL_CAPTURE_FRAMES = 20

# Motion-analysis and gating thresholds.
ARROW_THRESHOLD_PX = 3.0
ANCHOR_ALIGNMENT_ERROR_THRESHOLD_PX = 8.0
SOURCE_ACTIVITY_THRESHOLD_PX = 2.0

# Patch exploration defaults.
DEFAULT_RADIUS_PX = 35.0
RADIUS_STEP_PX = 5.0
MIN_RADIUS_PX = 10.0
MAX_RADIUS_PX = 120.0
CLICK_SELECT_MAX_DIST_PX = 18.0

WINDOW_NAME = "Neighbor Model Viewer"

# Display toggles.
SHOW_ALL_LANDMARKS = True
SHOW_LANDMARK_IDS = False
SHOW_ONLY_LEFT_CHEEK_IDS = False
SHOW_ANCHORS = False
SHOW_CHEEK_MESH = True

# Drawing colors (BGR)
COLOR_SOURCE = (0, 255, 255)          # yellow
COLOR_PATCH = (255, 200, 0)           # orange-ish highlight for active patch
COLOR_ACTUAL = (0, 255, 0)            # green
COLOR_PREDICTED = (255, 0, 255)       # magenta
COLOR_NEUTRAL = (120, 120, 120)       # gray
COLOR_TEXT = (255, 255, 255)
COLOR_ERROR = (0, 0, 255)
COLOR_INFO = (180, 180, 180)
COLOR_CONNECTOR = (230, 230, 230)


# -----------------------------------------------------------------------------
# Enums / simple data containers
# -----------------------------------------------------------------------------


class PatchMode(Enum):
    """Supported patch-definition modes for the interactive viewer."""

    DIRECT = "direct"
    TWO_RING = "two_ring"
    RADIUS = "radius"


@dataclass
class LoadedCheckpoint:
    """Container for a loaded one-point checkpoint and its training metadata."""

    model: nn.Module | None
    source_point_index: int | None
    neighbor_indices: list[int] | None
    input_dim: int | None
    output_dim: int | None
    message: str
    checkpoint_path: str | None = None


# -----------------------------------------------------------------------------
# PyTorch model
# -----------------------------------------------------------------------------


class NeighborMLP(nn.Module):
    """Same model structure used during training."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------------------------------------------------------
# MediaPipe setup
# -----------------------------------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
)


# -----------------------------------------------------------------------------
# Landmark utilities
# -----------------------------------------------------------------------------


def get_landmark_points_px(face_landmarks, ids, width, height):
    """Extract selected normalized MediaPipe landmarks as pixel-space points."""
    points = []
    for idx in ids:
        lm = face_landmarks[idx]
        points.append([lm.x * width, lm.y * height])
    return np.array(points, dtype=np.float32)



def draw_all_landmarks(frame, face_landmarks, width, height):
    """Draw all detected face landmarks for debugging."""
    for lm in face_landmarks:
        x = int(lm.x * width)
        y = int(lm.y * height)
        cv2.circle(frame, (x, y), 1, (0, 100, 0), -1)



def draw_landmark_ids(frame, face_landmarks, width, height, only_ids=None):
    """Draw landmark index numbers on the detected face points."""
    for index, lm in enumerate(face_landmarks):
        if only_ids is not None and index not in only_ids:
            continue

        x = int(lm.x * width)
        y = int(lm.y * height)
        cv2.putText(
            frame,
            str(index),
            (x + 3, y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )



def draw_tracked_cheek_mesh(frame, face_landmarks, cheek_ids, connections, width, height,
                            color=(180, 0, 0), thickness=1):
    """Draw the tracked left-cheek landmark mesh."""
    for conn in connections:
        a = conn.start
        b = conn.end

        if a in cheek_ids and b in cheek_ids:
            lm1 = face_landmarks[a]
            lm2 = face_landmarks[b]
            x1, y1 = int(lm1.x * width), int(lm1.y * height)
            x2, y2 = int(lm2.x * width), int(lm2.y * height)
            cv2.line(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    for idx in cheek_ids:
        lm = face_landmarks[idx]
        x, y = int(lm.x * width), int(lm.y * height)
        cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)



def draw_anchor_points(frame, anchor_points):
    """Draw head anchor points used for pose compensation."""
    for pt in anchor_points:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)


# -----------------------------------------------------------------------------
# Head-motion compensation
# -----------------------------------------------------------------------------


def estimate_head_motion_transform(current_anchor_points, neutral_anchor_points):
    """Estimate a partial affine transform mapping current anchors -> neutral anchors."""
    if current_anchor_points is None or neutral_anchor_points is None:
        return None

    if len(current_anchor_points) < 3 or len(neutral_anchor_points) < 3:
        return None

    transform, _ = cv2.estimateAffinePartial2D(
        current_anchor_points,
        neutral_anchor_points,
        method=cv2.LMEDS,
    )
    return transform



def apply_affine_to_points(points, transform):
    """Apply a 2x3 affine transform to an (N, 2) point array."""
    if transform is None:
        return None

    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    homogeneous = np.hstack([points, ones])
    transformed = (transform @ homogeneous.T).T
    return transformed.astype(np.float32)



def compute_mean_alignment_error(current_anchor_points, neutral_anchor_points, transform):
    """Measure mean post-alignment anchor error in pixels."""
    corrected_anchor_points = apply_affine_to_points(current_anchor_points, transform)
    if corrected_anchor_points is None:
        return float("inf")

    errors = np.linalg.norm(corrected_anchor_points - neutral_anchor_points, axis=1)
    return float(np.mean(errors))


# -----------------------------------------------------------------------------
# Patch utilities
# -----------------------------------------------------------------------------


def build_local_patch_edges(landmark_ids, connections):
    """Build local cheek-patch edges from the MediaPipe tessellation."""
    id_to_local = {landmark_id: i for i, landmark_id in enumerate(landmark_ids)}
    edges = set()

    for conn in connections:
        a = conn.start
        b = conn.end
        if a in id_to_local and b in id_to_local:
            ia = id_to_local[a]
            ib = id_to_local[b]
            if ia > ib:
                ia, ib = ib, ia
            edges.add((ia, ib))

    return sorted(edges)



def build_adjacency(num_points: int, patch_edges: list[tuple[int, int]]) -> list[set[int]]:
    """Build adjacency list from local patch edges."""
    adjacency: list[set[int]] = [set() for _ in range(num_points)]
    for a, b in patch_edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    return adjacency



def get_direct_neighbors(adjacency: list[set[int]], source_index: int) -> list[int]:
    """Return direct neighbors of a source point."""
    return sorted(adjacency[source_index])



def get_k_ring_patch(adjacency: list[set[int]], source_index: int, ring_count: int) -> list[int]:
    """Return all points inside a k-ring patch, excluding the source point itself."""
    visited = {source_index}
    frontier = {source_index}
    patch = set()

    for _ in range(ring_count):
        next_frontier = set()
        for node in frontier:
            for nbr in adjacency[node]:
                if nbr not in visited:
                    visited.add(nbr)
                    next_frontier.add(nbr)
                    patch.add(nbr)
        frontier = next_frontier
        if not frontier:
            break

    return sorted(patch)



def get_radius_patch(neutral_points: np.ndarray, source_index: int, radius_px: float) -> list[int]:
    """Return all cheek points within a neutral-space radius from the source point."""
    source_pt = neutral_points[source_index]
    deltas = neutral_points - source_pt
    dists = np.linalg.norm(deltas, axis=1)
    patch = [i for i, dist in enumerate(dists) if 0 < dist <= radius_px]
    return sorted(patch)



def get_active_patch_indices(
    patch_mode: PatchMode,
    source_index: int,
    adjacency: list[set[int]],
    neutral_points: np.ndarray,
    radius_px: float,
) -> list[int]:
    """Compute the currently active patch point list from the chosen patch rule."""
    if patch_mode == PatchMode.DIRECT:
        return get_direct_neighbors(adjacency, source_index)
    if patch_mode == PatchMode.TWO_RING:
        return get_k_ring_patch(adjacency, source_index, ring_count=2)
    if patch_mode == PatchMode.RADIUS:
        return get_radius_patch(neutral_points, source_index, radius_px=radius_px)
    raise ValueError(f"Unsupported patch mode: {patch_mode}")


# -----------------------------------------------------------------------------
# Drawing helpers
# -----------------------------------------------------------------------------


def draw_arrow(frame, start_pt, end_pt, color, thickness=2, threshold_px=0.0):
    """Draw an arrow only if the motion is large enough."""
    motion = float(np.linalg.norm(end_pt - start_pt))
    if motion < threshold_px:
        return

    start = tuple(np.round(start_pt).astype(int))
    end = tuple(np.round(end_pt).astype(int))
    cv2.arrowedLine(frame, start, end, color, thickness, tipLength=0.22)


def draw_visual_legend(frame):
    """Draw a compact legend so actual vs predicted motion is easier to read."""
    x0, y0 = 12, 48
    w, h = 220, 58
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (30, 30, 30), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (90, 90, 90), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Source
    cv2.circle(frame, (x0 + 12, y0 + 14), 4, COLOR_SOURCE, -1)
    cv2.putText(frame, "Source", (x0 + 25, y0 + 18), font, 0.38, COLOR_SOURCE, 1, cv2.LINE_AA)

    # Actual
    cv2.circle(frame, (x0 + 12, y0 + 32), 4, COLOR_ACTUAL, -1)
    cv2.putText(frame, "Actual", (x0 + 25, y0 + 36), font, 0.38, COLOR_ACTUAL, 1, cv2.LINE_AA)

    # Predicted
    cv2.circle(frame, (x0 + 12, y0 + 49), 4, COLOR_PREDICTED, -1)
    cv2.putText(frame, "Predicted", (x0 + 25, y0 + 53), font, 0.38, COLOR_PREDICTED, 1, cv2.LINE_AA)

    cv2.putText(frame, "Line = prediction error gap", (x0 + 112, y0 + 53), font, 0.30, COLOR_CONNECTOR, 1, cv2.LINE_AA)



def draw_active_patch_multi(frame, neutral_points, corrected_points, source_indices, patch_indices):
    """Draw all selected source points and the current union patch points."""
    for source_index in source_indices:
        source_neutral = neutral_points[source_index]
        source_current = corrected_points[source_index]

        cv2.circle(frame, tuple(np.round(source_neutral).astype(int)), 5, COLOR_NEUTRAL, -1)
        cv2.circle(frame, tuple(np.round(source_current).astype(int)), 4, COLOR_SOURCE, -1)
        draw_arrow(frame, source_neutral, source_current, COLOR_SOURCE, thickness=2, threshold_px=ARROW_THRESHOLD_PX)

        cv2.putText(
            frame,
            f"S:{source_index}",
            tuple((np.round(source_current).astype(int) + np.array([6, -6])).tolist()),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOR_SOURCE,
            1,
            cv2.LINE_AA,
        )

    for patch_index in patch_indices:
        neutral_pt = neutral_points[patch_index]
        actual_pt = corrected_points[patch_index]

        cv2.circle(frame, tuple(np.round(neutral_pt).astype(int)), 4, COLOR_NEUTRAL, -1)
        cv2.circle(frame, tuple(np.round(actual_pt).astype(int)), 4, COLOR_PATCH, -1)
        draw_arrow(frame, neutral_pt, actual_pt, COLOR_ACTUAL, thickness=1, threshold_px=ARROW_THRESHOLD_PX)

        cv2.putText(
            frame,
            f"P:{patch_index}",
            tuple((np.round(actual_pt).astype(int) + np.array([4, -4])).tolist()),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            COLOR_PATCH,
            1,
            cv2.LINE_AA,
        )



def draw_prediction_overlay(frame, neutral_points, corrected_points, predicted_patch_points, patch_indices):
    """Draw actual tracked patch motion and blended model-predicted patch motion."""
    for local_idx, patch_index in enumerate(patch_indices):
        neutral_pt = neutral_points[patch_index]
        actual_pt = corrected_points[patch_index]
        pred_pt = predicted_patch_points[local_idx]

        actual_xy = tuple(np.round(actual_pt).astype(int))
        cv2.circle(frame, actual_xy, 4, COLOR_ACTUAL, -1)
        draw_arrow(frame, neutral_pt, actual_pt, COLOR_ACTUAL, thickness=2, threshold_px=ARROW_THRESHOLD_PX)

        pred_xy = tuple(np.round(pred_pt).astype(int))
        cv2.circle(frame, pred_xy, 4, COLOR_PREDICTED, -1)
        draw_arrow(frame, neutral_pt, pred_pt, COLOR_PREDICTED, thickness=1, threshold_px=ARROW_THRESHOLD_PX)

        err = float(np.linalg.norm(actual_pt - pred_pt))
        connector_color = COLOR_ERROR if err > 6.0 else COLOR_CONNECTOR
        cv2.line(frame, actual_xy, pred_xy, connector_color, 1, cv2.LINE_AA)



def draw_side_panel(
    frame,
    cheek_disp,
    source_indices,
    patch_mode,
    patch_indices,
    radius_px,
    checkpoint_info,
    prediction_enabled,
    predicted_patch_displacements,
    contributing_source_count,
):
    """Draw a compact bottom-right panel for blended multi-source Mode 1 viewing."""
    h, w, _ = frame.shape
    panel_width = 255
    panel_height = 165 if prediction_enabled else 135
    panel_x0 = max(w - panel_width - 10, 0)
    panel_y0 = max(h - panel_height - 10, 0)

    cv2.rectangle(frame, (panel_x0, panel_y0), (panel_x0 + panel_width, panel_y0 + panel_height), (30, 30, 30), -1)
    cv2.rectangle(frame, (panel_x0, panel_y0), (panel_x0 + panel_width, panel_y0 + panel_height), (90, 90, 90), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = panel_y0 + 16

    def put(text, color=COLOR_TEXT, scale=0.33, advance=14):
        nonlocal y
        cv2.putText(frame, text, (panel_x0 + 8, y), font, scale, color, 1, cv2.LINE_AA)
        y += advance

    source_indices = list(source_indices)
    source_text = ",".join(str(i) for i in source_indices[:6])
    if len(source_indices) > 6:
        source_text += ",..."
    source_mags = [float(np.linalg.norm(cheek_disp[idx])) for idx in source_indices]
    mean_source_mag = float(np.mean(source_mags)) if source_mags else 0.0

    put(f"Srcs {len(source_indices)} | {source_text}", color=COLOR_SOURCE, scale=0.36, advance=16)
    put(f"Mean src mag={mean_source_mag:.1f}px")
    put(f"Mode={patch_mode.value} | patch={len(patch_indices)} pts", color=COLOR_PATCH)
    if patch_mode == PatchMode.RADIUS:
        put(f"Radius={radius_px:.1f}px")
    else:
        put("Radius=n/a", color=COLOR_INFO)
    put(checkpoint_info, color=COLOR_INFO, scale=0.28, advance=12)

    if not prediction_enabled:
        put("Prediction unavailable", color=COLOR_ERROR, scale=0.34, advance=16)
        return

    if len(patch_indices) == 0:
        put("Prediction enabled, but patch empty", color=COLOR_INFO, scale=0.30, advance=12)
        return

    pred_subset = predicted_patch_displacements[: len(patch_indices)]
    actual_subset = cheek_disp[patch_indices]
    per_point_err = np.linalg.norm(actual_subset - pred_subset, axis=1)
    mean_err = float(np.mean(per_point_err)) if len(per_point_err) > 0 else 0.0
    best_local_idx = int(np.argmin(per_point_err)) if len(per_point_err) > 0 else 0
    worst_local_idx = int(np.argmax(per_point_err)) if len(per_point_err) > 0 else 0

    put(f"Pred ON | src used={contributing_source_count}", color=COLOR_PREDICTED, scale=0.34, advance=14)
    put(f"Mean err={mean_err:.2f}px", color=COLOR_PREDICTED, scale=0.34, advance=16)
    if len(per_point_err) > 0:
        put(f"Best match: P{patch_indices[best_local_idx]} err={per_point_err[best_local_idx]:.2f}")
        put(f"Worst match: P{patch_indices[worst_local_idx]} err={per_point_err[worst_local_idx]:.2f}", color=COLOR_ERROR)


# -----------------------------------------------------------------------------
# Mouse-selection helpers
# -----------------------------------------------------------------------------


def pick_nearest_point_index(points: np.ndarray | None, x: int, y: int, max_dist_px: float) -> int | None:
    """Return the nearest local cheek-point index to a mouse click if within threshold."""
    if points is None or len(points) == 0:
        return None

    click_pt = np.array([x, y], dtype=np.float32)
    dists = np.linalg.norm(points - click_pt, axis=1)
    best_idx = int(np.argmin(dists))
    if float(dists[best_idx]) <= max_dist_px:
        return best_idx
    return None


def handle_mouse_click(event, x, y, flags, state):
    """Left click toggles a source point; right click makes the clicked point the only source."""
    if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
        return

    points = state.get("latest_clickable_points")
    selected = pick_nearest_point_index(points, x, y, CLICK_SELECT_MAX_DIST_PX)
    if selected is None:
        return

    current = list(state.get("selected_source_indices", [INITIAL_SOURCE_POINT_INDEX]))

    if event == cv2.EVENT_RBUTTONDOWN:
        state["selected_source_indices"] = [selected]
        return

    if selected in current:
        if len(current) > 1:
            current.remove(selected)
    else:
        current.append(selected)
        current = sorted(set(current))

    state["selected_source_indices"] = current


def draw_pre_neutral_sources(frame, current_cheek_points, source_indices):
    """Draw all currently selected source points even before neutral capture."""
    for source_index in source_indices:
        source_current = current_cheek_points[source_index]
        source_xy = tuple(np.round(source_current).astype(int))
        cv2.circle(frame, source_xy, 4, COLOR_SOURCE, -1)
        cv2.putText(
            frame,
            f"S:{source_index}",
            tuple((np.round(source_current).astype(int) + np.array([6, -6])).tolist()),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOR_SOURCE,
            1,
            cv2.LINE_AA,
        )


# -----------------------------------------------------------------------------
# Model utilities
# -----------------------------------------------------------------------------


def get_mode1_checkpoint_path(source_index: int) -> Path:
    """Return the expected checkpoint path for one Mode 1 source point."""
    return MODE1_MODELS_DIR / MODE1_FILENAME_TEMPLATE.format(source_index)



def load_trained_model(model_path: Path, device: str) -> LoadedCheckpoint:
    """Load trained one-point-neighbors model checkpoint if available."""
    if not model_path.exists():
        return LoadedCheckpoint(
            model=None,
            source_point_index=None,
            neighbor_indices=None,
            input_dim=None,
            output_dim=None,
            message=f"No checkpoint at {model_path.name}",
            checkpoint_path=str(model_path),
        )

    checkpoint = torch.load(model_path, map_location=device)
    input_dim = int(checkpoint["input_dim"])
    output_dim = int(checkpoint["output_dim"])

    model = NeighborMLP(input_dim, output_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    trained_source = checkpoint.get("source_point_index", None)
    trained_neighbors = checkpoint.get("neighbor_indices", None)
    trained_neighbors = list(trained_neighbors) if trained_neighbors is not None else None

    message = (
        f"{Path(model_path).name} | src={trained_source}"
        if trained_source is not None else f"{Path(model_path).name} | missing metadata"
    )

    return LoadedCheckpoint(
        model=model,
        source_point_index=trained_source,
        neighbor_indices=trained_neighbors,
        input_dim=input_dim,
        output_dim=output_dim,
        message=message,
        checkpoint_path=str(model_path),
    )



def prediction_matches_selection(
    checkpoint: LoadedCheckpoint,
    source_index: int,
    patch_indices: list[int],
) -> bool:
    """Return True only when the current selection matches the loaded checkpoint."""
    if checkpoint.model is None:
        return False
    if checkpoint.source_point_index != source_index:
        return False
    if checkpoint.neighbor_indices is None:
        return False
    return list(patch_indices) == list(checkpoint.neighbor_indices)



def get_mode1_checkpoint_for_source(
    source_index: int,
    device: str,
    cache: dict[int, LoadedCheckpoint],
) -> LoadedCheckpoint:
    """Load and cache the Mode 1 checkpoint for a given source point."""
    if source_index in cache:
        return cache[source_index]

    ckpt_path = get_mode1_checkpoint_path(source_index)
    checkpoint = load_trained_model(ckpt_path, device)
    cache[source_index] = checkpoint
    return checkpoint


# -----------------------------------------------------------------------------
# Main application
# -----------------------------------------------------------------------------


def main():
    """Run the interactive patch explorer / blended multi-source Mode 1 viewer."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    direct_checkpoint_cache: dict[int, LoadedCheckpoint] = {}

    # Local display state.
    show_landmark_ids = SHOW_LANDMARK_IDS
    show_only_left_cheek_ids = SHOW_ONLY_LEFT_CHEEK_IDS

    # Interactive selection state.
    mouse_state = {
        "selected_source_indices": [INITIAL_SOURCE_POINT_INDEX],
        "latest_clickable_points": None,
    }
    active_patch_mode = PatchMode.DIRECT
    active_radius_px = DEFAULT_RADIUS_PX

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, handle_mouse_click, mouse_state)

    # Neutral tracking references.
    neutral_cheek_points = None
    neutral_anchor_points = None

    capture_requested = False
    cheek_capture_buffer = []
    anchor_capture_buffer = []

    patch_edges = build_local_patch_edges(
        LEFT_CHEEK_IDS,
        FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
    )
    adjacency = build_adjacency(len(LEFT_CHEEK_IDS), patch_edges)

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            height, width, _ = frame.shape
            current_cheek_points = None
            current_anchor_points = None

            if result.face_landmarks:
                face_landmarks = result.face_landmarks[0]

                if SHOW_ALL_LANDMARKS:
                    draw_all_landmarks(frame, face_landmarks, width, height)

                if show_landmark_ids:
                    if show_only_left_cheek_ids:
                        draw_landmark_ids(frame, face_landmarks, width, height, only_ids=set(LEFT_CHEEK_IDS))
                    else:
                        draw_landmark_ids(frame, face_landmarks, width, height)

                current_cheek_points = get_landmark_points_px(face_landmarks, LEFT_CHEEK_IDS, width, height)
                current_anchor_points = get_landmark_points_px(face_landmarks, HEAD_ANCHOR_IDS, width, height)

                mouse_state["latest_clickable_points"] = current_cheek_points
                selected_source_indices = list(mouse_state["selected_source_indices"])

                if SHOW_CHEEK_MESH:
                    draw_tracked_cheek_mesh(
                        frame,
                        face_landmarks,
                        set(LEFT_CHEEK_IDS),
                        FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                        width,
                        height,
                        color=(180, 0, 0),
                        thickness=1,
                    )

                if SHOW_ANCHORS:
                    draw_anchor_points(frame, current_anchor_points)

                if neutral_cheek_points is None:
                    draw_pre_neutral_sources(frame, current_cheek_points, selected_source_indices)
                    cv2.putText(
                        frame,
                        "L-click add/remove | R-click solo | press N for neutral",
                        (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        COLOR_INFO,
                        2,
                    )

                # Neutral capture
                if capture_requested:
                    cheek_capture_buffer.append(current_cheek_points.copy())
                    anchor_capture_buffer.append(current_anchor_points.copy())

                    cv2.putText(
                        frame,
                        f"Capturing neutral {len(cheek_capture_buffer)}/{NEUTRAL_CAPTURE_FRAMES}",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        COLOR_TEXT,
                        2,
                    )

                    if len(cheek_capture_buffer) >= NEUTRAL_CAPTURE_FRAMES:
                        neutral_cheek_points = np.mean(cheek_capture_buffer, axis=0).astype(np.float32)
                        neutral_anchor_points = np.mean(anchor_capture_buffer, axis=0).astype(np.float32)
                        cheek_capture_buffer.clear()
                        anchor_capture_buffer.clear()
                        capture_requested = False

                # Head-motion compensated cheek analysis + patch viewer
                if neutral_cheek_points is not None and neutral_anchor_points is not None:
                    transform = estimate_head_motion_transform(current_anchor_points, neutral_anchor_points)
                    if transform is None:
                        cv2.putText(frame, "Alignment failed", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ERROR, 2)
                    else:
                        anchor_alignment_error = compute_mean_alignment_error(
                            current_anchor_points,
                            neutral_anchor_points,
                            transform,
                        )

                        cv2.putText(
                            frame,
                            f"Anchor error: {anchor_alignment_error:.2f}px",
                            (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            COLOR_TEXT,
                            2,
                        )

                        if anchor_alignment_error > ANCHOR_ALIGNMENT_ERROR_THRESHOLD_PX:
                            cv2.putText(
                                frame,
                                "Pose too large for patch comparison",
                                (20, 130),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                COLOR_ERROR,
                                2,
                            )
                        else:
                            corrected_cheek_points = apply_affine_to_points(current_cheek_points, transform)

                            if corrected_cheek_points is not None:
                                cheek_disp = corrected_cheek_points - neutral_cheek_points

                                per_source_patch_indices = {}
                                active_patch_union = set()
                                source_mags = []
                                for source_index in selected_source_indices:
                                    patch_for_source = get_active_patch_indices(
                                        patch_mode=active_patch_mode,
                                        source_index=source_index,
                                        adjacency=adjacency,
                                        neutral_points=neutral_cheek_points,
                                        radius_px=active_radius_px,
                                    )
                                    per_source_patch_indices[source_index] = patch_for_source
                                    active_patch_union.update(patch_for_source)
                                    source_mags.append(float(np.linalg.norm(cheek_disp[source_index])))

                                active_patch_indices = sorted(active_patch_union)

                                draw_active_patch_multi(
                                    frame,
                                    neutral_cheek_points,
                                    corrected_cheek_points,
                                    selected_source_indices,
                                    active_patch_indices,
                                )

                                sources_text = ",".join(str(i) for i in selected_source_indices)
                                mean_source_mag = float(np.mean(source_mags)) if source_mags else 0.0
                                cv2.putText(
                                    frame,
                                    f"Sources [{sources_text}] | mean mag {mean_source_mag:.2f}px",
                                    (20, 130),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,
                                    COLOR_TEXT,
                                    2,
                                )

                                prediction_enabled = False
                                predicted_patch_points = None
                                pred_patch_disp = np.zeros((len(active_patch_indices), 2), dtype=np.float32)
                                contributing_source_count = 0

                                if active_patch_mode == PatchMode.DIRECT:
                                    patch_index_to_local = {patch_idx: i for i, patch_idx in enumerate(active_patch_indices)}
                                    pred_sum = np.zeros((len(active_patch_indices), 2), dtype=np.float32)
                                    pred_count = np.zeros(len(active_patch_indices), dtype=np.float32)
                                    status_parts = []

                                    for source_index in selected_source_indices:
                                        source_vec = cheek_disp[source_index]
                                        source_mag = float(np.linalg.norm(source_vec))
                                        source_patch_indices = per_source_patch_indices[source_index]
                                        checkpoint = get_mode1_checkpoint_for_source(
                                            source_index,
                                            device,
                                            direct_checkpoint_cache,
                                        )

                                        if source_mag < SOURCE_ACTIVITY_THRESHOLD_PX:
                                            status_parts.append(f"S{source_index}:low")
                                            continue
                                        if not prediction_matches_selection(
                                            checkpoint,
                                            source_index,
                                            source_patch_indices,
                                        ):
                                            status_parts.append(f"S{source_index}:no model")
                                            continue

                                        with torch.no_grad():
                                            x = torch.from_numpy(source_vec.astype(np.float32).reshape(1, 2)).to(device)
                                            source_pred = checkpoint.model(x).cpu().numpy().reshape(len(source_patch_indices), 2).astype(np.float32)

                                        for local_idx, patch_idx in enumerate(source_patch_indices):
                                            union_local_idx = patch_index_to_local[patch_idx]
                                            pred_sum[union_local_idx] += source_pred[local_idx]
                                            pred_count[union_local_idx] += 1.0

                                        contributing_source_count += 1
                                        status_parts.append(f"S{source_index}:ok")

                                    valid_mask = pred_count > 0
                                    if np.any(valid_mask):
                                        pred_patch_disp[valid_mask] = pred_sum[valid_mask] / pred_count[valid_mask, None]
                                        predicted_patch_points = neutral_cheek_points[active_patch_indices] + pred_patch_disp
                                        prediction_enabled = True

                                    checkpoint_info = f"Blend {contributing_source_count}/{len(selected_source_indices)} src | {'; '.join(status_parts[:3])}"
                                else:
                                    checkpoint_info = "Mode 2/3 prediction not trained yet"

                                if prediction_enabled and predicted_patch_points is not None:
                                    draw_prediction_overlay(
                                        frame,
                                        neutral_cheek_points,
                                        corrected_cheek_points,
                                        predicted_patch_points,
                                        active_patch_indices,
                                    )
                                elif active_patch_mode != PatchMode.DIRECT:
                                    cv2.putText(
                                        frame,
                                        "Blended prediction is supported only in Mode 1",
                                        (20, 160),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        COLOR_INFO,
                                        2,
                                    )
                                else:
                                    cv2.putText(
                                        frame,
                                        "No active blended prediction for current source set",
                                        (20, 160),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        COLOR_INFO,
                                        2,
                                    )

                                draw_side_panel(
                                    frame,
                                    cheek_disp=cheek_disp,
                                    source_indices=selected_source_indices,
                                    patch_mode=active_patch_mode,
                                    patch_indices=active_patch_indices,
                                    radius_px=active_radius_px,
                                    checkpoint_info=checkpoint_info,
                                    prediction_enabled=prediction_enabled,
                                    predicted_patch_displacements=pred_patch_disp,
                                    contributing_source_count=contributing_source_count,
                                )

                cv2.putText(
                    frame,
                    "Interactive Cheek Patch Explorer",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    COLOR_TEXT,
                    2,
                )
                draw_visual_legend(frame)

            cv2.putText(
                frame,
                "L-click add/remove | R-click solo | N neutral | 1/2/3 patch | -/= radius | I/O ids | Q quit",
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                COLOR_TEXT,
                1,
            )

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("n"):
                if current_cheek_points is not None and current_anchor_points is not None:
                    capture_requested = True
                    cheek_capture_buffer.clear()
                    anchor_capture_buffer.clear()
            elif key == ord("c"):
                neutral_cheek_points = None
                neutral_anchor_points = None
                capture_requested = False
                cheek_capture_buffer.clear()
                anchor_capture_buffer.clear()
            elif key == ord("i"):
                show_landmark_ids = not show_landmark_ids
            elif key == ord("o"):
                show_only_left_cheek_ids = not show_only_left_cheek_ids
            elif key == ord("1"):
                active_patch_mode = PatchMode.DIRECT
            elif key == ord("2"):
                active_patch_mode = PatchMode.TWO_RING
            elif key == ord("3"):
                active_patch_mode = PatchMode.RADIUS
            elif key in (ord('-'), ord('_')):
                active_radius_px = max(MIN_RADIUS_PX, active_radius_px - RADIUS_STEP_PX)
            elif key in (ord('='), ord('+')):
                active_radius_px = min(MAX_RADIUS_PX, active_radius_px + RADIUS_STEP_PX)
            elif key == 27 or key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
