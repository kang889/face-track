from __future__ import annotations

"""
face_press_multipass_recorder.py

Purpose
-------
Prototype recorder for a multi-pass finger-press cheek deformation experiment.

Experiment idea
---------------
- Choose one cheek source point / press location.
- Capture a neutral face with no finger touching the cheek.
- Press the same cheek source point multiple times.
- Slightly change finger angle across passes so different cheek landmarks become
  visible or blocked.
- For each frame, record only the cheek-landmark deformation that is still
  visible. Blocked cheek landmarks are kept as missing data (NaN) for that
  frame instead of being treated as valid measurements.

What this recorder captures
---------------------------
- Left-cheek face landmarks (MediaPipe Face Landmarker)
- Hand landmarks (MediaPipe Hands)
- A visible mask and blocked mask for the cheek patch
- Visible-only cheek deformation per frame
- Session / trial / pressure-tier metadata
- Fixed source-point label for the press experiment

What this recorder does NOT do
------------------------------
- It does not measure true physical force in Newtons.
- It does not guarantee that the hidden cheek surface under the finger is known.
- It is a visible-region multi-pass prototype recorder.

Controls
--------
Left click : choose the cheek source point / press location
N          : capture neutral face
R          : start/stop recording current trial
T          : next trial (only when not recording)
1 / 2 / 3  : pressure tier = light / medium / hard
I          : toggle all face landmark IDs
O          : toggle cheek-only IDs
C          : clear neutral and reset current trial buffers
Q / ESC    : quit
"""

import os
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

WINDOW_NAME = "Face Press Multi-Pass Recorder"
MODEL_PATH = "models/face_landmarker.task"
HAND_MODEL_PATH = "models/hand_landmarker.task"
DATASET_ROOT_DIR = Path("ml_dataset_face_press_multipass")
SESSION_PREFIX = "face_press_session"
SAVE_EVERY_N_VALID_FRAMES = 1

# Left cheek landmark subset.
LEFT_CHEEK_IDS = [
    36, 50, 101, 111, 116, 117, 118, 123, 135, 137, 138, 147, 177,
    187, 192, 203, 205, 206, 207, 212, 213, 214, 215, 216, 227,
]

# Expanded rigid-face anchors used for head-motion compensation.
# Chosen around the eyes, brows, and nose bridge so they are less affected by
# cheek pressing than the cheek patch itself.
HEAD_ANCHOR_IDS = [
    33, 133, 159, 145,      # right eye
    362, 263, 386, 374,     # left eye
    70, 63, 105, 66, 107,   # right brow
    336, 296, 334, 293, 300,# left brow
    6, 168, 197, 195, 5, 4  # nose bridge / central nose
]
MIN_VISIBLE_ANCHORS = 6
ANCHOR_HAND_POINT_BLOCK_RADIUS_PX = 20.0
ANCHOR_HAND_SEGMENT_BLOCK_RADIUS_PX = 16.0
TRANSFORM_EMA_ALPHA = 0.35
Z_SHIFT_EMA_ALPHA = 0.35

# Neutral capture.
NEUTRAL_CAPTURE_FRAMES = 20

# Recording / gating thresholds.
ANCHOR_ALIGNMENT_ERROR_THRESHOLD_PX = 8.0
ARROW_THRESHOLD_PX = 3.0
MIN_VISIBLE_CHEEK_POINTS = 8

# Finger-occlusion heuristic.
# We use only the index-finger chain as the occluder for this prototype.
HAND_OCCLUSION_IDS = [5, 6, 7, 8]
HAND_OCCLUSION_CONNECTIONS = [(5, 6), (6, 7), (7, 8)]
CONTACT_TIP_ID = 8
HAND_POINT_BLOCK_RADIUS_PX = 18.0
HAND_SEGMENT_BLOCK_RADIUS_PX = 14.0
CLICK_SELECT_MAX_DIST_PX = 24.0

# Hand tracker parameters.
HAND_MAX_NUM_HANDS = 1
HAND_MIN_DET_CONF = 0.5
HAND_MIN_TRACK_CONF = 0.5

# Display toggles.
SHOW_ALL_LANDMARKS = True
SHOW_LANDMARK_IDS = False
SHOW_ONLY_LEFT_CHEEK_IDS = False
SHOW_CHEEK_MESH = True
SHOW_CORRECTED_CHEEK_POINTS = True
SHOW_HAND_LANDMARKS = True
SHOW_BLOCKED_CHEEK_POINTS = True

PRESSURE_LABELS = ["light", "medium", "hard"]
DEFAULT_PRESSURE_INDEX = 1

# Colors (BGR)
COLOR_SOURCE = (0, 255, 255)        # yellow
COLOR_VISIBLE = (0, 255, 0)         # green
COLOR_BLOCKED = (0, 0, 255)         # red
COLOR_NEUTRAL = (140, 140, 140)     # gray
COLOR_HAND = (255, 200, 0)          # orange
COLOR_TEXT = (255, 255, 255)
COLOR_INFO = (180, 180, 180)
COLOR_ERROR = (0, 0, 255)
COLOR_WARN = (0, 255, 255)


# -----------------------------------------------------------------------------
# MediaPipe setup
# -----------------------------------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

FACE_OPTIONS = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
)

HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions

HAND_OPTIONS = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=HAND_MAX_NUM_HANDS,
    min_hand_detection_confidence=HAND_MIN_DET_CONF,
    min_hand_presence_confidence=HAND_MIN_DET_CONF,
    min_tracking_confidence=HAND_MIN_TRACK_CONF,
)


# -----------------------------------------------------------------------------
# Helpers: face / hand landmarks
# -----------------------------------------------------------------------------


def get_landmark_points_px(face_landmarks, ids, width, height) -> np.ndarray:
    """Extract selected normalized MediaPipe landmarks as pixel-space points."""
    points = []
    for idx in ids:
        lm = face_landmarks[idx]
        points.append([lm.x * width, lm.y * height])
    return np.array(points, dtype=np.float32)



def get_landmark_points_xyz(face_landmarks, ids) -> np.ndarray:
    """Extract selected MediaPipe landmarks as normalized x/y/z values."""
    points = []
    for idx in ids:
        lm = face_landmarks[idx]
        points.append([lm.x, lm.y, lm.z])
    return np.array(points, dtype=np.float32)



def draw_all_landmarks(frame, face_landmarks, width, height):
    """Draw all detected face landmarks for debugging."""
    for lm in face_landmarks:
        x = int(lm.x * width)
        y = int(lm.y * height)
        cv2.circle(frame, (x, y), 1, (0, 100, 0), -1)



def draw_landmark_ids(frame, face_landmarks, width, height, only_ids=None):
    """Draw face landmark index numbers."""
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



def extract_first_hand_points(result_hands, width: int, height: int):
    """Return the first hand's landmarks in pixel and normalized xyz space."""
    if not result_hands.hand_landmarks:
        return None, None

    hand_landmarks = result_hands.hand_landmarks[0]
    hand_px = []
    hand_xyz = []
    for lm in hand_landmarks:
        hand_px.append([lm.x * width, lm.y * height])
        hand_xyz.append([lm.x, lm.y, lm.z])

    return np.array(hand_px, dtype=np.float32), np.array(hand_xyz, dtype=np.float32)



def draw_hand(frame, result_hands, width: int, height: int):
    """Draw the tracked hand landmarks and skeleton without mp.solutions."""
    if not result_hands.hand_landmarks:
        return

    for hand_landmarks in result_hands.hand_landmarks:
        for conn in HandLandmarksConnections.HAND_CONNECTIONS:
            a = conn.start
            b = conn.end
            lm1 = hand_landmarks[a]
            lm2 = hand_landmarks[b]
            x1, y1 = int(lm1.x * width), int(lm1.y * height)
            x2, y2 = int(lm2.x * width), int(lm2.y * height)
            cv2.line(frame, (x1, y1), (x2, y2), COLOR_HAND, 2, cv2.LINE_AA)

        for idx, lm in enumerate(hand_landmarks):
            x, y = int(lm.x * width), int(lm.y * height)
            radius = 4 if idx == CONTACT_TIP_ID else 3
            cv2.circle(frame, (x, y), radius, COLOR_HAND, -1)


# -----------------------------------------------------------------------------
# Head-motion compensation helpers
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
# Patch / visibility helpers
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



def point_to_segment_distance(point: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray) -> float:
    """Distance from a point to a 2D segment."""
    ab = seg_b - seg_a
    denom = float(np.dot(ab, ab))
    if denom < 1e-6:
        return float(np.linalg.norm(point - seg_a))
    t = float(np.dot(point - seg_a, ab) / denom)
    t = max(0.0, min(1.0, t))
    closest = seg_a + t * ab
    return float(np.linalg.norm(point - closest))



def compute_blocked_cheek_mask(
    cheek_points_px: np.ndarray,
    hand_points_px: np.ndarray | None,
    point_radius_px: float,
    segment_radius_px: float,
) -> np.ndarray:
    """
    Mark cheek points as blocked if they lie too close to the detected index-finger
    landmarks or index-finger segments.
    """
    blocked = np.zeros(len(cheek_points_px), dtype=bool)
    if hand_points_px is None:
        return blocked

    # Finger landmarks themselves.
    for local_i, cheek_pt in enumerate(cheek_points_px):
        for hand_idx in HAND_OCCLUSION_IDS:
            if np.linalg.norm(cheek_pt - hand_points_px[hand_idx]) <= point_radius_px:
                blocked[local_i] = True
                break

    # Finger segments.
    for local_i, cheek_pt in enumerate(cheek_points_px):
        if blocked[local_i]:
            continue
        for a, b in HAND_OCCLUSION_CONNECTIONS:
            dist = point_to_segment_distance(cheek_pt, hand_points_px[a], hand_points_px[b])
            if dist <= segment_radius_px:
                blocked[local_i] = True
                break

    return blocked


def smooth_affine_transform(prev_transform: np.ndarray | None, current_transform: np.ndarray, alpha: float) -> np.ndarray:
    """Exponentially smooth the affine transform to reduce frame-to-frame jitter."""
    if prev_transform is None:
        return current_transform.astype(np.float32)
    return ((1.0 - alpha) * prev_transform + alpha * current_transform).astype(np.float32)


def smooth_scalar(prev_value: float | None, current_value: float, alpha: float) -> float:
    """Exponentially smooth a scalar measurement."""
    if prev_value is None:
        return float(current_value)
    return float((1.0 - alpha) * prev_value + alpha * current_value)


def compute_anchor_visibility_mask(anchor_points_px: np.ndarray, hand_points_px: np.ndarray | None) -> np.ndarray:
    """Mark rigid-face anchors blocked if the finger crosses them."""
    return ~compute_blocked_cheek_mask(
        anchor_points_px,
        hand_points_px,
        ANCHOR_HAND_POINT_BLOCK_RADIUS_PX,
        ANCHOR_HAND_SEGMENT_BLOCK_RADIUS_PX,
    )


def compute_corrected_cheek_xyz(
    current_cheek_xyz: np.ndarray,
    neutral_cheek_xyz: np.ndarray,
    current_anchor_xyz: np.ndarray,
    neutral_anchor_xyz: np.ndarray,
    anchor_visible_mask: np.ndarray,
    prev_z_shift: float | None,
) -> tuple[np.ndarray, float]:
    """Compensate global forward/back head motion by removing the median anchor z shift."""
    if np.sum(anchor_visible_mask) < 1:
        z_shift = 0.0
    else:
        z_shift = float(np.median(current_anchor_xyz[anchor_visible_mask, 2] - neutral_anchor_xyz[anchor_visible_mask, 2]))

    z_shift = smooth_scalar(prev_z_shift, z_shift, Z_SHIFT_EMA_ALPHA)
    corrected = current_cheek_xyz.copy().astype(np.float32)
    corrected[:, 2] = corrected[:, 2] - z_shift
    return corrected, z_shift


# -----------------------------------------------------------------------------
# Drawing helpers
# -----------------------------------------------------------------------------


def draw_arrow(frame, start_pt, end_pt, color, thickness=1, threshold_px=0.0):
    """Draw an arrow only if the motion is large enough."""
    motion = float(np.linalg.norm(end_pt - start_pt))
    if motion < threshold_px:
        return
    start = tuple(np.round(start_pt).astype(int))
    end = tuple(np.round(end_pt).astype(int))
    cv2.arrowedLine(frame, start, end, color, thickness, tipLength=0.22)



def draw_cheek_status(
    frame,
    neutral_points_px,
    corrected_points_px,
    visible_mask,
    blocked_mask,
    source_index,
):
    """Draw visible and blocked cheek points plus the chosen source point."""
    source_neutral = neutral_points_px[source_index]
    source_current = corrected_points_px[source_index]
    cv2.circle(frame, tuple(np.round(source_current).astype(int)), 5, COLOR_SOURCE, -1)
    draw_arrow(frame, source_neutral, source_current, COLOR_SOURCE, 2, ARROW_THRESHOLD_PX)
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

    for i, (neutral_pt, corrected_pt) in enumerate(zip(neutral_points_px, corrected_points_px)):
        if i == source_index:
            continue
        if blocked_mask[i] and SHOW_BLOCKED_CHEEK_POINTS:
            cv2.circle(frame, tuple(np.round(corrected_pt).astype(int)), 4, COLOR_BLOCKED, -1)
            continue
        if visible_mask[i]:
            cv2.circle(frame, tuple(np.round(corrected_pt).astype(int)), 3, COLOR_VISIBLE, -1)
            draw_arrow(frame, neutral_pt, corrected_pt, COLOR_VISIBLE, 1, ARROW_THRESHOLD_PX)



def draw_pre_neutral_source(frame, current_cheek_points_px, source_index):
    """Draw the selected source point before neutral capture."""
    if current_cheek_points_px is None or len(current_cheek_points_px) == 0:
        return
    source_current = current_cheek_points_px[source_index]
    source_xy = tuple(np.round(source_current).astype(int))
    cv2.circle(frame, source_xy, 5, COLOR_SOURCE, -1)
    cv2.putText(
        frame,
        f"S:{source_index}",
        (source_xy[0] + 6, source_xy[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        COLOR_SOURCE,
        1,
        cv2.LINE_AA,
    )


# -----------------------------------------------------------------------------
# Mouse selection
# -----------------------------------------------------------------------------


def pick_nearest_point_index(points: np.ndarray | None, x: int, y: int, max_dist_px: float) -> int | None:
    if points is None or len(points) == 0:
        return None
    click = np.array([x, y], dtype=np.float32)
    dists = np.linalg.norm(points - click, axis=1)
    best_idx = int(np.argmin(dists))
    if float(dists[best_idx]) <= max_dist_px:
        return best_idx
    return None



def handle_mouse_click(event, x, y, flags, state):
    """Left click chooses the cheek source point. Disabled during recording."""
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if state.get("recording", False):
        state["status_message"] = "Stop recording before changing source point"
        return

    points = state.get("latest_clickable_points")
    selected = pick_nearest_point_index(points, x, y, CLICK_SELECT_MAX_DIST_PX)
    if selected is not None:
        state["active_source_index"] = selected
        state["status_message"] = f"Selected source point {selected} (MP {LEFT_CHEEK_IDS[selected]})"


# -----------------------------------------------------------------------------
# Dataset recording
# -----------------------------------------------------------------------------


def save_face_press_sample(
    output_dir: Path,
    sample_index: int,
    session_label: str,
    trial_index: int,
    pressure_label: str,
    source_index: int,
    neutral_cheek_points_px: np.ndarray,
    corrected_cheek_points_px: np.ndarray,
    neutral_cheek_xyz: np.ndarray,
    current_cheek_xyz: np.ndarray,
    corrected_cheek_xyz: np.ndarray,
    visible_mask: np.ndarray,
    blocked_mask: np.ndarray,
    hand_present: bool,
    hand_landmarks_px: np.ndarray,
    hand_landmarks_xyz: np.ndarray,
    fingertip_px: np.ndarray,
    fingertip_xyz: np.ndarray,
    patch_edges: list[tuple[int, int]],
    anchor_alignment_error: float,
):
    """
    Save one frame of the multi-pass face-press experiment.

    Blocked cheek points are not treated as valid measurements for that frame.
    We keep full arrays for indexing consistency, but visible-only arrays store
    NaN at blocked positions.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cheek_displacement_xy = (corrected_cheek_points_px - neutral_cheek_points_px).astype(np.float32)
    cheek_displacement_xyz = (corrected_cheek_xyz - neutral_cheek_xyz).astype(np.float32)

    visible_cheek_displacement_xy = cheek_displacement_xy.copy()
    visible_cheek_displacement_xy[blocked_mask] = np.nan

    visible_cheek_displacement_xyz = cheek_displacement_xyz.copy()
    visible_cheek_displacement_xyz[blocked_mask] = np.nan

    point_motion_magnitude_visible_xy = np.linalg.norm(
        np.nan_to_num(visible_cheek_displacement_xy, nan=0.0),
        axis=1,
    ).astype(np.float32)

    np.savez_compressed(
        output_dir / f"sample_{sample_index:06d}.npz",
        session_label=np.array(session_label),
        trial_index=np.int32(trial_index),
        pressure_label=np.array(pressure_label),
        source_local_index=np.int32(source_index),
        source_mp_id=np.int32(LEFT_CHEEK_IDS[source_index]),
        visible_mask=visible_mask.astype(np.uint8),
        blocked_mask=blocked_mask.astype(np.uint8),
        neutral_cheek_points_px=neutral_cheek_points_px.astype(np.float32),
        corrected_cheek_points_px=corrected_cheek_points_px.astype(np.float32),
        cheek_displacement_xy=cheek_displacement_xy.astype(np.float32),
        visible_cheek_displacement_xy=visible_cheek_displacement_xy.astype(np.float32),
        neutral_cheek_xyz=neutral_cheek_xyz.astype(np.float32),
        current_cheek_xyz=current_cheek_xyz.astype(np.float32),
        corrected_cheek_xyz=corrected_cheek_xyz.astype(np.float32),
        cheek_displacement_xyz=cheek_displacement_xyz.astype(np.float32),
        visible_cheek_displacement_xyz=visible_cheek_displacement_xyz.astype(np.float32),
        point_motion_magnitude_visible_xy=point_motion_magnitude_visible_xy,
        hand_present=np.uint8(hand_present),
        hand_landmarks_px=hand_landmarks_px.astype(np.float32),
        hand_landmarks_xyz=hand_landmarks_xyz.astype(np.float32),
        fingertip_px=fingertip_px.astype(np.float32),
        fingertip_xyz=fingertip_xyz.astype(np.float32),
        patch_edges=np.array(patch_edges, dtype=np.int32),
        left_cheek_ids=np.array(LEFT_CHEEK_IDS, dtype=np.int32),
        anchor_alignment_error=np.array(anchor_alignment_error, dtype=np.float32),
    )


# -----------------------------------------------------------------------------
# Main application
# -----------------------------------------------------------------------------


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    global SHOW_LANDMARK_IDS, SHOW_ONLY_LEFT_CHEEK_IDS

    session_label = f"{SESSION_PREFIX}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = DATASET_ROOT_DIR / session_label

    mouse_state = {
        "active_source_index": 13,
        "latest_clickable_points": None,
        "recording": False,
        "status_message": "Click a cheek source point, then press N for neutral",
    }

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, handle_mouse_click, mouse_state)

    neutral_cheek_points_px = None
    neutral_anchor_points_px = None
    neutral_cheek_xyz = None
    neutral_anchor_xyz = None

    smoothed_transform = None
    smoothed_z_shift = None

    patch_edges = build_local_patch_edges(
        LEFT_CHEEK_IDS,
        FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
    )

    capture_requested = False
    cheek_capture_buffer_px = []
    anchor_capture_buffer_px = []
    cheek_capture_buffer_xyz = []
    anchor_capture_buffer_xyz = []

    record_dataset = False
    valid_frame_index = 0
    saved_sample_count = 0
    current_trial_index = 1
    pressure_index = DEFAULT_PRESSURE_INDEX

    with FaceLandmarker.create_from_options(FACE_OPTIONS) as landmarker, \
         HandLandmarker.create_from_options(HAND_OPTIONS) as hand_landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int(time.time() * 1000)
            face_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            height, width, _ = frame.shape
            current_cheek_points_px = None
            current_anchor_points_px = None
            current_cheek_xyz = None
            current_anchor_xyz = None
            hand_landmarks_px = np.full((21, 2), np.nan, dtype=np.float32)
            hand_landmarks_xyz = np.full((21, 3), np.nan, dtype=np.float32)
            fingertip_px = np.array([np.nan, np.nan], dtype=np.float32)
            fingertip_xyz = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
            hand_present = False

            if face_result.face_landmarks:
                face_landmarks = face_result.face_landmarks[0]

                if SHOW_ALL_LANDMARKS:
                    draw_all_landmarks(frame, face_landmarks, width, height)

                if SHOW_LANDMARK_IDS:
                    if SHOW_ONLY_LEFT_CHEEK_IDS:
                        draw_landmark_ids(frame, face_landmarks, width, height, only_ids=set(LEFT_CHEEK_IDS))
                    else:
                        draw_landmark_ids(frame, face_landmarks, width, height)

                current_cheek_points_px = get_landmark_points_px(face_landmarks, LEFT_CHEEK_IDS, width, height)
                current_anchor_points_px = get_landmark_points_px(face_landmarks, HEAD_ANCHOR_IDS, width, height)
                current_cheek_xyz = get_landmark_points_xyz(face_landmarks, LEFT_CHEEK_IDS)
                current_anchor_xyz = get_landmark_points_xyz(face_landmarks, HEAD_ANCHOR_IDS)
                mouse_state["latest_clickable_points"] = current_cheek_points_px

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

                if hand_result.hand_landmarks:
                    hand_present = True
                    hand_landmarks_px, hand_landmarks_xyz = extract_first_hand_points(hand_result, width, height)
                    fingertip_px = hand_landmarks_px[CONTACT_TIP_ID].copy()
                    fingertip_xyz = hand_landmarks_xyz[CONTACT_TIP_ID].copy()
                    if SHOW_HAND_LANDMARKS:
                        draw_hand(frame, hand_result, width, height)
                        if not np.isnan(fingertip_px).any():
                            cv2.circle(frame, tuple(np.round(fingertip_px).astype(int)), 6, COLOR_HAND, -1)

                # Pre-neutral source display.
                active_source_index = int(mouse_state["active_source_index"])
                if neutral_cheek_points_px is None:
                    draw_pre_neutral_source(frame, current_cheek_points_px, active_source_index)
                    cv2.putText(
                        frame,
                        "Click source point, then N for neutral. Use same source for all passes.",
                        (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        COLOR_INFO,
                        2,
                    )

                # Neutral capture.
                if capture_requested:
                    cheek_capture_buffer_px.append(current_cheek_points_px.copy())
                    anchor_capture_buffer_px.append(current_anchor_points_px.copy())
                    cheek_capture_buffer_xyz.append(current_cheek_xyz.copy())
                    anchor_capture_buffer_xyz.append(current_anchor_xyz.copy())

                    cv2.putText(
                        frame,
                        f"Capturing neutral {len(cheek_capture_buffer_px)}/{NEUTRAL_CAPTURE_FRAMES}",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        COLOR_TEXT,
                        2,
                    )

                    if len(cheek_capture_buffer_px) >= NEUTRAL_CAPTURE_FRAMES:
                        neutral_cheek_points_px = np.mean(cheek_capture_buffer_px, axis=0).astype(np.float32)
                        neutral_anchor_points_px = np.mean(anchor_capture_buffer_px, axis=0).astype(np.float32)
                        neutral_cheek_xyz = np.mean(cheek_capture_buffer_xyz, axis=0).astype(np.float32)
                        neutral_anchor_xyz = np.mean(anchor_capture_buffer_xyz, axis=0).astype(np.float32)

                        cheek_capture_buffer_px.clear()
                        anchor_capture_buffer_px.clear()
                        cheek_capture_buffer_xyz.clear()
                        anchor_capture_buffer_xyz.clear()
                        smoothed_transform = None
                        smoothed_z_shift = None
                        capture_requested = False
                        mouse_state["status_message"] = "Neutral captured"

                # Main deformation recording stage.
                if neutral_cheek_points_px is not None and neutral_anchor_points_px is not None and neutral_cheek_xyz is not None and neutral_anchor_xyz is not None:
                    anchor_visible_mask = compute_anchor_visibility_mask(
                        current_anchor_points_px,
                        hand_landmarks_px if hand_present else None,
                    )
                    visible_anchor_count = int(np.sum(anchor_visible_mask))

                    if visible_anchor_count < MIN_VISIBLE_ANCHORS:
                        cv2.putText(frame, "Not enough visible rigid anchors", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ERROR, 2)
                    else:
                        transform = estimate_head_motion_transform(
                            current_anchor_points_px[anchor_visible_mask],
                            neutral_anchor_points_px[anchor_visible_mask],
                        )

                        if transform is None:
                            cv2.putText(frame, "Alignment failed", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ERROR, 2)
                        else:
                            transform = smooth_affine_transform(smoothed_transform, transform, TRANSFORM_EMA_ALPHA)
                            smoothed_transform = transform

                            anchor_alignment_error = compute_mean_alignment_error(
                                current_anchor_points_px[anchor_visible_mask],
                                neutral_anchor_points_px[anchor_visible_mask],
                                transform,
                            )

                            cv2.putText(
                                frame,
                                f"Anchor error: {anchor_alignment_error:.2f}px | visible anchors: {visible_anchor_count}",
                                (20, 130),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                COLOR_TEXT,
                                2,
                            )

                            if anchor_alignment_error > ANCHOR_ALIGNMENT_ERROR_THRESHOLD_PX:
                                cv2.putText(frame, "Pose too large for valid capture", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_ERROR, 2)
                            else:
                                corrected_cheek_points_px = apply_affine_to_points(current_cheek_points_px, transform)
                                if corrected_cheek_points_px is not None:
                                    corrected_cheek_xyz, smoothed_z_shift = compute_corrected_cheek_xyz(
                                        current_cheek_xyz,
                                        neutral_cheek_xyz,
                                        current_anchor_xyz,
                                        neutral_anchor_xyz,
                                        anchor_visible_mask,
                                        smoothed_z_shift,
                                    )

                                    blocked_mask = compute_blocked_cheek_mask(
                                        current_cheek_points_px,
                                        hand_landmarks_px if hand_present else None,
                                        HAND_POINT_BLOCK_RADIUS_PX,
                                        HAND_SEGMENT_BLOCK_RADIUS_PX,
                                    )
                                    visible_mask = ~blocked_mask
                                    visible_count = int(np.sum(visible_mask))

                                    if SHOW_CORRECTED_CHEEK_POINTS:
                                        draw_cheek_status(
                                            frame,
                                            neutral_cheek_points_px,
                                            corrected_cheek_points_px,
                                            visible_mask,
                                            blocked_mask,
                                            active_source_index,
                                        )

                                    cheek_disp_xy = corrected_cheek_points_px - neutral_cheek_points_px
                                    visible_disp = np.linalg.norm(np.where(visible_mask[:, None], cheek_disp_xy, np.nan), axis=1)
                                    mean_visible_disp = float(np.nanmean(visible_disp)) if np.any(visible_mask) else 0.0
                                    mean_visible_dz = float(np.nanmean(np.where(visible_mask, corrected_cheek_xyz[:, 2] - neutral_cheek_xyz[:, 2], np.nan))) if np.any(visible_mask) else 0.0

                                    cv2.putText(
                                        frame,
                                        f"Source S:{active_source_index} MP:{LEFT_CHEEK_IDS[active_source_index]}",
                                        (20, 190),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.55,
                                        COLOR_SOURCE,
                                        2,
                                    )
                                    cv2.putText(
                                        frame,
                                        f"Visible cheek points: {visible_count}/{len(LEFT_CHEEK_IDS)}",
                                        (20, 220),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.55,
                                        COLOR_VISIBLE if visible_count >= MIN_VISIBLE_CHEEK_POINTS else COLOR_WARN,
                                        2,
                                    )
                                    cv2.putText(
                                        frame,
                                        f"Mean visible disp: {mean_visible_disp:.2f}px | mean dz: {mean_visible_dz:.4f}",
                                        (20, 250),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.55,
                                        COLOR_TEXT,
                                        2,
                                    )

                                    valid_frame_index += 1
                                    if record_dataset and visible_count >= MIN_VISIBLE_CHEEK_POINTS and valid_frame_index % SAVE_EVERY_N_VALID_FRAMES == 0:
                                        save_face_press_sample(
                                            output_dir=output_dir,
                                            sample_index=saved_sample_count,
                                            session_label=session_label,
                                            trial_index=current_trial_index,
                                            pressure_label=PRESSURE_LABELS[pressure_index],
                                            source_index=active_source_index,
                                            neutral_cheek_points_px=neutral_cheek_points_px,
                                            corrected_cheek_points_px=corrected_cheek_points_px,
                                            neutral_cheek_xyz=neutral_cheek_xyz,
                                            current_cheek_xyz=current_cheek_xyz,
                                            corrected_cheek_xyz=corrected_cheek_xyz,
                                            visible_mask=visible_mask,
                                            blocked_mask=blocked_mask,
                                            hand_present=hand_present,
                                            hand_landmarks_px=hand_landmarks_px,
                                            hand_landmarks_xyz=hand_landmarks_xyz,
                                            fingertip_px=fingertip_px,
                                            fingertip_xyz=fingertip_xyz,
                                            patch_edges=patch_edges,
                                            anchor_alignment_error=anchor_alignment_error,
                                        )
                                        saved_sample_count += 1

            cv2.putText(frame, "Face Press Multi-Pass Recorder (Compensated)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)
            cv2.putText(frame, f"Session: {session_label}", (20, height - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_INFO, 1)
            cv2.putText(frame, f"Trial: {current_trial_index} | Pressure: {PRESSURE_LABELS[pressure_index]} | Recording: {'ON' if record_dataset else 'OFF'} | Saved: {saved_sample_count}", (20, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1)
            cv2.putText(frame, mouse_state["status_message"], (20, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_INFO, 1)
            cv2.putText(frame, "Click source | N neutral | R record | T next trial | 1/2/3 pressure | I/O IDs | C clear | Q quit", (20, height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT, 1)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            mouse_state["recording"] = record_dataset

            if key == ord("n"):
                if current_cheek_points_px is not None and current_anchor_points_px is not None and not record_dataset:
                    capture_requested = True
                    cheek_capture_buffer_px.clear()
                    anchor_capture_buffer_px.clear()
                    cheek_capture_buffer_xyz.clear()
                    anchor_capture_buffer_xyz.clear()
                    smoothed_transform = None
                    smoothed_z_shift = None
                    mouse_state["status_message"] = "Capturing neutral..."

            elif key == ord("r"):
                if neutral_cheek_points_px is None:
                    mouse_state["status_message"] = "Capture neutral first"
                else:
                    record_dataset = not record_dataset
                    mouse_state["recording"] = record_dataset
                    mouse_state["status_message"] = "Recording ON" if record_dataset else "Recording OFF"

            elif key == ord("t"):
                if record_dataset:
                    mouse_state["status_message"] = "Stop recording before moving to next trial"
                else:
                    current_trial_index += 1
                    mouse_state["status_message"] = f"Moved to trial {current_trial_index}"

            elif key == ord("1"):
                pressure_index = 0
                mouse_state["status_message"] = "Pressure tier = light"

            elif key == ord("2"):
                pressure_index = 1
                mouse_state["status_message"] = "Pressure tier = medium"

            elif key == ord("3"):
                pressure_index = 2
                mouse_state["status_message"] = "Pressure tier = hard"

            elif key == ord("i"):
                SHOW_LANDMARK_IDS = not SHOW_LANDMARK_IDS

            elif key == ord("o"):
                SHOW_ONLY_LEFT_CHEEK_IDS = not SHOW_ONLY_LEFT_CHEEK_IDS

            elif key == ord("c"):
                neutral_cheek_points_px = None
                neutral_anchor_points_px = None
                neutral_cheek_xyz = None
                neutral_anchor_xyz = None
                smoothed_transform = None
                smoothed_z_shift = None
                capture_requested = False
                cheek_capture_buffer_px.clear()
                anchor_capture_buffer_px.clear()
                cheek_capture_buffer_xyz.clear()
                anchor_capture_buffer_xyz.clear()
                record_dataset = False
                mouse_state["recording"] = False
                mouse_state["status_message"] = "Neutral cleared"

            elif key == 27 or key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
