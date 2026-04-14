"""
webcam_landmarks_linear_skinning.py

Left-cheek facial tracking prototype with a simple 2D linear skinning stage.

Overview
--------
This program keeps the original project scope:
- left cheek only
- neutral cheek capture
- neutral anchor capture
- anchor-based head-motion compensation
- corrected cheek motion analysis with pose/error gating

New deformation stage
---------------------
Instead of using Similarity MLS, this version deforms a separate 2D cheek patch
using simple linear skinning.

Conceptual roles
----------------
- landmarks: tracked input from MediaPipe
- anchor landmarks: used only for global head-motion compensation
- drivers / handles: a few stable cheek control points extracted from the
  corrected cheek motion
- cheek patch: separate 2D output geometry, initialized from the neutral cheek
  shape and deformed by linear skinning

What linear skinning means here
-------------------------------
Each patch vertex has fixed weights to the cheek drivers. Every frame:
1. The program computes driver displacement from neutral.
2. Each patch vertex blends those driver displacements using its own weights.
3. The blended displacement is added to the neutral patch vertex.

For this simple 2D prototype, the deformation formula is:
    v' = v + w1*d1 + w2*d2 + ... + wk*dk

where:
- v  = neutral patch vertex
- di = displacement of driver i from its neutral position
- wi = fixed weight of that vertex to driver i
- v' = deformed patch vertex

This is a lightweight translation-only skinning stage. It is intentionally
simpler than the previous Similarity MLS version and is easier to explain and
extend toward future skinning work.

This file also includes an optional dataset-recording mode for future ML work.
When recording is enabled, valid frames are saved as compressed NumPy samples
containing cheek motion inputs and the matching linear-skinning patch output.
"""

import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MODEL_PATH = "models/face_landmarker.task"

# Left cheek landmark subset.
LEFT_CHEEK_IDS = [
    36, 50, 101, 111, 118, 117, 116, 123, 147,
    192, 216, 206, 207, 205, 203, 212, 214, 187,
]

# Relatively stable landmarks used only for head-motion compensation.
HEAD_ANCHOR_IDS = [33, 133, 362, 263, 6, 168]

# Neutral capture.
NEUTRAL_CAPTURE_FRAMES = 20

# Driver extraction.
DRIVER_REGION_COUNT = 3  # upper / middle / lower cheek

# Linear skinning weight construction.
# Weights are computed once from the neutral patch and neutral driver positions.
# A higher power makes each vertex follow the nearest driver more strongly.
SKINNING_WEIGHT_POWER = 2.0
SKINNING_EPSILON = 1e-6

# Motion-analysis and gating thresholds.
ARROW_THRESHOLD_PX = 3.0
REGION_ACTIVITY_THRESHOLD_PX = 4.0
ANCHOR_ALIGNMENT_ERROR_THRESHOLD_PX = 8.0
DRIVER_ARROW_THRESHOLD_PX = 1.5

# Display toggles.
SHOW_ALL_LANDMARKS = True
SHOW_ANCHORS = False
SHOW_DRIVER_HANDLES = True
SHOW_SKINNED_PATCH = True

# Dataset recording for future ML work.
# When recording is enabled, the program saves one .npz file per sampled frame.
# Each file stores the cheek motion input and the corresponding linear-skinning
# patch deformation target.
DATASET_OUTPUT_DIR = Path("ml_dataset_linear_skinning")
SAVE_EVERY_N_VALID_FRAMES = 2


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
    """
    Extract selected normalized MediaPipe landmarks as pixel-space points.

    Parameters
    ----------
    face_landmarks : sequence
        MediaPipe face landmark list for one face.
    ids : list[int]
        Landmark indices to extract.
    width, height : int
        Frame dimensions.

    Returns
    -------
    np.ndarray
        Array of shape (N, 2), dtype float32.
    """
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



def draw_tracked_cheek_mesh(frame, face_landmarks, cheek_ids, connections, width, height,
                            color=(180, 0, 0), thickness=1):
    """
    Draw the tracked left-cheek landmark mesh.

    This is still the measured input geometry, not the skinned output patch.
    """
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
    """
    Estimate a partial affine transform mapping:
        current anchors -> neutral anchors

    This removes most global translation / rotation / scale before cheek motion
    is compared.
    """
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
    """
    Measure the mean post-alignment anchor error in pixels.

    A lower value means the current face pose aligns to the neutral pose more
    reliably. This is used as a gate to suppress misleading cheek comparisons.
    """
    corrected_anchor_points = apply_affine_to_points(current_anchor_points, transform)
    if corrected_anchor_points is None:
        return float("inf")

    errors = np.linalg.norm(corrected_anchor_points - neutral_anchor_points, axis=1)
    return float(np.mean(errors))


# -----------------------------------------------------------------------------
# Motion display helpers
# -----------------------------------------------------------------------------


def draw_displacement_arrows(frame, neutral_points, corrected_points, threshold_px):
    """
    Draw yellow arrows from neutral cheek points to corrected current cheek
    points and return per-point displacement magnitudes.
    """
    deltas = corrected_points - neutral_points
    magnitudes = np.linalg.norm(deltas, axis=1)

    for neutral_pt, corrected_pt, magnitude in zip(neutral_points, corrected_points, magnitudes):
        if magnitude < threshold_px:
            continue

        start = tuple(np.round(neutral_pt).astype(int))
        end = tuple(np.round(corrected_pt).astype(int))
        cv2.arrowedLine(frame, start, end, (0, 255, 255), 1, tipLength=0.25)

    return magnitudes



def draw_driver_handles(frame, rest_points, current_points, threshold_px=1.0):
    """
    Draw the neutral and current cheek driver handles.

    Each current handle is shown with an arrow from its neutral handle position.
    """
    for index, (rest_pt, current_pt) in enumerate(zip(rest_points, current_points), start=1):
        start = tuple(np.round(rest_pt).astype(int))
        end = tuple(np.round(current_pt).astype(int))
        motion = float(np.linalg.norm(current_pt - rest_pt))

        cv2.circle(frame, start, 4, (100, 100, 100), -1)
        cv2.putText(
            frame,
            f"R{index}",
            (start[0] + 5, start[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (160, 160, 160),
            1,
        )

        cv2.circle(frame, end, 4, (255, 255, 0), -1)
        cv2.putText(
            frame,
            f"D{index}",
            (end[0] + 5, end[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
        )

        if motion >= threshold_px:
            cv2.arrowedLine(frame, start, end, (255, 255, 0), 1, tipLength=0.25)


# -----------------------------------------------------------------------------
# Patch / driver construction
# -----------------------------------------------------------------------------


def build_local_patch_edges(landmark_ids, connections):
    """
    Build local cheek-patch edges from the MediaPipe global tesselation.

    The returned edges index into the local cheek patch vertex array.
    """
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



def build_driver_groups_from_neutral(neutral_points, num_groups=3):
    """
    Split the neutral cheek patch into simple vertical cheek regions.

    For the current prototype this yields a minimal, stable set of local cheek
    drivers: upper / middle / lower.
    """
    sorted_indices = np.argsort(neutral_points[:, 1])
    chunks = np.array_split(sorted_indices, num_groups)

    groups = []
    for chunk in chunks:
        if len(chunk) > 0:
            groups.append(np.array(chunk, dtype=np.int32))

    return groups



def compute_driver_rest_points(neutral_patch_points, driver_groups):
    """Compute neutral handle positions from the neutral cheek patch."""
    centers = []
    for group in driver_groups:
        centers.append(np.mean(neutral_patch_points[group], axis=0))
    return np.array(centers, dtype=np.float32)



def compute_driver_current_points(corrected_points, driver_groups):
    """Compute current handle positions from corrected cheek points."""
    centers = []
    for group in driver_groups:
        centers.append(np.mean(corrected_points[group], axis=0))
    return np.array(centers, dtype=np.float32)



def compute_driver_offsets(driver_rest_points, driver_current_points):
    """
    Compute driver displacement from neutral.

    Returns
    -------
    np.ndarray
        Array of shape (K, 2) where each row is the current driver displacement
        relative to its neutral position.
    """
    return (driver_current_points - driver_rest_points).astype(np.float32)



def build_patch_weights(neutral_patch_points, driver_rest_points, power=2.0, eps=1e-6):
    """
    Build fixed per-vertex skinning weights from the neutral patch geometry.

    For this prototype, weights are created once from inverse distance to the
    neutral driver positions and then normalized so each vertex's weights sum to 1.

    Parameters
    ----------
    neutral_patch_points : np.ndarray, shape (N, 2)
        Patch vertices in neutral space.
    driver_rest_points : np.ndarray, shape (K, 2)
        Neutral driver positions.
    power : float
        Inverse-distance falloff power.
    eps : float
        Small numerical constant.

    Returns
    -------
    np.ndarray
        Weight matrix of shape (N, K).
    """
    weights = []

    for vertex in neutral_patch_points:
        dist_sq = np.sum((driver_rest_points - vertex) ** 2, axis=1)
        inv = 1.0 / np.maximum(dist_sq, eps * eps) ** (power / 2.0)
        inv /= np.sum(inv)
        weights.append(inv)

    return np.array(weights, dtype=np.float32)


# -----------------------------------------------------------------------------
# Linear skinning deformation
# -----------------------------------------------------------------------------


def linear_skinning_deform_points(neutral_patch_points, patch_weights, driver_offsets):
    """
    Deform 2D vertices using simple linear skinning.

    Parameters
    ----------
    neutral_patch_points : np.ndarray, shape (N, 2)
        Patch vertices in neutral/rest space.
    patch_weights : np.ndarray, shape (N, K)
        Fixed per-vertex weights to K drivers.
    driver_offsets : np.ndarray, shape (K, 2)
        Current driver displacement vectors.

    Returns
    -------
    np.ndarray
        Deformed patch vertices of shape (N, 2).

    Notes
    -----
    For each vertex v, this function applies:
        v' = v + sum_i( w_i * d_i )

    This is a simple translation-based skinning step. It is intentionally kept
    minimal for clarity and for easy comparison against the earlier MLS version.
    """
    blended_offsets = patch_weights @ driver_offsets
    return (neutral_patch_points + blended_offsets).astype(np.float32)



def save_training_sample(
    output_dir,
    sample_index,
    corrected_cheek_points,
    neutral_cheek_points,
    driver_rest_points,
    driver_current_points,
    driver_offsets,
    neutral_patch_points,
    patch_weights,
    skinned_patch_points,
    patch_edges,
):
    """
    Save one ML training sample as a compressed NumPy .npz file.

    Saved content
    -------------
    cheek_displacement : (N, 2)
        Corrected cheek motion relative to the neutral cheek.
    driver_rest_points : (K, 2)
        Neutral driver positions.
    driver_current_points : (K, 2)
        Current driver positions.
    driver_offsets : (K, 2)
        Current driver displacement from neutral.
    neutral_patch_points : (N, 2)
        Rest patch geometry.
    patch_weights : (N, K)
        Fixed per-vertex skinning weights.
    skinned_patch_points : (N, 2)
        Final patch output for this frame.
    patch_displacement : (N, 2)
        Final patch displacement relative to the neutral patch.
    patch_edges : (E, 2)
        Local connectivity of the cheek patch.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cheek_displacement = (corrected_cheek_points - neutral_cheek_points).astype(np.float32)
    patch_displacement = (skinned_patch_points - neutral_patch_points).astype(np.float32)

    np.savez_compressed(
        output_dir / f"sample_{sample_index:06d}.npz",
        cheek_displacement=cheek_displacement,
        driver_rest_points=driver_rest_points.astype(np.float32),
        driver_current_points=driver_current_points.astype(np.float32),
        driver_offsets=driver_offsets.astype(np.float32),
        neutral_patch_points=neutral_patch_points.astype(np.float32),
        patch_weights=patch_weights.astype(np.float32),
        skinned_patch_points=skinned_patch_points.astype(np.float32),
        patch_displacement=patch_displacement,
        patch_edges=np.array(patch_edges, dtype=np.int32),
    )


# -----------------------------------------------------------------------------
# Patch drawing
# -----------------------------------------------------------------------------


def draw_patch_from_points(frame, points, edges, color=(255, 0, 255), thickness=2):
    """Draw a cheek patch from point positions and local patch edges."""
    for a, b in edges:
        p1 = tuple(np.round(points[a]).astype(int))
        p2 = tuple(np.round(points[b]).astype(int))
        cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)

    for pt in points:
        x, y = np.round(pt).astype(int)
        cv2.circle(frame, (x, y), 2, color, -1)


# -----------------------------------------------------------------------------
# Main application
# -----------------------------------------------------------------------------


def main():
    """Run the real-time left-cheek tracking and linear skinning demo."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    # Neutral tracking references.
    neutral_cheek_points = None
    neutral_anchor_points = None

    # Separate cheek patch and driver state.
    neutral_patch_points = None
    driver_groups = None
    driver_rest_points = None
    patch_weights = None

    patch_edges = build_local_patch_edges(
        LEFT_CHEEK_IDS,
        FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
    )

    capture_requested = False
    cheek_capture_buffer = []
    anchor_capture_buffer = []

    # Dataset recording state for future ML training.
    record_dataset = False
    valid_frame_index = 0
    saved_sample_count = 0

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

                current_cheek_points = get_landmark_points_px(
                    face_landmarks,
                    LEFT_CHEEK_IDS,
                    width,
                    height,
                )
                current_anchor_points = get_landmark_points_px(
                    face_landmarks,
                    HEAD_ANCHOR_IDS,
                    width,
                    height,
                )

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

                # -------------------------------------------------------------
                # Neutral capture
                # -------------------------------------------------------------
                if capture_requested:
                    cheek_capture_buffer.append(current_cheek_points.copy())
                    anchor_capture_buffer.append(current_anchor_points.copy())

                    cv2.putText(
                        frame,
                        f"Capturing neutral {len(cheek_capture_buffer)}/{NEUTRAL_CAPTURE_FRAMES}",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                    if len(cheek_capture_buffer) >= NEUTRAL_CAPTURE_FRAMES:
                        neutral_cheek_points = np.mean(cheek_capture_buffer, axis=0).astype(np.float32)
                        neutral_anchor_points = np.mean(anchor_capture_buffer, axis=0).astype(np.float32)

                        # The separate cheek patch is initialized from the neutral
                        # cheek geometry, and is later deformed by linear skinning.
                        neutral_patch_points = neutral_cheek_points.copy()

                        driver_groups = build_driver_groups_from_neutral(
                            neutral_patch_points,
                            num_groups=DRIVER_REGION_COUNT,
                        )
                        driver_rest_points = compute_driver_rest_points(
                            neutral_patch_points,
                            driver_groups,
                        )
                        patch_weights = build_patch_weights(
                            neutral_patch_points,
                            driver_rest_points,
                            power=SKINNING_WEIGHT_POWER,
                            eps=SKINNING_EPSILON,
                        )

                        cheek_capture_buffer.clear()
                        anchor_capture_buffer.clear()
                        capture_requested = False

                # -------------------------------------------------------------
                # Head-motion compensated cheek analysis + linear skinning
                # -------------------------------------------------------------
                if neutral_cheek_points is not None and neutral_anchor_points is not None:
                    transform = estimate_head_motion_transform(
                        current_anchor_points,
                        neutral_anchor_points,
                    )

                    if transform is None:
                        cv2.putText(
                            frame,
                            "Alignment failed",
                            (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
                        )
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
                            (255, 255, 255),
                            2,
                        )

                        if anchor_alignment_error > ANCHOR_ALIGNMENT_ERROR_THRESHOLD_PX:
                            cv2.putText(
                                frame,
                                "Pose too large for cheek comparison",
                                (20, 130),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 0, 255),
                                2,
                            )
                            cv2.putText(
                                frame,
                                "Cheek analysis and skinned patch suppressed",
                                (20, 160),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                (180, 180, 180),
                                2,
                            )
                        else:
                            corrected_cheek_points = apply_affine_to_points(
                                current_cheek_points,
                                transform,
                            )

                            if corrected_cheek_points is not None:
                                magnitudes = np.linalg.norm(
                                    corrected_cheek_points - neutral_cheek_points,
                                    axis=1,
                                )
                                mean_disp = float(np.mean(magnitudes))
                                max_disp = float(np.max(magnitudes))

                                if mean_disp >= REGION_ACTIVITY_THRESHOLD_PX:
                                    draw_displacement_arrows(
                                        frame,
                                        neutral_cheek_points,
                                        corrected_cheek_points,
                                        ARROW_THRESHOLD_PX,
                                    )
                                    status_text = "Cheek activity detected"
                                    status_color = (0, 255, 255)
                                else:
                                    status_text = "No significant local cheek activity"
                                    status_color = (180, 180, 180)

                                cv2.putText(
                                    frame,
                                    f"Mean disp: {mean_disp:.2f}px",
                                    (20, 130),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (255, 255, 255),
                                    2,
                                )
                                cv2.putText(
                                    frame,
                                    f"Max disp: {max_disp:.2f}px",
                                    (20, 160),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (255, 255, 255),
                                    2,
                                )
                                cv2.putText(
                                    frame,
                                    status_text,
                                    (20, 190),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    status_color,
                                    2,
                                )
                                cv2.putText(
                                    frame,
                                    "Head motion compensated",
                                    (20, 220),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,
                                    (0, 255, 0),
                                    2,
                                )

                                if (
                                    SHOW_SKINNED_PATCH
                                    and neutral_patch_points is not None
                                    and driver_groups is not None
                                    and driver_rest_points is not None
                                    and patch_weights is not None
                                ):
                                    driver_current_points = compute_driver_current_points(
                                        corrected_cheek_points,
                                        driver_groups,
                                    )
                                    driver_offsets = compute_driver_offsets(
                                        driver_rest_points,
                                        driver_current_points,
                                    )

                                    skinned_patch_points = linear_skinning_deform_points(
                                        neutral_patch_points,
                                        patch_weights,
                                        driver_offsets,
                                    )

                                    valid_frame_index += 1

                                    if (
                                        record_dataset
                                        and valid_frame_index % SAVE_EVERY_N_VALID_FRAMES == 0
                                    ):
                                        save_training_sample(
                                            DATASET_OUTPUT_DIR,
                                            saved_sample_count,
                                            corrected_cheek_points,
                                            neutral_cheek_points,
                                            driver_rest_points,
                                            driver_current_points,
                                            driver_offsets,
                                            neutral_patch_points,
                                            patch_weights,
                                            skinned_patch_points,
                                            patch_edges,
                                        )
                                        saved_sample_count += 1

                                    draw_patch_from_points(
                                        frame,
                                        skinned_patch_points,
                                        patch_edges,
                                        color=(255, 0, 255),
                                        thickness=2,
                                    )

                                    if SHOW_DRIVER_HANDLES:
                                        draw_driver_handles(
                                            frame,
                                            driver_rest_points,
                                            driver_current_points,
                                            threshold_px=DRIVER_ARROW_THRESHOLD_PX,
                                        )

                cv2.putText(
                    frame,
                    "Left Cheek Motion + Linear Skinning Patch",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

            recording_status = "ON" if record_dataset else "OFF"
            cv2.putText(
                frame,
                f"Recording: {recording_status} | Saved: {saved_sample_count}",
                (20, height - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                frame,
                "N = capture neutral | C = clear neutral | R = record dataset | Q / ESC = quit",
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            cv2.imshow("Face Landmarks", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("n"):
                if current_cheek_points is not None and current_anchor_points is not None:
                    capture_requested = True
                    cheek_capture_buffer.clear()
                    anchor_capture_buffer.clear()

            elif key == ord("c"):
                neutral_cheek_points = None
                neutral_anchor_points = None
                neutral_patch_points = None
                driver_groups = None
                driver_rest_points = None
                patch_weights = None
                capture_requested = False
                cheek_capture_buffer.clear()
                anchor_capture_buffer.clear()

            elif key == ord("r"):
                record_dataset = not record_dataset

            elif key == 27 or key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
