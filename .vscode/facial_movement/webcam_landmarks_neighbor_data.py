"""
webcam_landmarks_neighbor_data.py

Left-cheek facial tracking and raw point-motion dataset recorder.

Purpose
-------
This version removes the linear-skinning stage completely.
It is designed for the new project direction:
    study how movement at one cheek point relates to movement at nearby points.

What this program does
----------------------
- tracks only the left-cheek landmark subset
- captures a neutral cheek reference and neutral anchor reference
- compensates for global head motion using anchor alignment
- computes corrected cheek-point displacement relative to neutral
- records raw cheek-point motion and patch connectivity to .npz files

Why this version is better for the new goal
-------------------------------------------
The previous linear-skinning version saved model-generated patch deformation.
For the point-to-neighbor goal, that is not the main target anymore.
This version instead saves the real tracked cheek-point displacement:
    cheek_displacement
plus the cheek patch connectivity:
    patch_edges

So later analysis can answer questions like:
    "When point i moves, how do its neighboring points usually move?"

Saved .npz content
------------------
Each valid recorded sample stores:
- cheek_displacement : (N, 2)
    Corrected cheek motion relative to the neutral cheek.
- corrected_cheek_points : (N, 2)
    Current cheek points after anchor-based head-motion compensation.
- neutral_patch_points : (N, 2)
    Neutral cheek patch / neutral cheek landmark positions.
- point_motion_magnitude : (N,)
    Per-point displacement magnitude.
- patch_edges : (E, 2)
    Local connectivity of the cheek patch.
- left_cheek_ids : (N,)
    MediaPipe landmark IDs used in this patch.
- anchor_alignment_error : scalar
    Alignment quality for this sample.

This file keeps the useful geometry and tracking parts, but intentionally avoids
any driver or linear-skinning deformation logic.
"""

from __future__ import annotations

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

SHOW_LANDMARK_IDS = False
SHOW_ONLY_LEFT_CHEEK_IDS = False
MODEL_PATH = "models/face_landmarker.task"

# Left cheek landmark subset.
LEFT_CHEEK_IDS = [
    36, 50, 101, 111, 116, 117, 118, 123, 135, 137, 138, 147, 177,
    187, 192, 203, 205, 206, 207, 212, 213, 214, 215, 216, 227,
]

# Relatively stable landmarks used only for head-motion compensation.
HEAD_ANCHOR_IDS = [33, 133, 362, 263, 6, 168]

# Neutral capture.
NEUTRAL_CAPTURE_FRAMES = 20

# Motion-analysis and gating thresholds.
ARROW_THRESHOLD_PX = 3.0
REGION_ACTIVITY_THRESHOLD_PX = 4.0
ANCHOR_ALIGNMENT_ERROR_THRESHOLD_PX = 8.0

# Display toggles.
SHOW_ALL_LANDMARKS = True
SHOW_ANCHORS = False
SHOW_TRACKED_CHEEK_MESH = True
SHOW_CORRECTED_CHEEK_POINTS = True

# Dataset recording.
DATASET_OUTPUT_DIR = Path("ml_dataset_face_press")
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
    """Draw landmark index numbers on top of the detected face points."""
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
            (255, 255, 255),
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



def draw_corrected_cheek_points(frame, corrected_points, color=(255, 0, 255)):
    """Draw corrected cheek points after head-motion compensation."""
    for pt in corrected_points:
        x, y = np.round(pt).astype(int)
        cv2.circle(frame, (x, y), 2, color, -1)


# -----------------------------------------------------------------------------
# Head-motion compensation
# -----------------------------------------------------------------------------


def estimate_head_motion_transform(current_anchor_points, neutral_anchor_points):
    """
    Estimate a partial affine transform mapping current anchors to neutral anchors.
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
    """Measure the mean post-alignment anchor error in pixels."""
    corrected_anchor_points = apply_affine_to_points(current_anchor_points, transform)
    if corrected_anchor_points is None:
        return float("inf")

    errors = np.linalg.norm(corrected_anchor_points - neutral_anchor_points, axis=1)
    return float(np.mean(errors))


# -----------------------------------------------------------------------------
# Patch construction
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


# -----------------------------------------------------------------------------
# Dataset recording
# -----------------------------------------------------------------------------


def save_training_sample(
    output_dir,
    sample_index,
    corrected_cheek_points,
    neutral_cheek_points,
    neutral_patch_points,
    patch_edges,
    anchor_alignment_error,
):
    """Save one raw point-motion sample as a compressed NumPy .npz file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cheek_displacement = (corrected_cheek_points - neutral_cheek_points).astype(np.float32)
    point_motion_magnitude = np.linalg.norm(cheek_displacement, axis=1).astype(np.float32)

    np.savez_compressed(
        output_dir / f"sample_{sample_index:06d}.npz",
        cheek_displacement=cheek_displacement,
        corrected_cheek_points=corrected_cheek_points.astype(np.float32),
        neutral_patch_points=neutral_patch_points.astype(np.float32),
        point_motion_magnitude=point_motion_magnitude,
        patch_edges=np.array(patch_edges, dtype=np.int32),
        left_cheek_ids=np.array(LEFT_CHEEK_IDS, dtype=np.int32),
        anchor_alignment_error=np.array(anchor_alignment_error, dtype=np.float32),
    )


# -----------------------------------------------------------------------------
# Main application
# -----------------------------------------------------------------------------


def main():
    """Run the real-time left-cheek tracking and raw point-motion recorder."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    global SHOW_LANDMARK_IDS, SHOW_ONLY_LEFT_CHEEK_IDS

    neutral_cheek_points = None
    neutral_anchor_points = None
    neutral_patch_points = None

    patch_edges = build_local_patch_edges(
        LEFT_CHEEK_IDS,
        FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
    )

    capture_requested = False
    cheek_capture_buffer = []
    anchor_capture_buffer = []

    record_dataset = False
    valid_frame_index = 0
    saved_sample_count = 0

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #cvtcolor rearrange the bytes, because cv2 uses bgr
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

                if SHOW_LANDMARK_IDS:
                    if SHOW_ONLY_LEFT_CHEEK_IDS:
                        draw_landmark_ids(
                            frame,
                            face_landmarks,
                            width,
                            height,
                            only_ids=set(LEFT_CHEEK_IDS),
                        )
                    else:
                        draw_landmark_ids(frame, face_landmarks, width, height)

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

                if SHOW_TRACKED_CHEEK_MESH:
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
                        neutral_patch_points = neutral_cheek_points.copy()

                        cheek_capture_buffer.clear()
                        anchor_capture_buffer.clear()
                        capture_requested = False

                # -------------------------------------------------------------
                # Head-motion compensated cheek analysis + raw dataset recording
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
                                "Raw point-motion sample suppressed",
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
                                magnitudes = draw_displacement_arrows(
                                    frame,
                                    neutral_cheek_points,
                                    corrected_cheek_points,
                                    ARROW_THRESHOLD_PX,
                                )
                                mean_disp = float(np.mean(magnitudes))
                                max_disp = float(np.max(magnitudes))

                                if SHOW_CORRECTED_CHEEK_POINTS:
                                    draw_corrected_cheek_points(frame, corrected_cheek_points)

                                if mean_disp >= REGION_ACTIVITY_THRESHOLD_PX:
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

                                valid_frame_index += 1
                                if record_dataset and valid_frame_index % SAVE_EVERY_N_VALID_FRAMES == 0:
                                    save_training_sample(
                                        DATASET_OUTPUT_DIR,
                                        saved_sample_count,
                                        corrected_cheek_points,
                                        neutral_cheek_points,
                                        neutral_patch_points,
                                        patch_edges,
                                        anchor_alignment_error,
                                    )
                                    saved_sample_count += 1

                cv2.putText(
                    frame,
                    "Left Cheek Motion Recorder (Raw Point Data)",
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
                "N = capture neutral | C = clear neutral | R = record dataset | I = show IDs | O = cheek-only IDs | Q / ESC = quit",
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
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
                capture_requested = False
                cheek_capture_buffer.clear()
                anchor_capture_buffer.clear()

            elif key == ord("r"):
                record_dataset = not record_dataset

            elif key == ord("i"):
                SHOW_LANDMARK_IDS = not SHOW_LANDMARK_IDS

            elif key == ord("o"):
                SHOW_ONLY_LEFT_CHEEK_IDS = not SHOW_ONLY_LEFT_CHEEK_IDS

            elif key == 27 or key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
