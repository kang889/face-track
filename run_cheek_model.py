import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from torch import nn
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections

"""
run_cheek_model.py

Runtime inference script for the left-cheek project.

What this script does
---------------------
- Uses the same left-cheek tracking + neutral capture + anchor compensation
  pipeline as the current linear-skinning version.
- Loads a trained PyTorch model from cheek_mlp.pt.
- Replaces the hand-written linear-skinning deformation step with model
  prediction.
- Draws the predicted cheek patch in purple.

Important
---------
This script only works correctly if:
1. The trained model was produced from the SAME cheek landmark layout.
2. The training data used the same input format:
      input  = cheek_displacement + driver_offsets
      output = patch_displacement
3. The checkpoint file cheek_mlp.pt is present.
"""

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MODEL_PATH = "models/face_landmarker.task"
CHECKPOINT_PATH = Path("cheek_mlp.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SHOW_LANDMARK_IDS = False
SHOW_ONLY_LEFT_CHEEK_IDS = False

# Must match the landmark layout used to record/train the current dataset.
LEFT_CHEEK_IDS = [
    36, 50, 101, 111, 116, 117, 118, 123, 135, 137, 138, 147, 177,
    187, 192, 203, 205, 206, 207, 212, 213, 214, 215, 216, 227,
]

HEAD_ANCHOR_IDS = [33, 133, 362, 263, 6, 168]

NEUTRAL_CAPTURE_FRAMES = 20
DRIVER_REGION_COUNT = 3

ARROW_THRESHOLD_PX = 3.0
REGION_ACTIVITY_THRESHOLD_PX = 4.0
ANCHOR_ALIGNMENT_ERROR_THRESHOLD_PX = 8.0
DRIVER_ARROW_THRESHOLD_PX = 1.5

SHOW_ALL_LANDMARKS = True
SHOW_ANCHORS = False
SHOW_DRIVER_HANDLES = True
SHOW_PREDICTED_PATCH = True


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
# Model definition and loading
# -----------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_trained_model(checkpoint_path: Path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Could not find model checkpoint: {checkpoint_path.resolve()}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    input_dim = int(checkpoint["input_dim"])
    output_dim = int(checkpoint["output_dim"])

    model = MLP(input_dim, output_dim).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, input_dim, output_dim


# -----------------------------------------------------------------------------
# Landmark utilities
# -----------------------------------------------------------------------------


def get_landmark_points_px(face_landmarks, ids, width, height):
    points = []
    for idx in ids:
        lm = face_landmarks[idx]
        points.append([lm.x * width, lm.y * height])
    return np.array(points, dtype=np.float32)



def draw_all_landmarks(frame, face_landmarks, width, height):
    for lm in face_landmarks:
        x = int(lm.x * width)
        y = int(lm.y * height)
        cv2.circle(frame, (x, y), 1, (0, 100, 0), -1)



def draw_landmark_ids(frame, face_landmarks, width, height, only_ids=None):
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
    for pt in anchor_points:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)


# -----------------------------------------------------------------------------
# Head-motion compensation
# -----------------------------------------------------------------------------


def estimate_head_motion_transform(current_anchor_points, neutral_anchor_points):
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
    if transform is None:
        return None

    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    homogeneous = np.hstack([points, ones])
    transformed = (transform @ homogeneous.T).T
    return transformed.astype(np.float32)



def compute_mean_alignment_error(current_anchor_points, neutral_anchor_points, transform):
    corrected_anchor_points = apply_affine_to_points(current_anchor_points, transform)
    if corrected_anchor_points is None:
        return float("inf")

    errors = np.linalg.norm(corrected_anchor_points - neutral_anchor_points, axis=1)
    return float(np.mean(errors))


# -----------------------------------------------------------------------------
# Motion display helpers
# -----------------------------------------------------------------------------


def draw_displacement_arrows(frame, neutral_points, corrected_points, threshold_px):
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
    for index, (rest_pt, current_pt) in enumerate(zip(rest_points, current_points), start=1):
        start = tuple(np.round(rest_pt).astype(int))
        end = tuple(np.round(current_pt).astype(int))
        motion = float(np.linalg.norm(current_pt - rest_pt))

        cv2.circle(frame, start, 4, (100, 100, 100), -1)
        cv2.putText(frame, f"R{index}", (start[0] + 5, start[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

        cv2.circle(frame, end, 4, (255, 255, 0), -1)
        cv2.putText(frame, f"D{index}", (end[0] + 5, end[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        if motion >= threshold_px:
            cv2.arrowedLine(frame, start, end, (255, 255, 0), 1, tipLength=0.25)


# -----------------------------------------------------------------------------
# Patch / driver construction
# -----------------------------------------------------------------------------


def build_local_patch_edges(landmark_ids, connections):
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
    sorted_indices = np.argsort(neutral_points[:, 1])
    chunks = np.array_split(sorted_indices, num_groups)

    groups = []
    for chunk in chunks:
        if len(chunk) > 0:
            groups.append(np.array(chunk, dtype=np.int32))

    return groups



def compute_driver_rest_points(neutral_patch_points, driver_groups):
    centers = []
    for group in driver_groups:
        centers.append(np.mean(neutral_patch_points[group], axis=0))
    return np.array(centers, dtype=np.float32)



def compute_driver_current_points(corrected_points, driver_groups):
    centers = []
    for group in driver_groups:
        centers.append(np.mean(corrected_points[group], axis=0))
    return np.array(centers, dtype=np.float32)



def compute_driver_offsets(driver_rest_points, driver_current_points):
    return (driver_current_points - driver_rest_points).astype(np.float32)


# -----------------------------------------------------------------------------
# Model inference helpers
# -----------------------------------------------------------------------------


def predict_patch_points(model, input_dim, output_dim,
                         corrected_cheek_points, neutral_cheek_points,
                         driver_offsets, neutral_patch_points):
    cheek_displacement = (corrected_cheek_points - neutral_cheek_points).astype(np.float32)

    x = np.concatenate([
        cheek_displacement.reshape(-1),
        driver_offsets.reshape(-1),
    ], axis=0).astype(np.float32)

    if x.size != input_dim:
        raise RuntimeError(
            f"Model input mismatch. Expected {input_dim} values, got {x.size}. "
            "This usually means the model was trained with a different landmark layout."
        )

    with torch.no_grad():
        x_tensor = torch.from_numpy(x).unsqueeze(0).to(DEVICE)
        pred = model(x_tensor).squeeze(0).cpu().numpy().astype(np.float32)

    expected_output_size = neutral_patch_points.size
    if output_dim != expected_output_size:
        raise RuntimeError(
            f"Model output mismatch. Checkpoint output_dim={output_dim}, but the current "
            f"patch needs {expected_output_size} values."
        )

    pred_patch_displacement = pred.reshape(neutral_patch_points.shape)
    pred_patch_points = neutral_patch_points + pred_patch_displacement
    return pred_patch_points


# -----------------------------------------------------------------------------
# Patch drawing
# -----------------------------------------------------------------------------


def draw_patch_from_points(frame, points, edges, color=(255, 0, 255), thickness=2):
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
    global SHOW_LANDMARK_IDS, SHOW_ONLY_LEFT_CHEEK_IDS

    model, input_dim, output_dim = load_trained_model(CHECKPOINT_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    neutral_cheek_points = None
    neutral_anchor_points = None

    neutral_patch_points = None
    driver_groups = None
    driver_rest_points = None

    patch_edges = build_local_patch_edges(
        LEFT_CHEEK_IDS,
        FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
    )

    capture_requested = False
    cheek_capture_buffer = []
    anchor_capture_buffer = []

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

                if SHOW_LANDMARK_IDS:
                    if SHOW_ONLY_LEFT_CHEEK_IDS:
                        draw_landmark_ids(frame, face_landmarks, width, height,
                                          only_ids=set(LEFT_CHEEK_IDS))
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
                        driver_groups = build_driver_groups_from_neutral(
                            neutral_patch_points,
                            num_groups=DRIVER_REGION_COUNT,
                        )
                        driver_rest_points = compute_driver_rest_points(
                            neutral_patch_points,
                            driver_groups,
                        )

                        cheek_capture_buffer.clear()
                        anchor_capture_buffer.clear()
                        capture_requested = False

                if neutral_cheek_points is not None and neutral_anchor_points is not None:
                    transform = estimate_head_motion_transform(
                        current_anchor_points,
                        neutral_anchor_points,
                    )

                    if transform is None:
                        cv2.putText(frame, "Alignment failed", (20, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
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
                                "Cheek analysis and model patch suppressed",
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

                                cv2.putText(frame, f"Mean disp: {mean_disp:.2f}px", (20, 130),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                cv2.putText(frame, f"Max disp: {max_disp:.2f}px", (20, 160),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                cv2.putText(frame, status_text, (20, 190),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                                cv2.putText(frame, "Head motion compensated", (20, 220),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

                                if (
                                    SHOW_PREDICTED_PATCH
                                    and neutral_patch_points is not None
                                    and driver_groups is not None
                                    and driver_rest_points is not None
                                ):
                                    driver_current_points = compute_driver_current_points(
                                        corrected_cheek_points,
                                        driver_groups,
                                    )
                                    driver_offsets = compute_driver_offsets(
                                        driver_rest_points,
                                        driver_current_points,
                                    )

                                    predicted_patch_points = predict_patch_points(
                                        model,
                                        input_dim,
                                        output_dim,
                                        corrected_cheek_points,
                                        neutral_cheek_points,
                                        driver_offsets,
                                        neutral_patch_points,
                                    )

                                    draw_patch_from_points(
                                        frame,
                                        predicted_patch_points,
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
                    "Left Cheek Motion + Trained MLP Patch",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

            cv2.putText(
                frame,
                "N = capture neutral | C = clear neutral | I = show IDs | O = cheek-only IDs | Q / ESC = quit",
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
                driver_groups = None
                driver_rest_points = None
                capture_requested = False
                cheek_capture_buffer.clear()
                anchor_capture_buffer.clear()

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
