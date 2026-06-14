import time
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
TRAINED_MODEL_PATH = Path("one_point_neighbors_model.pt")

# Current left-cheek landmark subset.
LEFT_CHEEK_IDS = [
    36, 50, 101, 111, 116, 117, 118, 123, 135, 137, 138, 147, 177,
    187, 192, 203, 205, 206, 207, 212, 213, 214, 215, 216, 227,
]

# Local source point and its local neighbor indices.
# SOURCE_POINT_INDEX = 13 means LEFT_CHEEK_IDS[13] == 187.
SOURCE_POINT_INDEX = 13
NEIGHBOR_INDICES = [1, 7, 11, 14, 16, 18, 20, 21]

# Relatively stable landmarks used only for head-motion compensation.
HEAD_ANCHOR_IDS = [33, 133, 362, 263, 6, 168]

# Neutral capture.
NEUTRAL_CAPTURE_FRAMES = 20

# Motion-analysis and gating thresholds.
ARROW_THRESHOLD_PX = 3.0
ANCHOR_ALIGNMENT_ERROR_THRESHOLD_PX = 8.0
SOURCE_ACTIVITY_THRESHOLD_PX = 2.0

# Display toggles.
SHOW_ALL_LANDMARKS = True
SHOW_LANDMARK_IDS = False
SHOW_ONLY_LEFT_CHEEK_IDS = False
SHOW_ANCHORS = False
SHOW_CHEEK_MESH = True

# Drawing colors (BGR)
COLOR_SOURCE = (0, 255, 255)          # yellow
COLOR_NEIGHBOR = (0, 200, 0)          # green
COLOR_PREDICTED = (255, 0, 255)       # magenta
COLOR_ACTUAL = (0, 255, 0)            # bright green
COLOR_NEUTRAL = (120, 120, 120)       # gray
COLOR_TEXT = (255, 255, 255)
COLOR_ERROR = (0, 0, 255)


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



def draw_tracked_cheek_mesh(
    frame,
    face_landmarks,
    cheek_ids,
    connections,
    width,
    height,
    color=(180, 0, 0),
    thickness=1,
):
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
    """
    Estimate a partial affine transform mapping current anchors -> neutral anchors.
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
    """Measure mean post-alignment anchor error in pixels."""
    corrected_anchor_points = apply_affine_to_points(current_anchor_points, transform)
    if corrected_anchor_points is None:
        return float("inf")

    errors = np.linalg.norm(corrected_anchor_points - neutral_anchor_points, axis=1)
    return float(np.mean(errors))


# -----------------------------------------------------------------------------
# Drawing helpers for source / neighbors / predictions
# -----------------------------------------------------------------------------


def draw_arrow(frame, start_pt, end_pt, color, thickness=2, threshold_px=0.0):
    """Draw an arrow only if the motion is large enough."""
    motion = float(np.linalg.norm(end_pt - start_pt))
    if motion < threshold_px:
        return

    start = tuple(np.round(start_pt).astype(int))
    end = tuple(np.round(end_pt).astype(int))
    cv2.arrowedLine(frame, start, end, color, thickness, tipLength=0.22)



def draw_source_and_neighbors(
    frame,
    neutral_points,
    corrected_points,
    predicted_neighbor_points,
    source_index,
    neighbor_indices,
):
    """
    Draw source point, actual neighbor motion, and predicted neighbor motion.

    - Source point: yellow arrow from neutral to corrected.
    - Actual neighbors: green arrows from neutral to corrected.
    - Predicted neighbors: magenta arrows from neutral to predicted.
    """
    source_neutral = neutral_points[source_index]
    source_current = corrected_points[source_index]

    # Source point marker and arrow.
    cv2.circle(frame, tuple(np.round(source_neutral).astype(int)), 5, COLOR_NEUTRAL, -1)
    cv2.circle(frame, tuple(np.round(source_current).astype(int)), 6, COLOR_SOURCE, -1)
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

    for list_idx, neighbor_index in enumerate(neighbor_indices):
        neutral_pt = neutral_points[neighbor_index]
        actual_pt = corrected_points[neighbor_index]
        predicted_pt = predicted_neighbor_points[list_idx]

        # Base neutral marker.
        cv2.circle(frame, tuple(np.round(neutral_pt).astype(int)), 4, COLOR_NEUTRAL, -1)

        # Actual tracked neighbor movement.
        cv2.circle(frame, tuple(np.round(actual_pt).astype(int)), 4, COLOR_NEIGHBOR, -1)
        draw_arrow(frame, neutral_pt, actual_pt, COLOR_ACTUAL, thickness=2, threshold_px=ARROW_THRESHOLD_PX)

        # Model-predicted neighbor movement.
        cv2.circle(frame, tuple(np.round(predicted_pt).astype(int)), 3, COLOR_PREDICTED, -1)
        draw_arrow(frame, neutral_pt, predicted_pt, COLOR_PREDICTED, thickness=1, threshold_px=ARROW_THRESHOLD_PX)

        cv2.putText(
            frame,
            f"N:{neighbor_index}",
            tuple((np.round(actual_pt).astype(int) + np.array([5, -5])).tolist()),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            COLOR_NEIGHBOR,
            1,
            cv2.LINE_AA,
        )



def draw_side_panel(frame, cheek_disp, source_index, neighbor_indices, predicted_neighbor_displacements):
    """Draw a smaller text summary on the right side of the frame."""
    h, w, _ = frame.shape
    panel_width = 220
    panel_x0 = max(w - panel_width, 0)
    cv2.rectangle(frame, (panel_x0, 0), (w - 1, h - 1), (30, 30, 30), -1)
    cv2.line(frame, (panel_x0, 0), (panel_x0, h - 1), (90, 90, 90), 1)

    y = 20
    line_h = 15
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame, "Neighbor Viewer", (panel_x0 + 8, y), font, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)
    y += 20

    source_vec = cheek_disp[source_index]
    source_mag = float(np.linalg.norm(source_vec))
    cv2.putText(frame, f"Src {source_index}  mag {source_mag:4.1f}", (panel_x0 + 8, y), font, 0.38, COLOR_SOURCE, 1, cv2.LINE_AA)
    y += line_h
    cv2.putText(frame, f"dx {source_vec[0]:5.1f}  dy {source_vec[1]:5.1f}", (panel_x0 + 8, y), font, 0.36, COLOR_TEXT, 1, cv2.LINE_AA)
    y += 18

    cv2.putText(frame, "N   act(dx,dy)   pred(dx,dy)  err", (panel_x0 + 8, y), font, 0.28, (200, 200, 200), 1, cv2.LINE_AA)
    y += 14

    for idx, neighbor_index in enumerate(neighbor_indices):
        actual_vec = cheek_disp[neighbor_index]
        pred_vec = predicted_neighbor_displacements[idx]
        err = float(np.linalg.norm(actual_vec - pred_vec))

        line = (
            f"{neighbor_index:>2d} "
            f"({actual_vec[0]:4.0f},{actual_vec[1]:4.0f}) "
            f"({pred_vec[0]:4.0f},{pred_vec[1]:4.0f}) "
            f"{err:4.1f}"
        )
        color = COLOR_ERROR if err > 3.0 else COLOR_TEXT
        cv2.putText(frame, line, (panel_x0 + 8, y), font, 0.28, color, 1, cv2.LINE_AA)
        y += 13


# -----------------------------------------------------------------------------
# Model utilities
# -----------------------------------------------------------------------------


def load_trained_model(model_path: Path, device: str):
    """Load trained one-point-neighbors model checkpoint."""
    if not model_path.exists():
        raise FileNotFoundError(f"Could not find trained model: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    input_dim = int(checkpoint["input_dim"])
    output_dim = int(checkpoint["output_dim"])

    model = NeighborMLP(input_dim, output_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    trained_source = checkpoint.get("source_point_index", None)
    trained_neighbors = checkpoint.get("neighbor_indices", None)

    return model, trained_source, trained_neighbors


# -----------------------------------------------------------------------------
# Main application
# -----------------------------------------------------------------------------


def main():
    cap = cv2.VideoCapture(0) # tells opencv to connect to the default webcam, cap stores the image data sort of like a ifstream logic
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    device = "cuda" if torch.cuda.is_available() else "cpu" # checks if pytorch can use GPU if not uses cpu.
    model, trained_source, trained_neighbors = load_trained_model(TRAINED_MODEL_PATH, device) # works like a tuple, the three viable will store the 3 different elements returned from the load_trained_model

    if trained_source is not None and trained_source != SOURCE_POINT_INDEX:
        raise ValueError(
            f"Checkpoint was trained for source point {trained_source}, "
            f"but script SOURCE_POINT_INDEX={SOURCE_POINT_INDEX}."
        )

    if trained_neighbors is not None and list(trained_neighbors) != NEIGHBOR_INDICES: #list() is for safety, to ensure the contianer is a list
        raise ValueError(
            f"Checkpoint was trained for neighbors {trained_neighbors}, "
            f"but script NEIGHBOR_INDICES={NEIGHBOR_INDICES}."
        )

    # Local display state.
    show_landmark_ids = SHOW_LANDMARK_IDS
    show_only_left_cheek_ids = SHOW_ONLY_LEFT_CHEEK_IDS

    # Neutral tracking references.
    neutral_cheek_points = None
    neutral_anchor_points = None

    capture_requested = False
    cheek_capture_buffer = []
    anchor_capture_buffer = []

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #color fix, opencv reads images in BGR, but mediapipe reads it in RGB, cvtcolor rearranges those bytes in memory
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms) #passes the image buffer from the webcam into the mediapipe model

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
                        COLOR_TEXT,
                        2,
                    )

                    if len(cheek_capture_buffer) >= NEUTRAL_CAPTURE_FRAMES:
                        neutral_cheek_points = np.mean(cheek_capture_buffer, axis=0).astype(np.float32)
                        neutral_anchor_points = np.mean(anchor_capture_buffer, axis=0).astype(np.float32)

                        cheek_capture_buffer.clear()
                        anchor_capture_buffer.clear()
                        capture_requested = False

                # -------------------------------------------------------------
                # Head-motion compensated cheek analysis + neighbor model
                # -------------------------------------------------------------
                # It calculates a 2D Affine Transformation Matrix (transform) using stable points like your eyes. It applies that matrix to "cancel out" your head rotation.

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
                                "Pose too large for neighbor comparison",
                                (20, 130),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                COLOR_ERROR,
                                2,
                            )
                        else:
                            corrected_cheek_points = apply_affine_to_points(current_cheek_points, transform)

                            if corrected_cheek_points is not None:
                                cheek_disp = corrected_cheek_points - neutral_cheek_points # subtract the neutral position from the current position, yields the pure displacement vector (how far the muscles actually stretch)
                                source_vec = cheek_disp[SOURCE_POINT_INDEX]
                                source_mag = float(np.linalg.norm(source_vec))

                                cv2.putText(
                                    frame,
                                    f"Source point {SOURCE_POINT_INDEX} | mag {source_mag:.2f}px",
                                    (20, 130),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    COLOR_TEXT,
                                    2,
                                )

                                # Model prediction from current source point movement.
                                if source_mag >= SOURCE_ACTIVITY_THRESHOLD_PX:
                                    # this turns of pytorch training memory for maximum speed
                                    with torch.no_grad():
                                        x = torch.from_numpy(source_vec.astype(np.float32).reshape(1, 2)).to(device) #converts standard cpu memmory into a pytorch tensorflow, .to(device): Ships that tensor across the PCIe bus into GPU memory.
                                        pred = model(x).cpu().numpy().reshape(len(NEIGHBOR_INDICES), 2).astype(np.float32)

                                    neutral_neighbor_points = neutral_cheek_points[NEIGHBOR_INDICES]
                                    predicted_neighbor_points = neutral_neighbor_points + pred

                                    draw_source_and_neighbors(
                                        frame,
                                        neutral_cheek_points,
                                        corrected_cheek_points,
                                        predicted_neighbor_points,
                                        SOURCE_POINT_INDEX,
                                        NEIGHBOR_INDICES,
                                    )

                                    draw_side_panel(
                                        frame,
                                        cheek_disp,
                                        SOURCE_POINT_INDEX,
                                        NEIGHBOR_INDICES,
                                        pred,
                                    )
                                else:
                                    cv2.putText(
                                        frame,
                                        "Source movement too small for prediction",
                                        (20, 160),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.55,
                                        (180, 180, 180),
                                        2,
                                    )

                cv2.putText(
                    frame,
                    "One-Point Neighbor Model Viewer",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    COLOR_TEXT,
                    2,
                )

            cv2.putText(
                frame,
                "N = capture neutral | C = clear neutral | I = show IDs | O = cheek-only IDs | Q / ESC = quit",
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLOR_TEXT,
                1,
            )

            cv2.imshow("Neighbor Model Viewer", frame) #like swapbuffers in openfl, takes fully drawn array and pushes it to the os windoow

            key = cv2.waitKey(1) & 0xFF #Pauses the loop for exactly 1 millisecond to poll the OS for keyboard events. A bitwise mask. It strips out extra NumLock/Capslock modifier bits so an "n" is always recognized as an "n", regardless of the operating system.

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

            elif key == 27 or key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
