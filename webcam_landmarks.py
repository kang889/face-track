import cv2
import time
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections

MODEL_PATH = "models/face_landmarker.task"

# Left cheek only
LEFT_CHEEK_IDS = [
    36, 50, 101, 111, 118, 117, 116, 123, 147,
    192, 216, 206, 207, 205, 203, 212, 214, 187
]

# Relatively stable landmarks used to compensate for whole-head movement
# (eye corners / nose bridge area)
HEAD_ANCHOR_IDS = [33, 133, 362, 263, 6, 168]

NEUTRAL_CAPTURE_FRAMES = 20
ARROW_THRESHOLD_PX = 3.0
REGION_ACTIVITY_THRESHOLD_PX = 4.0
ANCHOR_ALIGNMENT_ERROR_THRESHOLD_PX = 8.0
SHOW_ANCHORS = False

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
)

def get_landmark_points_px(face_landmarks, ids, w, h):
    points = []
    for idx in ids:
        lm = face_landmarks[idx]
        points.append([lm.x * w, lm.y * h])
    return np.array(points, dtype=np.float32)

def draw_all_landmarks(frame, face_landmarks, w, h):
    for lm in face_landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(frame, (x, y), 1, (0, 100, 0), -1)

def draw_cheek_mesh(frame, face_landmarks, cheek_ids, connections, w, h,
                    color=(180, 0, 0), thickness=1):
    for conn in connections:
        a = conn.start
        b = conn.end

        if a in cheek_ids and b in cheek_ids:
            lm1 = face_landmarks[a]
            lm2 = face_landmarks[b]

            x1, y1 = int(lm1.x * w), int(lm1.y * h)
            x2, y2 = int(lm2.x * w), int(lm2.y * h)

            cv2.line(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    for idx in cheek_ids:
        lm = face_landmarks[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

def draw_anchor_points(frame, anchor_points):
    for pt in anchor_points:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)

def estimate_head_motion_transform(current_anchor_points, neutral_anchor_points):
    if current_anchor_points is None or neutral_anchor_points is None:
        return None

    if len(current_anchor_points) < 3 or len(neutral_anchor_points) < 3:
        return None

    transform, _ = cv2.estimateAffinePartial2D(
        current_anchor_points,
        neutral_anchor_points,
        method=cv2.LMEDS
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

    errors = np.linalg.norm(
        corrected_anchor_points - neutral_anchor_points,
        axis=1
    )
    return float(np.mean(errors))

def draw_displacement_arrows(frame, neutral_points, corrected_points, threshold_px):
    deltas = corrected_points - neutral_points
    magnitudes = np.linalg.norm(deltas, axis=1)

    for neutral_pt, corrected_pt, mag in zip(neutral_points, corrected_points, magnitudes):
        if mag < threshold_px:
            continue

        start = tuple(np.round(neutral_pt).astype(int))
        end = tuple(np.round(corrected_pt).astype(int))
        cv2.arrowedLine(frame, start, end, (0, 255, 255), 1, tipLength=0.25)

    return magnitudes

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

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

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        h, w, _ = frame.shape
        current_cheek_points = None
        current_anchor_points = None

        if result.face_landmarks:
            face_landmarks = result.face_landmarks[0]

            draw_all_landmarks(frame, face_landmarks, w, h)

            current_cheek_points = get_landmark_points_px(
                face_landmarks, LEFT_CHEEK_IDS, w, h
            )
            current_anchor_points = get_landmark_points_px(
                face_landmarks, HEAD_ANCHOR_IDS, w, h
            )

            draw_cheek_mesh(
                frame,
                face_landmarks,
                set(LEFT_CHEEK_IDS),
                FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                w,
                h,
                color=(180, 0, 0),
                thickness=1
            )

            if SHOW_ANCHORS:
                draw_anchor_points(frame, current_anchor_points)

            # Capture neutral cheek + neutral anchors
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
                    2
                )

                if len(cheek_capture_buffer) >= NEUTRAL_CAPTURE_FRAMES:
                    neutral_cheek_points = np.mean(cheek_capture_buffer, axis=0).astype(np.float32)
                    neutral_anchor_points = np.mean(anchor_capture_buffer, axis=0).astype(np.float32)

                    cheek_capture_buffer.clear()
                    anchor_capture_buffer.clear()
                    capture_requested = False

            # Head-motion compensated cheek displacement + pose validity gate
            if neutral_cheek_points is not None and neutral_anchor_points is not None:
                transform = estimate_head_motion_transform(
                    current_anchor_points,
                    neutral_anchor_points
                )

                if transform is None:
                    cv2.putText(
                        frame,
                        "Alignment failed",
                        (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )
                else:
                    anchor_alignment_error = compute_mean_alignment_error(
                        current_anchor_points,
                        neutral_anchor_points,
                        transform
                    )

                    cv2.putText(
                        frame,
                        f"Anchor error: {anchor_alignment_error:.2f}px",
                        (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

                    if anchor_alignment_error > ANCHOR_ALIGNMENT_ERROR_THRESHOLD_PX:
                        cv2.putText(
                            frame,
                            "Pose too large for cheek comparison",
                            (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2
                        )

                        cv2.putText(
                            frame,
                            "Cheek arrows suppressed",
                            (20, 160),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (180, 180, 180),
                            2
                        )
                    else:
                        corrected_cheek_points = apply_affine_to_points(
                            current_cheek_points,
                            transform
                        )

                        if corrected_cheek_points is not None:
                            magnitudes = np.linalg.norm(
                                corrected_cheek_points - neutral_cheek_points,
                                axis=1
                            )

                            mean_disp = float(np.mean(magnitudes))
                            max_disp = float(np.max(magnitudes))

                            if mean_disp >= REGION_ACTIVITY_THRESHOLD_PX:
                                draw_displacement_arrows(
                                    frame,
                                    neutral_cheek_points,
                                    corrected_cheek_points,
                                    ARROW_THRESHOLD_PX
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
                                2
                            )

                            cv2.putText(
                                frame,
                                f"Max disp: {max_disp:.2f}px",
                                (20, 160),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 255),
                                2
                            )

                            cv2.putText(
                                frame,
                                status_text,
                                (20, 190),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                status_color,
                                2
                            )

                            cv2.putText(
                                frame,
                                "Head motion compensated",
                                (20, 220),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                (0, 255, 0),
                                2
                            )

            cv2.putText(
                frame,
                "Left Cheek Motion",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        cv2.putText(
            frame,
            "N = capture neutral | C = clear neutral | Q / ESC = quit",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
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
            capture_requested = False
            cheek_capture_buffer.clear()
            anchor_capture_buffer.clear()

        elif key == 27 or key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()