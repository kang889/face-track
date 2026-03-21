import cv2
import time
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections



MODEL_PATH = "models/face_landmarker.task"

# or FACEMESH_CONTOURS if you want fewer lines

def draw_cheek_mesh(frame, face_landmarks, cheek_ids, connections, w, h,
                    color=(180, 0, 0), thickness=0.5):
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
LEFT_CHEEK_POLYGON_IDS = [
    36, 50, 101, 111, 118, 117, 116, 123, 147, 192, 216, 206, 207, 205, 187, 203, 212, 214, 187,
]

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
)

def get_region_points(face_landmarks, ids, w, h):
    points = []
    for idx in ids:
        lm = face_landmarks[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))
    return points

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        h, w, _ = frame.shape

        if result.face_landmarks:
            for face_landmarks in result.face_landmarks:
                # draw all landmarks faintly
                for lm in face_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 1, (0, 100, 0), -1)

                
                draw_cheek_mesh(
                 frame,
                 face_landmarks,
                 set(LEFT_CHEEK_POLYGON_IDS),
                FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                 w,
                 h,
                 color=(180, 0, 0),
                 thickness=1
                 )

               
                cv2.putText(
                    frame,
                    "Left Cheek Polygon",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

        cv2.imshow("Face Landmarks", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()