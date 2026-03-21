import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

MODEL_PATH = "models/face_landmarker.task"

LEFT_CHEEK_IDS = [
    36, 50, 101, 119, 120, 121, 47, 100, 118, 117, 123, 147, 187, 205, 206, 207,
    216, 192, 147, 123, 116, 111, 123
    ]

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

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

        # OpenCV gives BGR, MediaPipe expects RGB
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

        # extract left cheek region
        cheek_points = get_region_points(face_landmarks, LEFT_CHEEK_IDS, w, h)

        # draw cheek landmarks more clearly
        for pt in cheek_points:
            cv2.circle(frame, pt, 3, (0, 0, 255), -1)

        # draw cheek boundary
        cheek_array = np.array(cheek_points, dtype=np.int32)
        cv2.polylines(frame, [cheek_array], isClosed=True, color=(255, 0, 0), thickness=2)

        cv2.putText(
            frame,
            "Left Cheek Region",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow("Face Landmarks", frame)

        key = cv2.waitKey(1)
        if key == 27 or key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()