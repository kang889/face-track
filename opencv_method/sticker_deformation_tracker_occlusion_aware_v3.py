from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

WINDOW_NAME = "Sticker Deformation Tracker (Occlusion Aware)"
DATASET_OUTPUT_DIR = Path("sticker_deformation_dataset_occlusion_aware")
SAVE_EVERY_N_VALID_FRAMES = 1

MIN_MARKERS_REQUIRED = 6
MIN_VISIBLE_FOR_ALIGN = 3
MIN_VISIBLE_TO_SAVE = 6

K_NEIGHBORS = 3
CLICK_SELECT_MAX_DIST_PX = 20.0
MATCH_MAX_DIST_PX = 35.0

# HSV sampling / detection
HUE_TOL = 8
SAT_TOL = 60
VAL_TOL = 70
MIN_CONTOUR_AREA = 20
MAX_CONTOUR_AREA = 3000
MIN_CIRCULARITY = 0.35

# Motion / tracking
NEUTRAL_CAPTURE_FRAMES = 20
ALIGNMENT_ERROR_THRESHOLD_PX = 12.0
ARROW_THRESHOLD_PX = 2.0

# Colors (BGR)
COLOR_SOURCE = (0, 255, 255)
COLOR_VISIBLE = (0, 255, 0)
COLOR_BLOCKED = (0, 0, 255)
COLOR_NEUTRAL = (140, 140, 140)
COLOR_POINT = (255, 200, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_ERROR = (0, 0, 255)
COLOR_INFO = (180, 180, 180)
COLOR_WARN = (0, 255, 255)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def order_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    order = np.lexsort((pts[:, 0], pts[:, 1]))
    return pts[order]


def sample_hsv(frame_bgr: np.ndarray, x: int, y: int, radius: int = 6) -> tuple[int, int, int]:
    h, w = frame_bgr.shape[:2]
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    patch = frame_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mean_hsv = np.mean(hsv.reshape(-1, 3), axis=0)
    return int(mean_hsv[0]), int(mean_hsv[1]), int(mean_hsv[2])


def threshold_hsv(frame_bgr: np.ndarray, hsv_target: tuple[int, int, int]) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv_target

    lower_s = max(30, s - SAT_TOL)
    lower_v = max(30, v - VAL_TOL)
    upper_s = 255
    upper_v = 255

    if h - HUE_TOL < 0:
        low1 = np.array([0, lower_s, lower_v], dtype=np.uint8)
        high1 = np.array([h + HUE_TOL, upper_s, upper_v], dtype=np.uint8)
        low2 = np.array([180 + (h - HUE_TOL), lower_s, lower_v], dtype=np.uint8)
        high2 = np.array([179, upper_s, upper_v], dtype=np.uint8)
        mask = cv2.inRange(hsv, low1, high1) | cv2.inRange(hsv, low2, high2)
    elif h + HUE_TOL > 179:
        low1 = np.array([h - HUE_TOL, lower_s, lower_v], dtype=np.uint8)
        high1 = np.array([179, upper_s, upper_v], dtype=np.uint8)
        low2 = np.array([0, lower_s, lower_v], dtype=np.uint8)
        high2 = np.array([(h + HUE_TOL) - 180, upper_s, upper_v], dtype=np.uint8)
        mask = cv2.inRange(hsv, low1, high1) | cv2.inRange(hsv, low2, high2)
    else:
        low = np.array([h - HUE_TOL, lower_s, lower_v], dtype=np.uint8)
        high = np.array([h + HUE_TOL, upper_s, upper_v], dtype=np.uint8)
        mask = cv2.inRange(hsv, low, high)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detect_markers(frame_bgr: np.ndarray, hsv_target: tuple[int, int, int] | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if hsv_target is None:
        return None, None

    mask = threshold_hsv(frame_bgr, hsv_target)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers: list[list[float]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 1e-6:
            continue

        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < MIN_CIRCULARITY:
            continue

        m = cv2.moments(cnt)
        if abs(m["m00"]) < 1e-6:
            continue

        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        centers.append([cx, cy])

    if not centers:
        return None, mask

    pts = np.array(centers, dtype=np.float32)
    return pts, mask


def estimate_affine(current_points: np.ndarray, neutral_points: np.ndarray) -> np.ndarray | None:
    if len(current_points) < 3 or len(neutral_points) < 3:
        return None
    transform, _ = cv2.estimateAffinePartial2D(current_points, neutral_points, method=cv2.LMEDS)
    return transform


def apply_affine(points: np.ndarray, transform: np.ndarray | None) -> np.ndarray | None:
    if transform is None:
        return None
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    homog = np.hstack([points, ones])
    return (transform @ homog.T).T.astype(np.float32)


def mean_alignment_error(current_points: np.ndarray, neutral_points: np.ndarray, transform: np.ndarray | None) -> float:
    corrected = apply_affine(current_points, transform)
    if corrected is None:
        return float("inf")
    return float(np.mean(np.linalg.norm(corrected - neutral_points, axis=1)))


def build_knn_edges(neutral_points: np.ndarray, k_neighbors: int) -> list[tuple[int, int]]:
    n = len(neutral_points)
    if n < 2:
        return []
    edges = set()
    dmat = np.linalg.norm(neutral_points[:, None, :] - neutral_points[None, :, :], axis=2)
    for i in range(n):
        order = np.argsort(dmat[i])
        nbrs = [j for j in order if j != i][:k_neighbors]
        for j in nbrs:
            a, b = (i, j) if i < j else (j, i)
            edges.add((a, b))
    return sorted(edges)


def build_adjacency(num_points: int, edges: list[tuple[int, int]]) -> list[set[int]]:
    adj = [set() for _ in range(num_points)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def pick_nearest_point_index(points: np.ndarray | None, x: int, y: int, max_dist_px: float) -> int | None:
    if points is None or len(points) == 0:
        return None
    click = np.array([x, y], dtype=np.float32)
    dists = np.linalg.norm(points - click, axis=1)
    best_idx = int(np.argmin(dists))
    if float(dists[best_idx]) <= max_dist_px:
        return best_idx
    return None


def draw_arrow(frame: np.ndarray, start_pt: np.ndarray, end_pt: np.ndarray, color, thickness=1, threshold_px=0.0):
    motion = float(np.linalg.norm(end_pt - start_pt))
    if motion < threshold_px:
        return
    s = tuple(np.round(start_pt).astype(int))
    e = tuple(np.round(end_pt).astype(int))
    cv2.arrowedLine(frame, s, e, color, thickness, tipLength=0.22)


def greedy_match_points(predicted_points: np.ndarray, detected_points: np.ndarray, max_dist_px: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Match detections to sticker IDs using nearest predicted position.
    Returns:
        matched_points: (N,2) with matched positions or predicted positions if unmatched
        visible_mask: (N,) true where a detection matched this point ID
    """
    n = len(predicted_points)
    matched_points = predicted_points.copy()
    visible_mask = np.zeros(n, dtype=bool)

    if detected_points is None or len(detected_points) == 0:
        return matched_points, visible_mask

    pairs = []
    for i, pred in enumerate(predicted_points):
        dists = np.linalg.norm(detected_points - pred, axis=1)
        for j, d in enumerate(dists):
            if d <= max_dist_px:
                pairs.append((float(d), i, j))

    pairs.sort(key=lambda x: x[0])
    used_i = set()
    used_j = set()

    for _, i, j in pairs:
        if i in used_i or j in used_j:
            continue
        matched_points[i] = detected_points[j]
        visible_mask[i] = True
        used_i.add(i)
        used_j.add(j)

    return matched_points, visible_mask


def save_sample(
    output_dir: Path,
    sample_index: int,
    source_index: int,
    current_points: np.ndarray,
    corrected_points: np.ndarray,
    neutral_points: np.ndarray,
    visible_mask: np.ndarray,
    blocked_mask: np.ndarray,
    patch_edges: list[tuple[int, int]],
    alignment_error: float,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    displacement = (corrected_points - neutral_points).astype(np.float32)
    visible_displacement = displacement.copy()
    visible_displacement[blocked_mask] = np.nan

    point_motion_magnitude_visible = np.linalg.norm(
        np.nan_to_num(visible_displacement, nan=0.0),
        axis=1,
    ).astype(np.float32)

    np.savez_compressed(
        output_dir / f"sample_{sample_index:06d}.npz",
        source_local_index=np.int32(source_index),
        visible_mask=visible_mask.astype(np.uint8),
        blocked_mask=blocked_mask.astype(np.uint8),
        neutral_points=neutral_points.astype(np.float32),
        current_points=current_points.astype(np.float32),
        corrected_points=corrected_points.astype(np.float32),
        point_displacement=displacement.astype(np.float32),
        visible_point_displacement=visible_displacement.astype(np.float32),
        point_motion_magnitude_visible=point_motion_magnitude_visible,
        patch_edges=np.array(patch_edges, dtype=np.int32),
        alignment_error=np.array(alignment_error, dtype=np.float32),
    )


# -----------------------------------------------------------------------------
# Mouse callback
# -----------------------------------------------------------------------------

def handle_mouse(event, x, y, flags, state):
    if event == cv2.EVENT_RBUTTONDOWN:
        frame = state.get("latest_frame")
        if frame is not None:
            state["hsv_target"] = sample_hsv(frame, x, y)
            state["status_message"] = f"Sampled sticker HSV at ({x}, {y})"
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        points = state.get("latest_clickable_points")
        selected = pick_nearest_point_index(points, x, y, CLICK_SELECT_MAX_DIST_PX)
        if selected is not None:
            state["active_source_index"] = selected
            state["status_message"] = f"Selected source point {selected}"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    cv2.namedWindow(WINDOW_NAME)

    state = {
        "active_source_index": 0,
        "latest_clickable_points": None,
        "latest_frame": None,
        "hsv_target": None,
        "status_message": "Right-click a sticker to sample its color",
    }
    cv2.setMouseCallback(WINDOW_NAME, handle_mouse, state)

    neutral_points = None
    patch_edges: list[tuple[int, int]] = []
    adjacency: list[set[int]] = []

    capture_requested = False
    neutral_buffer: list[np.ndarray] = []
    neutral_expected_count = None
    recording = False
    valid_frame_index = 0
    saved_sample_count = 0

    prev_gray = None
    prev_points = None           # last known image-space points for all IDs
    last_seen_points = None      # frozen last visible image-space points
    last_corrected_points = None # frozen last corrected points

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        state["latest_frame"] = frame.copy()
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detected_points, mask = detect_markers(frame, state["hsv_target"])

        if mask is not None:
            mask_small = cv2.resize(mask, (160, 120))
            mask_bgr = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
            display[10:130, display.shape[1] - 170:display.shape[1] - 10] = mask_bgr
            cv2.rectangle(display, (display.shape[1] - 170, 10), (display.shape[1] - 10, 130), (90, 90, 90), 1)
            cv2.putText(display, "Sticker mask", (display.shape[1] - 160, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)

        if neutral_points is None:
            if detected_points is not None:
                ordered_detected = order_points(detected_points)
                state["latest_clickable_points"] = ordered_detected
                for idx, pt in enumerate(ordered_detected):
                    xy = tuple(np.round(pt).astype(int))
                    cv2.circle(display, xy, 4, COLOR_POINT, -1)
                    cv2.putText(display, str(idx), (xy[0] + 5, xy[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_POINT, 1, cv2.LINE_AA)

                src = int(state["active_source_index"])
                src = max(0, min(src, len(ordered_detected) - 1))
                state["active_source_index"] = src
                src_xy = tuple(np.round(ordered_detected[src]).astype(int))
                cv2.circle(display, src_xy, 5, COLOR_SOURCE, -1)
                cv2.putText(display, f"S:{src}", (src_xy[0] + 6, src_xy[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_SOURCE, 1, cv2.LINE_AA)
            else:
                state["latest_clickable_points"] = None

            if capture_requested:
                if detected_points is None or len(detected_points) < MIN_MARKERS_REQUIRED:
                    cv2.putText(display, f"Need at least {MIN_MARKERS_REQUIRED} markers for neutral", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ERROR, 2)
                else:
                    ordered_detected = order_points(detected_points)
                    current_count = len(ordered_detected)
                    if neutral_expected_count is None:
                        neutral_expected_count = current_count
                        state["status_message"] = f"Neutral locked to {neutral_expected_count} stickers"

                    if current_count != neutral_expected_count:
                        cv2.putText(display, f"Neutral needs exactly {neutral_expected_count} stickers, currently {current_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WARN, 2)
                    else:
                        neutral_buffer.append(ordered_detected)
                        cv2.putText(display, f"Capturing neutral {len(neutral_buffer)}/{NEUTRAL_CAPTURE_FRAMES} | stickers: {neutral_expected_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
                        if len(neutral_buffer) >= NEUTRAL_CAPTURE_FRAMES:
                            neutral_points = np.mean(np.stack(neutral_buffer, axis=0), axis=0).astype(np.float32)
                            neutral_points = order_points(neutral_points)
                            prev_points = neutral_points.copy()
                            last_seen_points = neutral_points.copy()
                            last_corrected_points = neutral_points.copy()
                            prev_gray = gray.copy()
                            patch_edges = build_knn_edges(neutral_points, K_NEIGHBORS)
                            adjacency = build_adjacency(len(neutral_points), patch_edges)
                            neutral_buffer.clear()
                            neutral_expected_count = None
                            capture_requested = False
                            state["active_source_index"] = min(int(state["active_source_index"]), len(neutral_points) - 1)
                            state["status_message"] = f"Neutral captured with {len(neutral_points)} markers"

        else:
            # Predict current sticker positions with optical flow from last frame.
            predicted_points = prev_points.copy()
            if prev_gray is not None and prev_points is not None:
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray,
                    gray,
                    prev_points.reshape(-1, 1, 2),
                    None,
                    winSize=(21, 21),
                    maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                )
                if next_pts is not None and status is not None:
                    flow_pts = next_pts.reshape(-1, 2).astype(np.float32)
                    flow_status = status.reshape(-1).astype(bool)
                    predicted_points[flow_status] = flow_pts[flow_status]

            current_points, visible_mask = greedy_match_points(
                predicted_points,
                detected_points,
                MATCH_MAX_DIST_PX,
            )
            blocked_mask = ~visible_mask
            visible_count = int(np.sum(visible_mask))

            # Freeze blocked points at last seen / last corrected positions.
            current_points[blocked_mask] = last_seen_points[blocked_mask]

            # Estimate rigid motion only from visible points.
            align_err = float("inf")
            corrected_points = last_corrected_points.copy()

            if visible_count >= MIN_VISIBLE_FOR_ALIGN:
                transform = estimate_affine(current_points[visible_mask], neutral_points[visible_mask])
                if transform is not None:
                    align_err = mean_alignment_error(current_points[visible_mask], neutral_points[visible_mask], transform)
                    corrected_visible = apply_affine(current_points, transform)
                    if corrected_visible is not None:
                        corrected_points[visible_mask] = corrected_visible[visible_mask]

            # Update frozen state only for visible points.
            last_seen_points[visible_mask] = current_points[visible_mask]
            last_corrected_points[visible_mask] = corrected_points[visible_mask]
            prev_points = current_points.copy()
            prev_gray = gray.copy()

            state["latest_clickable_points"] = current_points

            src = int(state["active_source_index"])
            src = max(0, min(src, len(neutral_points) - 1))
            state["active_source_index"] = src
            patch_indices = sorted(adjacency[src]) if adjacency else []

            cv2.putText(display, f"Visible: {visible_count}/{len(neutral_points)} | Align err: {align_err:.2f}px", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 2)
            if visible_count < MIN_VISIBLE_FOR_ALIGN:
                cv2.putText(display, "Not enough visible stickers for alignment", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ERROR, 2)
            elif align_err > ALIGNMENT_ERROR_THRESHOLD_PX:
                cv2.putText(display, "Alignment unstable - frame not saved", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WARN, 2)

            # Draw all points:
            for idx, npt in enumerate(neutral_points):
                nxy = tuple(np.round(npt).astype(int))
                cv2.circle(display, nxy, 4, COLOR_NEUTRAL, -1)

                cpt = corrected_points[idx]
                cxy = tuple(np.round(cpt).astype(int))

                if blocked_mask[idx]:
                    cv2.circle(display, cxy, 4, COLOR_BLOCKED, -1)
                else:
                    draw_color = COLOR_VISIBLE if idx not in patch_indices else COLOR_POINT
                    cv2.circle(display, cxy, 4, draw_color, -1)
                    draw_arrow(display, npt, cpt, COLOR_VISIBLE, thickness=1, threshold_px=ARROW_THRESHOLD_PX)

                cv2.putText(display, str(idx), (cxy[0] + 5, cxy[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_TEXT, 1, cv2.LINE_AA)

            # Draw source point.
            src_c = corrected_points[src]
            src_xy = tuple(np.round(src_c).astype(int))
            if blocked_mask[src]:
                cv2.circle(display, src_xy, 5, COLOR_BLOCKED, -1)
                cv2.putText(display, f"S:{src}", (src_xy[0] + 6, src_xy[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_SOURCE, 1, cv2.LINE_AA)
            else:
                cv2.circle(display, src_xy, 5, COLOR_SOURCE, -1)
                draw_arrow(display, neutral_points[src], corrected_points[src], COLOR_SOURCE, thickness=2, threshold_px=ARROW_THRESHOLD_PX)
                cv2.putText(display, f"S:{src}", (src_xy[0] + 6, src_xy[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_SOURCE, 1, cv2.LINE_AA)

            # Save only when there are enough visible points and alignment is acceptable.
            if recording and visible_count >= MIN_VISIBLE_TO_SAVE and align_err <= ALIGNMENT_ERROR_THRESHOLD_PX:
                valid_frame_index += 1
                if valid_frame_index % SAVE_EVERY_N_VALID_FRAMES == 0:
                    save_sample(
                        DATASET_OUTPUT_DIR,
                        saved_sample_count,
                        source_index=src,
                        current_points=current_points,
                        corrected_points=corrected_points,
                        neutral_points=neutral_points,
                        visible_mask=visible_mask,
                        blocked_mask=blocked_mask,
                        patch_edges=patch_edges,
                        alignment_error=align_err,
                    )
                    saved_sample_count += 1

        # UI text
        hsv_text = f"HSV target: {state['hsv_target']}" if state["hsv_target"] is not None else "HSV target: none"
        cv2.putText(display, "Sticker Deformation Tracker (Occlusion Aware)", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, COLOR_TEXT, 2)
        detected_count_text = f"Detected stickers: {0 if detected_points is None else len(detected_points)}"
        cv2.putText(display, hsv_text, (20, display.shape[0] - 98), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_INFO, 1)
        cv2.putText(display, detected_count_text, (20, display.shape[0] - 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_INFO, 1)
        cv2.putText(display, f"Recording: {'ON' if recording else 'OFF'} | Saved: {saved_sample_count}", (20, display.shape[0] - 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1)
        cv2.putText(display, state["status_message"], (20, display.shape[0] - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_INFO, 1)
        cv2.putText(display, "Right-click sample sticker color | Left-click choose source | N neutral | R record | C reset | Q quit", (20, display.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.43, COLOR_TEXT, 1)

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("n"):
            if detected_points is not None and len(detected_points) >= MIN_MARKERS_REQUIRED and not recording:
                capture_requested = True
                neutral_buffer.clear()
                neutral_expected_count = None
                state["status_message"] = "Started neutral capture"
            else:
                state["status_message"] = "Need enough detected stickers before neutral capture"

        elif key == ord("r"):
            if neutral_points is None:
                state["status_message"] = "Capture neutral first"
            else:
                recording = not recording
                state["status_message"] = f"Recording {'ON' if recording else 'OFF'}"

        elif key == ord("c"):
            neutral_points = None
            patch_edges = []
            adjacency = []
            capture_requested = False
            neutral_buffer.clear()
            neutral_expected_count = None
            recording = False
            valid_frame_index = 0
            prev_gray = None
            prev_points = None
            last_seen_points = None
            last_corrected_points = None
            state["status_message"] = "Reset neutral/tracking state"

        elif key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
