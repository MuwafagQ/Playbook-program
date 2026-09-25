from __future__ import annotations
import cv2
import numpy as np
import supervision as sv

BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3


def tiny_box_filter(
    detections: sv.Detections,
    frame_w: int,
    frame_h: int,
    min_area_ratio_people: float,
    min_area_ratio_ball: float,
) -> sv.Detections:
    if len(detections) == 0:
        return detections

    areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * (detections.xyxy[:, 3] - detections.xyxy[:, 1])
    frame_area = float(frame_w * frame_h)
    area_ratio = areas / max(frame_area, 1.0)

    thr = np.where(detections.class_id == BALL_ID, min_area_ratio_ball, min_area_ratio_people)
    keep = area_ratio >= thr
    return detections[keep]


def infer_players_and_ball(player_model, frame: np.ndarray, conf: float) -> sv.Detections:
    result = player_model.infer(frame, confidence=conf)[0]
    det = sv.Detections.from_inference(result)
    return det


def infer_players_and_ball_upscaled(
    player_model,
    frame: np.ndarray,
    conf: float,
    upscale: float = 1.0,
) -> sv.Detections:
    upscale = float(max(1.0, upscale))
    if upscale <= 1.01:
        return infer_players_and_ball(player_model, frame, conf=conf)

    h, w = frame.shape[:2]
    up = cv2.resize(frame, (int(w * upscale), int(h * upscale)), interpolation=cv2.INTER_LINEAR)
    det = infer_players_and_ball(player_model, up, conf=conf)
    if len(det) > 0:
        det.xyxy = det.xyxy / upscale
    return det


def _empty_ball_detections() -> sv.Detections:
    return sv.Detections(
        xyxy=np.zeros((0, 4), dtype=np.float32),
        confidence=np.zeros((0,), dtype=np.float32),
        class_id=np.zeros((0,), dtype=np.int32),
    )


def recover_ball_in_roi(
    player_model,
    frame: np.ndarray,
    center_xy,
    roi_px: int = 320,
    upscale: float = 2.0,
    conf: float = 0.10,
) -> sv.Detections:
    """
    Second-chance ball detection in a zoomed crop around the predicted position.

    The ball is often only ~10 px in the full frame, below what the detector
    resolves reliably. Cropping roi_px around the trajectory prediction and
    upscaling makes the ball several times larger, recovering detections the
    full-frame pass missed. Returned boxes (ball class only) are mapped back to
    full-frame coordinates.

    The low conf here is safe because the result is spatially anchored: the
    caller's smoother still applies its distance/size/confidence gates.
    """
    h, w = frame.shape[:2]
    half = max(16, int(roi_px) // 2)
    cx = int(round(float(center_xy[0])))
    cy = int(round(float(center_xy[1])))
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)
    if (x2 - x1) < 32 or (y2 - y1) < 32:
        return _empty_ball_detections()

    crop = frame[y1:y2, x1:x2]
    up = float(max(1.0, upscale))
    if up > 1.01:
        crop = cv2.resize(
            crop,
            (int((x2 - x1) * up), int((y2 - y1) * up)),
            interpolation=cv2.INTER_LINEAR,
        )

    det = infer_players_and_ball(player_model, crop, conf=conf)
    if det.class_id is None or len(det) == 0:
        return _empty_ball_detections()
    det = det[det.class_id == BALL_ID]
    if len(det) == 0:
        return _empty_ball_detections()
    det.xyxy = det.xyxy / up + np.array([x1, y1, x1, y1], dtype=np.float32)
    return det


def tile_origins(width: int, height: int, tile: int = 720, min_overlap: int = 64) -> list[tuple[int, int]]:
    """Top-left corners of square tiles covering the frame with at least min_overlap px overlap."""
    def starts(n_px):
        if n_px <= tile:
            return [0]
        k = int(np.ceil((n_px - min_overlap) / (tile - min_overlap)))
        return [int(round(v)) for v in np.linspace(0, n_px - tile, k)]
    return [(x, y) for y in starts(height) for x in starts(width)]


def infer_ball_tiles(player_model, frame: np.ndarray, conf: float = 0.10, tile: int = 720) -> sv.Detections:
    """Ball candidates from square native-resolution tiles (the ball is ~2.5x larger to the
    model than in the full-frame pass). Boxes are mapped back to the frame; duplicates from
    overlapping tiles are merged by NMS."""
    h, w = frame.shape[:2]
    origins = tile_origins(w, h, tile)
    crops = [frame[y:y + tile, x:x + tile] for x, y in origins]
    try:
        results = player_model.infer(crops, confidence=conf)
        if len(results) != len(crops):
            raise ValueError("batch size mismatch")
    except Exception:
        results = [player_model.infer(c, confidence=conf)[0] for c in crops]
    parts = []
    for (x, y), res in zip(origins, results):
        det = sv.Detections.from_inference(res)
        if det.class_id is None or len(det) == 0:
            continue
        det = det[det.class_id == BALL_ID]
        if len(det) == 0:
            continue
        det.xyxy = det.xyxy + np.array([x, y, x, y], dtype=np.float32)
        parts.append(det)
    if not parts:
        return _empty_ball_detections()
    merged = sv.Detections.merge(parts)
    return merged.with_nms(threshold=0.3, class_agnostic=True)


_KP_CORRESPONDENCE_LOGGED = False


def _vertex_label(keypoint) -> int:
    """Pitch-vertex label for a Roboflow keypoint.

    class_name is the true vertex label (e.g. "20"); the model's internal
    class_id is a different ordering and must NOT be used for correspondence.
    Falls back to class_id only if class_name is missing/non-numeric.
    """
    name = getattr(keypoint, "class_name", None)
    try:
        return int(name)
    except (TypeError, ValueError):
        try:
            return int(keypoint.class_id)
        except (TypeError, ValueError, AttributeError):
            return -1


def infer_field_keypoints(field_model, frame: np.ndarray, conf: float) -> sv.KeyPoints:
    global _KP_CORRESPONDENCE_LOGGED
    result = field_model.infer(frame, confidence=conf)[0]
    preds = list(getattr(result, "predictions", []) or [])
    if len(preds) == 0:
        return sv.KeyPoints(xy=np.zeros((1, 0, 2), dtype=np.float32))

    # The field model detects a single "pitch" object, but at low FIELD_CONF it can
    # emit duplicate detections (Roboflow test: 3 "pitch" boxes at the 0.53 object
    # threshold). Use the HIGHEST-confidence detection, not predictions[0] — order is
    # not guaranteed, and a weaker duplicate would corrupt the homography. Its keypoint
    # list is the full vertex set; each keypoint carries its own class_id (-> pitch
    # vertex) and confidence, which sv.KeyPoints.from_inference throws away.
    pred = max(preds, key=lambda p: float(getattr(p, "confidence", 0.0)))
    kpts = list(getattr(pred, "keypoints", []) or [])

    xy = np.array([[float(k.x), float(k.y)] for k in kpts], dtype=np.float32).reshape(1, -1, 2)
    kp = sv.KeyPoints(xy=xy)
    # Store the pitch-vertex LABEL (class_name, e.g. "20"), NOT the model's internal
    # class_id. The model's class_id ordering is its own and does NOT match the pitch
    # vertex numbering (class_id 17 == vertex label "20" for this model). The
    # homography estimator maps label -> vertex index via the pitch config.
    # The KeyPoints constructor validates class_id as per-DETECTION; assign per-keypoint
    # arrays (shape [1, n_kp]) as attributes so the homography estimator can read them.
    kp.class_id = np.array([_vertex_label(k) for k in kpts], dtype=np.int32).reshape(1, -1)
    kp.confidence = np.array([float(k.confidence) for k in kpts], dtype=np.float32).reshape(1, -1)

    if not _KP_CORRESPONDENCE_LOGGED:
        labels = kp.class_id[0]
        cf = kp.confidence[0]
        print(
            f"[detect] keypoint vertex-labels injected: n={labels.shape[0]} "
            f"labels[0:8]={labels[:8].tolist()} conf[0:8]={np.round(cf[:8], 2).tolist()} "
            f"pitch_conf={float(getattr(pred, 'confidence', 0.0)):.2f} n_det={len(preds)}"
        )
        try:
            pairs = [(int(k.class_id), getattr(k, "class_name", "?")) for k in kpts[:12]]
            print(f"[detect] kp class_id <-> class_name (label) pairs: {pairs}")
        except Exception:
            pass
        _KP_CORRESPONDENCE_LOGGED = True
    return kp


def class_conf_filter(
    detections: sv.Detections,
    player_conf: float,
    referee_conf: float,
    goalkeeper_conf: float,
    ball_conf: float,
) -> sv.Detections:
    if len(detections) == 0 or detections.confidence is None or detections.class_id is None:
        return detections

    cls = detections.class_id.astype(np.int32)
    conf = detections.confidence.astype(np.float32)
    thr = np.full_like(conf, fill_value=player_conf, dtype=np.float32)
    thr = np.where(cls == PLAYER_ID, player_conf, thr)
    thr = np.where(cls == REFEREE_ID, referee_conf, thr)
    thr = np.where(cls == GOALKEEPER_ID, goalkeeper_conf, thr)
    thr = np.where(cls == BALL_ID, ball_conf, thr)
    keep = conf >= thr
    return detections[keep]
