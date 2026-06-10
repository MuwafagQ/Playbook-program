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


_KP_CORRESPONDENCE_LOGGED = False


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
    # The KeyPoints constructor validates class_id as per-DETECTION; assign per-keypoint
    # arrays (shape [1, n_kp]) as attributes so the homography estimator can read them.
    kp.class_id = np.array([int(k.class_id) for k in kpts], dtype=np.int32).reshape(1, -1)
    kp.confidence = np.array([float(k.confidence) for k in kpts], dtype=np.float32).reshape(1, -1)

    if not _KP_CORRESPONDENCE_LOGGED:
        ids = kp.class_id[0]
        cf = kp.confidence[0]
        print(
            f"[detect] keypoint correspondence injected: n={ids.shape[0]} "
            f"ids[0:8]={ids[:8].tolist()} conf[0:8]={np.round(cf[:8], 2).tolist()} "
            f"pitch_conf={float(getattr(pred, 'confidence', 0.0)):.2f} n_det={len(preds)}"
        )
        try:
            pairs = [(int(k.class_id), getattr(k, "class_name", "?")) for k in kpts[:12]]
            print(f"[detect] kp class_id <-> class_name pairs: {pairs}")
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
