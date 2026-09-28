"""Dedicated ball detector run on crops of the frame.

The model is trained on small tiles (tools/tile_dataset.py), so it must be run on tiles
too. `predict_tiles` covers the whole frame (search mode); a crop around a predicted
position can be passed through the same function with a single origin later on.
"""
from __future__ import annotations

import numpy as np
import supervision as sv

from tools.tile_dataset import grid_origins


def _empty() -> sv.Detections:
    return sv.Detections(xyxy=np.zeros((0, 4), np.float32), confidence=np.zeros((0,), np.float32),
                         class_id=np.zeros((0,), np.int32))


def predict_tiles(predict_fn, frame: np.ndarray, tile: int = 320, conf: float = 0.1,
                  batch: int = 32, min_overlap: int = 64, origins=None) -> sv.Detections:
    """Ball candidates in frame coordinates.

    predict_fn(list of BGR crops, conf) -> list of sv.Detections in crop coordinates.
    Every detection is taken as a ball (single-class model). Duplicates from overlapping
    tiles are merged by NMS.
    """
    h, w = frame.shape[:2]
    origins = origins if origins is not None else grid_origins(w, h, tile, min_overlap)
    parts = []
    for i in range(0, len(origins), batch):
        chunk = origins[i:i + batch]
        crops = [frame[y:y + tile, x:x + tile] for x, y in chunk]
        for (x, y), det in zip(chunk, predict_fn(crops, conf)):
            if det is None or len(det) == 0:
                continue
            det = sv.Detections(xyxy=det.xyxy.astype(np.float32) + np.array([x, y, x, y], np.float32),
                                confidence=np.asarray(det.confidence, np.float32),
                                class_id=np.zeros(len(det), np.int32))
            parts.append(det)
    if not parts:
        return _empty()
    return sv.Detections.merge(parts).with_nms(threshold=0.3, class_agnostic=True)


def rfdetr_predict_fn(model):
    """Adapter for an rfdetr model (RFDETRSmall etc.): BGR crops -> list of sv.Detections."""
    def fn(crops, conf):
        rgb = [c[:, :, ::-1].copy() for c in crops]
        out = model.predict(rgb, threshold=conf)
        return out if isinstance(out, list) else [out]
    return fn


def roi_origins(center, width: int, height: int, tile: int = 320, n: int = 2,
                min_overlap: int = 64) -> list[tuple[int, int]]:
    """n x n overlapping tiles centred on `center` (e.g. the predicted ball position),
    shifted to stay inside the frame."""
    step = tile - min_overlap
    span = tile + (n - 1) * step
    x0 = int(np.clip(float(center[0]) - span / 2, 0, max(width - span, 0)))
    y0 = int(np.clip(float(center[1]) - span / 2, 0, max(height - span, 0)))
    return [(min(x0 + i * step, max(width - tile, 0)), min(y0 + j * step, max(height - tile, 0)))
            for j in range(n) for i in range(n)]


def within_origins(det: sv.Detections, origins, tile: int) -> sv.Detections:
    """Candidates whose centre lies inside any of the tiles (used in replay to reproduce
    a search around the ball from full-frame candidates)."""
    if len(det) == 0:
        return det
    cx = (det.xyxy[:, 0] + det.xyxy[:, 2]) / 2
    cy = (det.xyxy[:, 1] + det.xyxy[:, 3]) / 2
    keep = np.zeros(len(det), bool)
    for x, y in origins:
        keep |= (cx >= x) & (cx < x + tile) & (cy >= y) & (cy < y + tile)
    return det[keep]


def load_ball_model(path: str):
    """The trained ball model (rfdetr checkpoint) as a predict_fn for predict_tiles."""
    from rfdetr.detr import RFDETR

    model = RFDETR.from_checkpoint(path, trust_checkpoint=True)
    try:
        model.optimize_for_inference()
    except Exception as e:  # optional speed-up; not available on every setup
        print(f"[ball-model] optimize_for_inference skipped: {e}")
    return rfdetr_predict_fn(model)
