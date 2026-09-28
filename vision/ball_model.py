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
