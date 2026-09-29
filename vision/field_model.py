"""Open RF-DETR pitch keypoint model (trained by notebooks/train_field_keypoints.ipynb) behind the
same `.infer(image, confidence)` interface as the Roboflow keypoint model, so
vision.detect.infer_field_keypoints uses it unchanged (FIELD_MODEL_PATH).

The model predicts the 32 pitch landmarks in the dataset's keypoint order, which
tools/keypoint_dataset.py fixes to labels "1".."32": keypoint index k is pitch vertex label k + 1.
Every keypoint is returned with its own confidence; KP_CONF filters them before the homography.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def _to_result(kp, width: int, height: int):
    preds = []
    if kp is not None and kp.xy is not None and len(kp.xy):
        det_conf = getattr(kp, "detection_confidence", None)
        det_conf = np.ones(len(kp.xy)) if det_conf is None else np.asarray(det_conf)
        kp_conf = getattr(kp, "keypoint_confidence", None)
        kp_conf = np.ones(kp.xy.shape[:2]) if kp_conf is None else np.asarray(kp_conf)
        for xy, dc, kc in zip(kp.xy, det_conf, kp_conf):
            pts = [SimpleNamespace(x=float(x), y=float(y), confidence=float(c), class_id=k, class_name=str(k + 1))
                   for k, ((x, y), c) in enumerate(zip(xy.tolist(), kc.tolist()))]
            preds.append(SimpleNamespace(confidence=float(dc), keypoints=pts))
    return SimpleNamespace(image=SimpleNamespace(width=width, height=height), predictions=preds)


class FieldModel:
    def __init__(self, model):
        self.model = model  # an rfdetr keypoint model (anything with .predict(rgb_images, threshold) -> KeyPoints)

    def infer(self, images, confidence: float = 0.3):
        batch = images if isinstance(images, list) else [images]
        rgb = [im[:, :, ::-1].copy() for im in batch]
        out = self.model.predict(rgb, threshold=confidence)
        out = out if isinstance(out, list) else [out]
        return [_to_result(k, im.shape[1], im.shape[0]) for k, im in zip(out, batch)]


def load_field_model(path: str, accel: str = "none") -> FieldModel:
    """TensorRT is not supported for keypoint models here (its decoder is detection-only)."""
    from vision.fast_rfdetr import load_rfdetr

    return FieldModel(load_rfdetr(path, "fp16" if accel == "tensorrt" else accel))
