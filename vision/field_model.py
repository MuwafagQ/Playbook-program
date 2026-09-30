"""Open RF-DETR pitch keypoint model (trained by notebooks/train_field_keypoints.ipynb) behind the
same `.infer(image, confidence)` interface as the Roboflow keypoint model, so
vision.detect.infer_field_keypoints uses it unchanged (FIELD_MODEL_PATH).

Two kinds of checkpoint (FIELD_MODEL_KIND):
  "ours"          trained by us; tools/keypoint_dataset.py fixes the order to labels "1".."32", so
                  keypoint index k is pitch vertex label k + 1.
  "roboflow_v10"  weights downloaded from Roboflow (football-field-detection/10, Apache-2.0 per
                  Roboflow's download page): keypoint index k follows Roboflow's name order
                  ROBOFLOW_KP_ORDER, and that version was trained on contrast-stretched frames.
Every keypoint is returned with its own confidence; KP_CONF filters them before the homography.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

# Roboflow's keypoint name order for the pitch skeleton ("01".."13", "15".."18", "20".."32", "14", "19")
ROBOFLOW_KP_ORDER = list(range(1, 14)) + [15, 16, 17, 18] + list(range(20, 33)) + [14, 19]


def contrast_stretch(img: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """Roboflow's "Contrast Stretching" preprocessing: rescale intensities between the 2nd and 98th
    percentiles of the image to the full 0-255 range. Percentiles from every 4th pixel and a lookup
    table: ~7 ms for a 1080p frame instead of ~0.5 s (same result within 3 grey levels)."""
    import cv2

    lo, hi = np.percentile(img[::4, ::4], (low, high))
    if hi <= lo:
        return img
    lut = np.clip((np.arange(256, dtype=np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def _to_result(kp, width: int, height: int, labels: list[int] | None = None):
    preds = []
    if kp is not None and kp.xy is not None and len(kp.xy):
        det_conf = getattr(kp, "detection_confidence", None)
        det_conf = np.ones(len(kp.xy)) if det_conf is None else np.asarray(det_conf)
        kp_conf = getattr(kp, "keypoint_confidence", None)
        kp_conf = np.ones(kp.xy.shape[:2]) if kp_conf is None else np.asarray(kp_conf)
        for xy, dc, kc in zip(kp.xy, det_conf, kp_conf):
            pts = [SimpleNamespace(x=float(x), y=float(y), confidence=float(c), class_id=k,
                                   class_name=str(labels[k] if labels else k + 1))
                   for k, ((x, y), c) in enumerate(zip(xy.tolist(), kc.tolist()))]
            preds.append(SimpleNamespace(confidence=float(dc), keypoints=pts))
    return SimpleNamespace(image=SimpleNamespace(width=width, height=height), predictions=preds)


class FieldModel:
    def __init__(self, model, labels: list[int] | None = None, stretch: bool = False):
        self.model = model  # an rfdetr keypoint model (anything with .predict(rgb_images, threshold) -> KeyPoints)
        self.labels = labels  # pitch vertex label of each output keypoint (None: index + 1)
        self.stretch = stretch  # apply Roboflow's contrast stretching first

    def infer(self, images, confidence: float = 0.3):
        batch = images if isinstance(images, list) else [images]
        rgb = [(contrast_stretch(im) if self.stretch else im)[:, :, ::-1].copy() for im in batch]
        out = self.model.predict(rgb, threshold=confidence)
        out = out if isinstance(out, list) else [out]
        return [_to_result(k, im.shape[1], im.shape[0], self.labels) for k, im in zip(out, batch)]


def load_field_model(path: str, accel: str = "none", kind: str = "ours", stretch: bool | None = None) -> FieldModel:
    """kind "ours": a checkpoint we trained. kind "roboflow_v10": Roboflow's downloaded weights.pt (weights
    only, so it is built as an RF-DETR keypoint model with those weights). TensorRT is not supported for
    keypoint models here (its decoder is detection-only)."""
    accel = "fp16" if accel == "tensorrt" else accel
    if kind == "roboflow_v10":
        from rfdetr import RFDETRKeypointPreview

        from vision.fast_rfdetr import accelerate_fp16

        model = RFDETRKeypointPreview(pretrain_weights=path, num_classes=1)
        if accel == "fp16":
            model = accelerate_fp16(model)
        return FieldModel(model, labels=ROBOFLOW_KP_ORDER, stretch=True if stretch is None else stretch)
    from vision.fast_rfdetr import load_rfdetr

    return FieldModel(load_rfdetr(path, accel), stretch=bool(stretch))
