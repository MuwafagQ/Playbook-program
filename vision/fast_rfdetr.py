"""Faster inference for our RF-DETR models (people model, ball model) on an NVIDIA GPU.

Three modes, all exposing `.predict(list_of_rgb_arrays, threshold) -> list[sv.Detections]` like an
rfdetr model, so vision/people_model.py and vision/ball_model.py use them unchanged:
  "none"      the plain rfdetr model
  "fp16"      rfdetr's own half-precision inference (model.inference(dtype=float16, compile=False))
  "tensorrt"  a TensorRT FP16 engine built from the model (rfdetr export, format="tensorrt"),
              run with rfdetr's TRTInference; images are resized/normalised on the GPU exactly
              like RFDETR.predict and outputs decoded with rfdetr's reference decoder.
TensorRT engines are tied to the GPU model and TensorRT version they were built on.
Check a mode against "none" (compare_detections) before trusting it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import supervision as sv

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def accelerate_fp16(model):
    import torch

    model.inference(compile=False, dtype=torch.float16)
    return model


def build_tensorrt_engine(model, out_dir: str, max_batch: int = 32, batch: int = 4) -> str:
    """Export the model to a TensorRT FP16 engine with a dynamic batch (1..max_batch)."""
    out = model.export(output_dir=str(out_dir), format="tensorrt", fp16=True, dynamic_batch=True,
                       batch_size=batch, max_batch_size=max_batch, verbose=False)
    out = Path(out)
    if out.is_dir():
        engines = sorted(out.glob("*.trt")) + sorted(out.glob("*.engine"))
        if not engines:
            raise FileNotFoundError(f"no TensorRT engine written in {out}")
        out = engines[0]
    return str(out)


class TensorRTModel:
    def __init__(self, engine_path: str, background_class_id: int | None = -1, device: str = "cuda:0",
                 num_select: int | None = None, max_batch: int = 32):
        from rfdetr.export._tensorrt.inference import TRTInference

        self.trt = TRTInference(engine_path, device=device, sync_mode=True)
        self.device = device
        self.input_name = self.trt.input_names[0]
        shape = self.trt.bindings[self.input_name].shape
        self.height, self.width = int(shape[-2]), int(shape[-1])
        # the engine's largest batch (dynamic engines report -1); bigger inputs are split
        self.max_batch = int(shape[0]) if int(shape[0]) > 0 else max_batch
        self.boxes_name = next(n for n in self.trt.output_names if "dets" in n)
        self.logits_name = next(n for n in self.trt.output_names if "labels" in n)
        self.background_class_id = background_class_id
        self.num_select = num_select

    def _preprocess(self, rgb_images):
        import torch
        import torch.nn.functional as F

        x = torch.from_numpy(np.ascontiguousarray(np.stack(rgb_images))).to(self.device)
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
        x = F.interpolate(x, size=(self.height, self.width), mode="bilinear", align_corners=False, antialias=False)
        mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        return ((x - mean) / std).contiguous()

    def predict(self, rgb_images, threshold: float = 0.5):
        from rfdetr.export._runtime.decode import decode_detections

        single = not isinstance(rgb_images, list)
        imgs = [rgb_images] if single else rgb_images
        # equal-size images are run as one batch; others one by one
        groups: dict[tuple, list[int]] = {}
        for i, im in enumerate(imgs):
            groups.setdefault(im.shape[:2], []).append(i)
        res: list[sv.Detections | None] = [None] * len(imgs)
        for (h, w), all_idx in groups.items():
            for start in range(0, len(all_idx), self.max_batch):
                idx = all_idx[start:start + self.max_batch]
                outs = self.trt({self.input_name: self._preprocess([imgs[i] for i in idx])})
                boxes = outs[self.boxes_name].float().cpu().numpy()
                logits = outs[self.logits_name].float().cpu().numpy()
                for k, i in enumerate(idx):
                    d = decode_detections(boxes[k], logits[k], (w, h), threshold=threshold,
                                          num_select=self.num_select, background_class_id=self.background_class_id)
                    res[i] = sv.Detections(xyxy=d.xyxy.astype(np.float32),
                                           confidence=d.confidence.astype(np.float32),
                                           class_id=d.class_id.astype(int))
        return res[0] if single else res


def compare_detections(ref: list, test: list, iou: float = 0.5) -> dict:
    """How closely `test` reproduces `ref` (lists of sv.Detections for the same images): share of
    reference boxes found again (IoU >= iou, same class), extra boxes, and confidence difference."""
    found = total = extra = 0
    dconf = []
    for a, b in zip(ref, test):
        total += len(a)
        if len(a) == 0:
            extra += len(b)
            continue
        if len(b) == 0:
            continue
        x1 = np.maximum(a.xyxy[:, None, 0], b.xyxy[None, :, 0]); y1 = np.maximum(a.xyxy[:, None, 1], b.xyxy[None, :, 1])
        x2 = np.minimum(a.xyxy[:, None, 2], b.xyxy[None, :, 2]); y2 = np.minimum(a.xyxy[:, None, 3], b.xyxy[None, :, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area = lambda d: (d.xyxy[:, 2] - d.xyxy[:, 0]) * (d.xyxy[:, 3] - d.xyxy[:, 1])  # noqa: E731
        m = inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)
        m = m * (a.class_id[:, None] == b.class_id[None, :])
        hit = m.max(1) >= iou
        found += int(hit.sum())
        extra += int(((m.max(0) < iou)).sum())
        j = m.argmax(1)
        dconf += list(np.abs(a.confidence[hit] - b.confidence[j[hit]]))
    return {"ref_boxes": total, "found": found / max(total, 1), "extra_boxes": extra,
            "mean_conf_diff": float(np.mean(dconf)) if dconf else None}


def load_rfdetr(checkpoint: str, accel: str = "none", max_batch: int = 32,
                background_class_id: int | None = -1):
    """An rfdetr checkpoint as a model with .predict(), in the requested acceleration mode.
    TensorRT engines are built once and cached next to the checkpoint (per GPU model)."""
    from rfdetr.detr import RFDETR

    model = RFDETR.from_checkpoint(checkpoint, trust_checkpoint=True)
    accel = (accel or "none").lower()
    if accel == "none":
        return model
    if accel == "fp16":
        return accelerate_fp16(model)
    if accel == "tensorrt":
        import torch

        gpu = torch.cuda.get_device_name(0).replace(" ", "_")
        engine_dir = Path(checkpoint).with_suffix("").as_posix() + f"_trt_{gpu}"
        engines = sorted(Path(engine_dir).glob("*.trt")) + sorted(Path(engine_dir).glob("*.engine"))
        engine = str(engines[0]) if engines else build_tensorrt_engine(model, engine_dir, max_batch=max_batch)
        return TensorRTModel(engine, background_class_id=background_class_id, max_batch=max_batch)
    raise ValueError(f"unknown acceleration mode {accel!r} (none | fp16 | tensorrt)")
