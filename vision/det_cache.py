"""Record model outputs once, replay tracking many times.

The detector and keypoint models are the expensive part of a run. Everything after
them (tracking, ID stabilisation, team colour, homography, projection) only needs
their outputs plus the video frames. Recording those outputs lets tracking
experiments be replayed without the models or a GPU.

Layout of a cache directory:
  meta.json        source video, window, frame size, fps, detection settings
  dets.csv.gz      frame, x1, y1, x2, y2, conf, class_id   (full-frame detector output)
  roi.csv.gz       same columns                              (zoomed ball re-detections)
  kp.csv.gz        frame, label, x, y, conf                  (field keypoints)
  frames.csv.gz    frame, det, kp                            (which model ran on which frame)
"""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

DET_COLS = ["frame", "x1", "y1", "x2", "y2", "conf", "class_id"]
KP_COLS = ["frame", "label", "x", "y", "conf"]

# Settings that change what the models return; a replay under different values
# would not reproduce what the models would have produced.
DETECTION_SETTINGS = (
    "PLAYER_MODEL_ID", "FIELD_MODEL_ID", "DET_CONF", "FIELD_CONF", "DETECT_UPSCALE",
    "PREPROCESS_ENABLED", "BALL_ROI_RECOVERY", "BALL_ROI_PX", "BALL_ROI_UPSCALE", "BALL_ROI_CONF",
    "FAST_MODE", "DETECT_EVERY_N", "HOMOGRAPHY_EVERY_N",
)


def _empty_dets() -> sv.Detections:
    return sv.Detections(
        xyxy=np.zeros((0, 4), dtype=np.float32),
        confidence=np.zeros((0,), dtype=np.float32),
        class_id=np.zeros((0,), dtype=np.int32),
    )


def _empty_kp() -> sv.KeyPoints:
    return sv.KeyPoints(xy=np.zeros((1, 0, 2), dtype=np.float32))


class DetCacheWriter:
    def __init__(self, cache_dir: str):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._files = {}
        self._writers = {}
        for name, cols in (("dets", DET_COLS), ("roi", DET_COLS), ("kp", KP_COLS), ("frames", ["frame", "det", "kp"])):
            f = gzip.open(self.dir / f"{name}.csv.gz", "wt", newline="")
            w = csv.writer(f)
            w.writerow(cols)
            self._files[name] = f
            self._writers[name] = w

    def _write_dets(self, name: str, frame: int, det: sv.Detections) -> None:
        if det is None or len(det) == 0:
            return
        conf = det.confidence if det.confidence is not None else np.ones(len(det), dtype=np.float32)
        cls = det.class_id if det.class_id is not None else np.zeros(len(det), dtype=np.int32)
        w = self._writers[name]
        for (x1, y1, x2, y2), c, k in zip(det.xyxy.tolist(), conf.tolist(), cls.tolist()):
            w.writerow([frame, f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}", f"{c:.4f}", int(k)])

    def add_frame(self, frame: int, det: sv.Detections | None, kp: sv.KeyPoints | None) -> None:
        self._writers["frames"].writerow([frame, int(det is not None), int(kp is not None)])
        if det is not None:
            self._write_dets("dets", frame, det)
        if kp is not None and kp.xy is not None and kp.xy.shape[1] > 0:
            labels = np.asarray(getattr(kp, "class_id", np.full((1, kp.xy.shape[1]), -1))).reshape(-1)
            conf = np.asarray(getattr(kp, "confidence", np.ones((1, kp.xy.shape[1])))).reshape(-1)
            w = self._writers["kp"]
            for (x, y), lab, c in zip(kp.xy[0].tolist(), labels.tolist(), conf.tolist()):
                w.writerow([frame, int(lab), f"{x:.2f}", f"{y:.2f}", f"{c:.4f}"])

    def add_roi(self, frame: int, det: sv.Detections) -> None:
        self._write_dets("roi", frame, det)

    def close(self, meta: dict) -> None:
        for f in self._files.values():
            f.close()
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2))


class DetCacheReader:
    def __init__(self, cache_dir: str):
        import pandas as pd

        self.dir = Path(cache_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self._dets = self._group(pd.read_csv(self.dir / "dets.csv.gz"))
        self._roi = self._group(pd.read_csv(self.dir / "roi.csv.gz"))
        kp = pd.read_csv(self.dir / "kp.csv.gz")
        self._kp = {int(f): g for f, g in kp.groupby("frame")}
        fr = pd.read_csv(self.dir / "frames.csv.gz")
        self.det_frames = set(fr.loc[fr.det == 1, "frame"].astype(int))
        self.kp_frames = set(fr.loc[fr.kp == 1, "frame"].astype(int))

    @staticmethod
    def _group(df) -> dict[int, np.ndarray]:
        return {int(f): g[["x1", "y1", "x2", "y2", "conf", "class_id"]].to_numpy(np.float64)
                for f, g in df.groupby("frame")}

    @staticmethod
    def _to_dets(arr: np.ndarray | None) -> sv.Detections:
        if arr is None or len(arr) == 0:
            return _empty_dets()
        return sv.Detections(
            xyxy=arr[:, 0:4].astype(np.float32),
            confidence=arr[:, 4].astype(np.float32),
            class_id=arr[:, 5].astype(np.int32),
        )

    def dets(self, frame: int) -> sv.Detections:
        return self._to_dets(self._dets.get(int(frame)))

    def roi(self, frame: int) -> sv.Detections:
        return self._to_dets(self._roi.get(int(frame)))

    def kp(self, frame: int) -> sv.KeyPoints:
        g = self._kp.get(int(frame))
        if g is None or len(g) == 0:
            return _empty_kp()
        kp = sv.KeyPoints(xy=g[["x", "y"]].to_numpy(np.float32).reshape(1, -1, 2))
        kp.class_id = g["label"].to_numpy(np.int32).reshape(1, -1)
        kp.confidence = g["conf"].to_numpy(np.float32).reshape(1, -1)
        return kp

    def settings_mismatch(self, settings) -> list[str]:
        rec = self.meta.get("settings", {})
        out = []
        for k in DETECTION_SETTINGS:
            if k in rec and str(getattr(settings, k, None)) != str(rec[k]):
                out.append(f"{k}: recorded {rec[k]!r}, now {getattr(settings, k, None)!r}")
        return out


def iter_frames(video_path: str, start_frame: int, count: int, out_size: tuple[int, int] | None = None):
    """Yield `count` frames starting at absolute `start_frame` (exact: grab() to skip).

    If `out_size` (w, h) is given and differs from the video, frames are resized to it,
    so a reduced-resolution proxy clip can stand in for the full-resolution source.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        for _ in range(int(start_frame)):
            if not cap.grab():
                return
        for _ in range(int(count)):
            ok, frame = cap.read()
            if not ok:
                return
            if out_size is not None and (frame.shape[1], frame.shape[0]) != tuple(out_size):
                frame = cv2.resize(frame, tuple(out_size), interpolation=cv2.INTER_LINEAR)
            yield frame
    finally:
        cap.release()
