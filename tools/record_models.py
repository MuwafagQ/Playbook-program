"""Run only the models over a video and save their raw outputs as replayable caches.

  python -m tools.record_models --video clip.mp4 --out runs/models \
      --detector old=footballs-player-detection-zkams-zia6c/2 \
      --detector nas=players-detection-my09y-3-rfdetr-nas-t1--10cb6f \
      --field football-field-detection-f07vi-it2xv/10

Writes <out>/<name>/ for every detector, in the vision/det_cache layout, so that
`main.py --replay-cache <out>/<name> --replay-video clip.mp4` runs the whole pipeline on it
without models or a GPU. The field keypoints are computed once and shared by all caches, so
the detectors are compared on identical homographies.

This needs only the `inference` package (any version that can load the models), not the
pipeline's pinned environment: newer Roboflow models (e.g. NAS) need a newer `inference`.
Class ids are mapped by class name to the pipeline's order (0 ball, 1 goalkeeper,
2 player, 3 referee); other classes are dropped. The zoomed ball re-detection is not
recorded (replays of these caches run without it), identically for every detector.
"""
from __future__ import annotations

import argparse
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import supervision as sv

from vision.det_cache import DetCacheWriter
from vision.preprocess import enhance_frame

CANONICAL = {"ball": 0, "goalkeeper": 1, "player": 2, "referee": 3}


def to_canonical(det: sv.Detections) -> sv.Detections:
    names = det.data.get("class_name") if det.data else None
    if names is None or len(det) == 0:
        return det
    ids = np.array([CANONICAL.get(str(n).lower(), -1) for n in names], dtype=np.int32)
    keep = ids >= 0
    out = det[keep]
    out.class_id = ids[keep]
    return out


class HostedModel:
    """Roboflow's hosted (serverless) API with the `.infer(image, confidence)` interface of an
    `inference` model, so no local model or GPU is needed. Detection results come back as
    dicts (supervision parses them); keypoint results as objects, like `inference` returns."""

    def __init__(self, model_id: str, api_key: str, keypoints: bool = False,
                 url: str = "https://serverless.roboflow.com", retries: int = 4, jpeg_quality: int = 95):
        self.model_id, self.api_key, self.keypoints = model_id, api_key, keypoints
        self.url, self.retries, self.q = url.rstrip("/"), retries, jpeg_quality

    def infer(self, image, confidence: float = 0.1):
        import requests

        body = base64.b64encode(cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.q])[1]).decode()
        for attempt in range(self.retries):
            try:
                r = requests.post(f"{self.url}/{self.model_id}", params={"api_key": self.api_key, "confidence": confidence},
                                  data=body, headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=120)
                r.raise_for_status()
                j = r.json()
                break
            except Exception:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt)
        if not self.keypoints:
            return [j]
        preds = [SimpleNamespace(confidence=float(p.get("confidence", 0.0)), keypoints=[
            SimpleNamespace(x=float(k["x"]), y=float(k["y"]), confidence=float(k.get("confidence", 0.0)),
                            class_id=int(k.get("class_id", -1)), class_name=str(k.get("class", "")))
            for k in p.get("keypoints", [])]) for p in j.get("predictions", [])]
        return [SimpleNamespace(predictions=preds)]


def record(video: str, out_dir: str, detectors: dict, field_model, field_id: str, det_conf: float = 0.10,
           field_conf: float = 0.30, preprocess: bool = True, max_frames: int = 0, log_every: int = 50,
           workers: int = 1, chunk: int = 32, field_every: int = 1, ball_fn=None) -> dict:
    """workers > 1 sends several frames at once (for the hosted API); results are written in frame order.
    field_every: run the keypoint model on every Nth frame only (the pipeline holds the homography in
    between). ball_fn(image) -> sv.Detections: the ball model's candidates, saved as ballm.csv.gz."""
    from vision.detect import infer_field_keypoints

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writers = {name: DetCacheWriter(str(Path(out_dir) / name)) for name in detectors}
    times = {name: [] for name in detectors}
    times["field"] = []
    unknown = {name: set() for name in detectors}

    times["ball"] = []

    def process(item):
        fi, frame = item
        img = enhance_frame(frame, enabled=preprocess)
        tt = {}
        kp = None
        if fi % max(1, field_every) == 0:
            t0 = time.perf_counter()
            kp = infer_field_keypoints(field_model, img, field_conf)
            tt["field"] = time.perf_counter() - t0
        balls = None
        if ball_fn is not None:
            t0 = time.perf_counter()
            balls = ball_fn(img)
            tt["ball"] = time.perf_counter() - t0
        dets = {}
        for name, (model, _) in detectors.items():
            t0 = time.perf_counter()
            dets[name] = sv.Detections.from_inference(model.infer(img, confidence=det_conf)[0])
            tt[name] = time.perf_counter() - t0
        return kp, dets, balls, tt

    f, w, h, done = 0, 0, 0, False
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        while not done:
            frames = []
            while len(frames) < chunk:
                ok, frame = cap.read()
                if not ok or (max_frames and f + len(frames) >= max_frames):
                    done = True
                    break
                frames.append(frame)
            if not frames:
                break
            h, w = frames[0].shape[:2]
            for kp, dets, balls, tt in pool.map(process, list(enumerate(frames, start=f))):
                for k, v in tt.items():
                    times[k].append(v)
                for name, det in dets.items():
                    if det.data and "class_name" in det.data:
                        unknown[name] |= {str(n) for n in det.data["class_name"] if str(n).lower() not in CANONICAL}
                    writers[name].add_frame(f, to_canonical(det), kp)
                    if balls is not None:
                        writers[name].add_ballm(f, balls)
                f += 1
                if log_every and f % log_every == 0:
                    print(f"{f}/{total} frames", flush=True)
    cap.release()
    summary = {}
    for name, (_, model_id) in detectors.items():
        settings = {"PLAYER_MODEL_ID": model_id, "FIELD_MODEL_ID": field_id, "DET_CONF": det_conf,
                    "FIELD_CONF": field_conf, "DETECT_UPSCALE": 1.0, "PREPROCESS_ENABLED": preprocess,
                    "BALL_ROI_RECOVERY": False, "FAST_MODE": False}
        writers[name].close({"source_video": str(video), "start_frame": 0, "frames": f, "width": w,
                             "height": h, "fps": float(fps), "total_frames": total, "settings": settings,
                             "recorded_by": "tools.record_models"})
        # skip the first frames (model warm-up) in the timing
        t = np.array(times[name][5:] or times[name]) * 1000
        summary[name] = {"model_id": model_id, "ms_per_frame_median": float(np.median(t)) if len(t) else None,
                         "dropped_classes": sorted(unknown[name])}
    if times["ball"]:
        tb = np.array(times["ball"][5:] or times["ball"]) * 1000
        summary["ball_model"] = {"ms_per_frame_median": float(np.median(tb))}
    tf = np.array(times["field"][5:] or times["field"]) * 1000
    summary["field"] = {"model_id": field_id, "ms_per_frame_median": float(np.median(tf)) if len(tf) else None}
    summary["frames"] = f
    (Path(out_dir) / "record_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--detector", action="append", required=True, help="name=roboflow_model_id (repeatable)")
    p.add_argument("--field", default="football-field-detection-f07vi-it2xv/10")
    p.add_argument("--api-key", default=None, help="default: ROBOFLOW_API_KEY env var")
    p.add_argument("--det-conf", type=float, default=0.10)
    p.add_argument("--field-conf", type=float, default=0.30)
    p.add_argument("--no-preprocess", action="store_true")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--hosted", action="store_true", help="use Roboflow's hosted API instead of local models")
    p.add_argument("--workers", type=int, default=8, help="parallel requests with --hosted")
    a = p.parse_args(argv)
    import os

    key = a.api_key or os.environ["ROBOFLOW_API_KEY"]
    if a.hosted:
        load = lambda mid, kp=False: HostedModel(mid, key, keypoints=kp)  # noqa: E731
    else:
        from inference import get_model
        load = lambda mid, kp=False: get_model(model_id=mid, api_key=key)  # noqa: E731
    dets = {}
    for spec in a.detector:
        name, mid = spec.split("=", 1)
        dets[name] = (load(mid), mid)
    field = load(a.field, True)
    s = record(a.video, a.out, dets, field, a.field, a.det_conf, a.field_conf, not a.no_preprocess, a.max_frames,
               workers=a.workers if a.hosted else 1)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
