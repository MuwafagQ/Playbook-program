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
import json
import time
from pathlib import Path

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


def record(video: str, out_dir: str, detectors: dict, field_model, field_id: str, det_conf: float = 0.10,
           field_conf: float = 0.30, preprocess: bool = True, max_frames: int = 0, log_every: int = 50) -> dict:
    from vision.detect import infer_field_keypoints

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writers = {name: DetCacheWriter(str(Path(out_dir) / name)) for name in detectors}
    times = {name: [] for name in detectors}
    times["field"] = []
    unknown = {name: set() for name in detectors}
    f, w, h = 0, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and f >= max_frames):
            break
        h, w = frame.shape[:2]
        img = enhance_frame(frame, enabled=preprocess)
        t0 = time.perf_counter()
        kp = infer_field_keypoints(field_model, img, field_conf)
        times["field"].append(time.perf_counter() - t0)
        for name, (model, _) in detectors.items():
            t0 = time.perf_counter()
            det = sv.Detections.from_inference(model.infer(img, confidence=det_conf)[0])
            times[name].append(time.perf_counter() - t0)
            if det.data and "class_name" in det.data:
                unknown[name] |= {str(n) for n in det.data["class_name"] if str(n).lower() not in CANONICAL}
            writers[name].add_frame(f, to_canonical(det), kp)
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
    a = p.parse_args(argv)
    import os

    from inference import get_model

    key = a.api_key or os.environ["ROBOFLOW_API_KEY"]
    dets = {}
    for spec in a.detector:
        name, mid = spec.split("=", 1)
        dets[name] = (get_model(model_id=mid, api_key=key), mid)
    field = get_model(model_id=a.field, api_key=key)
    s = record(a.video, a.out, dets, field, a.field, a.det_conf, a.field_conf, not a.no_preprocess, a.max_frames)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
