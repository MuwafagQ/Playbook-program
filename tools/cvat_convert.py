"""Turn a CVAT-for-video 1.1 export into tracking truth and a Roboflow training set.

  python -m tools.cvat_convert data/cvat/cvat_annotation1.zip --clip hilal_ahli_B --out data/cvat/hilal_ahli_B
  python -m tools.cvat_convert ... --video HILAL-AHLI_match_B-up8.mp4   # also extract frames + zip

Writes:
  tracks.csv            every box with its CVAT track id, in the pipeline's CSV schema, so
                        `tools.benchmark score-gt --gt tracks.csv` can score a run on this clip.
                        notes = cvat_keyframe (drawn) or cvat_interp (interpolated by CVAT).
  training_frames.csv   frames chosen for detector training: only frames where every box
                        was drawn by hand, at least --min-gap frames apart.
  coco.json             those frames' boxes in COCO (classes named as in the Roboflow project).
  <clip>_train.zip      with --video: the frames as JPEGs + _annotations.coco.json, ready to
                        upload to Roboflow.
"""
from __future__ import annotations

import argparse
import json
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from tools.frame_sampler import CLASSES

LABEL_TO_CLASS = {"Ball": 0, "GK": 1, "Player": 2, "Referee": 3}  # pipeline class ids


def _read_xml(path: str) -> ET.Element:
    p = Path(path)
    if p.suffix == ".zip":
        with zipfile.ZipFile(p) as z:
            return ET.fromstring(z.read("annotations.xml"))
    return ET.parse(p).getroot()


def parse_boxes(path: str) -> pd.DataFrame:
    """One row per visible box: frame, track, label, class_id, x1..y2, keyframe, occluded."""
    root = _read_xml(path)
    rows = []
    for t in root.findall("track"):
        label = t.get("label")
        if label not in LABEL_TO_CLASS:
            continue
        for b in t.findall("box"):
            if b.get("outside") == "1":
                continue
            rows.append({
                "frame": int(b.get("frame")), "track": int(t.get("id")), "label": label,
                "class_id": LABEL_TO_CLASS[label],
                "x1": float(b.get("xtl")), "y1": float(b.get("ytl")),
                "x2": float(b.get("xbr")), "y2": float(b.get("ybr")),
                "keyframe": b.get("keyframe") == "1", "occluded": b.get("occluded") == "1",
            })
    return pd.DataFrame(rows).sort_values(["frame", "track"]).reset_index(drop=True)


def frame_size(path: str) -> tuple[int, int]:
    root = _read_xml(path)
    w = root.findtext(".//original_size/width")
    h = root.findtext(".//original_size/height")
    return int(w), int(h)


def to_tracks_csv(boxes: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "frame": boxes.frame, "track_id": boxes.track, "display_track_id": boxes.track,
        "class_id": boxes.class_id, "conf": 1.0,
        "x1": boxes.x1, "y1": boxes.y1, "x2": boxes.x2, "y2": boxes.y2,
        "team_id": -1, "notes": np.where(boxes.keyframe, "cvat_keyframe", "cvat_interp"),
    })
    return out


def choose_training_frames(boxes: pd.DataFrame, min_gap: int = 30, min_share: float = 1.0) -> list[int]:
    """Frames where at least min_share of the boxes were drawn by hand, spread out.

    Greedy: take eligible frames in order of hand-drawn share (then more boxes first),
    skipping any within min_gap frames of one already taken.
    """
    g = boxes.groupby("frame").agg(n=("keyframe", "size"), kf=("keyframe", "sum"))
    g["share"] = g.kf / g.n
    cand = g[g.share >= min_share].sort_values(["share", "n"], ascending=False)
    taken: list[int] = []
    for f in cand.index:
        if all(abs(int(f) - t) >= min_gap for t in taken):
            taken.append(int(f))
    return sorted(taken)


def to_coco(boxes: pd.DataFrame, frames: list[int], clip: str, size: tuple[int, int]) -> dict:
    cats = [{"id": i, "name": n, "supercategory": "none"} for i, n in enumerate(CLASSES)]
    images, anns = [], []
    for i, f in enumerate(frames):
        images.append({"id": i, "file_name": f"{clip}__f{f:07d}.jpg", "width": size[0], "height": size[1],
                       "extra": {"clip": clip, "frame": f}})
        for r in boxes[boxes.frame == f].itertuples():
            w, h = r.x2 - r.x1, r.y2 - r.y1
            anns.append({"id": len(anns), "image_id": i, "category_id": int(r.class_id),
                         "bbox": [round(r.x1, 1), round(r.y1, 1), round(w, 1), round(h, 1)],
                         "area": round(w * h, 1), "iscrowd": 0})
    return {"images": images, "annotations": anns, "categories": cats}


def extract_zip(video: str, coco: dict, out_zip: str) -> int:
    """Read the chosen frames from the source video and zip them with the COCO file."""
    import cv2

    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want = {im["extra"]["frame"]: im["file_name"] for im in coco["images"]}
    written = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in range(max(want) + 1):
            ok, img = cap.read()  # sequential read: exact frame numbers, same as CVAT's
            if not ok:
                break
            if f in want:
                ok_enc, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                z.writestr(want[f], buf.tobytes())
                written += 1
        z.writestr("_annotations.coco.json", json.dumps(coco))
    cap.release()
    if written != len(want):
        raise RuntimeError(f"video has {n} frames; wrote {written} of {len(want)} — is this the right video?")
    return written


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("export", help="CVAT for video 1.1 export (.zip or annotations.xml)")
    p.add_argument("--clip", required=True, help="short name for this clip, used in file names")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--min-gap", type=int, default=30, help="minimum frames between training frames")
    p.add_argument("--video", default=None, help="source video: also extract frames and write the upload zip")
    a = p.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    boxes = parse_boxes(a.export)
    size = frame_size(a.export)
    to_tracks_csv(boxes).to_csv(out / "tracks.csv", index=False)
    frames = choose_training_frames(boxes, min_gap=a.min_gap)
    pd.DataFrame({"frame": frames}).to_csv(out / "training_frames.csv", index=False)
    coco = to_coco(boxes, frames, a.clip, size)
    (out / "coco.json").write_text(json.dumps(coco))
    counts = boxes.groupby("label").size().to_dict()
    print(f"{a.clip}: {boxes.frame.nunique()} frames, {boxes.track.nunique()} tracks, boxes {counts}")
    print(f"  training frames (all boxes hand-drawn, >= {a.min_gap} apart): {len(frames)} -> {frames}")
    print(f"  wrote {out}/tracks.csv, training_frames.csv, coco.json")
    if a.video:
        n = extract_zip(a.video, coco, str(out / f"{a.clip}_train.zip"))
        print(f"  wrote {out}/{a.clip}_train.zip ({n} frames)")


if __name__ == "__main__":
    main()
