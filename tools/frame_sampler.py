"""Choose frames worth annotating from a folder of match videos.

Stages (the Colab notebook notebooks/sample_frames_for_annotation.ipynb runs them):
  1. scan_video      sample one frame every N seconds as a thumbnail; measure how much
                     of it is pitch (drops close-ups, crowd shots, graphics) and sharpness
  2. select_diverse  per-match quotas, minimum time gap, and farthest-point sampling in
                     an appearance feature space, so frames differ from each other
  3. hardness        score detector predictions: uncertain boxes, small players, crowding
  4. export_coco     full-resolution frames + detector boxes as COCO pre-labels, one zip
                     per split, with whole matches held out for validation
"""
from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["ball", "goalkeeper", "player", "referee"]  # class_id order of the detector


@dataclass
class Candidate:
    match: str
    video: str
    frame: int
    time_s: float
    green: float
    sharpness: float
    feature: np.ndarray = field(repr=False)
    thumb: np.ndarray = field(repr=False)
    hardness: float = 0.0
    detections: list = field(default_factory=list, repr=False)  # [(x1,y1,x2,y2,conf,class_id)]


def read_frame(video: str, frame: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(video)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ok, img = cap.read()
        return img if ok else None
    finally:
        cap.release()


def pitch_green_ratio(img: np.ndarray) -> float:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return float(((h >= 30) & (h <= 90) & (s >= 40) & (v >= 40)).mean())


def sharpness(img: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def appearance_feature(img: np.ndarray) -> np.ndarray:
    """Colour layout + colour distribution: separates stadiums, kits, lighting and zoom."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    hist /= max(hist.sum(), 1e-9)
    layout = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2LAB), (16, 9), interpolation=cv2.INTER_AREA)
    layout = layout.astype(np.float32).flatten() / 255.0
    layout -= layout.mean()
    layout /= max(np.linalg.norm(layout), 1e-9)
    return np.concatenate([np.sqrt(hist), 0.5 * layout]).astype(np.float32)


def scan_video(video: str, match: str, every_s: float = 1.0, thumb_width: int = 320,
               skip_start_s: float = 0.0) -> list[Candidate]:
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(every_s * fps)))
    out = []
    try:
        for f in range(int(skip_start_s * fps), n, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                break
            h, w = img.shape[:2]
            thumb = cv2.resize(img, (thumb_width, int(round(h * thumb_width / w))), interpolation=cv2.INTER_AREA)
            out.append(Candidate(
                match=match, video=video, frame=f, time_s=f / fps,
                green=pitch_green_ratio(thumb), sharpness=sharpness(thumb),
                feature=appearance_feature(thumb), thumb=thumb,
            ))
    finally:
        cap.release()
    return out


def filter_broadcast(cands: list[Candidate], min_green: float = 0.35,
                     min_sharp_pct: float = 15.0) -> list[Candidate]:
    """Keep game-view shots: enough pitch visible and not among the blurriest frames."""
    keep = [c for c in cands if c.green >= min_green]
    if not keep:
        return []
    thr = np.percentile([c.sharpness for c in keep], min_sharp_pct)
    return [c for c in keep if c.sharpness >= thr]


def select_diverse(cands: list[Candidate], per_match: int, min_gap_s: float = 3.0,
                   hardness_weight: float = 0.0, already: list[Candidate] | None = None) -> list[Candidate]:
    """Farthest-point sampling with a per-match quota and a minimum time gap.

    Each pick maximises distance to everything already chosen (across all matches, so a
    match that looks like one already covered contributes less), multiplied by
    (1 + hardness_weight * hardness) when detector hardness is known.
    """
    chosen: list[Candidate] = list(already or [])
    picked: list[Candidate] = []
    quota = {m: per_match for m in {c.match for c in cands}}
    pool = list(cands)
    feats = np.stack([c.feature for c in pool]) if pool else np.zeros((0, 1))
    min_d = np.full(len(pool), np.inf)
    for c in chosen:
        min_d = np.minimum(min_d, np.linalg.norm(feats - c.feature, axis=1))
    alive = np.ones(len(pool), dtype=bool)
    while alive.any() and any(q > 0 for q in quota.values()):
        base = np.where(np.isfinite(min_d), min_d, 1e6)
        score = base * (1.0 + hardness_weight * np.array([c.hardness for c in pool]))
        score[~alive] = -np.inf
        i = int(np.argmax(score))
        if not np.isfinite(score[i]):
            break
        c = pool[i]
        alive[i] = False
        if quota[c.match] <= 0:
            continue
        picked.append(c)
        quota[c.match] -= 1
        if quota[c.match] <= 0:
            alive &= np.array([p.match != c.match for p in pool])
        for j, p in enumerate(pool):
            if alive[j] and p.match == c.match and abs(p.time_s - c.time_s) < min_gap_s:
                alive[j] = False
        min_d = np.minimum(min_d, np.linalg.norm(feats - c.feature, axis=1))
    return picked


def hardness(detections: list, frame_h: int, uncertain=(0.15, 0.5), small_frac: float = 0.045) -> float:
    """How much a frame can teach the detector.

    Counts people boxes the model is unsure about, small (distant) players, and
    overlapping pairs. Roughly 0 for easy frames, 1+ for hard ones.
    """
    people = [d for d in detections if int(d[5]) != 0]
    if not people:
        return 0.0
    conf = np.array([d[4] for d in people])
    boxes = np.array([d[:4] for d in people], dtype=np.float32)
    n_unsure = int(((conf >= uncertain[0]) & (conf < uncertain[1])).sum())
    n_small = int(((boxes[:, 3] - boxes[:, 1]) < small_frac * frame_h).sum())
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0]); y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2]); y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    iou = inter / (area[:, None] + area[None, :] - inter + 1e-9)
    n_overlap = int((np.triu(iou, 1) > 0.2).sum())
    return 0.15 * n_unsure + 0.05 * n_small + 0.10 * n_overlap


def split_matches(matches: list[str], valid_frac: float = 0.2, seed: int = 0,
                  valid_matches: list[str] | None = None) -> dict[str, str]:
    """Whole matches go to one split. At least one validation match when there are 2+.

    valid_matches: hold out exactly these (e.g. night games, the target condition)
    instead of a random choice.
    """
    ms = sorted(set(matches))
    if valid_matches:
        unknown = sorted(set(valid_matches) - set(ms))
        if unknown:
            raise ValueError(f"validation matches not found: {unknown}; known: {ms}")
        return {m: ("valid" if m in set(valid_matches) else "train") for m in ms}
    rng = np.random.default_rng(seed)
    rng.shuffle(ms)
    n_valid = 0 if len(ms) < 2 else max(1, int(round(valid_frac * len(ms))))
    return {m: ("valid" if i < n_valid else "train") for i, m in enumerate(ms)}


def export_coco(picks: list[Candidate], out_dir: str, splits: dict[str, str],
                min_label_conf: float = 0.3) -> dict[str, str]:
    """Write <split>.zip, each with the full-resolution frames and _annotations.coco.json.

    The zip layout is what Roboflow's uploader recognises as COCO. Boxes below
    min_label_conf are left out: pre-labels should be mostly right, and the missed
    players are what the annotator adds.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cats = [{"id": i, "name": n, "supercategory": "none"} for i, n in enumerate(CLASSES)]
    per_split: dict[str, dict] = {}
    for c in picks:
        split = splits.get(c.match, "train")
        d = per_split.setdefault(split, {"images": [], "annotations": [], "categories": cats, "files": []})
        img = read_frame(c.video, c.frame)
        if img is None:
            continue
        name = f"{c.match}__f{c.frame:07d}.jpg"
        path = out / split / name
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        image_id = len(d["images"])
        d["images"].append({"id": image_id, "file_name": name, "width": img.shape[1], "height": img.shape[0],
                            "extra": {"match": c.match, "frame": c.frame, "time_s": round(c.time_s, 2),
                                      "hardness": round(c.hardness, 3)}})
        for x1, y1, x2, y2, conf, cls in c.detections:
            if conf < min_label_conf:
                continue
            d["annotations"].append({
                "id": len(d["annotations"]), "image_id": image_id, "category_id": int(cls),
                "bbox": [round(float(x1), 1), round(float(y1), 1), round(float(x2 - x1), 1), round(float(y2 - y1), 1)],
                "area": round(float((x2 - x1) * (y2 - y1)), 1), "iscrowd": 0,
            })
        d["files"].append(path)
    zips = {}
    for split, d in per_split.items():
        zp = out / f"{split}.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for p in d["files"]:
                z.write(p, arcname=p.name)
            coco = {k: v for k, v in d.items() if k != "files"}
            z.writestr("_annotations.coco.json", json.dumps(coco))
        zips[split] = str(zp)
    return zips


def contact_sheet(picks: list[Candidate], path: str, cols: int = 6) -> None:
    """Grid of the chosen thumbnails, labelled match / time / hardness, for a quick review."""
    if not picks:
        return
    tw = max(c.thumb.shape[1] for c in picks)
    th = max(c.thumb.shape[0] for c in picks)
    rows = (len(picks) + cols - 1) // cols
    sheet = np.full((rows * (th + 22), cols * tw, 3), 255, np.uint8)
    for i, c in enumerate(picks):
        r, k = divmod(i, cols)
        y, x = r * (th + 22), k * tw
        sheet[y:y + c.thumb.shape[0], x:x + c.thumb.shape[1]] = c.thumb
        label = f"{c.match[:18]} {int(c.time_s // 60)}:{c.time_s % 60:04.1f} h={c.hardness:.2f}"
        cv2.putText(sheet, label, (x + 3, y + th + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(path, sheet)
