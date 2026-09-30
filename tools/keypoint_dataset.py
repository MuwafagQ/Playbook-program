"""Pitch keypoint dataset for training our open RF-DETR keypoint model, from a Roboflow COCO
export of the `football-field-detection` project.

  python -m tools.keypoint_dataset --src data/field_v11 --dst data/field_v11_split

1. Splits by match, not by Roboflow's random split, so near-duplicate frames cannot leak into
   validation: frames of match_video_11 / match_video_12 -> valid, HILAL-HAZM frames -> test,
   everything else (older frames included) -> train. Until frames of matches 11 / 12 are labelled
   (none in the export), Roboflow's own split is kept instead, so there still is a validation set.
2. Fixes the keypoint order to pitch vertex labels "1".."32" (matched by keypoint name), so the
   trained model's keypoint index k is vertex label k + 1 (vision/field_model.py). Slots Roboflow
   adds beyond the 32 labels are dropped.
3. Keeps only annotations with at least MIN_POINTS visible points.
Images are symlinked, not copied.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

ANN = "_annotations.coco.json"
LABELS = [str(i) for i in range(1, 33)]
VALID = ("match_video_11_", "match_video_12_")
TEST = ("hilal_hazm",)
MIN_POINTS = 4
# horizontal image flip == mirroring the pitch along its length: label pairs that swap. The pitch config
# lists its vertices in Roboflow's order ("01".."13", "15".."18", "20".."32", "14", "19"), so the pairs
# are looked up by label, not by index (an index-based version once taught the model wrong labels).
# 15-18, the halfway-line points, map to themselves.
MIRROR = [(1, 27), (2, 28), (3, 29), (4, 30), (5, 31), (6, 32), (7, 25), (8, 26), (9, 24),
          (10, 20), (11, 21), (12, 22), (13, 23), (14, 19)]
FLIP_PAIRS = [i for a, b in MIRROR for i in (a - 1, b - 1)]  # flat 0-based over labels 1..32 (our dataset order)


def split_of(file_name: str) -> str:
    stem = file_name.lower()
    if stem.startswith(VALID):
        return "valid"
    if stem.startswith(TEST):
        return "test"
    return "train"


def _label_number(name) -> int | None:
    try:
        return int(str(name).strip())  # Roboflow names points "01".."09", "10".."32"
    except ValueError:
        return None


def _reorder(kps: list, names: list) -> list:
    pos = {_label_number(n): i for i, n in enumerate(names)}
    out = []
    for lab in LABELS:
        i = pos.get(int(lab))
        out += [0, 0, 0] if i is None or 3 * i + 2 >= len(kps) else [float(kps[3 * i]), float(kps[3 * i + 1]),
                                                                      int(kps[3 * i + 2])]
    return out


def keypoint_dataset(src_root, dst_root, by_match: bool | None = None) -> dict:
    """by_match: split by match name (see above); None = only if the export has frames of matches 11 / 12."""
    src, dst = Path(src_root), Path(dst_root)
    if by_match is None:
        by_match = any(im["file_name"].lower().startswith(VALID)
                       for p in src.iterdir() if (p / ANN).exists()
                       for im in json.loads((p / ANN).read_text())["images"])
    out = {s: {"images": [], "annotations": []} for s in ("train", "valid", "test")}
    categories = None
    for split_dir in sorted(p for p in src.iterdir() if (p / ANN).exists()):
        coco = json.loads((split_dir / ANN).read_text())
        kp_cats = {c["id"]: c for c in coco["categories"] if c.get("keypoints")}
        if categories is None:
            categories = [dict(c, keypoints=LABELS, skeleton=[]) if c["id"] in kp_cats else c for c in coco["categories"]]
        anns = {}
        for a in coco["annotations"]:
            if a["category_id"] in kp_cats:
                anns.setdefault(a["image_id"], []).append(a)
        for im in coco["images"]:
            kept = []
            for a in anns.get(im["id"], []):
                kps = _reorder(a.get("keypoints", []), kp_cats[a["category_id"]]["keypoints"])
                n = sum(1 for i in range(2, 96, 3) if kps[i] > 0)
                if n >= MIN_POINTS:
                    kept.append({**a, "keypoints": kps, "num_keypoints": n})
            if not kept:
                continue
            split = split_of(im["file_name"]) if by_match else split_dir.name
            s = out[split]
            new_id = len(s["images"])
            s["images"].append({**im, "id": new_id})
            for a in kept:
                s["annotations"].append({**a, "id": len(s["annotations"]) + 1, "image_id": new_id})  # ids from 1: rfdetr uses 0 as "unmatched"
            d = dst / split
            d.mkdir(parents=True, exist_ok=True)
            link = d / im["file_name"]
            if not link.exists():
                os.symlink((split_dir / im["file_name"]).resolve(), link)
    stats = {}
    for s, data in out.items():
        (dst / s).mkdir(parents=True, exist_ok=True)
        (dst / s / ANN).write_text(json.dumps({"images": data["images"], "annotations": data["annotations"],
                                               "categories": categories or []}))
        names = [im["file_name"] for im in data["images"]]
        stats[s] = {"images": len(names), "our_matches": sum(bool(re.match(r"(match_video_|hilal_hazm)", n)) for n in names)}
    stats["split_by"] = "match" if by_match else "roboflow"
    return stats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    a = p.parse_args(argv)
    print(json.dumps(keypoint_dataset(a.src, a.dst), indent=1))


if __name__ == "__main__":
    main()
