"""Cut a COCO detection dataset (Roboflow export layout) into square ball tiles.

  python -m tools.tile_dataset --src data/v3 --dst data/v3_ball_tiles --tile 320

Our ball is ~9 px wide in a 1920x1080 frame; a detector that resizes the whole frame to
~640 px sees 3 px and misses most balls. A ball model trained on small crops, run on crops
at inference time, sees the ball at full (or upscaled) resolution.

Input: <src>/{train,valid,test}/_annotations.coco.json plus the images (Roboflow "coco"
export). Output: the same layout with a single class, "ball".
  train   one tile per ball, the ball at a random position inside it (so the model does
          not learn "the ball is in the middle"), plus `neg_per_pos` tiles without a ball,
          centred on a random person's feet: boots are what the ball gets confused with.
  valid / test   every tile of a regular overlapping grid, the way the model is run over
          a full frame, so their scores reflect real use (most grid tiles hold no ball).
A ball cut by a tile border is kept (clipped) if at least `min_vis` of it is inside.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ANN = "_annotations.coco.json"
PEOPLE = ("player", "goalkeeper", "referee")
OUT_CATEGORIES = [{"id": 0, "name": "balls", "supercategory": "none"},
                  {"id": 1, "name": "ball", "supercategory": "balls"}]


def grid_origins(width: int, height: int, tile: int, min_overlap: int = 64) -> list[tuple[int, int]]:
    """Top-left corners of square tiles covering the frame with at least min_overlap px overlap."""
    def starts(n_px):
        if n_px <= tile:
            return [0]
        k = int(np.ceil((n_px - min_overlap) / (tile - min_overlap)))
        return [int(round(v)) for v in np.linspace(0, n_px - tile, k)]
    return [(x, y) for y in starts(height) for x in starts(width)]


def _clip(box, x0: int, y0: int, tile: int, min_vis: float):
    """COCO xywh box -> xywh inside the tile, or None if less than min_vis of it is inside."""
    x, y, w, h = box
    cx1, cy1 = max(x, x0), max(y, y0)
    cx2, cy2 = min(x + w, x0 + tile), min(y + h, y0 + tile)
    if cx2 <= cx1 or cy2 <= cy1 or w <= 0 or h <= 0:
        return None
    if (cx2 - cx1) * (cy2 - cy1) < min_vis * w * h:
        return None
    return [cx1 - x0, cy1 - y0, cx2 - cx1, cy2 - cy1]


def _inside(box, x0: int, y0: int, tile: int) -> bool:
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    return x0 <= cx < x0 + tile and y0 <= cy < y0 + tile


def _positive_origin(box, w: int, h: int, tile: int, rng, margin: int = 16) -> tuple[int, int]:
    """A tile that contains the ball centre at a random position (at least `margin` from the edge)."""
    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
    lo_x, hi_x = cx - tile + margin, cx - margin
    lo_y, hi_y = cy - tile + margin, cy - margin
    x0 = int(np.clip(rng.uniform(lo_x, hi_x), 0, max(w - tile, 0)))
    y0 = int(np.clip(rng.uniform(lo_y, hi_y), 0, max(h - tile, 0)))
    return x0, y0


def _feet_origin(box, w: int, h: int, tile: int, rng) -> tuple[int, int]:
    fx, fy = box[0] + box[2] / 2, box[1] + box[3]
    x0 = int(np.clip(fx - tile / 2 + rng.uniform(-tile / 4, tile / 4), 0, max(w - tile, 0)))
    y0 = int(np.clip(fy - tile / 2 + rng.uniform(-tile / 4, tile / 4), 0, max(h - tile, 0)))
    return x0, y0


def tile_split(src_dir, dst_dir, tile: int = 320, mode: str = "train", neg_per_pos: float = 1.0,
               min_vis: float = 0.5, seed: int = 0, grid_overlap: int = 64) -> dict:
    src, dst = Path(src_dir), Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    coco = json.loads((src / ANN).read_text())
    names = {c["id"]: c["name"].lower() for c in coco["categories"]}
    ball_ids = {i for i, n in names.items() if n == "ball"}
    people_ids = {i for i, n in names.items() if n in PEOPLE}
    by_img: dict[int, list] = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    rng = np.random.default_rng(seed)
    images, anns = [], []
    stats = {"source_images": 0, "tiles": 0, "tiles_with_ball": 0, "balls": 0}

    for im in coco["images"]:
        img = cv2.imread(str(src / im["file_name"]))
        if img is None:
            continue
        stats["source_images"] += 1
        h, w = img.shape[:2]
        a_im = by_img.get(im["id"], [])
        balls = [a["bbox"] for a in a_im if a["category_id"] in ball_ids]
        people = [a["bbox"] for a in a_im if a["category_id"] in people_ids]
        if mode == "train":
            origins = [_positive_origin(b, w, h, tile, rng) for b in balls]
            n_neg = int(round(neg_per_pos * max(len(balls), 1)))
            for _ in range(n_neg * 5):  # a few tries: a feet tile must not contain a ball
                if n_neg == 0 or not people:
                    break
                o = _feet_origin(people[rng.integers(len(people))], w, h, tile, rng)
                if not any(_inside(b, o[0], o[1], tile) for b in balls):
                    origins.append(o)
                    n_neg -= 1
        else:
            origins = grid_origins(w, h, tile, grid_overlap)

        stem = Path(im["file_name"]).stem
        for x0, y0 in origins:
            tid = len(images)
            name = f"{stem}_x{x0}_y{y0}.jpg"
            crop = img[y0:y0 + tile, x0:x0 + tile]
            if crop.shape[0] != tile or crop.shape[1] != tile:  # frame smaller than a tile: pad
                crop = cv2.copyMakeBorder(crop, 0, tile - crop.shape[0], 0, tile - crop.shape[1],
                                          cv2.BORDER_CONSTANT, value=0)
            cv2.imwrite(str(dst / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            images.append({"id": tid, "file_name": name, "width": tile, "height": tile,
                           "source": im["file_name"], "x0": x0, "y0": y0})
            n_here = 0
            for b in balls:
                c = _clip(b, x0, y0, tile, min_vis)
                if c is not None:
                    anns.append({"id": len(anns), "image_id": tid, "category_id": 1, "bbox": c,
                                 "area": c[2] * c[3], "iscrowd": 0})
                    n_here += 1
            stats["tiles"] += 1
            stats["tiles_with_ball"] += int(n_here > 0)
            stats["balls"] += n_here

    (dst / ANN).write_text(json.dumps({"images": images, "annotations": anns, "categories": OUT_CATEGORIES}))
    return stats


def tile_dataset(src_root, dst_root, tile: int = 320, neg_per_pos: float = 1.0, seed: int = 0) -> dict:
    out = {}
    for split in ("train", "valid", "test"):
        if (Path(src_root) / split / ANN).exists():
            out[split] = tile_split(Path(src_root) / split, Path(dst_root) / split, tile,
                                    "train" if split == "train" else "grid", neg_per_pos, seed=seed)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--tile", type=int, default=320)
    p.add_argument("--neg-per-pos", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    print(json.dumps(tile_dataset(a.src, a.dst, a.tile, a.neg_per_pos, a.seed), indent=2))


if __name__ == "__main__":
    main()
