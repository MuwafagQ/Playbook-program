"""Ball tiles: dataset cutting, tiled inference, candidate recall."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pandas as pd
import supervision as sv

from tools.ball_eval import candidate_recall
from tools.tile_dataset import grid_origins, tile_dataset
from vision.ball_model import predict_tiles

W, H = 1280, 720
BALL = (600.0, 400.0, 10.0, 10.0)  # xywh
PLAYER = (200.0, 300.0, 30.0, 80.0)


def _make_split(root, split, n=3):
    d = root / split
    d.mkdir(parents=True)
    images, anns = [], []
    for i in range(n):
        img = np.full((H, W, 3), 60, np.uint8)
        cv2.imwrite(str(d / f"f{i}.jpg"), img)
        images.append({"id": i, "file_name": f"f{i}.jpg", "width": W, "height": H})
        anns.append({"id": 2 * i, "image_id": i, "category_id": 1, "bbox": list(BALL), "area": 100, "iscrowd": 0})
        anns.append({"id": 2 * i + 1, "image_id": i, "category_id": 3, "bbox": list(PLAYER), "area": 2400, "iscrowd": 0})
    cats = [{"id": 0, "name": "football", "supercategory": "none"}, {"id": 1, "name": "ball", "supercategory": "football"},
            {"id": 2, "name": "goalkeeper", "supercategory": "football"}, {"id": 3, "name": "player", "supercategory": "football"}]
    (d / "_annotations.coco.json").write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}))


def test_tile_dataset(tmp_path):
    for s in ("train", "valid"):
        _make_split(tmp_path / "src", s)
    stats = tile_dataset(tmp_path / "src", tmp_path / "dst", tile=320, neg_per_pos=1.0)
    tr = json.loads((tmp_path / "dst/train/_annotations.coco.json").read_text())
    # one positive + one feet negative per source image
    assert stats["train"]["tiles"] == 6 and stats["train"]["tiles_with_ball"] == 3
    for a in tr["annotations"]:
        im = tr["images"][a["image_id"]]
        x, y, w, h = a["bbox"]
        assert a["category_id"] == 1 and 0 <= x and x + w <= 320 and 0 <= y and y + h <= 320
        # maps back to the original ball
        assert abs(im["x0"] + x - BALL[0]) < 1e-6 and abs(im["y0"] + y - BALL[1]) < 1e-6
    # valid: full grid; the ball lies in the overlap of two tiles in each direction at most
    va = stats["valid"]
    assert va["tiles"] == 3 * len(grid_origins(W, H, 320)) and va["tiles_with_ball"] >= 3


def test_predict_tiles_maps_back_and_merges():
    frame = np.zeros((H, W, 3), np.uint8)
    bx, by = 650.0, 410.0  # ball centre, lies in several overlapping tiles
    origins = grid_origins(W, H, 320)

    def fake(crops, conf):
        out = []
        for (x, y) in origins[fake.i:fake.i + len(crops)]:
            if x <= bx < x + 320 and y <= by < y + 320:
                out.append(sv.Detections(xyxy=np.array([[bx - x - 5, by - y - 5, bx - x + 5, by - y + 5]], np.float32),
                                         confidence=np.array([0.7], np.float32), class_id=np.array([1])))
            else:
                out.append(sv.Detections.empty())
        fake.i += len(crops)
        return out
    fake.i = 0
    det = predict_tiles(fake, frame, tile=320, batch=5)
    assert len(det) == 1
    assert np.allclose(det.xyxy[0], [bx - 5, by - 5, bx + 5, by + 5])


def test_candidate_recall():
    gt = pd.DataFrame([{"frame": f, "class_id": 0, "x1": 100, "y1": 100, "x2": 110, "y2": 110} for f in range(4)])
    c = pd.DataFrame([
        {"frame": 0, "class_id": 0, "x1": 101, "y1": 101, "x2": 111, "y2": 111, "conf": 0.9},
        {"frame": 1, "class_id": 0, "x1": 101, "y1": 101, "x2": 111, "y2": 111, "conf": 0.15},
        {"frame": 1, "class_id": 0, "x1": 500, "y1": 500, "x2": 510, "y2": 510, "conf": 0.8},
        {"frame": 2, "class_id": 0, "x1": 500, "y1": 500, "x2": 510, "y2": 510, "conf": 0.8},
    ])
    r = candidate_recall(c, gt, thresholds=(0.1, 0.3))
    assert r["0.10"]["ball_found"] == 0.5 and r["0.30"]["ball_found"] == 0.25
    assert r["0.10"]["candidates_per_frame"] == 1.0
