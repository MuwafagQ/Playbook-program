"""Tiled ball-candidate detection (vision/detect.infer_ball_tiles)."""
from __future__ import annotations

import numpy as np

from vision.detect import infer_ball_tiles, tile_origins

BALL_XY = (650.0, 420.0)  # inside two overlapping tiles: (600,0) and (600,360)


class FakeModel:
    """Returns the ball wherever a tile contains it, plus a player; accepts a batch."""

    def __init__(self, origins):
        self.origins = list(origins)
        self.calls = 0

    def infer(self, images, confidence=0.1):
        self.calls += 1
        out = []
        for (x0, y0), img in zip(self.origins, images):
            preds = [{"x": 100.0, "y": 100.0, "width": 30.0, "height": 80.0, "confidence": 0.9,
                      "class_id": 2, "class": "player"}]
            bx, by = BALL_XY[0] - x0, BALL_XY[1] - y0
            if 0 <= bx < img.shape[1] and 0 <= by < img.shape[0]:
                preds.append({"x": bx, "y": by, "width": 10.0, "height": 10.0, "confidence": 0.6,
                              "class_id": 0, "class": "ball"})
            out.append({"image": {"width": img.shape[1], "height": img.shape[0]}, "predictions": preds})
        return out


def test_tiles_cover_frame_and_merge_duplicates():
    origins = tile_origins(1920, 1080)
    assert len(origins) == 6
    covered = np.zeros((1080, 1920), bool)
    for x, y in origins:
        covered[y:y + 720, x:x + 720] = True
    assert covered.all()

    model = FakeModel(origins)
    det = infer_ball_tiles(model, np.zeros((1080, 1920, 3), np.uint8))
    assert model.calls == 1                       # one batched call for all tiles
    assert len(det) == 1                           # ball seen in 2 tiles, merged; players dropped
    cx, cy = (det.xyxy[0, 0] + det.xyxy[0, 2]) / 2, (det.xyxy[0, 1] + det.xyxy[0, 3]) / 2
    assert abs(cx - BALL_XY[0]) < 1e-3 and abs(cy - BALL_XY[1]) < 1e-3
