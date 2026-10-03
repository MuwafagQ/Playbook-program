"""Appearance model helpers: crops, sampling, training loop, retrieval metric (tiny fake backbone)."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import torch

from tools.reid_appearance import (H, W, Embedder, crops_from_video, pk_batches, piece_means, retrieval,
                                   sample_rows, train)

COLOURS = {1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0)}  # BGR per player id


def _video(path, n=12):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
    rows = []
    for f in range(n):
        img = np.zeros((240, 320, 3), np.uint8)
        for pid, col in COLOURS.items():
            x = 20 + 90 * (pid - 1) + f
            cv2.rectangle(img, (x, 60), (x + 40, 180), col, -1)
            rows.append({"frame": f, "pid": pid, "x1": x, "y1": 60, "x2": x + 40, "y2": 180})
        vw.write(img)
    vw.release()
    return pd.DataFrame(rows)


def test_crops_follow_the_boxes(tmp_path):
    rows = _video(tmp_path / "v.avi")
    pick = sample_rows(rows, "pid", every=2)
    assert set(pick.frame) == {0, 2, 4, 6, 8, 10}
    crops, r = crops_from_video(str(tmp_path / "v.avi"), pick)
    assert crops.shape == (len(pick), H, W, 3)
    for c, pid in zip(crops, r.pid):            # RGB: player 1 red, 2 green, 3 blue
        centre = c[H // 2, W // 2].astype(int)
        assert int(np.argmax(centre)) == {1: 0, 2: 1, 3: 2}[pid]
    assert len(sample_rows(rows, "pid", per_id=3)) == 9


def test_pk_batches_and_piece_means():
    labels = np.array([0] * 10 + [1] * 2 + [2] * 6)
    for b in pk_batches(labels, p=2, k=4, steps=5):
        assert len(b) == 8 and len(set(labels[b])) == 2
    pieces, m = piece_means(np.array([[1, 0], [1, 0], [0, 2.0]]), [7, 7, 9])
    assert pieces.tolist() == [7, 9] and np.allclose(m, [[1, 0], [0, 1]])


def test_retrieval_perfect_and_team_restricted():
    ids = np.repeat([1, 2, 3], 4)
    frames = np.tile([0, 100, 200, 300], 3)
    emb = np.eye(3)[ids - 1]
    r = retrieval(emb, ids, frames, chunk=60, min_gap=60)
    assert r["rank1"] == 1.0 and r["mAP"] == 1.0 and r["queries"] == 12
    noisy = emb + 0.0
    noisy[ids == 2] = emb[ids == 1][:4]           # 1 and 2 look identical ...
    assert retrieval(noisy, ids, frames)["rank1"] < 1.0
    assert retrieval(noisy, ids, frames, groups=ids)["rank1"] == 1.0   # ... but are in different teams


def test_training_runs_and_separates_colours(tmp_path):
    rows = _video(tmp_path / "v.avi", n=20)
    crops, r = crops_from_video(str(tmp_path / "v.avi"), rows)
    backbone = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 5, stride=4), torch.nn.ReLU(),
                                   torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(8, 16))
    m = Embedder(backbone, dim=16, device="cpu")
    log = train(m, crops, r.pid.to_numpy(), steps=30, p=3, k=4, lr=1e-3, log_every=10)
    assert len(log) == 3 and all(np.isfinite(x["loss"]) for x in log)
    e = m.embed(crops)
    assert e.shape == (len(crops), 16) and np.allclose(np.linalg.norm(e, axis=1), 1, atol=1e-4)
    assert retrieval(e, r.pid.to_numpy(), r.frame.to_numpy(), chunk=5, min_gap=5)["rank1"] == 1.0
