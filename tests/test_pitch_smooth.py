"""tools/pitch_smooth.py on a synthetic panning camera with jittery homographies and a cut."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from tools import pitch_smooth as ps

W, H = 1920, 1080
BASE = np.array([[6.0, 0.8, 0.0], [0.0, 9.0, -2000.0], [0.0, 0.0012, 1.0]])  # image -> pitch (cm)


def _true_h(f):
    """Camera pans smoothly for 60 frames, then a hard cut to a different view."""
    shift = np.eye(3)
    shift[0, 2] = 15.0 * f if f < 60 else 3000.0 + 10.0 * (f - 60)
    return shift @ BASE


def _noisy(Ht, rng, px=6.0):
    """What a per-frame fit from noisy keypoints gives: perturb 4 image points, refit."""
    src = np.array([[300, 500], [1600, 500], [1700, 1000], [200, 1000]], np.float64)
    dst = cv2.perspectiveTransform(src.reshape(-1, 1, 2), Ht).reshape(-1, 2)
    return cv2.findHomography((src + rng.normal(0, px, src.shape)).astype(np.float32), dst.astype(np.float32))[0]


def test_smoothing_cuts_jitter_and_keeps_the_cut():
    rng = np.random.default_rng(0)
    frames = range(120)
    Hs = {f: _noisy(_true_h(f), rng) for f in frames}
    sm, n_cuts = ps.smooth_homographies(Hs, W, H)
    assert n_cuts == 1

    foot = np.array([[[960.0, 800.0]]])
    true = np.array([cv2.perspectiveTransform(foot, _true_h(f))[0, 0] for f in frames])
    raw = np.array([cv2.perspectiveTransform(foot, Hs[f])[0, 0] for f in frames])
    smo = np.array([cv2.perspectiveTransform(foot, sm[f])[0, 0] for f in frames])

    step = lambda p: np.median(np.linalg.norm(np.diff(p - true, axis=0), axis=1))  # noqa: E731
    assert step(smo) < 0.35 * step(raw)                     # frame-to-frame noise mostly gone
    err_raw = np.median(np.linalg.norm(raw - true, axis=1))
    err_smo = np.median(np.linalg.norm(smo - true, axis=1))
    assert err_smo < err_raw                                # and closer to the truth
    # no blending across the cut: frames right after it are as accurate as elsewhere
    assert np.linalg.norm(smo[60] - true[60]) < 3 * err_raw
    assert np.linalg.norm(smo[59] - true[59]) < 3 * err_raw


def test_smooth_tracks_respects_gaps_and_ids():
    rows = []
    for f in range(40):
        if 15 <= f < 20:
            continue  # the person is missing for 5 frames
        rows.append({"frame": f, "track_id": 7, "class_id": 2, "x_m": 100.0 * f + (30 if f % 2 else -30), "y_m": 500.0})
        rows.append({"frame": f, "track_id": -1, "class_id": 0, "x_m": 1.0 * f, "y_m": 1.0})  # ball: untouched
    df = pd.DataFrame(rows)
    out = ps.smooth_tracks(df, window=5)
    p = out[out.track_id == 7].set_index("frame").x_m
    wobble_before = np.abs(np.diff(df[df.track_id == 7].x_m.to_numpy(), 2)).mean()
    wobble_after = np.abs(np.diff(p.loc[25:35].to_numpy(), 2)).mean()
    assert wobble_after < 0.3 * wobble_before
    ball = out[out.track_id == -1]
    assert (ball.x_m.to_numpy() == df[df.track_id == -1].x_m.to_numpy()).all()
