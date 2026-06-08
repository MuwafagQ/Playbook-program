"""Unit tests for the manual-homography keyframe-anchor core (geometry/manual_h.py)."""
from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np

from geometry.manual_h import (
    ManualHGuide,
    compute_homography,
    densify,
    load_anchors,
    load_guide,
    save_anchors,
    save_dense,
)

IMG = np.array([[100, 100], [500, 120], [480, 400], [120, 380]], float)
PITCH = np.array([[0, 0], [12000, 0], [12000, 7000], [0, 7000]], float)


def test_compute_homography_4pt_roundtrip():
    H = compute_homography(IMG, PITCH)
    proj = cv2.perspectiveTransform(
        IMG.reshape(-1, 1, 2).astype(np.float32), H.astype(np.float32)
    ).reshape(-1, 2)
    assert np.allclose(proj, PITCH, atol=1e-3)


def test_compute_homography_rejects_too_few():
    assert compute_homography(IMG[:3], PITCH[:3]) is None


def _true_H_chain(H0, M, t):
    Ht = H0.copy()
    for _ in range(t):
        Ht = Ht @ M
    return Ht / Ht[2, 2]


def test_densify_exact_chain():
    H0 = compute_homography(IMG, PITCH)
    M = np.eye(3)
    M[0, 2], M[1, 2] = -3.0, -1.0
    a, b = 0, 20
    anchor_H = {a: _true_H_chain(H0, M, a), b: _true_H_chain(H0, M, b)}
    step_M = {t: M for t in range(a + 1, b + 1)}
    dense = densify(anchor_H, step_M)
    assert set(dense.keys()) == set(range(a, b + 1))
    for t in range(a, b + 1):
        assert np.abs(dense[t] - _true_H_chain(H0, M, t)).max() < 1e-6


def test_densify_broken_link_is_partial_not_fatal():
    H0 = compute_homography(IMG, PITCH)
    M = np.eye(3)
    M[0, 2] = -3.0
    a, b = 0, 20
    anchor_H = {a: _true_H_chain(H0, M, a), b: _true_H_chain(H0, M, b)}
    step_M = {t: M for t in range(a + 1, b + 1)}
    step_M[10] = None
    dense = densify(anchor_H, step_M)
    assert dense[a] is not None and dense[b] is not None
    assert 5 in dense and 15 in dense


def test_guide_and_serialization_roundtrip():
    H0 = compute_homography(IMG, PITCH)
    M = np.eye(3)
    M[0, 2] = -2.0
    anchor_H = {0: H0, 12: _true_H_chain(H0, M, 12)}
    dense = densify(anchor_H, {t: M for t in range(1, 13)})
    g = ManualHGuide(dense)
    assert g.has_any() and g.frame_range() == (0, 12)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "sidecar.json")
        save_dense(p, dense, meta={"video": "x.mp4", "fps": 30})
        g2 = load_guide(p)
        assert g2.frame_range() == (0, 12)
        assert np.allclose(g2.get(6), dense[6])
        ap = os.path.join(d, "anchors.json")
        save_anchors(ap, {0: {"image_pts": IMG, "pitch_pts": PITCH, "H": H0}})
        la = load_anchors(ap)
        assert np.allclose(la[0]["H"], H0)
