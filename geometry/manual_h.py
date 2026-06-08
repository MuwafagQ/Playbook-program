from __future__ import annotations

import json
import numpy as np
import cv2


def normalize_h(H: np.ndarray) -> np.ndarray:
    """Scale a homography so H[2,2] == 1 (when non-degenerate)."""
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"expected 3x3 homography, got {H.shape}")
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def compute_homography(
    image_pts,
    pitch_pts,
    ransac_thresh: float = 15.0,
) -> np.ndarray | None:
    """Solve an image->pitch homography from hand-clicked correspondences.

    image_pts : Nx2 pixel coordinates clicked on the broadcast frame.
    pitch_pts : Nx2 pitch coordinates (cm) of the matching template vertices.

    With exactly 4 points an exact perspective transform is used; with more, a
    RANSAC fit tolerates a single misclick. Returns a normalized 3x3 H, or None
    if the correspondences are insufficient / degenerate.
    """
    image_pts = np.asarray(image_pts, dtype=np.float64).reshape(-1, 2)
    pitch_pts = np.asarray(pitch_pts, dtype=np.float64).reshape(-1, 2)
    if image_pts.shape[0] < 4 or image_pts.shape[0] != pitch_pts.shape[0]:
        return None

    if image_pts.shape[0] == 4:
        H = cv2.getPerspectiveTransform(
            image_pts.astype(np.float32), pitch_pts.astype(np.float32)
        )
    else:
        H, _ = cv2.findHomography(
            image_pts, pitch_pts, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh
        )
    if H is None or not np.all(np.isfinite(H)):
        return None
    return normalize_h(H)


def _blend(H_a: np.ndarray, H_b: np.ndarray, w: float) -> np.ndarray:
    """Weighted matrix blend, w in [0,1]; w=0 -> H_a, w=1 -> H_b."""
    H = (1.0 - w) * np.asarray(H_a, dtype=np.float64) + w * np.asarray(H_b, dtype=np.float64)
    return normalize_h(H)


def densify(
    anchor_H: dict[int, np.ndarray],
    step_M: dict[int, np.ndarray | None],
) -> dict[int, np.ndarray]:
    """Expand sparse manual anchors into a dense per-frame homography map.

    anchor_H : {frame_idx: 3x3 H} solved from clicked correspondences.
    step_M   : {k: 3x3 M} where M maps frame-k image coords -> frame-(k-1) image
               coords (exactly what CameraMotionEstimator.estimate(gray[k-1],
               gray[k]) returns). A None / missing entry marks a broken link
               (optical flow failed for that step).

    For every frame between two consecutive anchors a < t < b we propagate the
    left anchor forward (H_t = H_{t-1} @ M_t) and the right anchor backward
    (H_{t-1} = H_t @ inv(M_t)), then blend the two estimates by temporal weight.
    Because the forward chain is exact at `a` and the backward chain is exact at
    `b`, the blend pins both ends and shares accumulated optical-flow drift across
    the interval instead of letting it pile up at one side.

    Returns {frame_idx: 3x3 H} covering every frame in [min(anchor), max(anchor)]
    for which an unbroken chain reached it. Frames the chains couldn't reach are
    omitted (the caller falls back to its automatic estimator there).
    """
    if not anchor_H:
        return {}
    anchors = sorted(int(k) for k in anchor_H.keys())
    dense: dict[int, np.ndarray] = {}
    for a in anchors:
        dense[a] = normalize_h(anchor_H[a])

    for a, b in zip(anchors[:-1], anchors[1:]):
        if b <= a + 1:
            continue
        H_a = dense[a]
        H_b = dense[b]

        # Forward chain from a: fwd[t] = fwd[t-1] @ step_M[t]
        fwd: dict[int, np.ndarray] = {a: H_a}
        cur = H_a
        for t in range(a + 1, b + 1):
            M = step_M.get(t)
            if M is None:
                break
            cur = normalize_h(cur @ np.asarray(M, dtype=np.float64))
            fwd[t] = cur

        # Backward chain from b: bwd[t-1] = bwd[t] @ inv(step_M[t])
        bwd: dict[int, np.ndarray] = {b: H_b}
        cur = H_b
        for t in range(b, a, -1):
            M = step_M.get(t)
            if M is None:
                break
            Minv = np.linalg.inv(np.asarray(M, dtype=np.float64))
            if not np.all(np.isfinite(Minv)):
                break
            cur = normalize_h(cur @ Minv)
            bwd[t - 1] = cur

        for t in range(a + 1, b):
            w = (t - a) / float(b - a)
            hf = fwd.get(t)
            hb = bwd.get(t)
            if hf is not None and hb is not None:
                dense[t] = _blend(hf, hb, w)
            elif hf is not None:
                dense[t] = hf
            elif hb is not None:
                dense[t] = hb
            # else: neither chain reached t -> leave to the automatic estimator
    return dense


class ManualHGuide:
    """Per-frame manual homography lookup used by the pipeline."""

    def __init__(self, dense: dict[int, np.ndarray] | None = None):
        self._H: dict[int, np.ndarray] = {}
        for k, v in (dense or {}).items():
            self._H[int(k)] = np.asarray(v, dtype=np.float64).reshape(3, 3)

    def has_any(self) -> bool:
        return len(self._H) > 0

    def __len__(self) -> int:
        return len(self._H)

    def get(self, frame_idx: int) -> np.ndarray | None:
        return self._H.get(int(frame_idx))

    def frame_range(self) -> tuple[int, int] | None:
        if not self._H:
            return None
        keys = self._H.keys()
        return (min(keys), max(keys))


def save_dense(path: str, dense: dict[int, np.ndarray], meta: dict | None = None) -> None:
    """Write the dense per-frame homography sidecar consumed by the pipeline."""
    obj = {
        "meta": meta or {},
        "frames": {str(int(k)): np.asarray(v, dtype=np.float64).reshape(3, 3).tolist()
                   for k, v in dense.items()},
    }
    with open(path, "w") as f:
        json.dump(obj, f)


def load_guide(path: str) -> ManualHGuide:
    """Load a dense sidecar written by save_dense into a ManualHGuide."""
    with open(path) as f:
        obj = json.load(f)
    frames = obj.get("frames", {})
    dense = {int(k): np.asarray(v, dtype=np.float64) for k, v in frames.items()}
    return ManualHGuide(dense)


def save_anchors(path: str, anchors: dict[int, dict], meta: dict | None = None) -> None:
    """Persist the raw clicked correspondences so anchors can be re-edited later.

    anchors : {frame_idx: {"image_pts": [[px,py],...],
                           "pitch_pts": [[cx,cy],...],
                           "H": 3x3 or None}}
    """
    out = {"meta": meta or {}, "anchors": {}}
    for k, v in anchors.items():
        H = v.get("H")
        out["anchors"][str(int(k))] = {
            "image_pts": np.asarray(v["image_pts"], dtype=float).reshape(-1, 2).tolist(),
            "pitch_pts": np.asarray(v["pitch_pts"], dtype=float).reshape(-1, 2).tolist(),
            "H": (np.asarray(H, dtype=float).reshape(3, 3).tolist() if H is not None else None),
        }
    with open(path, "w") as f:
        json.dump(out, f)


def load_anchors(path: str) -> dict[int, dict]:
    with open(path) as f:
        obj = json.load(f)
    res: dict[int, dict] = {}
    for k, v in obj.get("anchors", {}).items():
        H = v.get("H")
        res[int(k)] = {
            "image_pts": np.asarray(v["image_pts"], dtype=float).reshape(-1, 2),
            "pitch_pts": np.asarray(v["pitch_pts"], dtype=float).reshape(-1, 2),
            "H": (np.asarray(H, dtype=float).reshape(3, 3) if H is not None else None),
        }
    return res
