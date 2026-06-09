from __future__ import annotations

import numpy as np
import cv2
import supervision as sv

from core.types import HomographyResult
from core.utils import normalize_h
from sports.configs.soccer import SoccerPitchConfiguration


class HomographyEstimator:
    """
    Pure per-frame image->pitch homography, replicating the original
    roboflow/sports `ViewTransformer` recipe exactly:

        filter = keypoints.confidence[0] > KP_CONF
        source = keypoints.xy[0][filter]
        target = np.array(CONFIG.vertices)[filter]
        H, _   = cv2.findHomography(source, target)

    There is NO temporal memory, NO EMA, NO RANSAC gating, NO inlier/reproj
    validation, NO jump gate, and NO fallback to a previous H. Each frame is
    solved independently. If the solve fails (too few points, or
    cv2.findHomography returns None) the result is H=None and the radar is
    blank for that frame. This is the "pure model" baseline; any robustness
    layers are added back later, deliberately, on top of this.

    Correspondence is positional: keypoint slot i maps to pitch vertex i. The
    fine-tuned field model emits a fixed, vertex-ordered 32-slot keypoint array
    (undetected slots carry near-zero confidence and are dropped by the KP_CONF
    filter), so `vertices[filter]` pairs each surviving point with its vertex —
    exactly as in the original notebook.
    """

    def __init__(self, config: SoccerPitchConfiguration, kp_conf: float):
        self.config = config
        self.kp_conf = float(kp_conf)
        self._vertices = np.asarray(config.vertices, dtype=np.float32)

    def estimate(self, keypoints: sv.KeyPoints) -> HomographyResult:
        if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
            return HomographyResult(H=None, ok=False, n_points=0, inlier_ratio=0.0, reproj_err=1e9)

        frame_all = np.asarray(keypoints.xy[0], dtype=np.float32)
        if frame_all.ndim != 2 or frame_all.shape[1] != 2 or frame_all.shape[0] == 0:
            return HomographyResult(H=None, ok=False, n_points=0, inlier_ratio=0.0, reproj_err=1e9)

        if keypoints.confidence is not None and len(keypoints.confidence) > 0:
            conf_all = np.asarray(keypoints.confidence[0], dtype=np.float32)
        else:
            conf_all = np.ones((frame_all.shape[0],), dtype=np.float32)

        # Positional pairing: slot i -> vertex i (original ViewTransformer recipe).
        n_common = min(frame_all.shape[0], conf_all.shape[0], self._vertices.shape[0])
        frame_all = frame_all[:n_common]
        conf_all = conf_all[:n_common]
        pitch_all = self._vertices[:n_common]

        keep = conf_all > self.kp_conf
        frame_pts = frame_all[keep]
        pitch_pts = pitch_all[keep]

        n = int(frame_pts.shape[0])
        if n < 4:
            return HomographyResult(H=None, ok=False, n_points=n, inlier_ratio=0.0, reproj_err=1e9)

        H, _ = cv2.findHomography(frame_pts, pitch_pts)
        if H is None:
            return HomographyResult(H=None, ok=False, n_points=n, inlier_ratio=0.0, reproj_err=1e9)

        H = normalize_h(H)
        return HomographyResult(H=H, ok=True, n_points=n, inlier_ratio=1.0, reproj_err=0.0)

    @staticmethod
    def transform_points(H: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
        """Apply homography H to Nx2 image points -> Nx2 pitch points."""
        if H is None or points_xy is None or points_xy.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        pts = points_xy.astype(np.float32).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        return out.astype(np.float32)
