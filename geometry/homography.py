from __future__ import annotations

import numpy as np
import cv2
import supervision as sv

from core.types import HomographyResult
from core.utils import normalize_h
from sports.configs.soccer import SoccerPitchConfiguration


class HomographyEstimator:
    """
    Pure per-frame image->pitch homography, matching the original notebook recipe:

        filter = keypoints.confidence[0] > KP_CONF
        source = keypoints.xy[0][filter]
        target = np.array(CONFIG.vertices)[filter]
        H, _   = cv2.findHomography(source, target)

    NO EMA, NO RANSAC gating, NO inlier/reproj validation, NO jump gate.

    Degeneracy guards (for the ~29 frames where only collinear halfway-line
    keypoints are visible and plain DLT produces a rank-deficient H):
    - PCA collinearity: minor-axis spread of frame_pts < min_kp_spread_px → degenerate
    - Condition number: cond(H) > 1e8 after the solve → degenerate

    Bounded short hold: when a guard fires, the last good H is returned (ok=True)
    for up to max_hold_frames consecutive degenerate frames; after that, H=None.

    Correspondence uses the per-keypoint vertex LABEL injected by vision/detect.py
    as kp.class_id (taken from each keypoint's class_name). The model's internal
    class_id ordering does NOT match the pitch vertex numbering, so we map the
    label -> vertex index via config.labels.
    """

    def __init__(
        self,
        config: SoccerPitchConfiguration,
        kp_conf: float,
        min_kp_spread_px: float = 12.0,
        max_hold_frames: int = 15,
    ):
        self.config = config
        self.kp_conf = float(kp_conf)
        self.min_kp_spread_px = float(min_kp_spread_px)
        self.max_hold_frames = int(max_hold_frames)
        self._vertices = np.asarray(config.vertices, dtype=np.float32)
        # Map pitch-vertex LABEL (e.g. 20) -> index into config.vertices.
        # detect.py injects each keypoint's class_NAME (the vertex label) as
        # kp.class_id, because the model's internal class_id ordering does NOT
        # match the vertex numbering (model class_id 17 == vertex label "20").
        try:
            self._label_to_vertex_idx = {
                int(label): i for i, label in enumerate(config.labels)
            }
        except Exception:
            self._label_to_vertex_idx = None

        self._H_prev: np.ndarray | None = None
        self._hold_remaining: int = 0

    # ------------------------------------------------------------------
    # Helpers expected by hstate.py / main.py
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._H_prev = None
        self._hold_remaining = 0

    def set_prev_h(self, H: np.ndarray | None) -> None:
        self._H_prev = H

    def report_failure_summary(self) -> None:
        pass  # pure model has no accumulated failure state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _kp_spread_px(pts: np.ndarray) -> float:
        """Spread of the weaker PCA axis of a 2-D point cloud, in pixels."""
        if pts.shape[0] < 3:
            return 0.0
        centered = pts - pts.mean(axis=0, keepdims=True)
        cov = np.cov(centered.T)
        eig = np.linalg.eigvalsh(cov)   # ascending; eig[0] = weaker-axis variance
        return float(np.sqrt(max(eig[0], 0.0)))

    def _hold_or_none(self, n_pts: int) -> HomographyResult:
        """Called when a degeneracy guard fires. Return held H (ok=True) if budget
        remains; otherwise H=None (ok=False)."""
        if self._H_prev is not None and self._hold_remaining > 0:
            self._hold_remaining -= 1
            return HomographyResult(
                H=self._H_prev, ok=True,
                n_points=n_pts, inlier_ratio=0.0, reproj_err=0.0,
            )
        return HomographyResult(
            H=None, ok=False,
            n_points=n_pts, inlier_ratio=0.0, reproj_err=1e9,
        )

    # ------------------------------------------------------------------
    # Main estimation
    # ------------------------------------------------------------------

    def estimate(self, keypoints: sv.KeyPoints) -> HomographyResult:
        if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
            return HomographyResult(H=None, ok=False, n_points=0, inlier_ratio=0.0, reproj_err=1e9)

        frame_all = np.asarray(keypoints.xy[0], dtype=np.float32)
        if frame_all.ndim != 2 or frame_all.shape[1] != 2 or frame_all.shape[0] == 0:
            return HomographyResult(H=None, ok=False, n_points=0, inlier_ratio=0.0, reproj_err=1e9)

        if keypoints.confidence is not None and len(keypoints.confidence) > 0:
            conf_all = np.asarray(keypoints.confidence[0], dtype=np.float32)
            if conf_all.ndim == 0 or conf_all.shape[0] != frame_all.shape[0]:
                conf_all = np.ones((frame_all.shape[0],), dtype=np.float32)
        else:
            conf_all = np.ones((frame_all.shape[0],), dtype=np.float32)

        # Correspondence: vertex LABEL (kp.class_id, injected from class_name) ->
        # vertex index. Required for the it2xv model family, whose keypoint order
        # and internal class_id do NOT match the pitch vertex numbering.
        pitch_all = None
        if (
            hasattr(keypoints, "class_id")
            and keypoints.class_id is not None
            and self._label_to_vertex_idx
        ):
            labels = np.asarray(keypoints.class_id)
            if labels.ndim > 1:
                labels = labels[0]
            labels = labels.astype(np.int32)
            if labels.shape[0] == frame_all.shape[0]:
                vtx_idx = np.array(
                    [self._label_to_vertex_idx.get(int(l), -1) for l in labels],
                    dtype=np.int32,
                )
                has_vtx = vtx_idx >= 0
                if np.any(has_vtx):
                    frame_all = frame_all[has_vtx]
                    conf_all = conf_all[has_vtx]
                    pitch_all = self._vertices[vtx_idx[has_vtx]]

        if pitch_all is None:
            # Positional fallback (only correct if model emits vertex-ordered slots)
            n = min(frame_all.shape[0], conf_all.shape[0], self._vertices.shape[0])
            if n == 0:
                return HomographyResult(H=None, ok=False, n_points=0, inlier_ratio=0.0, reproj_err=1e9)
            frame_all = frame_all[:n]
            conf_all = conf_all[:n]
            pitch_all = self._vertices[:n]

        keep = conf_all > self.kp_conf
        frame_pts = frame_all[keep]
        pitch_pts = pitch_all[keep]

        n = int(frame_pts.shape[0])
        if n < 4:
            return HomographyResult(H=None, ok=False, n_points=n, inlier_ratio=0.0, reproj_err=1e9)

        # PCA collinearity guard: mid-pan camera may expose only halfway-line
        # keypoints (all at pitch-x≈6000), yielding a rank-deficient DLT solve.
        if self.min_kp_spread_px > 0.0:
            spread = self._kp_spread_px(frame_pts)
            if spread < self.min_kp_spread_px:
                return self._hold_or_none(n)

        H, _ = cv2.findHomography(frame_pts, pitch_pts)
        if H is None:
            return HomographyResult(H=None, ok=False, n_points=n, inlier_ratio=0.0, reproj_err=1e9)

        # Condition-number guard: a degenerate solve can still return a matrix;
        # cond >> 1 means tiny input noise maps to huge output error.
        cond = float(np.linalg.cond(H))
        if not np.isfinite(cond) or cond > 1e8:
            return self._hold_or_none(n)

        H = normalize_h(H)
        self._H_prev = H
        self._hold_remaining = self.max_hold_frames  # reset budget for next degeneracy window
        return HomographyResult(H=H, ok=True, n_points=n, inlier_ratio=1.0, reproj_err=0.0)

    @staticmethod
    def transform_points(H: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
        if H is None or points_xy is None or points_xy.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        pts = points_xy.astype(np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, H).reshape(-1, 2).astype(np.float32)
