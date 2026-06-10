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

    Stateless: NO EMA, NO RANSAC gating, NO inlier/reproj validation,
    NO jump gate, NO fallback to a previous H. Failed solve -> H=None.

    Correspondence uses the per-keypoint vertex LABEL injected by vision/detect.py
    as kp.class_id (taken from each keypoint's class_name). The model's internal
    class_id ordering does NOT match the pitch vertex numbering, so we map the
    label -> vertex index via config.labels.
    """

    def __init__(self, config: SoccerPitchConfiguration, kp_conf: float):
        self.config = config
        self.kp_conf = float(kp_conf)
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

        H, _ = cv2.findHomography(frame_pts, pitch_pts)
        if H is None:
            return HomographyResult(H=None, ok=False, n_points=n, inlier_ratio=0.0, reproj_err=1e9)

        H = normalize_h(H)
        return HomographyResult(H=H, ok=True, n_points=n, inlier_ratio=1.0, reproj_err=0.0)

    @staticmethod
    def transform_points(H: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
        if H is None or points_xy is None or points_xy.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        pts = points_xy.astype(np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, H).reshape(-1, 2).astype(np.float32)
