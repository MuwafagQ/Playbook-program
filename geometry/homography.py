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

    Correspondence uses the per-keypoint class_id injected by vision/detect.py
    (not positional pairing). The it2xv model emits keypoints in label order,
    so slot 0 is NOT vertex 0 — it's whatever vertex its class_id maps to.
    class_id -> vertex_index: int(config.labels[class_id]) - 1.
    """

    def __init__(self, config: SoccerPitchConfiguration, kp_conf: float):
        self.config = config
        self.kp_conf = float(kp_conf)
        self._vertices = np.asarray(config.vertices, dtype=np.float32)
        # class_id -> vertex_index: label "20" -> index 19, etc.
        try:
            self._class_id_to_vertex_idx = np.array(
                [int(label) - 1 for label in config.labels],
                dtype=np.int32,
            )
        except Exception:
            self._class_id_to_vertex_idx = None

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

        # Correspondence: class_id -> vertex index (required for it2xv model family).
        # Fall back to positional pairing only if class_ids are absent.
        pitch_all = None
        if (
            hasattr(keypoints, "class_id")
            and keypoints.class_id is not None
            and self._class_id_to_vertex_idx is not None
        ):
            kp_idx = np.asarray(keypoints.class_id)
            if kp_idx.ndim > 1:
                kp_idx = kp_idx[0]
            kp_idx = kp_idx.astype(np.int32)
            if kp_idx.shape[0] == frame_all.shape[0]:
                # Map class_id -> vertex_index
                valid = (kp_idx >= 0) & (kp_idx < len(self._class_id_to_vertex_idx))
                vtx_idx = np.full_like(kp_idx, -1)
                vtx_idx[valid] = self._class_id_to_vertex_idx[kp_idx[valid]]
                # Keep only slots with a valid vertex mapping
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
