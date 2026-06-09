from __future__ import annotations

import numpy as np
import cv2
import supervision as sv

from core.types import HomographyResult
from core.utils import normalize_h
from sports.configs.soccer import SoccerPitchConfiguration


class HomographyEstimator:
    """
    Stateless image->pitch homography from Roboflow field keypoints.

    This is a deliberate replica of the original "pure model" pipeline (notebook
    cell 60): every frame is solved independently with plain least-squares
    findHomography, exactly like sports.common.view.ViewTransformer. There is NO
    temporal memory of any kind — no EMA, no RANSAC inlier gating, no jump-gate,
    no condition-number guard, no propagation, no fallback to a previous H. The
    raw per-frame solve self-corrects within a handful of frames during camera
    pans, which is more stable on this footage than any temporal scheme.

    Keypoint->pitch correspondence:
      The f07vi field model emits keypoints whose per-keypoint class_id encodes
      which pitch vertex each point is. sv.KeyPoints.from_inference does NOT keep
      that mapping in slot order (slots follow the model's label order, not
      config.vertices order), so vision.detect re-injects the per-keypoint ids
      and we remap class_id -> vertex_index here. This adapter REPLACES the
      positional `vertices[filter]` trick the original it2xv model gave us for
      free; it is required, not optional.

    Confidence:
      Keep only keypoints with conf > kp_conf (the original used 0.5). At that
      threshold the surviving points are clean and well-distributed, so a plain
      least-squares fit is well-conditioned without RANSAC.

    Constructor keeps the historical signature so existing call sites / the
    (now-bypassed) state machine still construct it; the temporal-tuning
    arguments are accepted and ignored.
    """

    def __init__(
        self,
        config: SoccerPitchConfiguration,
        kp_conf: float,
        ema_alpha: float = 0.0,
        ransac_reproj_thresh: float = 0.0,
        min_kp: int = 4,
        min_inlier_ratio: float = 0.0,
        max_reproj_err: float = 0.0,
        inlier_hysteresis: float = 0.0,
        max_jump_m: float = 0.0,
        min_kp_spread_px: float = 0.0,
        min_inliers_abs: int = 0,
    ):
        self.config = config
        self.kp_conf = float(kp_conf)
        self.min_kp = max(4, int(min_kp))

        # Retained only so the bypassed state machine can still poke the estimator
        # without AttributeError. Not used by the stateless solve.
        self._H_prev: np.ndarray | None = None
        self._have_lock = False
        self._warned_kp_mismatch = False
        self._fail_counts: dict[str, int] = {}
        self._diag_max_per_reason = 5

        # class_id -> vertex_index: the model labels are the vertex names ("1".."32"),
        # so vertex_index = int(label) - 1. This is the required correspondence
        # adapter for the f07vi model family (see class docstring).
        try:
            self._class_id_to_vertex_idx = np.array(
                [int(label) - 1 for label in config.labels],
                dtype=np.int32,
            )
        except Exception:
            self._class_id_to_vertex_idx = None

    def _diag(self, reason: str, **kwargs) -> None:
        n = self._fail_counts.get(reason, 0) + 1
        self._fail_counts[reason] = n
        if n <= self._diag_max_per_reason:
            extras = " ".join(f"{k}={v}" for k, v in kwargs.items())
            print(f"[homography] skip reason={reason} ({extras})")

    def report_failure_summary(self) -> None:
        if not self._fail_counts:
            return
        print("[homography] skip summary:", dict(self._fail_counts))

    # Kept as no-ops for compatibility with any external caller that still
    # references them (the state machine). The stateless solve ignores them.
    def reset(self) -> None:
        self._H_prev = None
        self._have_lock = False

    def set_prev_h(self, H: np.ndarray | None) -> None:
        self._H_prev = H
        self._have_lock = H is not None

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

        vertices = np.asarray(self.config.vertices, dtype=np.float32)

        # --- Correspondence: per-keypoint class_id -> pitch vertex. ---
        pitch_all = None
        kp_idx = None
        if hasattr(keypoints, "class_id") and keypoints.class_id is not None:
            kp_idx = np.asarray(keypoints.class_id)
            if kp_idx.ndim > 1:
                kp_idx = kp_idx[0]
            kp_idx = kp_idx.astype(np.int32)
            # Map model class_id -> vertex index when the label-order remap exists.
            if self._class_id_to_vertex_idx is not None:
                valid = (kp_idx >= 0) & (kp_idx < len(self._class_id_to_vertex_idx))
                vtx_idx = np.full_like(kp_idx, fill_value=-1)
                vtx_idx[valid] = self._class_id_to_vertex_idx[kp_idx[valid]]
                kp_idx = vtx_idx

        if kp_idx is not None and kp_idx.shape[0] == frame_all.shape[0]:
            valid_idx = (kp_idx >= 0) & (kp_idx < len(vertices))
            if np.any(valid_idx):
                frame_all = frame_all[valid_idx]
                conf_all = conf_all[valid_idx]
                pitch_all = vertices[kp_idx[valid_idx]]

        if pitch_all is None:
            # No usable per-keypoint ids: fall back to positional alignment on the
            # common prefix (only correct if the model returns vertex-ordered slots).
            n_common = min(frame_all.shape[0], conf_all.shape[0], len(vertices))
            if n_common == 0:
                return HomographyResult(H=None, ok=False, n_points=0, inlier_ratio=0.0, reproj_err=1e9)
            if not self._warned_kp_mismatch and frame_all.shape[0] != len(vertices):
                print(
                    f"[WARN] No per-keypoint ids and kp/vertex count mismatch "
                    f"(kp={frame_all.shape[0]}, pitch={len(vertices)}). Using first {n_common} pairs."
                )
                self._warned_kp_mismatch = True
            frame_all = frame_all[:n_common]
            conf_all = conf_all[:n_common]
            pitch_all = vertices[:n_common]

        # --- Confidence filter ---
        keep = conf_all > self.kp_conf
        frame_pts = frame_all[keep]
        pitch_pts = pitch_all[keep]

        n = frame_pts.shape[0]
        if n < self.min_kp:
            self._diag("too_few_kp", n=n, min_kp=self.min_kp)
            return HomographyResult(H=None, ok=False, n_points=n, inlier_ratio=0.0, reproj_err=1e9)

        # --- Stateless RANSAC solve. ---
        # Using conf>0.25 admits 8-14 points/frame (vs exactly 4 at 0.50), so the
        # set contains some noisy low-confidence points. RANSAC selects the largest
        # geometrically consistent subset per-frame, exactly as the original solve
        # did — but without any temporal carry-over or acceptance gating. A frame
        # whose inliers disagree (degenerate pan mix or genuine model failure) just
        # returns H=None and the radar is blank that frame; the next frame tries
        # fresh. Threshold is in pitch space (cm): ~150cm ≈ 5-15 px of image error.
        H, inliers = cv2.findHomography(
            frame_pts,
            pitch_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(self.ransac_reproj_thresh),
        )
        if H is None or inliers is None:
            self._diag("ransac_returned_none", n=n)
            return HomographyResult(H=None, ok=False, n_points=n, inlier_ratio=0.0, reproj_err=1e9)

        inliers_bool = inliers.reshape(-1).astype(bool)
        n_inliers = int(inliers_bool.sum())
        if n_inliers < 4:
            self._diag("too_few_inliers", n=n, inliers=n_inliers)
            return HomographyResult(H=None, ok=False, n_points=n, inlier_ratio=float(n_inliers) / n, reproj_err=1e9)

        H = normalize_h(H)

        # Reprojection error on RANSAC inliers — for logging only.
        proj_in = cv2.perspectiveTransform(frame_pts[inliers_bool].reshape(-1, 1, 2), H).reshape(-1, 2)
        if np.all(np.isfinite(proj_in)):
            reproj_err = float(np.linalg.norm(proj_in - pitch_pts[inliers_bool], axis=1).mean())
        else:
            reproj_err = 1e9

        return HomographyResult(H=H, ok=True, n_points=n_inliers, inlier_ratio=float(n_inliers) / n, reproj_err=reproj_err)

    @staticmethod
    def transform_points(H: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
        """Apply homography H to Nx2 image points -> Nx2 pitch points."""
        if H is None or points_xy is None or points_xy.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        pts = points_xy.astype(np.float32).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        return out.astype(np.float32)
