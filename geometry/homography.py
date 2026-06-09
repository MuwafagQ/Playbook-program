from __future__ import annotations

import numpy as np
import cv2
import supervision as sv

from core.types import HomographyResult
from core.utils import normalize_h, ema_matrix
from sports.configs.soccer import SoccerPitchConfiguration


class HomographyEstimator:
    """
    Estimates image->pitch homography using Roboflow field keypoints.
    - Filters keypoints by confidence
    - Uses RANSAC homography
    - Validates via inlier ratio and mean reprojection error
    - EMA smooths H
    - Falls back to last good H (self._H_prev) when current estimate fails
    """
    def __init__(
        self,
        config: SoccerPitchConfiguration,
        kp_conf: float,
        ema_alpha: float,
        ransac_reproj_thresh: float,
        min_kp: int,
        min_inlier_ratio: float,
        max_reproj_err: float,
        inlier_hysteresis: float = 0.0,
        max_jump_m: float = 0.0,
        min_kp_spread_px: float = 0.0,
        min_inliers_abs: int = 0,
    ):
        self.config = config
        self.kp_conf = kp_conf
        self.ema_alpha = ema_alpha
        self.ransac_reproj_thresh = ransac_reproj_thresh
        self.min_kp = min_kp
        self.min_inlier_ratio = min_inlier_ratio
        self.max_reproj_err = max_reproj_err
        # Hysteresis: once a lock is held, tolerate a slightly lower inlier ratio
        # to maintain it. Stops the per-frame flip-flop caused by the coarse
        # inlier-ratio quantization with only 6-9 keypoints (e.g. 0.50 vs 0.571).
        self.inlier_hysteresis = float(inlier_hysteresis)
        # Frame-to-frame discontinuity gate: reject a new H whose projected points
        # jump implausibly far (in cm) from the previous H. Catches ill-conditioned
        # matrices that pass the inlier test but fling points across the pitch.
        # 0 = disabled.
        self.max_jump_m = float(max_jump_m)
        # Reject geometrically degenerate (near-collinear / tightly-clustered)
        # keypoint sets before RANSAC. Value is the minimum required spread of the
        # weaker principal axis, in pixels. 0 = disabled.
        self.min_kp_spread_px = float(min_kp_spread_px)
        # Absolute inlier-count acceptance: a fit with this many RANSAC inliers is
        # well-determined regardless of how many low-confidence keypoints inflated
        # the denominator and dragged the inlier *ratio* below the bar. Accepted
        # fits still pass the cond / soft-reproj / jump guards. 0 = disabled.
        self.min_inliers_abs = int(min_inliers_abs)
        self._have_lock = False
        self._H_prev: np.ndarray | None = None
        self._warned_kp_mismatch = False
        self._fail_counts: dict[str, int] = {}
        self._diag_max_per_reason = 5

        # The keypoint-detection model emits class_ids in the order of config.labels,
        # NOT in the order of config.vertices. Specifically labels "14" and "19" are
        # at the END of the labels list (indices 30, 31), so a naive vertices[class_id]
        # lookup is wrong for any class_id >= 13. Build a class_id -> vertex_index map
        # using the label numerical name (vertex "15" -> index 14, etc.).
        try:
            self._class_id_to_vertex_idx = np.array(
                [int(label) - 1 for label in config.labels],
                dtype=np.int32,
            )
        except Exception:
            self._class_id_to_vertex_idx = None

        # Bilateral symmetry disambiguation: the pitch is left/right symmetric so the
        # model sometimes assigns the wrong vertex ID to symmetric landmark pairs
        # (e.g. left penalty-box corner ↔ right penalty-box corner). We detect both
        # assignments at runtime and prefer whichever RANSAC finds more inliers for.
        self._sym_pairs = self._build_sym_pairs(np.asarray(config.vertices, dtype=np.float64))

    @staticmethod
    def _build_sym_pairs(vertices: np.ndarray) -> np.ndarray:
        """For each vertex, find its bilateral mirror across the pitch centre-x.

        Returns an int32 array of length N where result[i] is the index of the
        mirror vertex (same as i when no mirror exists within 200 cm tolerance).
        """
        cx = float(vertices[:, 0].mean())
        n = len(vertices)
        result = np.arange(n, dtype=np.int32)
        for i in range(n):
            mirror_x = 2.0 * cx - vertices[i, 0]
            mirror_y = float(vertices[i, 1])
            dists = np.hypot(vertices[:, 0] - mirror_x, vertices[:, 1] - mirror_y)
            j = int(np.argmin(dists))
            if j != i and dists[j] < 200.0:
                result[i] = j
        return result

    def _diag(self, reason: str, **kwargs) -> None:
        n = self._fail_counts.get(reason, 0) + 1
        self._fail_counts[reason] = n
        if n <= self._diag_max_per_reason:
            extras = " ".join(f"{k}={v}" for k, v in kwargs.items())
            print(f"[homography] reject reason={reason} ({extras})")

    def report_failure_summary(self) -> None:
        if not self._fail_counts:
            return
        print("[homography] failure summary:", dict(self._fail_counts))

    def reset(self) -> None:
        self._H_prev = None
        self._have_lock = False

    def set_prev_h(self, H: np.ndarray | None) -> None:
        """Override the EMA / jump-gate baseline (used by external H propagation)."""
        self._H_prev = H
        self._have_lock = H is not None

    @staticmethod
    def _kp_spread_px(frame_pts: np.ndarray) -> float:
        """Spread of the weaker principal axis of the keypoint cloud, in pixels.

        A near-collinear or tightly-clustered set produces an ill-conditioned H,
        so we use the smaller standard deviation along the PCA axes as a guard.
        """
        if frame_pts.shape[0] < 3:
            return 0.0
        centered = frame_pts - frame_pts.mean(axis=0, keepdims=True)
        cov = np.cov(centered.T)
        eig = np.linalg.eigvalsh(cov)  # ascending; eig[0] = weaker axis variance
        return float(np.sqrt(max(eig[0], 0.0)))

    def estimate(self, keypoints: sv.KeyPoints) -> HomographyResult:
        if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
            return HomographyResult(H=self._H_prev, ok=False, n_points=0, inlier_ratio=0.0, reproj_err=1e9)

        frame_all = np.asarray(keypoints.xy[0], dtype=np.float32)
        if frame_all.ndim != 2 or frame_all.shape[1] != 2 or frame_all.shape[0] == 0:
            return HomographyResult(H=self._H_prev, ok=False, n_points=0, inlier_ratio=0.0, reproj_err=1e9)

        if keypoints.confidence is not None and len(keypoints.confidence) > 0:
            conf_all = np.asarray(keypoints.confidence[0], dtype=np.float32)
        else:
            conf_all = np.ones((frame_all.shape[0],), dtype=np.float32)

        vertices = np.asarray(self.config.vertices, dtype=np.float32)

        # Prefer explicit keypoint ids from detector output (if available), then fallback to aligned prefix.
        pitch_all = None
        kp_idx = None
        if hasattr(keypoints, "class_id") and keypoints.class_id is not None:
            kp_idx = np.asarray(keypoints.class_id)
            if kp_idx.ndim > 1:
                kp_idx = kp_idx[0]
            kp_idx = kp_idx.astype(np.int32)

        active_vidx: np.ndarray | None = None  # vertex indices for the kept points
        if kp_idx is not None and kp_idx.shape[0] == frame_all.shape[0]:
            valid_idx = (kp_idx >= 0) & (kp_idx < len(vertices))
            if np.any(valid_idx):
                active_vidx_pre_conf = kp_idx[valid_idx]
                frame_all = frame_all[valid_idx]
                conf_all = conf_all[valid_idx]
                pitch_all = vertices[active_vidx_pre_conf]
            else:
                active_vidx_pre_conf = None
        else:
            active_vidx_pre_conf = None

        if pitch_all is None:
            n_common = min(frame_all.shape[0], conf_all.shape[0], len(vertices))
            if n_common == 0:
                return HomographyResult(H=self._H_prev, ok=False, n_points=0, inlier_ratio=0.0, reproj_err=1e9)

            if not self._warned_kp_mismatch and frame_all.shape[0] != len(vertices):
                print(
                    f"[WARN] Keypoint/vertex count mismatch (kp={frame_all.shape[0]}, pitch={len(vertices)}). "
                    f"Using first {n_common} pairs."
                )
                self._warned_kp_mismatch = True

            frame_all = frame_all[:n_common]
            conf_all = conf_all[:n_common]
            pitch_all = vertices[:n_common]

        keep = conf_all > self.kp_conf
        frame_pts = frame_all[keep]
        pitch_pts = pitch_all[keep]
        if active_vidx_pre_conf is not None:
            active_vidx = active_vidx_pre_conf[keep]

        n = frame_pts.shape[0]
        if n < 4 or n < self.min_kp:
            self._diag("too_few_kp", n=n, min_kp=self.min_kp)
            return HomographyResult(H=self._H_prev, ok=False, n_points=n, inlier_ratio=0.0, reproj_err=1e9)

        # Degeneracy guard: reject near-collinear / clustered keypoints up front.
        if self.min_kp_spread_px > 0.0:
            spread = self._kp_spread_px(frame_pts)
            if spread < self.min_kp_spread_px:
                self._diag("kp_degenerate", n=n, spread=f"{spread:.1f}", min=self.min_kp_spread_px)
                return HomographyResult(H=self._H_prev, ok=False, n_points=n, inlier_ratio=0.0, reproj_err=1e9)

        H, inliers = cv2.findHomography(
            frame_pts,
            pitch_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(self.ransac_reproj_thresh),
        )

        if H is None or inliers is None:
            self._diag("ransac_returned_none", n=n)
            return HomographyResult(H=self._H_prev, ok=False, n_points=n, inlier_ratio=0.0, reproj_err=1e9)

        inliers = inliers.reshape(-1).astype(bool)
        n_inliers = int(inliers.sum())

        # Symmetry disambiguation: try the mirror-flipped vertex assignment and prefer
        # it if RANSAC finds strictly more inliers. This corrects the common failure
        # where the model swaps symmetric landmark pairs (left↔right penalty corners,
        # etc.) producing a "consistent but wrong" homography that passes RANSAC.
        if active_vidx is not None:
            flipped_vidx = self._sym_pairs[active_vidx]
            if not np.array_equal(flipped_vidx, active_vidx):
                pitch_flip = vertices[flipped_vidx]
                H_f, inl_f = cv2.findHomography(
                    frame_pts, pitch_flip,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=float(self.ransac_reproj_thresh),
                )
                if H_f is not None and inl_f is not None:
                    inl_f = inl_f.reshape(-1).astype(bool)
                    n_f = int(inl_f.sum())
                    if n_f > n_inliers:
                        H, inliers, n_inliers, pitch_pts = H_f, inl_f, n_f, pitch_flip

        inlier_ratio = float(inliers.mean()) if len(inliers) else 0.0
        # Hysteresis: a held lock is maintained at a slightly lower bar than is
        # required to acquire one, so frames hovering at the quantization boundary
        # don't flicker in and out of "ok".
        accept_bar = self.min_inlier_ratio
        if self._have_lock and self._H_prev is not None:
            accept_bar = max(0.0, self.min_inlier_ratio - self.inlier_hysteresis)
        # Accept if the inlier RATIO clears the bar, OR the absolute inlier COUNT is
        # high enough that the fit is well-determined on its own. The latter rescues
        # frames where many low-confidence keypoints inflate the denominator (lower
        # KP_CONF) and mechanically depress the ratio even though plenty of points
        # agree. Both paths still face the cond / soft-reproj / jump guards below.
        ratio_ok = inlier_ratio >= accept_bar
        count_ok = self.min_inliers_abs > 0 and n_inliers >= self.min_inliers_abs
        if not (ratio_ok or count_ok):
            self._diag("inlier_ratio_low", n=n, inl=n_inliers, ratio=f"{inlier_ratio:.2f}", min=f"{accept_bar:.2f}")
            return HomographyResult(H=self._H_prev, ok=False, n_points=n, inlier_ratio=inlier_ratio, reproj_err=1e9)

        # Reject ill-conditioned H — a degenerate matrix projects most of the image to infinity
        # even when the RANSAC inlier set has zero reprojection error.
        cond = float(np.linalg.cond(H))
        if not np.isfinite(cond) or cond > 1e9:
            self._diag("ill_conditioned", n=n, cond=f"{cond:.2e}")
            return HomographyResult(H=self._H_prev, ok=False, n_points=n, inlier_ratio=inlier_ratio, reproj_err=1e9)

        # Full-keypoint reprojection check: RANSAC inlier error is circular (RANSAC minimized it),
        # so check ALL confident keypoints. A minimal 4-point H that diverges everywhere else
        # will fail this test. Use a generous soft threshold to stay robust to genuine outliers.
        all_proj = cv2.perspectiveTransform(frame_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        if not np.all(np.isfinite(all_proj)):
            self._diag("nonfinite_proj", n=n)
            return HomographyResult(H=self._H_prev, ok=False, n_points=n, inlier_ratio=inlier_ratio, reproj_err=1e9)
        all_err = np.linalg.norm(all_proj - pitch_pts, axis=1)
        soft_thresh = self.ransac_reproj_thresh * 6.0
        soft_inlier_ratio = float(np.mean(all_err <= soft_thresh))
        if soft_inlier_ratio < (self.min_inlier_ratio - 0.10):
            self._diag(
                "soft_inlier_low",
                n=n,
                soft=f"{soft_inlier_ratio:.2f}",
                med_err=f"{float(np.median(all_err)):.1f}",
            )
            return HomographyResult(H=self._H_prev, ok=False, n_points=n, inlier_ratio=inlier_ratio, reproj_err=float(np.median(all_err)))

        # Reprojection error on RANSAC inliers only (for logging/threshold).
        frame_in = frame_pts[inliers]
        pitch_in = pitch_pts[inliers]
        proj = cv2.perspectiveTransform(frame_in.reshape(-1, 1, 2), H).reshape(-1, 2)
        err = np.linalg.norm(proj - pitch_in, axis=1)
        reproj_err = float(err.mean()) if err.size else 1e9
        if reproj_err > self.max_reproj_err:
            self._diag("reproj_err_high", n=n, err=f"{reproj_err:.1f}", max=self.max_reproj_err)
            return HomographyResult(H=self._H_prev, ok=False, n_points=n, inlier_ratio=inlier_ratio, reproj_err=reproj_err)

        H = normalize_h(H)

        # Frame-to-frame discontinuity gate: a valid-looking H that nonetheless
        # projects the keypoints to a wildly different pitch location than the
        # previous H is a discontinuity (camera cut, or an ill-conditioned fit
        # that survived the inlier test). Reject and hold the previous H.
        if self.max_jump_m > 0.0 and self._H_prev is not None:
            proj_prev = cv2.perspectiveTransform(frame_pts.reshape(-1, 1, 2), self._H_prev).reshape(-1, 2)
            jump = float(np.median(np.linalg.norm(all_proj - proj_prev, axis=1)))
            if jump > self.max_jump_m:
                self._diag("h_discontinuity", n=n, jump=f"{jump:.0f}", max=self.max_jump_m)
                return HomographyResult(H=self._H_prev, ok=False, n_points=n, inlier_ratio=inlier_ratio, reproj_err=reproj_err)

        # EMA smoothing (note: alpha here means "weight of previous")
        if self._H_prev is None:
            H_smooth = H
        else:
            H_smooth = ema_matrix(self._H_prev, H, alpha=self.ema_alpha)
            H_smooth = normalize_h(H_smooth)

        self._H_prev = H_smooth
        self._have_lock = True
        return HomographyResult(H=H_smooth, ok=True, n_points=n, inlier_ratio=inlier_ratio, reproj_err=reproj_err)

    @staticmethod
    def transform_points(H: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
        """
        Apply homography H to Nx2 points (image coords) -> Nx2 points (pitch coords).
        """
        if H is None or points_xy is None or points_xy.size == 0:
            return np.zeros((0, 2), dtype=np.float32)

        pts = points_xy.astype(np.float32).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        return out.astype(np.float32)
