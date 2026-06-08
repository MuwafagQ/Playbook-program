from __future__ import annotations

import numpy as np

from core.types import HomographyResult
from core.utils import normalize_h
from geometry.homography import HomographyEstimator
from geometry.motion import CameraMotionEstimator


class HomographyStateMachine:
    """
    Controls homography acceptance/hold/reinit so jitter and long stale holds are reduced.

    When a fresh keypoint-based homography cannot be solved (too few or
    geometrically degenerate pitch keypoints, e.g. a zoom onto the halfway line),
    the last good homography is *propagated* using camera motion estimated from
    background optical flow, instead of being held statically. This keeps player
    projections moving with the camera through midfield zooms. Propagation is
    capped (it drifts) and is reset whenever a strong keypoint lock returns.

    last_action exposes which path produced the current H: "ok" | "propagated" |
    "hold" | "none".
    """
    def __init__(
        self,
        estimator: HomographyEstimator,
        reinit_frames: int = 90,
        motion_estimator: CameraMotionEstimator | None = None,
        max_propagation_frames: int = 0,
    ):
        self.estimator = estimator
        self.reinit_frames = int(reinit_frames)
        self.motion = motion_estimator
        self.max_propagation_frames = int(max_propagation_frames)
        self.H_current: np.ndarray | None = None
        self.fail_streak = 0
        self.ok_count = 0
        self.prop_count = 0
        self.prev_gray: np.ndarray | None = None
        self.last_action = "none"

    def update(
        self,
        keypoints,
        gray: np.ndarray | None = None,
        exclude_boxes: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, bool, HomographyResult]:
        hres = self.estimator.estimate(keypoints)
        if hres.ok and hres.H is not None:
            self.H_current = hres.H
            self.fail_streak = 0
            self.prop_count = 0
            self.ok_count += 1
            if gray is not None:
                self.prev_gray = gray
            self.last_action = "ok"
            return self.H_current, True, hres

        self.fail_streak += 1

        # --- Optical-flow propagation of the last good H ---
        if (
            self.motion is not None
            and self.max_propagation_frames > 0
            and self.H_current is not None
            and gray is not None
            and self.prev_gray is not None
            and self.prop_count < self.max_propagation_frames
        ):
            M = self.motion.estimate(self.prev_gray, gray, exclude_boxes=exclude_boxes)
            if M is not None:
                H_prop = normalize_h(self.H_current @ M)
                cond = float(np.linalg.cond(H_prop))
                if np.all(np.isfinite(H_prop)) and np.isfinite(cond) and cond < 1e9:
                    self.H_current = H_prop
                    # Keep the estimator's EMA / jump-gate baseline aligned with the
                    # propagated H so the next keypoint solve blends/validates against
                    # the current camera pose rather than a stale one.
                    self.estimator.set_prev_h(H_prop)
                    self.prop_count += 1
                    self.prev_gray = gray
                    self.last_action = "propagated"
                    return self.H_current, False, hres

        # --- Reinit after a long failure with no recovery ---
        if self.fail_streak > self.reinit_frames:
            self.estimator.reset()
            self.H_current = None
            self.fail_streak = 0
            self.prop_count = 0
            self.prev_gray = None
            self.last_action = "none"
            return None, False, hres

        # --- Static hold (covers any fail streak up to reinit_frames; H_current
        # stays cached and projectable the whole time, so the radar keeps showing
        # the last good positions instead of going blank in this window) ---
        if self.H_current is not None:
            if gray is not None:
                self.prev_gray = gray
            self.last_action = "hold"
            return self.H_current, False, hres

        self.last_action = "none"
        return None, False, hres
