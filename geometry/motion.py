from __future__ import annotations

import cv2
import numpy as np


class CameraMotionEstimator:
    """
    Estimates the frame-to-frame camera motion as an image->image homography
    using sparse optical flow on background features.

    This is used to *propagate* the last good pitch homography through frames
    where the pitch keypoints are too few or too geometrically degenerate to
    solve a fresh homography (e.g. a zoom onto the halfway line + center circle).
    The stands, billboards and stadium structure are feature-rich exactly when
    the grass is feature-poor, so tracking them recovers the camera motion.

    Moving foreground (players, ball) and screen-fixed graphics (scoreboard,
    broadcaster logo) are masked out so they don't corrupt the estimate.
    """

    def __init__(
        self,
        max_corners: int = 600,
        quality_level: float = 0.01,
        min_distance: int = 8,
        ransac_thresh: float = 3.0,
        min_tracked: int = 25,
        min_inliers: int = 20,
        fb_error_thresh: float = 2.0,
        overlay_top_frac: float = 0.12,
    ):
        self.max_corners = int(max_corners)
        self.quality_level = float(quality_level)
        self.min_distance = int(min_distance)
        self.ransac_thresh = float(ransac_thresh)
        self.min_tracked = int(min_tracked)
        self.min_inliers = int(min_inliers)
        self.fb_error_thresh = float(fb_error_thresh)
        # Top band that usually holds screen-fixed overlays (scoreboard/logo).
        self.overlay_top_frac = float(overlay_top_frac)
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    def _feature_mask(self, gray: np.ndarray, exclude_boxes: np.ndarray | None) -> np.ndarray:
        h, w = gray.shape[:2]
        mask = np.full((h, w), 255, dtype=np.uint8)
        # Drop the top overlay band (scoreboard / broadcaster logo are screen-fixed
        # and would inject false "zero motion" against a panning camera).
        top = int(h * self.overlay_top_frac)
        if top > 0:
            mask[:top, :] = 0
        # Drop moving foreground (players/ball), dilated a little for safety.
        if exclude_boxes is not None and len(exclude_boxes) > 0:
            pad = 6
            for box in exclude_boxes:
                x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
                x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = 0
        return mask

    def estimate(
        self,
        prev_gray: np.ndarray,
        cur_gray: np.ndarray,
        exclude_boxes: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """
        Returns a 3x3 homography M mapping CURRENT-frame image coords to
        PREVIOUS-frame image coords (so a pitch homography H_prev can be
        propagated as H_cur = H_prev @ M), or None if motion can't be recovered.
        """
        if prev_gray is None or cur_gray is None:
            return None
        if prev_gray.shape != cur_gray.shape:
            return None

        mask = self._feature_mask(prev_gray, exclude_boxes)
        pts0 = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            mask=mask,
        )
        if pts0 is None or len(pts0) < self.min_tracked:
            return None

        # Forward flow prev -> cur.
        pts1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, pts0, None, **self._lk_params)
        if pts1 is None:
            return None
        # Backward flow cur -> prev for a forward-backward consistency check.
        pts0r, st2, _ = cv2.calcOpticalFlowPyrLK(cur_gray, prev_gray, pts1, None, **self._lk_params)
        if pts0r is None:
            return None

        st1 = st1.reshape(-1).astype(bool)
        st2 = st2.reshape(-1).astype(bool)
        fb_err = np.linalg.norm((pts0 - pts0r).reshape(-1, 2), axis=1)
        good = st1 & st2 & (fb_err < self.fb_error_thresh)
        if int(good.sum()) < self.min_tracked:
            return None

        p0 = pts0.reshape(-1, 2)[good]
        p1 = pts1.reshape(-1, 2)[good]

        # M maps current-frame points (p1) back to previous-frame points (p0).
        M, inliers = cv2.findHomography(p1, p0, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_thresh)
        if M is None or inliers is None:
            return None
        if int(inliers.sum()) < self.min_inliers:
            return None
        if not np.all(np.isfinite(M)):
            return None
        return M
