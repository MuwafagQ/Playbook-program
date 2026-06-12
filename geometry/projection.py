from __future__ import annotations

import numpy as np
import supervision as sv
from geometry.homography import HomographyEstimator


def project_anchors_to_pitch(H: np.ndarray, detections: sv.Detections, anchor: sv.Position) -> np.ndarray:
    """
    Extract anchor points from detections and project them onto pitch coordinates using homography H.
    """
    if H is None or len(detections) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    pts = detections.get_anchors_coordinates(anchor)
    return HomographyEstimator.transform_points(H, pts)


def on_pitch_mask(
    H: np.ndarray,
    detections: sv.Detections,
    bounds: tuple[float, float, float, float],
    margin_x: float,
    margin_y: float,
    anchor: sv.Position = sv.Position.BOTTOM_CENTER,
) -> np.ndarray:
    """
    Boolean mask of detections whose projected anchor lies within the pitch
    rectangle expanded by (margin_x, margin_y). The pitch rectangle is defined by
    the field keypoints (corners) via the homography H, so this is exactly an
    "inside the lines + margin" test in pitch coordinates.

    bounds = (xmin, xmax, ymin, ymax) in pitch units (cm).

    Detections that cannot be projected (no H, or a non-finite projection from a
    degenerate solve) are KEPT — we never drop a detection just because the
    geometry is momentarily unavailable.
    """
    n = len(detections)
    if H is None or n == 0:
        return np.ones((n,), dtype=bool)
    xy = project_anchors_to_pitch(H, detections, anchor=anchor)
    if xy.shape[0] != n:
        return np.ones((n,), dtype=bool)
    xmin, xmax, ymin, ymax = bounds
    gx = xy[:, 0]
    gy = xy[:, 1]
    finite = np.isfinite(gx) & np.isfinite(gy)
    inside = (
        (gx >= xmin - margin_x)
        & (gx <= xmax + margin_x)
        & (gy >= ymin - margin_y)
        & (gy <= ymax + margin_y)
    )
    return inside | (~finite)

