from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.configs.soccer import SoccerPitchConfiguration

from vision.detect import PLAYER_ID, REFEREE_ID, GOALKEEPER_ID


# Palette matches the Roboflow "football-ai" radar example.
TEAM_COLOR_HEX = {0: "00BFFF", 1: "FF1493"}
REFEREE_COLOR_HEX = "FFD700"
UNKNOWN_COLOR_HEX = "CCCCCC"
BALL_FACE = sv.Color.WHITE
BALL_EDGE = sv.Color.BLACK
EDGE_COLOR = sv.Color.BLACK


def _finite_xy(pts: np.ndarray) -> np.ndarray:
    if pts is None or pts.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    mask = np.isfinite(pts).all(axis=1)
    return pts[mask].astype(np.float32)


def render_radar(
    config: SoccerPitchConfiguration,
    pitch_xy_tracks: np.ndarray,
    track_class_ids: np.ndarray,
    track_ids: np.ndarray,
    team_by_track: dict[int, int],
    pitch_xy_ball: np.ndarray,
    scale: float = 0.1,
    padding: int = 50,
) -> np.ndarray:
    """
    Render a top-down "radar" of the pitch using the roboflow/sports annotators.
    Returns the pitch image (BGR) with team/referee/ball points drawn.
    """
    pitch = draw_pitch(config=config, padding=padding, scale=scale)

    # Bucket projected tracks by team / role.
    team0: list[list[float]] = []
    team1: list[list[float]] = []
    refs: list[list[float]] = []
    unknown: list[list[float]] = []

    n = len(track_class_ids)
    for i in range(n):
        x = float(pitch_xy_tracks[i, 0])
        y = float(pitch_xy_tracks[i, 1])
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        cls_id = int(track_class_ids[i])
        tid = int(track_ids[i]) if i < len(track_ids) else -1
        if cls_id in (PLAYER_ID, GOALKEEPER_ID):
            team = int(team_by_track.get(tid, -1))
            if team == 0:
                team0.append([x, y])
            elif team == 1:
                team1.append([x, y])
            else:
                unknown.append([x, y])
        elif cls_id == REFEREE_ID:
            refs.append([x, y])
        else:
            unknown.append([x, y])

    def _draw(points: list[list[float]], face_hex: str, radius: int) -> None:
        nonlocal pitch
        if not points:
            return
        pitch = draw_points_on_pitch(
            config=config,
            xy=np.asarray(points, dtype=np.float32),
            face_color=sv.Color.from_hex(face_hex),
            edge_color=EDGE_COLOR,
            radius=radius,
            padding=padding,
            scale=scale,
            pitch=pitch,
        )

    _draw(unknown, UNKNOWN_COLOR_HEX, 14)
    _draw(refs, REFEREE_COLOR_HEX, 14)
    _draw(team0, TEAM_COLOR_HEX[0], 16)
    _draw(team1, TEAM_COLOR_HEX[1], 16)

    ball_xy = _finite_xy(pitch_xy_ball)
    if ball_xy.shape[0] > 0:
        pitch = draw_points_on_pitch(
            config=config,
            xy=ball_xy,
            face_color=BALL_FACE,
            edge_color=BALL_EDGE,
            radius=9,
            padding=padding,
            scale=scale,
            pitch=pitch,
        )

    return pitch


def _fit_into(canvas_w: int, canvas_h: int, radar: np.ndarray) -> np.ndarray:
    """Letterbox the radar into a (canvas_h, canvas_w) dark panel, preserving aspect."""
    panel = np.full((canvas_h, canvas_w, 3), (24, 24, 24), dtype=np.uint8)
    rh, rw = radar.shape[:2]
    if rh == 0 or rw == 0:
        return panel
    fit = min(canvas_w / rw, canvas_h / rh)
    new_w = max(1, int(rw * fit))
    new_h = max(1, int(rh * fit))
    resized = cv2.resize(radar, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x0 = (canvas_w - new_w) // 2
    y0 = (canvas_h - new_h) // 2
    panel[y0:y0 + new_h, x0:x0 + new_w] = resized
    return panel


def _draw_status(panel: np.ndarray, homography_ok: bool) -> None:
    status = "H: OK" if homography_ok else "H: HOLD"
    cv2.putText(panel, status, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(panel, status, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def compose_with_radar(
    frame: np.ndarray,
    radar: np.ndarray,
    left_ratio: float = 0.62,
    homography_ok: bool = False,
) -> np.ndarray:
    """Side-by-side: annotated frame on the left, radar panel on the right. Output keeps frame size."""
    h, w = frame.shape[:2]
    left_w = int(np.clip(left_ratio, 0.45, 0.80) * w)
    right_w = max(1, w - left_w)

    left = cv2.resize(frame, (left_w, h), interpolation=cv2.INTER_AREA)
    panel = _fit_into(right_w, h, radar)
    _draw_status(panel, homography_ok)

    out = np.zeros_like(frame)
    out[:, :left_w] = left
    out[:, left_w:] = panel
    return out


def overlay_radar(
    frame: np.ndarray,
    radar: np.ndarray,
    homography_ok: bool = False,
    width_frac: float = 0.38,
    alpha: float = 0.75,
) -> np.ndarray:
    """Overlay the radar in the right portion of the frame (when not using side-by-side)."""
    h, w = frame.shape[:2]
    panel_w = max(1, int(w * width_frac))
    panel = _fit_into(panel_w, h, radar)
    _draw_status(panel, homography_ok)
    x0 = w - panel_w
    out = frame.copy()
    out[:, x0:w] = cv2.addWeighted(out[:, x0:w], 1.0 - alpha, panel, alpha, 0.0)
    return out
