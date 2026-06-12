from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import supervision as sv


@dataclass
class BallTrackState:
    center: np.ndarray | None = None      # last accepted/held center (px)
    velocity: np.ndarray | None = None    # estimated px/frame
    box_wh: np.ndarray | None = None       # last accepted box size (px)
    area_ema: float = 0.0                  # running ball-area estimate (px^2)
    missing: int = 0                       # consecutive frames without an accept
    vel_measured: bool = False             # True once velocity from a real displacement exists


class BallSmoother:
    """
    Plausibility filter for the ball track.

    The detector frequently fires on boots, the centre spot, line intersections,
    and other small round objects. Because the real ball is tiny and fast, those
    false positives often have HIGHER confidence than the genuine ball, so a
    naive "pick the most confident box" approach latches onto them.

    This filter keeps a single coherent ball trajectory by:
      1. Predicting the next position from the last position + velocity.
      2. Accepting only candidates inside a velocity-aware gate around the
         prediction (the gate widens as the ball moves faster, up to a hard
         pixel ceiling). Candidates outside the gate are REJECTED, not
         redirected to the most-confident box.
      3. Rejecting candidates whose box is implausibly large vs the running
         ball-size estimate (a boot/limb is bigger than a distant ball).
      4. Bridging short detection gaps by extrapolating along the last velocity
         (with decay), so a fast ball keeps moving instead of freezing.

    Returns (detections, imputed) where `imputed` is True when the returned box
    was synthesised by extrapolation rather than a real detection.

    This operates only on ball detections; it is independent of player tracking
    and reID.
    """

    def __init__(
        self,
        ball_id: int,
        max_missing: int = 10,
        max_interp_frames: int = 8,
        max_jump_px: float = 600.0,
        min_conf: float = 0.12,
        gate_base_px: float = 90.0,
        gate_vel_k: float = 3.0,
        acquire_gate_px: float = 300.0,
        size_max_ratio: float = 5.0,
        vel_alpha: float = 0.5,
        hold_decay: float = 0.85,
    ):
        self.ball_id = int(ball_id)
        self.max_missing = int(max_missing)
        self.max_interp_frames = int(max_interp_frames)
        # Hard ceiling: never accept a candidate this far from the prediction,
        # regardless of velocity. Bounds the damage from a wild false positive.
        self.max_jump_px = float(max_jump_px)
        self.min_conf = float(min_conf)
        # Velocity-aware gate radius = gate_base_px + gate_vel_k * speed, capped
        # at max_jump_px. base covers detection jitter on a slow/stationary ball;
        # the velocity term admits genuinely fast (kicked) balls.
        self.gate_base_px = float(gate_base_px)
        self.gate_vel_k = float(gate_vel_k)
        # Cold-start gate used until a velocity has been measured from a real
        # frame-to-frame displacement. Wider than gate_base_px so a fast ball's
        # FIRST move (before any speed estimate exists) isn't rejected outright.
        self.acquire_gate_px = float(acquire_gate_px)
        # Reject a candidate whose box area exceeds this multiple of the running
        # ball-area estimate (catches boots / limbs misread as the ball). 0 = off.
        self.size_max_ratio = float(size_max_ratio)
        # EMA weight for the velocity estimate (weight of the NEW measurement).
        self.vel_alpha = float(vel_alpha)
        # Per-frame velocity decay while extrapolating through a gap.
        self.hold_decay = float(hold_decay)
        self.state = BallTrackState()

    @staticmethod
    def _centers_xyxy(xyxy: np.ndarray) -> np.ndarray:
        cx = 0.5 * (xyxy[:, 0] + xyxy[:, 2])
        cy = 0.5 * (xyxy[:, 1] + xyxy[:, 3])
        return np.stack([cx, cy], axis=1)

    def _make_detection(self, center: np.ndarray, wh: np.ndarray) -> sv.Detections:
        half = 0.5 * wh
        x1y1 = center - half
        x2y2 = center + half
        xyxy = np.array([[x1y1[0], x1y1[1], x2y2[0], x2y2[1]]], dtype=np.float32)
        return sv.Detections(
            xyxy=xyxy,
            confidence=np.array([1.0], dtype=np.float32),
            class_id=np.array([self.ball_id], dtype=np.int32),
        )

    @staticmethod
    def _empty() -> sv.Detections:
        return sv.Detections(
            xyxy=np.zeros((0, 4), dtype=np.float32),
            confidence=np.zeros((0,), dtype=np.float32),
            class_id=np.zeros((0,), dtype=np.int32),
        )

    def _accept(self, center: np.ndarray, wh: np.ndarray) -> None:
        """Commit an accepted detection: update velocity, size and clear misses."""
        center = center.astype(np.float32)
        wh = np.maximum(wh.astype(np.float32), 2.0)
        if self.state.center is not None:
            # Displacement spans (missing + 1) frames since the last accept.
            elapsed = max(1, self.state.missing + 1)
            measured_v = (center - self.state.center) / float(elapsed)
            if self.state.velocity is None:
                new_v = measured_v
            else:
                new_v = self.vel_alpha * measured_v + (1.0 - self.vel_alpha) * self.state.velocity
            self.state.velocity = new_v.astype(np.float32)
            self.state.vel_measured = True
        else:
            self.state.velocity = np.zeros(2, dtype=np.float32)
        area = float(wh[0] * wh[1])
        if self.state.area_ema <= 0.0:
            self.state.area_ema = area
        else:
            self.state.area_ema = 0.7 * self.state.area_ema + 0.3 * area
        self.state.center = center
        self.state.box_wh = wh
        self.state.missing = 0

    def _predicted_center(self) -> np.ndarray:
        if self.state.center is None:
            return None
        v = self.state.velocity if self.state.velocity is not None else np.zeros(2, dtype=np.float32)
        return self.state.center + v

    def predicted_center(self) -> np.ndarray | None:
        """One-step prediction of the ball position, or None when no track is
        active. Used by ROI re-detection to centre the search crop."""
        return self._predicted_center()

    def _gate_radius(self) -> float:
        # Until we've measured a real velocity, use the wider acquisition gate so
        # a fast ball's first post-acquisition move can be picked up.
        if not self.state.vel_measured:
            return min(self.max_jump_px, self.acquire_gate_px)
        speed = float(np.linalg.norm(self.state.velocity)) if self.state.velocity is not None else 0.0
        return min(self.max_jump_px, self.gate_base_px + self.gate_vel_k * speed)

    def _hold(self) -> tuple[sv.Detections, bool]:
        """Extrapolate through a gap, or give up once the gap is too long."""
        self.state.missing += 1
        if self.state.center is None or self.state.missing > self.max_missing:
            self.state = BallTrackState()
            return self._empty(), False
        if self.state.missing <= self.max_interp_frames:
            v = self.state.velocity if self.state.velocity is not None else np.zeros(2, dtype=np.float32)
            v = (v * self.hold_decay).astype(np.float32)
            self.state.velocity = v
            self.state.center = (self.state.center + v).astype(np.float32)
            return self._make_detection(self.state.center, self.state.box_wh), True
        return self._empty(), False

    def update(self, ball_det: sv.Detections) -> tuple[sv.Detections, bool]:
        # No detections at all -> bridge the gap from state.
        if len(ball_det) == 0:
            return self._hold()

        centers = self._centers_xyxy(ball_det.xyxy)
        widths = ball_det.xyxy[:, 2] - ball_det.xyxy[:, 0]
        heights = ball_det.xyxy[:, 3] - ball_det.xyxy[:, 1]
        wh = np.stack([widths, heights], axis=1)
        conf = ball_det.confidence if ball_det.confidence is not None else np.ones((len(ball_det),), dtype=np.float32)

        # Confidence floor.
        ok_conf = conf >= self.min_conf
        # Size guard: drop boxes far larger than the established ball size.
        if self.size_max_ratio > 0.0 and self.state.area_ema > 0.0:
            areas = widths * heights
            ok_size = areas <= (self.size_max_ratio * self.state.area_ema)
        else:
            ok_size = np.ones((len(ball_det),), dtype=bool)
        valid = ok_conf & ok_size
        if not np.any(valid):
            return self._hold()

        valid_idx = np.where(valid)[0]

        # First acquisition: no prior trajectory -> trust the most confident box.
        if self.state.center is None:
            idx = int(valid_idx[np.argmax(conf[valid_idx])])
            self._accept(centers[idx], wh[idx])
            return ball_det[idx:idx + 1], False

        # Established trajectory: match against the PREDICTED position and accept
        # only inside the velocity-aware gate. No fallback to most-confident.
        pred = self._predicted_center()
        d = np.linalg.norm(centers[valid_idx] - pred.reshape(1, 2), axis=1)
        best = int(np.argmin(d))
        if d[best] <= self._gate_radius():
            idx = int(valid_idx[best])
            self._accept(centers[idx], wh[idx])
            return ball_det[idx:idx + 1], False

        # Nothing plausible near the prediction -> treat as a miss and bridge.
        return self._hold()
