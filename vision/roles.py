"""Per-track role (goalkeeper / player / referee) from the detector's votes over time.

With ROLE_BY_TRACK the pipeline tracks all people with one tracker, so a player whose
per-frame label flickers (e.g. a yellow kit the detector sometimes calls "referee") keeps
one track. This class then gives each track the role it was labelled with most often
(confidence-weighted, slowly decaying so a wrong early start can still be corrected).
"""
from __future__ import annotations

import numpy as np
import supervision as sv

ROLES = (1, 2, 3)  # goalkeeper, player, referee


class RoleVoter:
    def __init__(self, min_votes: float = 1.5, decay: float = 0.995, switch_ratio: float = 1.5):
        self.min_votes = float(min_votes)
        self.decay = float(decay)
        self.switch_ratio = float(switch_ratio)  # a set role changes only when another has this many times its votes
        self.votes: dict[int, np.ndarray] = {}
        self.role: dict[int, int] = {}

    def update(self, tracks: sv.Detections) -> np.ndarray:
        """Role per track row; the current label is kept until a track has min_votes."""
        n = len(tracks)
        if n == 0 or tracks.class_id is None:
            return np.zeros((0,), np.int32)
        cls = tracks.class_id.astype(np.int32)
        if tracks.tracker_id is None:
            return cls.copy()
        conf = tracks.confidence if tracks.confidence is not None else np.ones(n, np.float32)
        out = cls.copy()
        for i in range(n):
            tid = int(tracks.tracker_id[i])
            if tid < 0 or int(cls[i]) not in ROLES:
                continue
            v = self.votes.get(tid)
            if v is None:
                v = np.zeros(len(ROLES), np.float64)
            v *= self.decay
            v[ROLES.index(int(cls[i]))] += float(conf[i])
            self.votes[tid] = v
            if v.sum() < self.min_votes:
                continue
            best = int(np.argmax(v))
            cur = self.role.get(tid)
            if cur is None or v[best] >= self.switch_ratio * v[cur]:
                self.role[tid] = cur = best
            out[i] = ROLES[cur]
        return out
