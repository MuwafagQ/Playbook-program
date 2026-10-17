"""Team of a goalkeeper, whose kit matches neither team's colours: the team whose players are, on
average, nearer to him (each team's centre of mass), voted over several seconds. Over time each
team's players sit deeper toward their own goal, so the goalkeeper's own team is the nearer one.
"""
from __future__ import annotations

import numpy as np


def nearest_team(gk_xy: np.ndarray, player_xy: np.ndarray, player_team: np.ndarray, min_players: int = 2) -> np.ndarray:
    """For each goalkeeper position (N, 2): the team (0/1) whose players' centre is nearer, or -1 when
    a team has fewer than min_players known positions this frame. Positions must share one space
    (pitch coordinates, or image coordinates)."""
    gk_xy = np.asarray(gk_xy, dtype=np.float64).reshape(-1, 2)
    out = np.full((len(gk_xy),), -1, dtype=np.int32)
    player_xy = np.asarray(player_xy, dtype=np.float64).reshape(-1, 2)
    player_team = np.asarray(player_team).reshape(-1)
    ok = np.isfinite(player_xy).all(axis=1) & (player_team >= 0)
    centres = []
    for t in (0, 1):
        m = ok & (player_team == t)
        if m.sum() < min_players:
            return out
        centres.append(player_xy[m].mean(axis=0))
    for i, p in enumerate(gk_xy):
        if not np.isfinite(p).all():
            continue
        d0, d1 = np.linalg.norm(p - centres[0]), np.linalg.norm(p - centres[1])
        out[i] = 0 if d0 < d1 else 1
    return out
