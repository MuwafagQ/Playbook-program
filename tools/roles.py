"""Automatic tactical roles per frame: line (defence / midfield / attack) x side (left / centre / right).

For every frame, each team's outfield players are compared with their own team's shape at that moment:
  depth = distance ahead of the team's mean position along its attacking direction (metres)
  width = distance to the left / right of the team's mean position, as the team sees the pitch
The attacking direction per half comes from where the team's players are (teams swap ends at half time).
Roles describe where a player is right now, so they are consistent but not identities: a full back who
pushes up is "attack" while he is up there. Frames with fewer than `min_players` visible teammates get
no role (the shape of two or three players says little).

  tracks = roles(tracks, half_start_frame)    # adds role_line, role_side, role ("A left defence")
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PITCH = (105.0, 68.0)


def attack_sign(tracks: pd.DataFrame, half_start: int) -> dict[tuple[int, int], int]:
    """(team, half) -> +1 if the team attacks towards +x, else -1: a team's players stand mostly in
    their own half, so a team whose mean x is below the centre attacks +x."""
    t = tracks[tracks.team_id.isin((0, 1)) & tracks.x_m.notna()]
    half = np.where(t.frame >= half_start, 2, 1)
    mean_x = (t.x_m / 100).groupby([t.team_id, half]).mean()
    return {(int(k[0]), int(k[1])): (1 if v < PITCH[0] / 2 else -1) for k, v in mean_x.items()}


def roles(tracks: pd.DataFrame, half_start: int, letters=("A", "B"), min_players: int = 4,
          depth_cut: float = 6.0, width_cut: float = 6.0) -> pd.DataFrame:
    t = tracks.copy()
    t["role_line"] = None
    t["role_side"] = None
    t["role"] = None
    sel = t.class_id.eq(2) & t.team_id.isin((0, 1)) & t.x_m.notna()
    p = t.loc[sel, ["frame", "team_id", "x_m", "y_m"]].copy()
    p["x"], p["y"] = p.x_m / 100, p.y_m / 100
    g = p.groupby(["frame", "team_id"])
    p["n"] = g.x.transform("size")
    p["cx"], p["cy"] = g.x.transform("mean"), g.y.transform("mean")
    sign = attack_sign(t, half_start)
    half = np.where(p.frame >= half_start, 2, 1)
    s = np.array([sign.get((int(a), int(b)), 1) for a, b in zip(p.team_id, half)])
    depth = (p.x - p.cx) * s
    width = (p.cy - p.y) * s          # facing +x, the left is +y on the pitch... flipped with the direction
    ok = p.n >= min_players
    line = np.where(depth < -depth_cut, "defence", np.where(depth > depth_cut, "attack", "midfield"))
    side = np.where(width > width_cut, "left", np.where(width < -width_cut, "right", "centre"))
    idx = p.index[ok]
    t.loc[idx, "role_line"] = line[ok.to_numpy()]
    t.loc[idx, "role_side"] = side[ok.to_numpy()]
    team_letter = p.team_id.map({0: letters[0], 1: letters[1]})
    t.loc[idx, "role"] = (team_letter + " " + pd.Series(side, index=p.index) + " "
                          + pd.Series(line, index=p.index))[ok]
    return t
