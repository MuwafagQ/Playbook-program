"""Name a team's players by relaying names from track to track (the app's Relay mode).

The tracker breaks each player into many tracks (pieces). Every time a new piece of the team starts, the
pieces of that team that ended shortly before, nearby, are its likely predecessors. A person names the new
piece with one tap (usually the nearest predecessor's name), and the name is carried forward from there.
Clear hand-overs (one predecessor that ended a moment ago right where the new piece starts, no rival
close) can be accepted without a tap.

  events = relay_events(tracks, team_pieces)   # one per new piece: start frame, box, candidate predecessors
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def relay_events(tracks: pd.DataFrame, team_pieces, min_frames: int = 15, max_gap: int = 90,
                 max_overlap: int = 5, max_dist: float = 6.0, top: int = 3) -> list[dict]:
    """tracks: frame, raw_tracker_id, x1, y1, x2, y2. team_pieces: piece ids of the team.
    Distances are in body heights between the predecessor's last foot point and the new piece's first."""
    t = tracks[tracks.raw_tracker_id.isin(set(team_pieces))].sort_values("frame")
    g = t.groupby("raw_tracker_id")
    n = g.size()
    first, last = g.head(1).set_index("raw_tracker_id"), g.tail(1).set_index("raw_tracker_id")
    keep = n[n >= min_frames].index
    first, last = first.loc[first.index.intersection(keep)], last.loc[last.index.intersection(keep)]
    foot = lambda d: np.c_[(d.x1 + d.x2) / 2, d.y2]
    lf, lh = foot(last), (last.y2 - last.y1).clip(lower=8).to_numpy()
    lend, lids = last.frame.to_numpy(), last.index.to_numpy()
    out = []
    for p, r in first.sort_values("frame").iterrows():
        s = r.frame
        m = (lend >= s - max_gap) & (lend <= s + max_overlap) & (lids != p)
        cands = []
        if m.any():
            h = max(r.y2 - r.y1, 8)
            d = np.hypot(lf[m, 0] - (r.x1 + r.x2) / 2, lf[m, 1] - r.y2) / ((lh[m] + h) / 2)
            order = np.argsort(d)
            for i in order[:top]:
                if d[i] <= max_dist:
                    cands.append([int(lids[m][i]), round(float(d[i]), 2), int(s - lend[m][i])])
        out.append({"piece": int(p), "start": int(s), "frames": int(n[p]),
                    "box": [int(r.x1), int(r.y1), int(r.x2), int(r.y2)], "cands": cands})
    return out


def clear(event: dict, near: float = 0.8, gap: int = 20, rival: float = 2.0) -> bool:
    """One predecessor, close and recent, and the next one clearly further away."""
    c = event["cands"]
    if not c or c[0][1] > near or c[0][2] > gap:
        return False
    return len(c) == 1 or c[1][1] >= rival * max(c[0][1], 0.1)
