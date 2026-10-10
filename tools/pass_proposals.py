"""Propose passes from the tracks, for a person to confirm or fix instead of finding every pass.

  python -m tools.pass_proposals --tracks match_tracks.csv.gz --out proposals.csv --fps 30

1. Possession per frame: the goalkeeper / player whose feet (box bottom-centre) are nearest the
   ball, if within `near` x their box height (the ball is at their feet); else nobody (ball in
   flight, or too far to tell).
2. Possession spells: runs of the same player, short flickers (< min_spell frames) dropped, short
   gaps (<= max_gap frames) of the same player bridged.
3. A pass = a spell of player A followed by a spell of a different player B within max_flight_s:
   kick = A's last frame, touch = B's first frame. Same-team / other-team is recorded (other team =
   possibly an interception or a lost ball; the person decides).

`player` is `joined_id` when the tracks have it (pieces joined into players), else the tracker id;
when consecutive spells are pieces of the same player that the tracker split (A ends where B starts,
same team, within `same_spot_m`), they are merged instead of proposed as a pass.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

PEOPLE = (1, 2)


def _id_col(t: pd.DataFrame) -> str:
    for c in ("joined_id", "raw_tracker_id", "track_id"):
        if c in t.columns and (t[c] > 0).any():
            return c
    return "track_id"


def possession(tracks: pd.DataFrame, near: float = 0.6) -> pd.DataFrame:
    """One row per frame with a ball: frame, holder (player id or -1), dist (box heights), team, feet xy."""
    col = _id_col(tracks)
    ball = tracks[tracks.class_id == 0].drop_duplicates("frame")
    ball = ball.assign(bx=(ball.x1 + ball.x2) / 2, by=(ball.y1 + ball.y2) / 2)[["frame", "bx", "by"]]
    ppl = tracks[tracks.class_id.isin(PEOPLE) & (tracks[col] > 0)]
    ppl = ppl.assign(fx=(ppl.x1 + ppl.x2) / 2, fy=ppl.y2, h=(ppl.y2 - ppl.y1).clip(lower=10))
    m = ppl.merge(ball, on="frame")
    # distance from the ball to the feet, with the vertical part halved (the ball sits in front of /
    # beside the feet in image space), in box heights
    m["d"] = np.hypot(m.bx - m.fx, 0.5 * (m.by - m.fy)) / m.h
    best = m.loc[m.groupby("frame").d.idxmin()]
    out = ball[["frame"]].merge(best[["frame", col, "d", "team_id", "x_m", "y_m", "fx", "fy"]], on="frame", how="left")
    out = out.rename(columns={col: "holder"})
    out.loc[~(out.d <= near), "holder"] = -1
    out["holder"] = out.holder.fillna(-1).astype(int)
    return out.sort_values("frame").reset_index(drop=True)


def spells(pos: pd.DataFrame, min_spell: int = 3, max_gap: int = 4) -> pd.DataFrame:
    """Runs of possession by one player: player, team, start, end (frames), feet at start / end."""
    p = pos[pos.holder >= 0]
    runs = []
    for r in p.itertuples():
        if runs and runs[-1]["player"] == r.holder and r.frame - runs[-1]["end"] <= max_gap + 1:
            runs[-1].update(end=r.frame, n=runs[-1]["n"] + 1, ex=r.x_m, ey=r.y_m, efx=r.fx, efy=r.fy)
        else:
            runs.append({"player": r.holder, "team": r.team_id, "start": r.frame, "end": r.frame, "n": 1,
                         "sx": r.x_m, "sy": r.y_m, "ex": r.x_m, "ey": r.y_m,
                         "sfx": r.fx, "sfy": r.fy, "efx": r.fx, "efy": r.fy})
    s = pd.DataFrame(runs)
    if not len(s):
        return s
    s = s[s.n >= min_spell].reset_index(drop=True)
    # merge back-to-back spells of the same player left after dropping flickers
    merged = []
    for r in s.to_dict("records"):
        if merged and merged[-1]["player"] == r["player"] and r["start"] - merged[-1]["end"] <= 2 * max_gap + 1:
            merged[-1].update(end=r["end"], n=merged[-1]["n"] + r["n"], ex=r["ex"], ey=r["ey"], efx=r["efx"], efy=r["efy"])
        else:
            merged.append(r)
    return pd.DataFrame(merged)


def propose(tracks: pd.DataFrame, fps: float = 30.0, near: float = 0.6, min_spell: int = 3,
            max_gap: int = 4, max_flight_s: float = 4.0, min_flight: int = 3,
            same_spot_m: float = 1.5) -> pd.DataFrame:
    s = spells(possession(tracks, near), min_spell, max_gap)
    rows = []
    i = 0
    while i < len(s) - 1:
        a, b = s.iloc[i], s.iloc[i + 1]
        flight = b.start - a.end
        moved = np.hypot(b.sx - a.ex, b.sy - a.ey) / 100 if pd.notna(a.ex) and pd.notna(b.sx) else np.nan
        if a.team == b.team and a.team >= 0 and flight <= 2 * max_gap and moved <= same_spot_m:
            i += 1          # the tracker split one player into two pieces: not a pass
            continue
        if min_flight <= flight <= max_flight_s * fps:
            rows.append({"kick_frame": int(a.end), "touch_frame": int(b.start),
                         "passer": int(a.player), "receiver": int(b.player),
                         "passer_team": int(a.team), "receiver_team": int(b.team),
                         "same_team": bool(a.team == b.team and a.team >= 0),
                         "passer_fx": a.efx, "passer_fy": a.efy, "receiver_fx": b.sfx, "receiver_fy": b.sfy,
                         "flight_s": round(flight / fps, 2),
                         "distance_m": round(float(moved), 1) if pd.notna(moved) else None})
        i += 1
    return pd.DataFrame(rows)


def score(proposals: pd.DataFrame, tagged: pd.DataFrame, tol: int = 10) -> dict:
    """Match proposals to hand-tagged passes (kick and touch both within tol frames, one-to-one)."""
    used, hits, offs = set(), 0, []
    for t in tagged.itertuples():
        c = proposals[(abs(proposals.kick_frame - t.departure_frame) <= tol) &
                      (abs(proposals.touch_frame - t.arrival_frame) <= tol) & ~proposals.index.isin(used)]
        if len(c):
            j = (abs(c.kick_frame - t.departure_frame) + abs(c.touch_frame - t.arrival_frame)).idxmin()
            used.add(j); hits += 1
            offs.append((int(c.loc[j, "kick_frame"] - t.departure_frame), int(c.loc[j, "touch_frame"] - t.arrival_frame)))
    return {"tagged": len(tagged), "proposed": len(proposals), "found": hits,
            "recall": round(hits / max(len(tagged), 1), 3), "precision": round(hits / max(len(proposals), 1), 3),
            "offsets_kick_touch": offs}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args(argv)
    p = propose(pd.read_csv(a.tracks, low_memory=False), a.fps)
    p.to_csv(a.out, index=False)
    print(len(p), "passes proposed")


if __name__ == "__main__":
    main()
