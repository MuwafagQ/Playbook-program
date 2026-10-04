"""Solve the ball's path for a whole segment at once from full-frame ball candidates.

The pipeline's online tracker looks for the ball near its last guess, so once it accepts a wrong blob it
can follow it for a long time (drift). Here every frame's candidates (tools/id_cards.ball_candidates:
the ball model over the whole picture) are known in advance, and the path is the cheapest chain through
them (dynamic programming over a DAG):

  cost of a candidate      -log(conf / base_conf): confident candidates make the path cheaper, so it
                           wants to pass through them; +static_cost if a candidate sits on the same
                           spot for seconds (line marks, heads, boots that the model likes)
  cost of a step           ((distance - allowed) / scale)^2 when the ball would move faster than
                           max_px_per_frame, so long passes are fine but teleports are not
  cost of skipping frames  gap_cost per skipped candidate frame (the ball hidden, e.g. in a keeper's hands)
  start / end anywhere     start_cost

Frames between two chosen candidates are filled by straight lines when the gap is at most max_fill
frames; longer gaps stay empty (no ball shown) instead of drifting.

  path = solve(cands)                    # frame, x, y, conf (chosen candidates)
  ball = fill(path, frames)              # one row per frame: x, y, filled (0 seen / 1 line / NaN none)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def static_mask(c: pd.DataFrame, offsets=(-60, -30, 30, 60), px: float = 4.0) -> np.ndarray:
    """True for candidates that have a candidate within px at every one of the frame offsets that fall
    inside the segment (something that does not move for seconds)."""
    by = {f: g[["x", "y"]].to_numpy(float) for f, g in c.groupby("frame")}
    frames = np.array(sorted(by))
    xy = c[["x", "y"]].to_numpy(float)
    f = c.frame.to_numpy()
    out = np.ones(len(c), bool)
    for off in offsets:
        hit = (f + off < frames[0]) | (f + off > frames[-1])     # outside the segment: no evidence
        for i in range(len(c)):
            k = np.searchsorted(frames, f[i] + off)
            for kk in (k - 1, k):        # nearest candidate frame to the offset
                if 0 <= kk < len(frames) and abs(frames[kk] - f[i] - off) <= 4:
                    if (np.abs(by[frames[kk]] - xy[i]).max(axis=1) <= px).any():
                        hit[i] = True
        out &= hit
    return out


def solve(cands: pd.DataFrame, max_skip: int = 30, max_px_per_frame: float = 45.0, scale: float = 25.0,
          gap_cost: float = 0.35, start_cost: float = 3.0, min_conf: float = 0.05, base_conf: float = 0.25,
          static_cost: float = 0.6) -> pd.DataFrame:
    """cands: frame, x, y, conf (several per frame). Returns the chosen candidates, one per used frame.
    max_skip is in frames of the candidate table (with candidates every 2nd frame, 30 frames = 1 s)."""
    c = cands[cands.conf >= min_conf].sort_values(["frame", "conf"], ascending=[True, False]).reset_index(drop=True)
    if not len(c):
        return c.iloc[:0][["frame", "x", "y", "conf"]]
    if "static" not in c:
        c["static"] = static_mask(c)
    f = c.frame.to_numpy()
    xy = c[["x", "y"]].to_numpy(float)
    node = -np.log(np.clip(c.conf.to_numpy(float), 1e-3, 1.0) / base_conf) + static_cost * c.static.to_numpy()
    frames = np.unique(f)
    stride = max(1, int(np.median(np.diff(frames)))) if len(frames) > 1 else 1
    starts = np.searchsorted(f, frames)
    ends = np.r_[starts[1:], len(f)]
    best = node + start_cost           # cost of the cheapest path ending at each candidate
    prev = np.full(len(c), -1)
    for k, fr in enumerate(frames):
        i0, i1 = starts[k], ends[k]
        j = k - 1
        while j >= 0 and fr - frames[j] <= max_skip:
            p0, p1 = starts[j], ends[j]
            gap = fr - frames[j]
            d = np.linalg.norm(xy[i0:i1, None, :] - xy[None, p0:p1, :], axis=2)       # here x before
            over = np.clip(d - max_px_per_frame * gap, 0, None) / scale
            cost = best[None, p0:p1] + over ** 2 + gap_cost * (gap / stride - 1) + node[i0:i1, None]
            a = np.argmin(cost, axis=1)
            cand = cost[np.arange(i1 - i0), a]
            better = cand < best[i0:i1]
            best[i0:i1] = np.where(better, cand, best[i0:i1])
            prev[i0:i1] = np.where(better, p0 + a, prev[i0:i1])
            j -= 1
    # the path may end anywhere: pick the end that is cheapest including skipping to the last frame
    tail = best + start_cost
    i = int(np.argmin(tail))
    path = []
    while i >= 0:
        path.append(i)
        i = prev[i]
    # a path covers only one stretch; solve the rest of the segment again around it
    chosen = c.iloc[path[::-1]][["frame", "x", "y", "conf"]]
    lo, hi = chosen.frame.min(), chosen.frame.max()
    kw = dict(max_skip=max_skip, max_px_per_frame=max_px_per_frame, scale=scale, gap_cost=gap_cost,
              start_cost=start_cost, min_conf=min_conf, base_conf=base_conf, static_cost=static_cost)
    before, after = c[c.frame < lo - max_skip], c[c.frame > hi + max_skip]
    rest = [solve(before, **kw) if len(before) else None, solve(after, **kw) if len(after) else None]
    return pd.concat([r for r in (rest[0], chosen, rest[1]) if r is not None]).reset_index(drop=True)


def fill(path: pd.DataFrame, frames: np.ndarray, max_fill: int = 30) -> pd.DataFrame:
    """One row per frame. filled: 0 = a candidate on the path, 1 = straight line between two
    candidates at most max_fill frames apart, NaN x/y = no ball."""
    out = pd.DataFrame({"frame": frames}).merge(path[["frame", "x", "y"]], on="frame", how="left")
    seen = out.x.notna()
    known = out.frame[seen].to_numpy()
    out["filled"] = np.where(seen, 0.0, np.nan)
    if len(known) >= 2:
        gap_after = np.diff(known)
        nxt = np.searchsorted(known, out.frame.to_numpy())
        ok = (~seen) & (nxt > 0) & (nxt < len(known))
        ok &= np.r_[gap_after, 10**9][np.clip(nxt - 1, 0, len(known) - 1)] <= max_fill
        interp_x = np.interp(out.frame, known, out.x[seen])
        interp_y = np.interp(out.frame, known, out.y[seen])
        out.loc[ok, "x"] = interp_x[ok]
        out.loc[ok, "y"] = interp_y[ok]
        out.loc[ok, "filled"] = 1.0
    return out


def at_player(ball: pd.DataFrame, tracks: pd.DataFrame, moments: list[tuple[int, int]], reach: float = 1.5) -> float:
    """Share of tagged moments (frame, joined player id) where the ball is within `reach` body heights
    of that player's box centre: at a pass's kick the ball is at the passer, at the touch at the receiver."""
    b = ball.dropna(subset=["x"]).set_index("frame")
    people = tracks[tracks.class_id.isin((1, 2, 3))].set_index(["frame", "joined_id"])
    hit = n = 0
    for f, jid in moments:
        if (f, jid) not in people.index:
            continue
        n += 1
        if f not in b.index:
            continue
        r = people.loc[(f, jid)]
        r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
        h = max(r.y2 - r.y1, 8)
        d = np.hypot(b.x[f] - (r.x1 + r.x2) / 2, b.y[f] - (r.y1 + r.y2) / 2) / h
        hit += bool(d <= reach)
    return hit / n if n else float("nan")
