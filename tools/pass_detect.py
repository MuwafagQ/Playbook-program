"""Pass suggestions from who is nearest the ball: the ball moves from one player to another.

Per frame, the ball's owner is the player whose feet are nearest the ball on screen, when close enough
(within `reach` x that player's box height). Runs of the same owner (short gaps bridged) are touches on
the ball; consecutive touches by two different players are a pass from the first to the second:

  kick  = last frame of the first player's run
  touch = first frame of the second player's run
  same team -> complete, other team -> intercepted

Players are the joined ids (one player across the tracker's pieces); a run must last `min_run` frames,
the ball must travel at least `min_travel` px and arrive within `max_flight` frames. Parameters were
tuned on match_video_2 against the 813 passes tagged by hand (tools/pass_detect.py --eval).

  P = detect(tracks)            # tracks_small rows of one segment (or the match): frame, raw_tracker_id,
                                # joined_id, class_id, team_id, x1, y1, x2, y2
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PARAMS = dict(reach=0.6, min_run=8, gap=12, min_travel=40, max_flight=120)   # tuned on match_video_2


def owners(t: pd.DataFrame, reach: float = 0.6) -> pd.DataFrame:
    """Per frame with a ball: the nearest player's id (joined id, else piece), piece, team, and the ball
    position; owner -1 when nobody is close."""
    b = t[t.class_id == 0].drop_duplicates("frame").set_index("frame")
    ball = pd.DataFrame({"bx": (b.x1 + b.x2) / 2, "by": (b.y1 + b.y2) / 2})
    p = t[t.class_id.isin([1, 2]) & (t.raw_tracker_id >= 0)].join(ball, on="frame", how="inner")
    who = p.joined_id.where(p.joined_id > 0, -p.raw_tracker_id) if "joined_id" in p else -p.raw_tracker_id
    p = p.assign(who=who, d=np.hypot((p.x1 + p.x2) / 2 - p.bx, p.y2 - p.by), h=p.y2 - p.y1)
    p = p[p.d <= reach * p.h]
    best = p.loc[p.groupby("frame").d.idxmin(), ["frame", "who", "raw_tracker_id", "team_id", "x1", "x2", "y2", "h"]]
    out = ball.reset_index().merge(best, on="frame", how="left")
    out["who"] = out.who.fillna(-1).astype(np.int64)
    return out


def runs(o: pd.DataFrame, min_run: int = 4, gap: int = 6) -> pd.DataFrame:
    """Touches: stretches where one player owns the ball (gaps of up to `gap` frames without an owner or
    with a one-off other owner are bridged), at least `min_run` frames with that owner."""
    o = o[o.who != -1].sort_values("frame")
    rows, cur = [], None
    for r in o.itertuples():
        if cur and r.who == cur["who"] and r.frame - cur["last"] <= gap:
            cur["last"] = r.frame; cur["n"] += 1; cur["lx"], cur["ly"] = r.bx, r.by; cur["hs"].append(r.h)
            continue
        if cur:
            rows.append(cur)
        cur = {"who": r.who, "piece": int(r.raw_tracker_id), "team": int(r.team_id) if not pd.isna(r.team_id) else -1,
               "first": r.frame, "last": r.frame, "n": 1, "fx": r.bx, "fy": r.by, "lx": r.bx, "ly": r.by, "hs": [r.h]}
    if cur:
        rows.append(cur)
    for c in rows:
        c["h"] = float(np.median(c.pop("hs")))
    R = pd.DataFrame(rows)
    return R[R.n >= min_run].reset_index(drop=True) if len(R) else R


def detect(t: pd.DataFrame, reach=None, min_run=None, gap=None, min_travel=None, max_flight=None) -> pd.DataFrame:
    P = {**PARAMS, **{k: v for k, v in dict(reach=reach, min_run=min_run, gap=gap, min_travel=min_travel,
                                            max_flight=max_flight).items() if v is not None}}
    R = runs(owners(t, P["reach"]), P["min_run"], P["gap"])
    out, prev_last = [], None
    for a, b in zip(R.itertuples(), R.iloc[1:].itertuples()):
        if a.who == b.who:
            prev_last = a.last
            continue
        flight = b.first - a.last
        travel = float(np.hypot(b.fx - a.lx, b.fy - a.ly))
        if flight <= 0 or flight > P["max_flight"] or travel < P["min_travel"]:
            continue
        out.append({"run_a": int(a.n), "run_b": int(b.n), "span_a": int(a.last - a.first), "span_b": int(b.last - b.first),
                    "h_a": float(a.h), "h_b": float(b.h), "travel_h": travel / max((a.h + b.h) / 2, 1.0),
                    "speed_h": travel / max(flight, 1) / max((a.h + b.h) / 2, 1.0), "same_team": int(a.team == b.team and a.team >= 0),
                    "gap_prev": int(a.first - prev_last) if prev_last is not None else 999,
                    "kick": int(a.last), "touch": int(b.first), "passer": int(a.who), "passer_piece": a.piece,
                    "passer_team": a.team, "receiver": int(b.who), "receiver_piece": b.piece, "receiver_team": b.team,
                    "outcome": "complete" if a.team == b.team and a.team >= 0 else "intercepted",
                    "flight": int(flight), "travel_px": round(travel)})
        prev_last = a.last
    return pd.DataFrame(out)


def evaluate(pred: pd.DataFrame, truth: pd.DataFrame, tol: int = 15) -> dict:
    """truth: kick, passer_piece (may be NaN). A suggestion is right when a tagged kick is within tol
    frames (each tagged pass used once)."""
    tk = np.sort(truth.kick.to_numpy()); used = np.zeros(len(tk), bool); hit = 0; same_passer = 0
    tp = truth.set_index("kick").passer_piece.to_dict() if "passer_piece" in truth else {}
    for r in pred.sort_values("kick").itertuples():
        i = np.searchsorted(tk, r.kick)
        cand = [j for j in (i - 1, i, i + 1) if 0 <= j < len(tk) and not used[j] and abs(tk[j] - r.kick) <= tol]
        if cand:
            j = min(cand, key=lambda j: abs(tk[j] - r.kick)); used[j] = True; hit += 1
            pp = tp.get(int(tk[j]))
            if pp is not None and not pd.isna(pp) and int(str(pp).split("_")[0]) == r.passer_piece:
                same_passer += 1
    return {"suggested": len(pred), "tagged": len(tk), "right": hit,
            "precision": round(hit / max(len(pred), 1), 3), "found": round(hit / max(len(tk), 1), 3),
            "passer_piece_agrees": round(same_passer / max(hit, 1), 3)}


def suggest(t: pd.DataFrame, model_path: str | None = None) -> pd.DataFrame:
    """Pass suggestions with a confidence (0-1) and a tier: 'confident' (>= 0.6: 82% were real passes on
    match_video_2, covering 42% of them) or 'check' (the rest: about one in three is real)."""
    import json, os
    M = json.load(open(model_path or os.path.join(os.path.dirname(__file__), "pass_detect_model.json")))
    P = detect(t, **M["params"])
    if not len(P):
        return P
    X = P[M["features"]].fillna(0).to_numpy(float)
    X = np.minimum(X, np.array(M["clip_hi"]))
    z = (X - np.array(M["mean"])) / np.array(M["std"])
    P["confidence"] = np.round(1 / (1 + np.exp(-(z @ np.array(M["coef"]) + M["intercept"]))), 3)
    P["tier"] = np.where(P.confidence >= 0.6, "confident", "check")
    return P
