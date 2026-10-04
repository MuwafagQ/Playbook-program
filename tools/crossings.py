"""Find where two players' boxes cross and decide whether the tracker swapped them there.

When two boxes overlap, the tracker can hand one player's track to the other (an ID switch). Each
crossing is judged from photos just before and just after it:
  colour     share of strongly red pixels on the shirt (match_video_2: team A plays in red); when the
             two players differ in colour before the crossing, colour after it says who is who
  appearance cosine similarity of appearance embeddings (tools/reid_appearance.py), before vs after;
             between only two people in the same light, even a weak model separates them often enough

  X = find_crossings(tracks)                  # p, q, f0, f1, iou, p_after, q_after
  frames = sample_frames(tracks, X)           # which boxes to photograph for each crossing
  score = swap_score(before_p, before_q, after_p, after_q)   # > 0: swapped, < 0: kept

A crossing needs both tracks on screen for `before` frames ahead of it and at least one of them still on
screen `after` frames after it. If only p goes on, the question is whether p's box after the crossing is
p's player or q's (q's track ended, its player went on under p's id).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _iou_matrix(B: np.ndarray) -> np.ndarray:
    x1 = np.maximum(B[:, None, 0], B[None, :, 0]); y1 = np.maximum(B[:, None, 1], B[None, :, 1])
    x2 = np.minimum(B[:, None, 2], B[None, :, 2]); y2 = np.minimum(B[:, None, 3], B[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / np.maximum(area[:, None] + area[None, :] - inter, 1e-9)


def find_crossings(tracks: pd.DataFrame, iou_min: float = 0.3, before: int = 15, after: int = 15,
                   merge_gap: int = 10, frame: str = "frame") -> pd.DataFrame:
    """tracks: frame, raw_tracker_id, x1, y1, x2, y2 (people only). One row per pair and stretch of
    overlap (overlaps less than merge_gap frames apart are one crossing)."""
    t = tracks[tracks.raw_tracker_id >= 0]
    span = t.groupby("raw_tracker_id")[frame].agg(["min", "max"])
    ev = []
    for f, g in t.groupby(frame):
        if len(g) < 2:
            continue
        U = _iou_matrix(g[["x1", "y1", "x2", "y2"]].to_numpy(float))
        P = g.raw_tracker_id.to_numpy()
        i, j = np.where(np.triu(U, 1) > iou_min)
        for a, b in zip(i, j):
            ev.append((int(f), int(min(P[a], P[b])), int(max(P[a], P[b])), float(U[a, b])))
    cols = ["p", "q", "f0", "f1", "iou", "p_after", "q_after"]
    if not ev:
        return pd.DataFrame(columns=cols)
    E = pd.DataFrame(ev, columns=["frame", "p", "q", "iou"]).sort_values(["p", "q", "frame"])
    E["run"] = (E.groupby(["p", "q"]).frame.diff().fillna(merge_gap + 1) > merge_gap).cumsum()
    X = E.groupby("run").agg(p=("p", "first"), q=("q", "first"), f0=("frame", "min"), f1=("frame", "max"),
                             iou=("iou", "max")).reset_index(drop=True)
    ok_before = (X.p.map(span["min"]) <= X.f0 - before) & (X.q.map(span["min"]) <= X.f0 - before)
    X["p_after"] = X.p.map(span["max"]) >= X.f1 + after
    X["q_after"] = X.q.map(span["max"]) >= X.f1 + after
    X = X[ok_before & (X.p_after | X.q_after)].sort_values("f0").reset_index(drop=True)
    return X[cols]


def sample_frames(tracks: pd.DataFrame, X: pd.DataFrame, offsets=(4, 8, 12), frame: str = "frame") -> pd.DataFrame:
    """Boxes to photograph: per crossing k, each track's box at f0 - offsets (side 'b') and, for tracks
    that go on, at f1 + offsets (side 'a'). Columns: k, piece, side ('p'/'q'), when ('b'/'a'), frame, box."""
    t = tracks[tracks.raw_tracker_id >= 0].set_index(["raw_tracker_id", frame]).sort_index()
    rows = []
    for k, r in enumerate(X.itertuples()):
        for side, piece, goes_on in (("p", r.p, r.p_after), ("q", r.q, r.q_after)):
            for when, fs in (("b", [r.f0 - o for o in offsets]), ("a", [r.f1 + o for o in offsets] if goes_on else [])):
                for f in fs:
                    if (piece, f) in t.index:
                        b = t.loc[(piece, f)]
                        b = b.iloc[0] if isinstance(b, pd.DataFrame) else b
                        rows.append((k, int(piece), side, when, int(f), float(b.x1), float(b.y1), float(b.x2), float(b.y2)))
    return pd.DataFrame(rows, columns=["k", "piece", "side", "when", "frame", "x1", "y1", "x2", "y2"])


def red_share(crop_bgr: np.ndarray) -> float:
    """Share of strongly red pixels in the shirt/shorts centre of a player crop (any size)."""
    import cv2

    h, w = crop_bgr.shape[:2]
    c = crop_bgr[int(h * 0.2):int(h * 0.6), int(w * 0.3):int(w * 0.7)]
    if c.size == 0:
        return 0.0
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hh, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return float((((hh <= 8) | (hh >= 170)) & (s > 90) & (v > 60)).mean())


def _mean_unit(e):
    if e is None or len(e) == 0:
        return None
    m = np.asarray(e, float).mean(0)
    return m / (np.linalg.norm(m) + 1e-9)


def swap_score(bp: dict, bq: dict, ap: dict | None, aq: dict | None, colour_gap: float = 0.25) -> tuple[float, str]:
    """Each argument: {'emb': (n, d) embeddings, 'red': [red shares]} of one track before (b) or after (a)
    the crossing; ap / aq None when that track does not go on. Returns (score, basis): score > 0 means
    swapped, < 0 kept; |score| is the confidence. basis is 'colour' when the two players' colours differ
    clearly before the crossing (then colour decides), else 'appearance'."""
    rp, rq = np.mean(bp["red"]), np.mean(bq["red"])
    if abs(rp - rq) >= colour_gap:
        s = 0.0
        for after, own, other in ((ap, rp, rq), (aq, rq, rp)):
            if after is not None and len(after["red"]):
                ra = np.mean(after["red"])
                s += (abs(ra - own) - abs(ra - other)) / abs(rp - rq)     # +1: looks like the other one
        n = (ap is not None) + (aq is not None)
        return float(s / max(n, 1)) * 2.0, "colour"
    ep, eq = _mean_unit(bp["emb"]), _mean_unit(bq["emb"])
    s, n = 0.0, 0
    for after, own, other in ((ap, ep, eq), (aq, eq, ep)):
        ea = _mean_unit(after["emb"]) if after is not None else None
        if ea is not None and own is not None and other is not None:
            s += float(ea @ other - ea @ own); n += 1
    return (s / n if n else 0.0), "appearance"
