"""Cut the tracker's pieces where it swapped two players or switched a box to the other team, before the
pieces are joined into players (notebooks/after_tracking.ipynb).

match_video_2 showed the two main tracking errors: at a crossing the tracker hands one player's track
to the other, and a track that drifts onto a player of the other team. When the two players wear
different kits, the shirt colour just before and just after says which happened (about 90% right on
match_video_2, where it was judged on red vs not-red). Here the colour test works for any two kits:
each team's kit is learnt from the segment itself.

  kits = team_kits(crops, teams)                     # mean colour signature per team (0 / 1)
  p0 = kit_prob(crop, kits)                          # 0..1: how much a crop looks like team 0's kit
  cuts = crossing_cuts(tracks, X, s, probs)          # {piece: [frame, ...]} for swaps across teams
  cuts |= flip_cuts(tracks, probs_by_piece_frame)    # team switches confirmed by kit colour
  tracks = apply_cuts(tracks, cuts)                  # each cut starts a new piece id

Only crossings between players of different teams are cut automatically; same-kit crossings stay as
they are (colour cannot tell them apart; the joining by appearance and the review app handle them).
A cut never merges anything: the worst a wrong cut does is split one player's track in two, which the
joining usually puts back together.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tools.crossings import swap_score

NEW_ID_BASE = 900_000      # tails of cut pieces get ids from here (segment-local ids stay below 1e6)


def kit_signature(crop_rgb: np.ndarray) -> np.ndarray:
    """Colour signature of the shirt/shorts centre of a player crop: hue histogram weighted by
    saturation (12 bins) + share of dark / mid / bright and of unsaturated (white/grey/black) pixels."""
    import cv2

    h, w = crop_rgb.shape[:2]
    c = crop_rgb[int(h * 0.2):int(h * 0.6), int(w * 0.3):int(w * 0.7)]
    if c.size == 0:
        return np.zeros(16, np.float32)
    hsv = cv2.cvtColor(np.ascontiguousarray(c), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    hue, sat, val = hsv[:, 0], hsv[:, 1] / 255, hsv[:, 2] / 255
    hist = np.bincount((hue / 15).astype(int).clip(0, 11), weights=sat, minlength=12)
    grey = (sat < 0.25).mean()
    tone = [(val < 0.33).mean(), ((val >= 0.33) & (val < 0.66)).mean(), (val >= 0.66).mean()]
    sig = np.concatenate([hist / max(hist.sum(), 1e-6) * (1 - grey), [grey], tone]).astype(np.float32)
    return sig


def team_kits(sigs: np.ndarray, teams: np.ndarray) -> dict:
    """Mean signature per team from crops whose team is known (0 / 1)."""
    return {t: sigs[teams == t].mean(0) for t in (0, 1) if (teams == t).sum() >= 10}


def kit_prob(sig: np.ndarray, kits: dict) -> float:
    """0..1: closeness to team 0's kit relative to team 1's (L1 distance between signatures)."""
    if 0 not in kits or 1 not in kits:
        return 0.5
    d0, d1 = np.abs(sig - kits[0]).sum(), np.abs(sig - kits[1]).sum()
    return float(d1 / max(d0 + d1, 1e-9))


def piece_teams(tracks: pd.DataFrame) -> dict:
    t = tracks[(tracks.raw_tracker_id >= 0) & (tracks.team_id >= 0)]
    return t.groupby("raw_tracker_id").team_id.agg(lambda s: int(s.mode().iloc[0])).to_dict()


def crossing_cuts(tracks: pd.DataFrame, X: pd.DataFrame, s: pd.DataFrame, probs: np.ndarray,
                  min_score: float = 0.5) -> tuple[dict, pd.DataFrame]:
    """X: tools.crossings.find_crossings; s: its sample_frames rows (k, piece, side, when, frame, box);
    probs: kit_prob of each row of s. A crossing between pieces of different teams whose colour score
    says 'swapped' (>= min_score) cuts both pieces in the middle of the overlap."""
    teams = piece_teams(tracks)
    cuts: dict = {}
    rows = []
    for k, r in X.iterrows():
        tp, tq = teams.get(r.p, -1), teams.get(r.q, -1)
        part = {}
        for side in ("p", "q"):
            for when in ("b", "a"):
                m = ((s.k == k) & (s.side == side) & (s.when == when)).to_numpy()
                part[side + when] = {"emb": None, "red": list(probs[m])} if m.any() else None
        if tp < 0 or tq < 0 or tp == tq or part["pb"] is None or part["qb"] is None or (part["pa"] is None and part["qa"] is None):
            rows.append((k, np.nan, "skipped")); continue
        sc, basis = swap_score(part["pb"], part["qb"], part["pa"], part["qa"])
        swapped = basis == "colour" and sc >= min_score
        rows.append((k, sc, "cut" if swapped else "kept"))
        if swapped:
            f = int((r.f0 + r.f1) // 2) + 1
            for piece, goes_on in ((r.p, r.p_after), (r.q, r.q_after)):
                if goes_on:
                    cuts.setdefault(int(piece), set()).add(f)
    return cuts, pd.DataFrame(rows, columns=["k", "score", "decision"])


def flip_candidates(tracks: pd.DataFrame, frame: str = "frame", window: int = 45, min_len: int = 90) -> pd.DataFrame:
    """Frames where a piece's team label switches and the new label holds (rolling majority over
    `window` frames): piece, frame, team_before, team_after. Labels alone are too noisy to cut on;
    each candidate is checked by kit colour first."""
    out = []
    t = tracks[(tracks.raw_tracker_id >= 0) & (tracks.team_id >= 0)].sort_values(["raw_tracker_id", frame])
    for p, g in t.groupby("raw_tracker_id"):
        if len(g) < min_len:
            continue
        lab = g.team_id.rolling(window, center=True, min_periods=window // 2).mean().round().to_numpy()
        f = g[frame].to_numpy()
        ch = np.nonzero(np.diff(lab) != 0)[0]
        for i in ch:
            before, after = lab[max(0, i - window):i + 1], lab[i + 1:i + 1 + window]
            if len(after) >= window // 2 and (after == lab[i + 1]).all() and (before == lab[i]).all():
                out.append((int(p), int(f[i + 1]), int(lab[i]), int(lab[i + 1])))
    return pd.DataFrame(out, columns=["piece", "frame", "team_before", "team_after"])


def flip_cuts(cands: pd.DataFrame, before_probs: dict, after_probs: dict, min_gap: float = 0.3) -> dict:
    """Confirmed team switches: the kit colour before and after the switch agrees with the labels
    (prob of team 0 moves by at least min_gap in the labelled direction). probs: {candidate index:
    [kit_prob, ...]} of crops just before / after."""
    cuts: dict = {}
    for i, r in cands.iterrows():
        b, a = before_probs.get(i), after_probs.get(i)
        if not b or not a:
            continue
        moved = np.mean(b) - np.mean(a)        # > 0: looks less like team 0 after the switch
        want = 1 if (r.team_before == 0 and r.team_after == 1) else -1
        if moved * want >= min_gap:
            cuts.setdefault(int(r.piece), set()).add(int(r.frame))
    return cuts


def apply_cuts(tracks: pd.DataFrame, cuts: dict, frame: str = "frame", start_id: int = NEW_ID_BASE) -> tuple[pd.DataFrame, dict]:
    """Give each stretch of a cut piece after a cut frame a new raw_tracker_id. Returns the tracks and
    {new id: (old piece, cut frame)}."""
    t = tracks.copy()
    new, nid = {}, start_id
    for p, fs in sorted(cuts.items()):
        rows = t.raw_tracker_id == p
        for f in sorted(fs):
            m = rows & (t[frame] >= f)
            if not m.any():
                continue
            t.loc[m, "raw_tracker_id"] = nid
            new[nid] = (int(p), int(f))
            rows = t.raw_tracker_id == nid
            nid += 1
    return t, new
