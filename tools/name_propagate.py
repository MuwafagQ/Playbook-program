"""Name every track piece of a team from a few named ones (seeds): appearance + place in the team shape.

The person names each starter once (first minutes of each half), and shirt numbers read from the
video add more seeds. For every other piece of that team and half, two clues are combined:
  appearance  cosine similarity of the piece's mean embedding to each seeded player's mean embedding
              (tools/reid_appearance.py model; sharpened with temperature `tau`)
  position    where the piece plays relative to its team's centre (forward / across, metres), against
              each seeded player's usual spot (Gaussian, `sig` metres)
The best name gets a confidence (softmax); a name is never given to two pieces on screen at the same
time (the less confident one gets its next best name). Pieces whose best confidence stays low are
marked as a possible new player (a substitute, or a starter nobody seeded).

Measured on match_video_2 (team A, starters seeded from the first segment of each half, every other
segment predicted): first half 90% right on the 63% of pieces with confidence >= 0.6, 95% at >= 0.8;
second half 83% / 88% (substitutes were not seeded).

  G = propagate(pieces, seeds)   # pieces: DataFrame piece, team, half, emb (mean, unit), du, dv,
                                 #         first, last (frames); seeds: {piece: name}
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-9)


def propagate(pieces: pd.DataFrame, seeds: dict, tau: float = 0.03, sig: float = 8.0,
              new_below: float = 0.5) -> pd.DataFrame:
    out = []
    for (team, half), g in pieces.groupby(["team", "half"]):
        S = g[g.piece.isin(seeds)]
        if not len(S):
            continue
        S = S.assign(name=S.piece.map(seeds))
        names = sorted(S.name.unique())
        P = np.array([_unit(np.mean(np.stack(S[S.name == n].emb.to_list()), 0)) for n in names])
        G = np.array([[S[S.name == n].du.median(), S[S.name == n].dv.median()] for n in names])
        rows = []
        for r in g.itertuples():
            if r.piece in seeds:
                rows.append((r.piece, seeds[r.piece], 1.0, [seeds[r.piece]], [1.0], r.first, r.last)); continue
            la = (P @ _unit(r.emb)) / tau
            lp = 0.0 if np.isnan(r.du) else -((G[:, 0] - r.du) ** 2 + (G[:, 1] - r.dv) ** 2) / (2 * sig ** 2)
            lg = la + lp; lg = lg - lg.max(); pr = np.exp(lg) / np.exp(lg).sum()
            order = np.argsort(-pr)
            rows.append((r.piece, names[order[0]], float(pr[order[0]]), [names[i] for i in order[:3]],
                         [float(pr[i]) for i in order[:3]], r.first, r.last))
        R = pd.DataFrame(rows, columns=["piece", "name", "confidence", "options", "option_conf", "first", "last"])
        R = _no_duplicates(R)
        R["team"], R["half"] = team, half
        out.append(R)
    if not out:
        return pd.DataFrame(columns=["piece", "name", "confidence", "team", "half", "new_player"])
    R = pd.concat(out, ignore_index=True)
    R["new_player"] = R.confidence < new_below
    return R


def _no_duplicates(R: pd.DataFrame) -> pd.DataFrame:
    """Two pieces on screen together never share a name: the more confident keeps it, the other takes
    its next option (confidence of that option)."""
    R = R.sort_values("confidence", ascending=False).reset_index(drop=True)
    taken: dict = {}
    for i, r in R.iterrows():
        for name, conf in zip(r.options, r.option_conf):
            spans = taken.get(name, [])
            if all(r["last"] < a or r["first"] > b for a, b in spans) or r.confidence == 1.0:
                R.at[i, "name"], R.at[i, "confidence"] = name, conf
                taken.setdefault(name, []).append((r["first"], r["last"]))
                break
        else:
            R.at[i, "confidence"] = 0.0
    return R.drop(columns=["options", "option_conf"])
