"""Propose a name for every track piece from the names the user already gave.

Labels come from the Pass Review app: every tagged pass names its passer and receiver at one frame,
which pins a name on the piece (raw tracker id) under that box; fixes made in Players mode / on the
photo cards name pieces directly. A projection of the appearance embeddings is trained on those
labelled pieces (linear discriminant analysis: this match's kits, light and players), and every other
piece gets the names of its nearest labelled pieces, with a confidence margin (best minus second best).

Held out on match_video_2 (810 passes, 692 labelled pieces): 80% of proposals agree with the user's
label, about 93% for the most confident half; the user's own labels disagree among themselves too
(same name on two pieces on screen together), so the true accuracy is higher.

  labels = labels_from_reviews(reviews, tracks)      # piece -> name
  pieces = piece_table(tracks)                       # start, end, frames, team per piece
  props = propose(emb, emb_piece, labels, pieces)    # piece -> name, margin, alternatives
  clashes(labels | confirmed, pieces)                # same name on two pieces on screen together
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PEOPLE = (1, 2)
TEAM_PENALTY = 0.15   # similarity taken off a name of the other team (tracker teams are ~90% right)


def piece_table(tracks: pd.DataFrame) -> pd.DataFrame:
    t = tracks[tracks.class_id.isin(PEOPLE) & (tracks.raw_tracker_id >= 0)]
    mode = lambda s: int(s.mode().iloc[0]) if len(s.mode()) else -1
    return t.groupby("raw_tracker_id").agg(start=("frame", "min"), end=("frame", "max"), frames=("frame", "size"),
                                           team=("team_id", mode))


def labels_from_reviews(reviews: dict, tracks: pd.DataFrame) -> pd.Series:
    """piece -> the name given most often to it in the tagged passes (passer at kick, receiver at touch)."""
    t = tracks[tracks.class_id.isin(PEOPLE) & (tracks.raw_tracker_id >= 0)]
    want = {}
    for r in reviews.values():
        for role in ("passer", "receiver"):
            w = r.get(role) or {}
            if w.get("name") and w.get("jid") is not None and w.get("frame") is not None:
                want.setdefault(int(w["frame"]), []).append((w.get("piece"), w["jid"], w["name"]))
    rows = []
    for f, g in t[t.frame.isin(list(want))].groupby("frame"):
        for piece, jid, name in want[f]:
            if piece is not None:
                rows.append((int(piece), name))
                continue
            hit = g[g.joined_id == jid]
            if len(hit):
                rows.append((int(hit.raw_tracker_id.iloc[0]), name))
    if not rows:
        return pd.Series(dtype=object)
    df = pd.DataFrame(rows, columns=["piece", "name"])
    return df.groupby("piece").name.agg(lambda s: s.mode().iloc[0])


def _piece_means(x: np.ndarray, piece: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame(x)
    df["p"] = piece
    m = df.groupby("p").mean()
    v = m.to_numpy(np.float32)
    return m.index.to_numpy(), v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


def propose(emb: np.ndarray, emb_piece: np.ndarray, labels: pd.Series, pieces: pd.DataFrame,
            k: int = 3, top: int = 3) -> pd.DataFrame:
    """For every piece with embeddings: the best names, scored by the mean of the k highest
    similarities to labelled pieces of that name (after a projection trained on the labels)."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    emb = np.asarray(emb, np.float32)
    lab = labels[labels.index.isin(np.unique(emb_piece))]
    m = np.isin(emb_piece, lab.index)
    y = pd.Series(emb_piece[m]).map(lab).to_numpy()
    n_cls = len(np.unique(y))
    if n_cls >= 3:
        lda = LinearDiscriminantAnalysis(n_components=min(n_cls - 1, emb.shape[1])).fit(emb[m], y)
        emb = lda.transform(emb)
    ids, vec = _piece_means(emb, emb_piece)
    pos = {p: i for i, p in enumerate(ids)}
    ref = np.array([pos[p] for p in lab.index])
    ref_names = lab.to_numpy()
    names = np.unique(ref_names)
    sims = vec @ vec[ref].T                                   # pieces x labelled pieces
    out = []
    for i, p in enumerate(ids):
        s = sims[i].copy()
        if p in lab.index:                                    # never vote for yourself
            s[ref == i] = -np.inf
        team = pieces.team.get(p, -1)
        letter = {0: "A", 1: "B"}.get(team)
        sc = []
        for n in names:
            v = s[ref_names == n]
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            val = float(np.sort(v)[-k:].mean()) - (TEAM_PENALTY if letter and n[:1] in "AB" and n[0] != letter else 0.0)
            sc.append((n, val))
        sc.sort(key=lambda x: -x[1])
        if not sc:
            continue
        margin = sc[0][1] - (sc[1][1] if len(sc) > 1 else 0.0)
        out.append({"piece": int(p), "name": sc[0][0], "score": round(sc[0][1], 3), "margin": round(margin, 3),
                    "alternatives": [n for n, _ in sc[1:top]], "labelled": lab.get(p),
                    "scores": [(n, round(v, 3)) for n, v in sc[:8]]})
    return pd.DataFrame(out)


def assign_exclusive(props: pd.DataFrame, fixed: pd.Series, pieces: pd.DataFrame, cut: float = 0.70,
                     overlap: int = 15) -> dict[int, tuple[str, float]]:
    """One name holds one piece at a time. The user's labels are placed first; then (piece, name, score)
    candidates are taken best first and kept while the name is free for the piece's whole time on screen.
    Pieces left without a name scoring >= cut stay unnamed. Held out on match_video_2: 71% of pieces
    named, 89% of those agree with the user (80% without the rule, with constant clashes)."""
    occ: dict[str, list[tuple[int, int]]] = {}
    out: dict[int, tuple[str, float]] = {}
    for p, n in fixed.items():
        if p in pieces.index:
            occ.setdefault(n, []).append((pieces.start[p], pieces.end[p]))
            out[int(p)] = (n, 1.0)
    cand = sorted(((v, int(r.piece), n) for r in props.itertuples() if int(r.piece) not in out
                   for n, v in r.scores if v >= cut), reverse=True)
    for v, p, n in cand:
        if p in out or p not in pieces.index:
            continue
        a, b = pieces.start[p], pieces.end[p]
        if all(b < s0 + overlap or a > e0 - overlap for s0, e0 in occ.get(n, [])):
            out[p] = (n, v)
            occ.setdefault(n, []).append((a, b))
    return out


def clashes(named: pd.Series, pieces: pd.DataFrame, min_overlap: int = 15) -> list[tuple[int, int, str]]:
    """Pairs of pieces that carry the same name while both are on screen (more than min_overlap frames)."""
    x = pieces.loc[pieces.index.intersection(named.index)].assign(name=named)
    out = []
    for n, g in x.groupby("name"):
        g = g.sort_values("start")
        s, e, p = g.start.to_numpy(), g.end.to_numpy(), g.index.to_numpy()
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                if s[j] > e[i]:
                    break
                if min(e[i], e[j]) - s[j] > min_overlap:
                    out.append((int(p[i]), int(p[j]), n))
    return out


TIERS = ((0.86, "likely"), (0.70, "maybe"))   # score cut-offs (held out: >=0.86 about 94% right)


def tier(score: float, labelled: bool) -> str:
    if labelled:
        return "likely"
    for cut, name in TIERS:
        if score >= cut:
            return name
    return "unsure"


def app_json(props: pd.DataFrame, pieces: pd.DataFrame, segments: list[dict], assigned: dict | None = None,
             min_frames: int = 30) -> dict:
    """The proposals for the app's Cards mode: names, and per piece
    [segment index, start, end, team, proposed name, tier, alternatives, labelled by the user].
    With `assigned` (assign_exclusive), a piece left without a name is 'unsure' with its best guesses."""
    seg_of = lambda f: next((i for i, s in enumerate(segments) if s["seg_start"] <= f < s["seg_start"] + s["frames"]), -1)
    p = props.merge(pieces, left_on="piece", right_index=True)
    p = p[p.frames >= min_frames]
    out = {}
    for r in p.itertuples():
        lab = r.labelled if isinstance(r.labelled, str) else None
        name, score = (r.name, r.score)
        if assigned is not None:
            name, score = assigned.get(int(r.piece), (r.name, -1.0))
        alts = [n for n, _ in r.scores if n != name][:2]
        out[str(r.piece)] = [seg_of(r.start), int(r.start), int(r.end), int(r.team), lab or name,
                             tier(score, lab is not None), alts, 1 if lab else 0]
    names = sorted({v[4] for v in out.values()} | {a for v in out.values() for a in v[6]})
    return {"names": names, "pieces": out}
