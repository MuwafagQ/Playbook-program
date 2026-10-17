"""Join the tracker's pieces into players by appearance (tools/reid_appearance.py model), team and
time: two pieces on screen at the same time are never the same player.

  python -m tools.join_pieces --video match.mp4 --tracks runs/x/per_frame_tracks.csv \
      --model MyDrive/Playbook/reid_appearance/appearance_model.pt --out runs/x/per_frame_tracks_joined.csv

Adds `joined_id` (1..N by first appearance; goalkeepers and players, -1 for everything else); the
pass-tagging tool names joined players instead of single pieces when the column is there.

Method: each piece (`raw_tracker_id`) gets the mean embedding of up to 12 of its crops; then
agglomerative clustering on cosine similarity of cluster centroids, merging the most similar pair
first, never across known teams or between goalkeeper and player, never when the two clusters share
more than `max_overlap` frames; stops below `thr`.
Measured on the hand-unified HIL-HAZ windows, each half scored with a model trained on the other:
IDF1 0.57 -> 0.80 (half 1) and 0.66 -> 0.73 (half 2) at thr 0.3 (perfect joining of these pieces: 0.89 / 0.79).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

PEOPLE = (1, 2)


def piece_table(tracks: pd.DataFrame) -> dict:
    """piece -> frames (sorted array), team (majority known team or -1), class (majority)."""
    t = tracks[tracks.class_id.isin(PEOPLE) & (tracks.raw_tracker_id >= 0)]
    out = {}
    for p, g in t.groupby("raw_tracker_id"):
        known = g.team_id[g.team_id >= 0] if "team_id" in g else pd.Series(dtype=int)
        out[int(p)] = {"frames": np.unique(g.frame.to_numpy()),
                       "team": int(known.mode().iloc[0]) if len(known) else -1,
                       "cls": int(g.class_id.mode().iloc[0])}
    return out


def cluster(pieces: dict, emb: np.ndarray, crop_piece, thr: float = 0.3, max_overlap: int = 3,
            k: int = 30) -> dict:
    """piece -> cluster key (a member piece). Pieces without crops stay on their own.
    Candidate pairs are each cluster's k most similar allowed clusters, in a priority queue (stale
    entries are skipped), so a full match (thousands of pieces) takes seconds."""
    import heapq

    crop_piece = np.asarray(crop_piece)
    order = np.argsort(crop_piece, kind="stable")
    cp_sorted = crop_piece[order]
    keys = [p for p in pieces if p in set(cp_sorted.tolist())]
    n = len(keys)
    lab = {p: p for p in pieces}
    if n < 2:
        return lab
    starts = np.searchsorted(cp_sorted, keys, side="left")
    ends = np.searchsorted(cp_sorted, keys, side="right")
    V = np.stack([emb[order[a:b]].astype(np.float64).sum(0) for a, b in zip(starts, ends)])
    U = V / np.linalg.norm(V, axis=1, keepdims=True)
    team = np.array([pieces[p]["team"] for p in keys])
    cls = np.array([pieces[p]["cls"] for p in keys])
    frames = [pieces[p]["frames"] for p in keys]
    members = [[p] for p in keys]
    alive = np.ones(n, bool)
    version = np.zeros(n, int)
    heap: list = []

    def push(i):
        s = U @ U[i]
        bad = ~alive | (cls != cls[i]) | ((team >= 0) & (team[i] >= 0) & (team != team[i]))
        bad[i] = True
        s[bad] = -np.inf
        top = np.argpartition(-s, min(k, n - 1))[:k]
        for j in top:
            if s[j] >= thr:
                heapq.heappush(heap, (-float(s[j]), int(i), int(j), int(version[i]), int(version[j])))

    for i in range(0, n, 512):           # initial candidates, in blocks
        blk = np.arange(i, min(n, i + 512))
        S = U[blk] @ U.T
        bad = (cls[blk][:, None] != cls[None, :]) | ((team[blk][:, None] >= 0) & (team[None, :] >= 0)
                                                    & (team[blk][:, None] != team[None, :]))
        S[bad] = -np.inf
        S[np.arange(len(blk)), blk] = -np.inf
        top = np.argpartition(-S, min(k, n - 1), axis=1)[:, :k]
        for r, i_ in enumerate(blk):
            for j in top[r]:
                if S[r, j] >= thr and i_ < j:
                    heapq.heappush(heap, (-float(S[r, j]), int(i_), int(j), 0, 0))
    while heap:
        _, i, j, vi, vj = heapq.heappop(heap)
        if not (alive[i] and alive[j]) or version[i] != vi or version[j] != vj:
            continue
        if (team[i] >= 0 and team[j] >= 0 and team[i] != team[j]) or cls[i] != cls[j]:
            continue
        if len(np.intersect1d(frames[i], frames[j], assume_unique=True)) > max_overlap:
            continue
        V[i] += V[j]
        U[i] = V[i] / np.linalg.norm(V[i])
        frames[i] = np.union1d(frames[i], frames[j])
        members[i] += members[j]
        if team[i] < 0:
            team[i] = team[j]
        alive[j] = False
        U[j] = 0.0
        version[i] += 1
        push(i)
    for i in np.where(alive)[0]:
        for p in members[i]:
            lab[p] = members[i][0]
    return lab


def joined_ids(tracks: pd.DataFrame, lab: dict) -> pd.Series:
    """joined_id per row: 1..N by first appearance, -1 for rows that are not goalkeeper/player pieces."""
    people = tracks.class_id.isin(PEOPLE) & (tracks.raw_tracker_id >= 0)
    key = tracks.raw_tracker_id.where(people).map(lab)
    first = tracks.assign(key=key)[people].groupby("key").frame.min().sort_values()
    num = {k: i for i, k in enumerate(first.index, start=1)}
    out = key.map(num).fillna(-1).astype(int)
    # joined pieces may share up to max_overlap frames: there, the longer piece keeps the id and the
    # other piece gets an id of its own, so an id is never on two boxes in one frame
    t = pd.DataFrame({"frame": tracks.frame, "piece": tracks.raw_tracker_id, "jid": out})[people]
    size = t.piece.map(t.piece.value_counts())
    t = t.assign(size=size).sort_values(["frame", "jid", "size"], ascending=[True, True, False])
    dup = t.duplicated(["frame", "jid"], keep="first")
    if dup.any():
        extra = {p: out.max() + i for i, p in enumerate(sorted(t.piece[dup].unique()), start=1)}
        out.loc[t.index[dup]] = t.piece[dup].map(extra).to_numpy()
    return out


def join(tracks: pd.DataFrame, video: str, model_path: str, thr: float = 0.3, per_piece: int = 12) -> tuple[pd.DataFrame, dict]:
    import torch

    from tools.reid_appearance import Embedder, crops_from_video, sample_rows

    t = tracks[tracks.class_id.isin(PEOPLE) & (tracks.raw_tracker_id >= 0)]
    crops, rows = crops_from_video(video, sample_rows(t, "raw_tracker_id", per_id=per_piece))
    model = Embedder()
    model.load_state_dict(torch.load(model_path, map_location=model.device))
    emb = model.embed(crops)
    pieces = piece_table(tracks)
    lab = cluster(pieces, emb, rows.raw_tracker_id.to_numpy(), thr=thr)
    out = tracks.copy()
    out["joined_id"] = joined_ids(out, lab)
    info = {"pieces": len(pieces), "players_joined": int(out.joined_id[out.joined_id > 0].nunique()),
            "crops": len(crops), "thr": thr}
    return out, info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.3)
    a = ap.parse_args(argv)
    out, info = join(pd.read_csv(a.tracks, low_memory=False), a.video, a.model, a.thr)
    out.to_csv(a.out, index=False)
    json.dump(info, open(a.out.rsplit(".", 1)[0] + "_info.json", "w"), indent=1)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
