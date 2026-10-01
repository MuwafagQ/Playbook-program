"""Full match from per-segment pipeline runs (notebooks/full_match.ipynb): one tracks table on one
match timeline, teams labelled the same in every segment, pieces joined into players.

  merge_segments(segments)          global `frame` (segments back to back), `segment` / `segment_frame`,
                                    tracker ids made unique per segment
  team_mapping(...)                 per segment: does its team 0/1 match the reference's 0/1 or are
                                    they swapped? (each segment fits its own kit colours, so its labels
                                    are arbitrary); decided by the appearance embeddings of its pieces
  apply_team_mapping(tracks, maps)  relabels team_id
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ID_STRIDE = 1_000_000  # tracker ids of segment i become i * ID_STRIDE + id
PEOPLE = (1, 2)


def merge_segments(segments: list[tuple[str, pd.DataFrame, int]]) -> pd.DataFrame:
    """segments: (name, tracks, number of frames) in match order."""
    out, offset = [], 0
    for i, (name, t, n_frames) in enumerate(segments):
        t = t.copy()
        t["segment"] = name
        t["segment_frame"] = t["frame"]
        t["frame"] = t["frame"] + offset
        for col in ("raw_tracker_id", "track_id"):
            if col in t.columns:
                t[col] = np.where(t[col] >= 0, t[col] + i * ID_STRIDE, t[col])
        if "display_track_id" in t.columns:
            t["display_track_id_segment"] = t["display_track_id"]
        out.append(t)
        offset += int(n_frames)
    return pd.concat(out, ignore_index=True)


def _team_centroids(emb, crop_piece, piece_team: dict, piece_clip: dict, clip):
    c = {}
    for team in (0, 1):
        pieces = [p for p, tm in piece_team.items() if tm == team and piece_clip[p] == clip]
        m = np.isin(crop_piece, pieces)
        if m.any():
            v = emb[m].astype(np.float64).mean(0)
            c[team] = v / np.linalg.norm(v)
    return c


def team_mapping(emb: np.ndarray, crop_piece, piece_team: dict, piece_clip: dict, clips: list,
                 rounds: int = 3) -> dict:
    """{clip: {0: a, 1: b}} mapping each segment's team ids onto common ones, plus the similarity
    margin of the decision ("margin": larger = clearer). Reference = the first segment, then the
    mean of all mapped segments."""
    crop_piece = np.asarray(crop_piece)
    cents = {c: _team_centroids(emb, crop_piece, piece_team, piece_clip, c) for c in clips}
    ref = next((cents[c] for c in clips if len(cents[c]) == 2), None)
    maps = {c: {0: 0, 1: 1, "margin": 0.0} for c in clips}
    if ref is None:
        return maps
    for _ in range(rounds):
        for c in clips:
            k = cents[c]
            if len(k) < 2:
                continue
            same = float(k[0] @ ref[0] + k[1] @ ref[1])
            swap = float(k[0] @ ref[1] + k[1] @ ref[0])
            maps[c] = ({0: 0, 1: 1} if same >= swap else {0: 1, 1: 0})
            maps[c]["margin"] = round(abs(same - swap), 4)
        acc = {0: [], 1: []}
        for c in clips:
            for t, v in cents[c].items():
                acc[maps[c][t]].append(v)
        ref = {t: (np.mean(v, 0) / np.linalg.norm(np.mean(v, 0))) for t, v in acc.items() if v}
        if len(ref) < 2:
            break
    return maps


def apply_team_mapping(tracks: pd.DataFrame, maps: dict) -> pd.DataFrame:
    out = tracks.copy()
    if "team_id" not in out.columns:
        return out
    tid = out["team_id"].to_numpy().copy()
    for c, m in maps.items():
        sel = (out["segment"] == c).to_numpy() & np.isin(tid, (0, 1))
        tid[sel] = np.where(tid[sel] == 0, m[0], m[1])
    out["team_id"] = tid
    return out


def piece_summary(tracks: pd.DataFrame) -> pd.DataFrame:
    """One row per tracker piece (goalkeepers / players): segment, class, team, first / last frame,
    rows, joined_id. Small enough to analyse the joining without the full tracks table."""
    t = tracks[tracks.class_id.isin(PEOPLE) & (tracks.raw_tracker_id >= 0)]
    g = t.groupby("raw_tracker_id")
    out = pd.DataFrame({
        "segment": g.segment.first() if "segment" in t else "",
        "class_id": g.class_id.agg(lambda s: s.mode().iloc[0]),
        "team_id": g.team_id.agg(lambda s: s[s >= 0].mode().iloc[0] if (s >= 0).any() else -1),
        "first": g.frame.min(), "last": g.frame.max(), "rows": g.size(),
        "joined_id": g.joined_id.agg(lambda s: s.mode().iloc[0]) if "joined_id" in t else -1,
    })
    return out.reset_index().rename(columns={"raw_tracker_id": "piece"})


def id_coverage(pieces: pd.DataFrame, id_col: str = "joined_id") -> dict:
    """How concentrated player-frames are on few ids: ids needed to cover 50 / 80 / 90 / 99 % of
    goalkeeper + player rows, and the number of ids with at least 1 / 5 minutes on screen (30 fps)."""
    rows = pieces.groupby(id_col).rows.sum().sort_values(ascending=False)
    cum = rows.cumsum() / rows.sum()
    out = {"ids": int(len(rows))}
    for q in (0.5, 0.8, 0.9, 0.99):
        out[f"ids_for_{int(q * 100)}pct"] = int((cum < q).sum() + 1)
    for m in (1, 5):
        out[f"ids_over_{m}min"] = int((rows >= m * 60 * 30).sum())
    return out
