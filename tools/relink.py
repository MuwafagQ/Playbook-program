"""Join track pieces of the same player after the tracker lost them (camera pan, long occlusion), by
pitch position, team and timing; everything uncertain becomes a ranked merge list for a person to
confirm.

  python -m tools.relink --run runs/x/per_frame_tracks.csv --out runs/x/relinked.csv --fps 30

A piece is one `track_id` (the pipeline already re-links gaps up to ID_RELINK_FRAMES inside it).
Piece A can continue as a later piece B when:
  - same class (player / goalkeeper) and, when both are known, the same team;
  - B starts after A ends, within max_gap_s;
  - B starts within reach of where A ended: margin_m + max_speed_mps * gap.
score = distance / reach (0 = exactly where A ended, 1 = at the edge of reach). Each piece gets at most
one predecessor and one successor (best score first). A link is applied automatically when it is
confident (score <= auto_score, gap <= auto_gap_s, both teams known); all other feasible links are
listed as suggestions, best first, for the review page. Automatic joining is OFF by default
(auto_score=0): on the 20 s CVAT clips all 3 automatic joins were wrong - there most ID errors are swaps at
crossings, not lost-and-found players; tune it on the long hand-unified windows first.

Output CSV: `display_track_id` becomes the joined player id (1..N per class, by first appearance), the
pipeline's own ids are kept in `piece_id` (= track_id) and `display_track_id_orig`. Output JSON: the
applied links and the suggestions.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

PEOPLE = (1, 2)  # goalkeeper, player (referees keep their ids)


def pieces(tracks: pd.DataFrame, ends_n: int = 10) -> pd.DataFrame:
    """One row per track_id: class, team, first/last frame, pitch position (m) at both ends."""
    rows = []
    t = tracks[tracks.class_id.isin(PEOPLE) & (tracks.track_id >= 0)]
    for tid, g in t.sort_values("frame").groupby("track_id"):
        v = g.dropna(subset=["x_m", "y_m"])
        teams = g.team_id[g.team_id >= 0] if "team_id" in g else pd.Series(dtype=int)
        rows.append({
            "piece": int(tid), "class_id": int(g.class_id.mode().iloc[0]),
            "team": int(teams.mode().iloc[0]) if len(teams) else -1,
            "start": int(g.frame.iloc[0]), "end": int(g.frame.iloc[-1]), "rows": len(g),
            "sx": float(v.x_m.head(ends_n).median() / 100) if len(v) else np.nan,
            "sy": float(v.y_m.head(ends_n).median() / 100) if len(v) else np.nan,
            "ex": float(v.x_m.tail(ends_n).median() / 100) if len(v) else np.nan,
            "ey": float(v.y_m.tail(ends_n).median() / 100) if len(v) else np.nan,
        })
    return pd.DataFrame(rows)


def candidates(p: pd.DataFrame, fps: float, max_gap_s: float = 60.0, max_speed_mps: float = 7.0,
               margin_m: float = 4.0) -> pd.DataFrame:
    """All feasible (A ends -> B starts) links with their score."""
    out = []
    for a in p.itertuples():
        later = p[(p.start > a.end) & (p.class_id == a.class_id) & ((p.start - a.end) / fps <= max_gap_s)]
        for b in later.itertuples():
            if a.team >= 0 and b.team >= 0 and a.team != b.team:
                continue
            gap = (b.start - a.end) / fps
            if np.isnan(a.ex) or np.isnan(b.sx):
                continue
            dist = float(np.hypot(b.sx - a.ex, b.sy - a.ey))
            reach = margin_m + max_speed_mps * gap
            if dist > reach:
                continue
            out.append({"from": a.piece, "to": b.piece, "gap_s": round(gap, 2), "dist_m": round(dist, 1),
                        "score": round(dist / reach, 3), "teams_known": bool(a.team >= 0 and b.team >= 0)})
    return pd.DataFrame(out, columns=["from", "to", "gap_s", "dist_m", "score", "teams_known"])


def link(c: pd.DataFrame, auto_score: float = 0.0, auto_gap_s: float = 10.0) -> tuple[list[dict], list[dict]]:
    """One-to-one links, best score first. Returns (applied, suggestions)."""
    applied, used_from, used_to = [], set(), set()
    for r in c.sort_values(["score", "gap_s"]).itertuples():
        if r.score > auto_score or r.gap_s > auto_gap_s or not r.teams_known:
            continue
        if r._1 in used_from or r.to in used_to:
            continue
        applied.append({"from": int(r._1), "to": int(r.to), "gap_s": r.gap_s, "dist_m": r.dist_m, "score": r.score})
        used_from.add(r._1); used_to.add(r.to)
    # suggestions: for every piece end not linked, its feasible continuations not already taken, best first
    sugg = []
    for f, g in c.sort_values("score").groupby("from"):
        if f in used_from:
            continue
        opts = g[~g.to.isin(used_to)].head(3)
        if len(opts):
            sugg.append({"from": int(f), "options": [{"to": int(o.to), "gap_s": o.gap_s, "dist_m": o.dist_m,
                                                       "score": o.score} for o in opts.itertuples()]})
    sugg.sort(key=lambda s: s["options"][0]["score"])
    return applied, sugg


def apply_links(tracks: pd.DataFrame, applied: list[dict]) -> pd.DataFrame:
    """display_track_id := joined player id (per class, 1..N by first appearance)."""
    succ = {a["from"]: a["to"] for a in applied}
    pred = {a["to"]: a["from"] for a in applied}
    root = {}
    for tid in tracks.track_id.unique():
        r = int(tid)
        while r in pred:
            r = pred[r]
        root[int(tid)] = r
    out = tracks.copy()
    out["piece_id"] = out.track_id
    out["display_track_id_orig"] = out.display_track_id
    people = out.class_id.isin(PEOPLE) & (out.track_id >= 0)
    first = (out[people].assign(root=out.loc[people, "track_id"].map(root))
             .groupby(["class_id", "root"]).frame.min().reset_index().sort_values("frame"))
    new_id = {}
    for cls, g in first.groupby("class_id"):
        for i, r in enumerate(g.root, start=1):
            new_id[(cls, r)] = i
    out.loc[people, "display_track_id"] = [new_id[(c, root[int(t)])]
                                           for c, t in zip(out.loc[people, "class_id"], out.loc[people, "track_id"])]
    return out


def relink(tracks: pd.DataFrame, fps: float, **kw) -> tuple[pd.DataFrame, dict]:
    ckw = {k: kw[k] for k in ("max_gap_s", "max_speed_mps", "margin_m") if k in kw}
    lkw = {k: kw[k] for k in ("auto_score", "auto_gap_s") if k in kw}
    p = pieces(tracks)
    c = candidates(p, fps, **ckw)
    applied, sugg = link(c, **lkw)
    return apply_links(tracks, applied), {"pieces": len(p), "applied": applied, "suggestions": sugg}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=25.0)
    a = ap.parse_args(argv)
    t = pd.read_csv(a.run, low_memory=False)
    out, info = relink(t, a.fps)
    out.to_csv(a.out, index=False)
    json.dump(info, open(a.out.rsplit(".", 1)[0] + "_links.json", "w"), indent=1)
    print(f"{info['pieces']} pieces, {len(info['applied'])} joined automatically, {len(info['suggestions'])} to review")


if __name__ == "__main__":
    main()
