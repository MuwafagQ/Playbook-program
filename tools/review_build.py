"""Data for the Pass Review web app (review/pass_review.html): per segment, the boxes of every frame
and the proposed passes; plus the passes already tagged by hand.

  python -m tools.review_build --export review_export/match_video_2 --out review/site \
      --tagged pass_events.csv --names player_names.csv

Input per segment (notebooks/review_export.ipynb): tracks_small.csv.gz (match frames) and segment.json.
Output: review_index.json (segments, proposals) and boxes_<segment>.json (per segment frame: people
[x1, y1, x2, y2, player, team, class, piece] in source-video pixels, and the ball [x, y]); `player` is
the joined player id, `piece` the raw tracker id (one unbroken track, nearly always one person). Also seed.json: the review and name documents to write into the app's store for the
passes tagged so far (tools seed them with the ArtifactData tool).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.pass_proposals import propose

PEOPLE = (1, 2, 3)


def boxes(t: pd.DataFrame, seg_start: int, frames: int) -> dict:
    people = [[] for _ in range(frames)]
    ball = [None] * frames
    pid = t.joined_id if "joined_id" in t.columns else t.raw_tracker_id
    t = t.assign(pid_=pid.fillna(-1).astype(int), team_=t.team_id.fillna(-1).astype(int),
                 piece_=t.raw_tracker_id.fillna(-1).astype(int))
    for r in t[t.class_id.isin(PEOPLE)].itertuples():
        i = int(r.frame) - seg_start
        if 0 <= i < frames:
            people[i].append([int(r.x1), int(r.y1), int(r.x2), int(r.y2), int(r.pid_), int(r.team_), int(r.class_id), int(r.piece_)])
    for r in t[t.class_id == 0].drop_duplicates("frame").itertuples():
        i = int(r.frame) - seg_start
        if 0 <= i < frames:
            ball[i] = [round((r.x1 + r.x2) / 2), round((r.y1 + r.y2) / 2)]
    return {"people": people, "ball": ball}


def segment_proposals(t: pd.DataFrame, seg: str, fps: float) -> list[dict]:
    p = propose(t, fps=fps)
    out = []
    for i, r in enumerate(p.itertuples()):
        out.append({"id": f"{seg}-{i:03d}", "kick": int(r.kick_frame), "touch": int(r.touch_frame), "source": "auto",
                    "passer": {"jid": int(r.passer), "team": int(r.passer_team), "x": float(r.passer_fx), "y": float(r.passer_fy)},
                    "receiver": {"jid": int(r.receiver), "team": int(r.receiver_team), "x": float(r.receiver_fx), "y": float(r.receiver_fy)},
                    "same_team": bool(r.same_team), "flight_s": float(r.flight_s)})
    return out


def tagged_reviews(ev: pd.DataFrame) -> dict:
    """Passes tagged in the Colab tool -> review documents (already reviewed)."""
    out = {}
    ev = ev.drop_duplicates(["departure_frame", "arrival_frame"]).sort_values("departure_frame")
    for i, e in enumerate(ev.itertuples()):
        def who(prefix, frame):
            piece = getattr(e, f"{prefix}_piece", np.nan)
            name = getattr(e, f"{prefix}_name", None)
            return {"jid": int(piece) if pd.notna(piece) else None, "team": -1, "frame": int(frame),
                    "x": None, "y": None, "name": name if isinstance(name, str) and name else None}
        out[f"tagged-{i:03d}"] = {"status": "pass", "kick": int(e.departure_frame), "touch": int(e.arrival_frame),
                                  "outcome": e.outcome if isinstance(e.outcome, str) else "complete",
                                  "passer": who("passer", e.departure_frame), "receiver": who("receiver", e.arrival_frame),
                                  "added": True, "source": "tagged", "notes": e.notes if isinstance(e.notes, str) else ""}
    return out


def name_docs(names: pd.DataFrame) -> dict:
    out = {}
    for piece, g in names.sort_values("from_frame").groupby("piece"):
        out[str(int(piece))] = {"entries": [[int(f), str(n)] for f, n in zip(g.from_frame, g.name)]}
    return out


def build(export: str, out: str, tagged: str | None = None, names: str | None = None, match: str = "match_video_2") -> dict:
    exp, dst = Path(export), Path(out)
    dst.mkdir(parents=True, exist_ok=True)
    segs = []
    fps = 30.0
    for d in sorted(p for p in exp.iterdir() if (p / "segment.json").exists()):
        info = json.loads((d / "segment.json").read_text())
        fps = float(info["fps"])
        t = pd.read_csv(d / "tracks_small.csv.gz")
        (dst / f"boxes_{info['segment']}.json").write_text(
            json.dumps(boxes(t, info["seg_start"], info["frames"]), separators=(",", ":")))
        segs.append({"name": info["segment"], "seg_start": info["seg_start"], "frames": info["frames"],
                     "width": info["width"], "height": info["height"], "boxes": f"boxes_{info['segment']}.json",
                     "proposals": segment_proposals(t, info["segment"], fps)})
    index = {"match": match, "fps": fps, "segments": segs}
    (dst / "review_index.json").write_text(json.dumps(index, separators=(",", ":")))
    seed = {"reviews": tagged_reviews(pd.read_csv(tagged)) if tagged else {},
            "names": name_docs(pd.read_csv(names)) if names else {}}
    (dst / "seed.json").write_text(json.dumps(seed, indent=1))
    return {"segments": len(segs), "proposals": sum(len(s["proposals"]) for s in segs),
            "tagged": len(seed["reviews"]), "named_players": len(seed["names"])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tagged")
    ap.add_argument("--names")
    a = ap.parse_args(argv)
    print(build(a.export, a.out, a.tagged, a.names))


if __name__ == "__main__":
    main()
