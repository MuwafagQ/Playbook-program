"""Collect the answers from the Pass Review web app into the match's pass data.

  # reviews read from the app's store (ArtifactData list ... out_dir=reviews_dir)
  python -m tools.review_collect --reviews reviews_dir/reviews --export review_export/match_video_2 \
      --index review/site_mv2/review_index.json --out passes_match_video_2.csv

Writes one row per confirmed pass: frames, times, segment, passer / receiver (name, team letter, joined
player id), outcome, note, and pitch positions in metres of passer and receiver at kick and touch
(from the tracks; for a player added by hand, from the image->pitch map fitted on that frame's
players). Also reports how good the proposals were: accepted as proposed, accepted after a fix,
rejected, and passes added by hand (missed by the detector).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_reviews(folder: str) -> dict:
    out = {}
    for f in Path(folder).glob("*.json"):
        d = json.loads(f.read_text())
        out[f.stem] = d.get("data", d)
    return out


def _tracks(export: str) -> pd.DataFrame:
    parts = [pd.read_csv(p) for p in sorted(Path(export).glob("*/tracks_small.csv.gz"))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def pitch_m(t_frame: pd.DataFrame, who: dict | None):
    """(x, y) in metres for a passer / receiver at its frame, or (None, None)."""
    if not who or t_frame is None or not len(t_frame):
        return None, None
    ppl = t_frame[t_frame.class_id.isin((1, 2))]
    if who.get("jid") is not None:
        r = ppl[ppl.joined_id == who["jid"]]
        if len(r) and pd.notna(r.x_m.iloc[0]):
            return round(float(r.x_m.iloc[0]) / 100, 2), round(float(r.y_m.iloc[0]) / 100, 2)
    if who.get("x") is None:
        return None, None
    g = ppl.dropna(subset=["x_m", "y_m"])
    if len(g) < 4:
        return None, None
    import cv2
    src = np.c_[(g.x1 + g.x2) / 2, g.y2].astype(np.float32)
    H, _ = cv2.findHomography(src, g[["x_m", "y_m"]].to_numpy(np.float32), cv2.RANSAC, 150.0)
    if H is None:
        return None, None
    p = cv2.perspectiveTransform(np.array([[[who["x"], who["y"]]]], np.float32), H)[0, 0]
    return round(float(p[0]) / 100, 2), round(float(p[1]) / 100, 2)


def team_letter(name) -> str:
    return name[0] if isinstance(name, str) and name[:1] in ("A", "B") else "?"


def collect(reviews: dict, tracks: pd.DataFrame, index: dict | None, fps: float = 30.0) -> tuple[pd.DataFrame, dict]:
    by_frame = {f: g for f, g in tracks.groupby("frame")} if len(tracks) else {}
    segs = (index or {}).get("segments", [])
    seg_of = lambda f: next((s["name"] for s in segs if s["seg_start"] <= f < s["seg_start"] + s["frames"]), "")
    rows = []
    for rid, r in sorted(reviews.items(), key=lambda kv: kv[1].get("kick", 0)):
        if r.get("status") != "pass":
            continue
        k, t = int(r["kick"]), int(r["touch"])
        ps, rc = r.get("passer") or {}, r.get("receiver") or {}
        pxm, pym = pitch_m(by_frame.get(k), ps)
        rxm, rym = pitch_m(by_frame.get(t), rc)
        rows.append({"review_id": rid, "segment": seg_of(k), "kick_frame": k, "kick_s": round(k / fps, 2),
                     "touch_frame": t, "touch_s": round(t / fps, 2), "duration_s": round((t - k) / fps, 2),
                     "passer": ps.get("name"), "passer_team": team_letter(ps.get("name")), "passer_player_id": ps.get("jid"),
                     "receiver": rc.get("name"), "receiver_team": team_letter(rc.get("name")), "receiver_player_id": rc.get("jid"),
                     "outcome": r.get("outcome"), "note": r.get("notes", ""), "source": r.get("source"),
                     "passer_x_m": pxm, "passer_y_m": pym, "receiver_x_m": rxm, "receiver_y_m": rym,
                     "length_m": round(float(np.hypot(rxm - pxm, rym - pym)), 1) if None not in (pxm, rxm) else None})
    passes = pd.DataFrame(rows)
    # proposal quality, only over segments where something was answered
    props = {p["id"]: p for s in segs for p in s["proposals"]}
    q = {"accepted_as_proposed": 0, "accepted_after_fix": 0, "rejected": 0, "added_by_hand": 0, "tagged_in_colab": 0}
    for rid, r in reviews.items():
        if rid in props:
            if r.get("status") != "pass":
                q["rejected"] += 1
                continue
            p = props[rid]
            same = (abs(r["kick"] - p["kick"]) <= 3 and abs(r["touch"] - p["touch"]) <= 3
                    and (r.get("passer") or {}).get("jid") == p["passer"]["jid"]
                    and (r.get("receiver") or {}).get("jid") == p["receiver"]["jid"])
            q["accepted_as_proposed" if same else "accepted_after_fix"] += 1
        elif r.get("source") == "tagged":
            q["tagged_in_colab"] += 1
        elif r.get("status") == "pass":
            q["added_by_hand"] += 1
    answered = q["accepted_as_proposed"] + q["accepted_after_fix"] + q["rejected"]
    q["proposal_precision"] = round((q["accepted_as_proposed"] + q["accepted_after_fix"]) / answered, 3) if answered else None
    found = q["accepted_as_proposed"] + q["accepted_after_fix"]
    q["proposal_recall"] = round(found / (found + q["added_by_hand"]), 3) if found + q["added_by_hand"] else None
    return passes, q


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reviews", required=True)
    ap.add_argument("--export", required=True)
    ap.add_argument("--index")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    index = json.loads(Path(a.index).read_text()) if a.index else None
    passes, q = collect(load_reviews(a.reviews), _tracks(a.export), index, (index or {}).get("fps", 30.0))
    passes.to_csv(a.out, index=False)
    Path(a.out).with_suffix(".quality.json").write_text(json.dumps(q, indent=1))
    print(len(passes), "passes;", q)


if __name__ == "__main__":
    main()
