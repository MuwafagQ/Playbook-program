"""Score the ball track against ground truth, and count boots taken for the ball.

  python -m tools.ball_eval --pred runs/X/per_frame_tracks.csv --gt data/cvat/hilal_ahli_B/tracks.csv
  python -m tools.ball_eval --pred runs/SNGS-021/per_frame_tracks.csv --gt-soccernet valid/SNGS-021/Labels-GameState.json

Compares ball centres (our ball boxes are padded, so box overlap would mislead). A predicted
ball is a hit if its centre is within max(min_px, k * ground-truth ball width) of the true
ball. Every frame falls in one class:
  hit         we output the ball, in the right place (split: real detection / interpolated)
  wrong_feet  we output a "ball" in the wrong place, at a player's feet (a boot, usually)
  wrong_other we output a "ball" in the wrong place, elsewhere
  miss        the ball is visible but we output nothing
  false_feet / false_other   no ball in the ground truth, but we output one
Feet = the bottom 30% of a ground-truth person box, widened by 20%.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BALL, PEOPLE = 0, (1, 2, 3)


def gt_from_soccernet(labels: dict) -> pd.DataFrame:
    fid = {str(im["image_id"]): int(Path(im["file_name"]).stem) - 1 for im in labels["images"]}
    cls = {"ball": 0, "goalkeeper": 1, "player": 2, "referee": 3, "other": 3}
    rows = []
    for a in labels["annotations"]:
        if a["supercategory"] != "object" or a["attributes"]["role"] not in cls:
            continue
        b = a["bbox_image"]
        rows.append({"frame": fid[str(a["image_id"])], "class_id": cls[a["attributes"]["role"]],
                     "x1": b["x"], "y1": b["y"], "x2": b["x"] + b["w"], "y2": b["y"] + b["h"]})
    return pd.DataFrame(rows)


def _centre(r):
    return np.array([(r.x1 + r.x2) / 2.0, (r.y1 + r.y2) / 2.0])


def _at_feet(pt, people: pd.DataFrame) -> bool:
    if people is None or people.empty:
        return False
    w = people.x2 - people.x1
    h = people.y2 - people.y1
    inside = ((pt[0] >= people.x1 - 0.2 * w) & (pt[0] <= people.x2 + 0.2 * w)
              & (pt[1] >= people.y2 - 0.3 * h) & (pt[1] <= people.y2 + 0.1 * h))
    return bool(inside.any())


def ball_eval(pred: pd.DataFrame, gt: pd.DataFrame, frames=None, min_px: float = 8.0, k: float = 1.5):
    """Returns (summary dict, per-frame DataFrame)."""
    pb = pred[pred.class_id == BALL]
    if "conf" in pb.columns:
        pb = pb.sort_values("conf", ascending=False)
    pb = pb.drop_duplicates("frame").set_index("frame")
    gb = gt[gt.class_id == BALL].drop_duplicates("frame").set_index("frame")
    gp = {f: g for f, g in gt[gt.class_id.isin(PEOPLE)].groupby("frame")}
    if frames is None:
        frames = sorted(set(gt.frame.unique()) | set(pb.index))
    rows = []
    for f in frames:
        has_gt, has_p = f in gb.index, f in pb.index
        interp = bool(pb.loc[f].get("ball_interpolated", 0)) if has_p else False
        dist = np.nan
        if has_gt and has_p:
            g, p = gb.loc[f], pb.loc[f]
            dist = float(np.linalg.norm(_centre(p) - _centre(g)))
            tol = max(min_px, k * float(g.x2 - g.x1))
            if dist <= tol:
                kind = "hit"
            else:
                kind = "wrong_feet" if _at_feet(_centre(p), gp.get(f)) else "wrong_other"
        elif has_gt:
            kind = "miss"
        elif has_p:
            kind = "false_feet" if _at_feet(_centre(pb.loc[f]), gp.get(f)) else "false_other"
        else:
            kind = "none"
        rows.append({"frame": f, "kind": kind, "interpolated": interp, "dist_px": dist})
    d = pd.DataFrame(rows)
    vis = d[d.kind.isin(["hit", "wrong_feet", "wrong_other", "miss"])]
    out_ball = d[d.kind.isin(["hit", "wrong_feet", "wrong_other", "false_feet", "false_other"])]
    n = max(len(vis), 1)
    summary = {
        "frames_ball_visible": int(len(vis)),
        "hit_rate": float((vis.kind == "hit").mean()) if len(vis) else None,
        "hit_rate_real_detection": float(((vis.kind == "hit") & ~vis.interpolated).sum() / n),
        "hit_rate_interpolated": float(((vis.kind == "hit") & vis.interpolated).sum() / n),
        "miss_rate": float((vis.kind == "miss").mean()) if len(vis) else None,
        "wrong_at_feet_rate": float((vis.kind == "wrong_feet").mean()) if len(vis) else None,
        "wrong_elsewhere_rate": float((vis.kind == "wrong_other").mean()) if len(vis) else None,
        "precision": float((out_ball.kind == "hit").mean()) if len(out_ball) else None,
        "false_balls_at_feet": int((d.kind == "false_feet").sum()),
        "false_balls_elsewhere": int((d.kind == "false_other").sum()),
        "median_hit_error_px": float(vis.loc[vis.kind == "hit", "dist_px"].median()) if (vis.kind == "hit").any() else None,
    }
    return summary, d


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--gt", help="ground truth in our CSV schema (e.g. tools.cvat_convert tracks.csv)")
    g.add_argument("--gt-soccernet", help="SoccerNet Labels-GameState.json")
    p.add_argument("--min-px", type=float, default=8.0)
    p.add_argument("--k", type=float, default=1.5)
    a = p.parse_args(argv)
    pred = pd.read_csv(a.pred, low_memory=False)
    gt = pd.read_csv(a.gt) if a.gt else gt_from_soccernet(json.loads(Path(a.gt_soccernet).read_text()))
    summary, _ = ball_eval(pred, gt, min_px=a.min_px, k=a.k)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
