"""Score team classification (per_frame_tracks.csv team_id) against per-track team labels.

  python -m tools.team_eval --gt data/cvat/hilal_hazm_B/tracks.csv --teams data/cvat/hilal_hazm_B/teams.json \
      --run runs/x/per_frame_tracks.csv

teams.json: {"team0": [gt track ids], "team1": [...], "goalkeepers": [...], "unknown": [...]} (outfield
players only are scored). Predicted players are matched to ground-truth boxes per frame (IoU >= 0.5);
the pipeline's team numbers are arbitrary, so the better of the two mappings is used.

  accuracy      share of matched player boxes with a team that have the right team
  coverage      share of matched player boxes that have a team at all
  wrong_boxes   player boxes shown in the other team's colour
  flip_ids      predicted IDs whose team changes during the clip (a player "joins the other team")
  worst_gt      ground-truth players most often given the wrong team
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from tools.benchmark import _iou_matrix


def match_players(pred: pd.DataFrame, gt: pd.DataFrame, teams: dict, iou: float = 0.5) -> pd.DataFrame:
    """One row per matched ground-truth outfield player box: frame, gt track, true team, predicted id/team."""
    truth = {int(t): 0 for t in teams["team0"]} | {int(t): 1 for t in teams["team1"]}
    g = gt[gt.track_id.isin(list(truth))]
    p = pred[pred.class_id == 2]
    pf = {f: d for f, d in p.groupby("frame")}
    rows = []
    for f, gg in g.groupby("frame"):
        pp = pf.get(f)
        if pp is None or len(pp) == 0:
            continue
        m = _iou_matrix(gg[["x1", "y1", "x2", "y2"]].to_numpy(float), pp[["x1", "y1", "x2", "y2"]].to_numpy(float))
        used = set()
        for i in np.argsort(-m.max(1)):  # greedy, best overlaps first
            j = int(np.argmax(np.where(np.isin(np.arange(m.shape[1]), list(used)), -1, m[i])))
            if m[i, j] < iou or j in used:
                continue
            used.add(j)
            gr, pr = gg.iloc[i], pp.iloc[j]
            rows.append({"frame": int(f), "gt_track": int(gr.track_id), "true": truth[int(gr.track_id)],
                         "pred_id": int(pr.display_track_id), "pred_team": int(pr.team_id)})
    return pd.DataFrame(rows)


def team_scores(pred: pd.DataFrame, gt: pd.DataFrame, teams: dict) -> dict:
    m = match_players(pred, gt, teams)
    if m.empty:
        return {"matched_boxes": 0}
    has = m[m.pred_team >= 0]
    same = int((has.pred_team == has.true).sum())
    flip = same < len(has) - same  # pipeline team numbers are arbitrary
    has = has.assign(ok=(has.pred_team != has.true) if flip else (has.pred_team == has.true))
    flips = 0
    for _, d in has.sort_values("frame").groupby("pred_id"):
        flips += int((d.pred_team.diff().fillna(0) != 0).any())
    worst = (has.groupby("gt_track").ok.agg(["mean", "size"]).query("mean < 1").sort_values("mean").head(5))
    return {"matched_boxes": len(m), "coverage": round(len(has) / len(m), 3),
            "accuracy": round(float(has.ok.mean()), 3) if len(has) else None,
            "wrong_boxes": int((~has.ok).sum()), "flip_ids": flips, "ids_with_team": int(has.pred_id.nunique()),
            "worst_gt": {int(k): round(float(v), 2) for k, v in worst["mean"].items()}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--teams", required=True)
    ap.add_argument("--run", action="append", required=True, help="[name=]per_frame_tracks.csv (repeatable)")
    a = ap.parse_args(argv)
    gt, teams = pd.read_csv(a.gt), json.load(open(a.teams))
    for spec in a.run:
        name, path = spec.split("=", 1) if "=" in spec else (spec, spec)
        print(name, json.dumps(team_scores(pd.read_csv(path, low_memory=False), gt, teams)))


if __name__ == "__main__":
    main()
