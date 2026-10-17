"""Score recorded detector caches (old vs NAS) against the HILAL-HAZM CVAT ground truth."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tools.ball_eval import ball_eval, candidate_recall
from tools.benchmark import _iou_matrix, score_against_gt

SP = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")  # holds nas_eval/<name>/ caches and nas_eval_runs/<name>/ replays
GT = pd.read_csv("/home/user/Playbook-program/data/cvat/hilal_hazm_B/tracks.csv")
NAMES = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}
PEOPLE = (1, 2, 3)
CONF = 0.30  # the pipeline's per-class thresholds (baseline)


def match_frame(p, g, thr=0.5):
    """Greedy IoU matching regardless of class. Returns list of (pi, gi)."""
    if len(p) == 0 or len(g) == 0:
        return []
    iou = _iou_matrix(p[["x1", "y1", "x2", "y2"]].to_numpy(float), g[["x1", "y1", "x2", "y2"]].to_numpy(float))
    pairs = []
    while True:
        i, j = np.unravel_index(np.argmax(iou), iou.shape)
        if iou[i, j] < thr:
            break
        pairs.append((i, j))
        iou[i, :] = -1
        iou[:, j] = -1
    return pairs


def detection_scores(dets):
    d = dets[(dets.class_id.isin(PEOPLE)) & (dets.conf >= CONF)]
    g = GT[GT.class_id.isin(PEOPLE)]
    gf = {f: x.reset_index(drop=True) for f, x in g.groupby("frame")}
    df = {f: x.reset_index(drop=True) for f, x in d.groupby("frame")}
    conf_mat = pd.DataFrame(0, index=[NAMES[c] for c in PEOPLE] + ["missed"],
                            columns=[NAMES[c] for c in PEOPLE] + ["false"])
    for f in sorted(set(gf) | set(df)):
        p, gg = df.get(f, d.iloc[:0]), gf.get(f, g.iloc[:0])
        pairs = match_frame(p, gg)
        mp, mg = {a for a, _ in pairs}, {b for _, b in pairs}
        for a, b in pairs:
            conf_mat.loc[NAMES[gg.class_id[b]], NAMES[p.class_id[a]]] += 1
        for b in set(range(len(gg))) - mg:
            conf_mat.loc["missed", NAMES[gg.class_id[b]]] += 1
        for a in set(range(len(p))) - mp:
            conf_mat.loc[NAMES[p.class_id[a]], "false"] += 1
    per = {}
    for c in PEOPLE:
        n = NAMES[c]
        gt_n = int((g.class_id == c).sum())
        found_any_class = int(conf_mat.loc[n, [NAMES[k] for k in PEOPLE]].sum())
        right_class = int(conf_mat.loc[n, n])
        pred_n = int((d.class_id == c).sum())
        per[n] = {"gt_boxes": gt_n, "recall_any_class": found_any_class / gt_n, "recall_right_class": right_class / gt_n,
                  "precision": right_class / max(pred_n, 1)}
    return per, conf_mat


def main():
    out = {}
    for name in ("old", "nas"):
        cache = SP / "nas_eval" / name
        dets = pd.read_csv(cache / "dets.csv.gz")
        per, cm = detection_scores(dets)
        balls = dets[dets.class_id == 0]
        cr = candidate_recall(balls, GT, thresholds=(0.1, 0.2, 0.3, 0.5))
        top, _ = ball_eval(balls, GT)
        res = {"people": per, "confusion": cm.to_dict(), "ball_candidates": cr, "ball_top1_raw": top}
        run = SP / "nas_eval_runs" / name / "per_frame_tracks.csv"
        if run.exists():
            pred = pd.read_csv(run, low_memory=False)
            ids, _, _ = score_against_gt(pred, GT, classes=(2,), fps=30.0)
            res["identity_players"] = {k: ids[k] for k in ("idf1", "id_switches", "mota", "idp", "idr") if k in ids}
            res["ball_pipeline"], _ = ball_eval(pred, GT)
        out[name] = res
        print(f"\n=== {name} ===")
        print(json.dumps(per, indent=1))
        print(cm)
        print("ball candidates:", json.dumps(cr))
        print("ball top-1 raw:", {k: round(v, 3) if isinstance(v, float) else v for k, v in top.items()})
        if "identity_players" in res:
            print("identity:", res["identity_players"])
            print("ball after pipeline:", {k: round(v, 3) if isinstance(v, float) else v for k, v in res["ball_pipeline"].items()})
    Path("data/nas_eval/hazm_old_vs_nas.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
