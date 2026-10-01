"""Compare pipeline runs made with faster settings: speed, and whether accuracy holds
(notebooks/speed_settings.ipynb).

  identity(pred, gt)       IDF1 of display ids, and of the tracker's own pieces joined perfectly
                           (the ceiling for tools/join_pieces.py), pieces per minute, piece purity
  agreement(pred, ref)     same players / same ball as a reference run: matched boxes, median pitch
                           position difference (m), ball found where the reference has one, ball px
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tools.benchmark import _assign_max, _iou_matrix, score_against_gt

PEOPLE = (1, 2)


def match_rows(pred: pd.DataFrame, ref: pd.DataFrame, ref_id: str, iou_thr: float = 0.5) -> pd.Series:
    """For each pred row: the matched ref row's `ref_id` (IoU >= thr, one-to-one per frame), else -1."""
    out = pd.Series(-1, index=pred.index, dtype=object)
    rb = {f: g for f, g in ref.groupby("frame")}
    for f, pf in pred.groupby("frame"):
        if f not in rb:
            continue
        gf = rb[f]
        iou = _iou_matrix(gf[["x1", "y1", "x2", "y2"]].to_numpy(float), pf[["x1", "y1", "x2", "y2"]].to_numpy(float))
        for r, c in _assign_max(iou):
            if iou[r, c] >= iou_thr:
                out.loc[pf.index[c]] = gf[ref_id].iloc[r]
    return out


def identity(pred: pd.DataFrame, gt: pd.DataFrame, fps: float) -> dict:
    s, _, _ = score_against_gt(pred, gt, fps=fps)
    res = {"IDF1_display": round(s["IDF1"], 4), "detection_recall": round(s["detection_recall"], 4),
           "detection_precision": round(s["detection_precision"], 4)}
    if "raw_tracker_id" not in pred.columns:
        return res
    p = pred[pred.class_id.isin(PEOPLE) & (pred.raw_tracker_id >= 0)].copy()
    g = gt[gt.class_id == 2].copy()
    g["gid"] = g.display_track_id.astype(int)
    p["gid"] = match_rows(p, g, "gid")
    m = p[p.gid.astype(int) >= 0]
    maj = m.groupby("raw_tracker_id").gid.agg(lambda x: x.value_counts().index[0])
    pur = m.groupby("raw_tracker_id").gid.agg(lambda x: x.value_counts().iloc[0] / len(x))
    n = m.groupby("raw_tracker_id").size()
    o = pred.copy()
    o["oracle"] = o.raw_tracker_id.map(maj)
    o["oracle"] = o.oracle.where(o.oracle.notna(), 100000 + o.raw_tracker_id).astype(int)
    so, _, _ = score_against_gt(o, gt, pred_id_col="oracle", fps=fps)
    minutes = (p.frame.max() - p.frame.min() + 1) / fps / 60 if len(p) else 1
    res.update({"IDF1_pieces_joined_perfectly": round(so["IDF1"], 4),
                "piece_purity": round(float((pur * n).sum() / max(n.sum(), 1)), 4),
                "pieces_per_minute": round(float(p.raw_tracker_id.nunique() / max(minutes, 1e-9)), 1)})
    return res


def agreement(pred: pd.DataFrame, ref: pd.DataFrame) -> dict:
    """How close a run is to a reference run of the same frames."""
    pp, rp = pred[pred.class_id.isin(PEOPLE)].copy(), ref[ref.class_id.isin(PEOPLE)].copy()
    rp["rid"] = np.arange(len(rp))
    pp["rid"] = match_rows(pp, rp, "rid").astype(int)
    m = pp[pp.rid >= 0]
    r = rp.set_index("rid").loc[m.rid]
    d = np.hypot(m.x_m.to_numpy() - r.x_m.to_numpy(), m.y_m.to_numpy() - r.y_m.to_numpy()) / 100.0
    d = d[np.isfinite(d)]
    pb = pred[pred.class_id == 0].drop_duplicates("frame").set_index("frame")
    rb = ref[ref.class_id == 0].drop_duplicates("frame").set_index("frame")
    both = rb.index.intersection(pb.index)
    bd = np.hypot(((pb.loc[both].x1 + pb.loc[both].x2) - (rb.loc[both].x1 + rb.loc[both].x2)) / 2,
                  ((pb.loc[both].y1 + pb.loc[both].y2) - (rb.loc[both].y1 + rb.loc[both].y2)) / 2)
    return {"people_matched": round(len(m) / max(len(rp), 1), 4),
            "position_diff_m_median": round(float(np.median(d)), 3) if len(d) else None,
            "position_diff_m_p90": round(float(np.percentile(d, 90)), 3) if len(d) else None,
            "ball_found_where_ref_has_one": round(len(both) / max(len(rb), 1), 4),
            "ball_px_diff_median": round(float(np.median(bd)), 1) if len(bd) else None,
            "ball_same_spot_share": round(float((bd <= 15).mean()), 4) if len(bd) else None}
