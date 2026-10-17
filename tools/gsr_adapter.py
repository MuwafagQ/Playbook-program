"""Convert our pipeline's per_frame_tracks.csv into SoccerNet GSR predictions.

  python -m tools.gsr_adapter --tracks runs/SNGS-021/per_frame_tracks.csv \
      --labels data/SoccerNetGS/valid/SNGS-021/Labels-GameState.json --out preds/ours/SNGS-021.json

What changes on the way:
- pitch position: our homography maps onto the roboflow/sports pitch template, in cm from a
  corner on a 120 x 70 m template. SoccerNet uses metres from the centre spot on a 105 x 68 m
  pitch (x: left goal line -52.5 -> right +52.5 as the camera sees it; y grows towards the
  camera). The template is mapped landmark by landmark (goal line, penalty-box edge, halfway
  line; touchline, box edge, centre line), piecewise-linearly, so the landmarks the
  homography is fitted to land exactly on SoccerNet's.
- role: goalkeeper / player / referee from our class; the ball is left out (not scored).
- team: our team 0/1 per track (majority vote) becomes left/right by where each team's
  players stand on average (the SoccerNet baseline's 'mean_position' rule). Goalkeepers,
  which our pipeline does not assign to a team, take the side of the goal they are nearer.
- jersey: null (our pipeline does not read numbers).
- rows without a pitch position (no homography that frame) are dropped, as TrackLab does.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

GK, PLAYER, REF = 1, 2, 3
ROLE = {GK: "goalkeeper", PLAYER: "player", REF: "referee"}

# (template cm, SoccerNet m) landmark pairs
X_KNOTS = ([0.0, 2015.0, 6000.0, 9985.0, 12000.0], [-52.5, -36.0, 0.0, 36.0, 52.5])
Y_KNOTS = ([0.0, 1450.0, 3500.0, 5550.0, 7000.0], [-34.0, -20.16, 0.0, 20.16, 34.0])


def _piecewise(v: np.ndarray, knots) -> np.ndarray:
    xs, ys = np.asarray(knots[0]), np.asarray(knots[1])
    out = np.interp(v, xs, ys)
    lo, hi = v < xs[0], v > xs[-1]  # extrapolate with the end segments' slopes
    out[lo] = ys[0] + (v[lo] - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
    out[hi] = ys[-1] + (v[hi] - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    return out


def template_to_soccernet(x_cm, y_cm, flip_x: bool = False, flip_y: bool = False):
    x = _piecewise(np.asarray(x_cm, float), X_KNOTS)
    y = _piecewise(np.asarray(y_cm, float), Y_KNOTS)
    return (-x if flip_x else x), (-y if flip_y else y)


def frame_to_image_id(labels: dict) -> dict[int, str]:
    """0-based video frame -> SoccerNet image_id (frame i is image file i+1)."""
    return {int(Path(im["file_name"]).stem) - 1: str(im["image_id"]) for im in labels["images"]}


def to_predictions(tracks: pd.DataFrame, labels: dict, id_col: str = "track_id",
                   flip_x: bool = False, flip_y: bool = False) -> dict:
    df = tracks[tracks.class_id.isin(ROLE)].copy()
    df = df.dropna(subset=["x_m", "y_m"])
    df = df[pd.to_numeric(df[id_col], errors="coerce") >= 0]
    img = frame_to_image_id(labels)
    df = df[df.frame.isin(img)]
    df["px"], df["py"] = template_to_soccernet(df.x_m.to_numpy(), df.y_m.to_numpy(), flip_x, flip_y)
    df["tid"] = pd.to_numeric(df[id_col]).astype(int)

    # team per track by majority vote, then left/right by mean x of each team's players
    team_of = {}
    if "team_id" in df.columns:
        t = df[(df.class_id == PLAYER) & (pd.to_numeric(df.team_id, errors="coerce") >= 0)]
        team_of = t.groupby("tid").team_id.agg(lambda s: int(s.mode().iloc[0])).to_dict()
    players = df[(df.class_id == PLAYER) & df.tid.map(team_of).notna()]
    side_of_team = {}
    if len(players):
        mean_x = players.assign(team=players.tid.map(team_of)).groupby("team").px.mean().sort_values()
        for i, team in enumerate(mean_x.index):
            side_of_team[int(team)] = "left" if i == 0 else "right"
    gk_side = df[df.class_id == GK].groupby("tid").px.median().map(lambda x: "left" if x < 0 else "right").to_dict()

    preds = []
    for r in df.itertuples():
        role = ROLE[int(r.class_id)]
        if role == "player":
            team = side_of_team.get(team_of.get(r.tid))
        elif role == "goalkeeper":
            team = gk_side.get(r.tid)
        else:
            team = None
        x, y = float(r.px), float(r.py)
        preds.append({
            "id": f"{img[int(r.frame)]}{len(preds):05d}", "image_id": img[int(r.frame)],
            "track_id": int(r.tid), "supercategory": "object", "category_id": 1.0,
            "confidence": float(getattr(r, "conf", 1.0)),
            "attributes": {"role": role, "jersey": None, "team": team},
            "bbox_image": {"x": float(r.x1), "y": float(r.y1), "w": float(r.x2 - r.x1), "h": float(r.y2 - r.y1)},
            "bbox_pitch": {"x_bottom_left": x, "y_bottom_left": y, "x_bottom_middle": x,
                           "y_bottom_middle": y, "x_bottom_right": x, "y_bottom_right": y},
        })
    return {"predictions": preds}


def orientation_check(tracks: pd.DataFrame, labels: dict, iou_thr: float = 0.5) -> dict[str, float]:
    """Median pitch distance (m) between our people and the ground truth they overlap in the
    image, for each axis flip. Used once to confirm the template's orientation convention."""
    from tools.benchmark import _assign_max, _iou_matrix

    img = frame_to_image_id(labels)
    gt = [a for a in labels["annotations"] if a["supercategory"] == "object"
          and a["attributes"]["role"] != "ball" and a.get("bbox_pitch")]
    by_img: dict[str, list] = {}
    for a in gt:
        by_img.setdefault(str(a["image_id"]), []).append(a)
    df = tracks[tracks.class_id.isin(ROLE)].dropna(subset=["x_m", "y_m"])
    pairs = []
    for f, g in df.groupby("frame"):
        anns = by_img.get(img.get(int(f), ""), [])
        if not anns:
            continue
        gb = np.array([[a["bbox_image"]["x"], a["bbox_image"]["y"], a["bbox_image"]["x"] + a["bbox_image"]["w"],
                        a["bbox_image"]["y"] + a["bbox_image"]["h"]] for a in anns], float)
        iou = _iou_matrix(g[["x1", "y1", "x2", "y2"]].to_numpy(float), gb)
        for r, c in _assign_max(iou):
            if iou[r, c] >= iou_thr:
                p = anns[c]["bbox_pitch"]
                pairs.append((g.x_m.iloc[r], g.y_m.iloc[r], p["x_bottom_middle"], p["y_bottom_middle"]))
    if not pairs:
        return {}
    a = np.array(pairs)
    out = {}
    for fx in (False, True):
        for fy in (False, True):
            x, y = template_to_soccernet(a[:, 0], a[:, 1], fx, fy)
            out[f"flip_x={fx},flip_y={fy}"] = float(np.median(np.hypot(x - a[:, 2], y - a[:, 3])))
    out["pairs"] = float(len(a))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tracks", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--id-col", default="track_id")
    p.add_argument("--flip-x", action="store_true")
    p.add_argument("--flip-y", action="store_true")
    p.add_argument("--check-orientation", action="store_true", help="print median distance per axis flip")
    a = p.parse_args(argv)
    tracks = pd.read_csv(a.tracks, low_memory=False)
    labels = json.loads(Path(a.labels).read_text())
    if a.check_orientation:
        print(json.dumps(orientation_check(tracks, labels), indent=2))
    preds = to_predictions(tracks, labels, a.id_col, a.flip_x, a.flip_y)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(preds))
    print(f"{len(preds['predictions'])} predictions -> {a.out}")


if __name__ == "__main__":
    main()
