"""tools/gsr_adapter.py on a synthetic SoccerNet clip."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("trackeval")
sys.path.insert(0, str(Path(__file__).parent))
import test_gsr_eval as tg  # noqa: E402
from tools import gsr_adapter as ad  # noqa: E402
from tools import gsr_eval as ge  # noqa: E402

CLS = {"goalkeeper": 1, "player": 2, "referee": 3}


def _tracks_from_gt(labels):
    """What a perfect run of our pipeline would write: template cm, our class ids, team 0/1."""
    fid = {a["image_id"]: int(a["file_name"][:-4]) - 1 for a in labels["images"]}
    inv = lambda v, k: float(np.interp(v, k[1], k[0]))  # noqa: E731
    rows = []
    for a in labels["annotations"]:
        b, p, at = a["bbox_image"], a["bbox_pitch"], a["attributes"]
        rows.append({"frame": fid[a["image_id"]], "track_id": a["track_id"], "class_id": CLS[at["role"]],
                     "conf": 0.9, "x1": b["x"], "y1": b["y"], "x2": b["x"] + b["w"], "y2": b["y"] + b["h"],
                     "x_m": inv(p["x_bottom_middle"], ad.X_KNOTS), "y_m": inv(p["y_bottom_middle"], ad.Y_KNOTS),
                     # our team ids are arbitrary labels: 'right' players get 0 here on purpose
                     "team_id": {"left": 1, "right": 0}.get(at["team"], -1) if at["role"] == "player" else -1})
    return pd.DataFrame(rows)


def test_landmarks_map_exactly():
    x, y = ad.template_to_soccernet([0, 2015, 6000, 9985, 12000], [0, 1450, 3500, 5550, 7000])
    assert np.allclose(x, [-52.5, -36.0, 0.0, 36.0, 52.5]) and np.allclose(y, [-34, -20.16, 0, 20.16, 34])
    x, _ = ad.template_to_soccernet([12100], [3500])  # beyond the goal line: extrapolated, not clamped
    assert x[0] > 52.5


def test_perfect_tracks_score_perfectly_except_jersey(tmp_path):
    labels = tg._gt()
    tracks = _tracks_from_gt(labels)
    chk = ad.orientation_check(tracks, labels)
    assert chk["flip_x=False,flip_y=False"] == pytest.approx(0.0, abs=1e-6)
    assert min(v for k, v in chk.items() if k.startswith("flip") and k != "flip_x=False,flip_y=False") > 5

    preds = ad.to_predictions(tracks, labels)
    roles = {p["attributes"]["role"] for p in preds["predictions"]}
    assert roles == {"player", "goalkeeper", "referee"}
    assert all(p["attributes"]["jersey"] is None for p in preds["predictions"])

    gt_dir = tmp_path / "gt" / "valid" / "SNGS-021"
    gt_dir.mkdir(parents=True)
    (gt_dir / "Labels-GameState.json").write_text(json.dumps(labels))
    (tmp_path / "pred").mkdir()
    (tmp_path / "pred" / "SNGS-021.json").write_text(json.dumps(preds))
    h = {r: v["HOTA"] for r, v in ge.evaluate(str(tmp_path / "gt" / "valid"),
                                              {"o": str(tmp_path / "pred")})["o"].items()}
    # team side recovered from positions despite arbitrary team ids; only jersey is missing
    for rung in ("image_hota", "pitch", "+role", "+team"):
        assert h[rung] == pytest.approx(100.0)
    assert h["gs_hota"] < 100.0
