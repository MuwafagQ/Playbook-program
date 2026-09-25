"""The GSR ladder scorer, on a synthetic clip in SoccerNet's label format."""
from __future__ import annotations

import copy
import json

import pytest

trackeval = pytest.importorskip("trackeval")
from tools import gsr_eval as ge  # noqa: E402

PEOPLE = [  # track_id, role, team, jersey, start x (m)
    (1, "player", "left", "7", -30.0),
    (2, "player", "left", None, -10.0),
    (3, "player", "right", "10", 10.0),
    (4, "goalkeeper", "right", "1", 45.0),
    (5, "referee", None, None, 0.0),
]
N = 20


def _gt():
    images, anns = [], []
    for f in range(1, N + 1):
        iid = f"2021{f:06d}"
        images.append({"image_id": iid, "file_name": f"{f:06d}.jpg", "height": 1080, "width": 1920,
                       "has_labeled_person": True, "has_labeled_pitch": True, "has_labeled_camera": True})
        for tid, role, team, jersey, x0 in PEOPLE:
            x, y = x0 + 0.2 * f, -20.0 + 8 * tid
            px = 200 + 300 * tid
            anns.append({
                "id": f"{iid}{tid:02d}", "image_id": iid, "track_id": tid, "supercategory": "object",
                "category_id": 1, "attributes": {"role": role, "team": team, "jersey": jersey},
                "bbox_image": {"x": px, "y": 500, "w": 40, "h": 100},
                "bbox_pitch": {"x_bottom_left": x - 0.3, "y_bottom_left": y, "x_bottom_middle": x,
                               "y_bottom_middle": y, "x_bottom_right": x + 0.3, "y_bottom_right": y},
            })
    return {"info": {"version": "1.3"}, "images": images, "annotations": anns,
            "categories": [{"id": 1, "name": "person", "supercategory": "object"}]}


def _preds(gt, mutate=None):
    preds = []
    for a in gt["annotations"]:
        p = copy.deepcopy(a)
        p["confidence"] = 1.0
        if mutate:
            mutate(p)
        preds.append(p)
    return {"predictions": preds}


def test_each_broken_attribute_drops_exactly_its_rung(tmp_path):
    gt = _gt()
    (tmp_path / "gt" / "valid" / "SNGS-021").mkdir(parents=True)
    (tmp_path / "gt" / "valid" / "SNGS-021" / "Labels-GameState.json").write_text(json.dumps(gt))

    def shift(p):  # 3 m off on the pitch, image box untouched
        for k in ("x_bottom_left", "x_bottom_middle", "x_bottom_right"):
            p["bbox_pitch"][k] += 3.0

    def no_jersey(p):
        p["attributes"]["jersey"] = None

    def swap_team(p):
        t = p["attributes"]["team"]
        p["attributes"]["team"] = {"left": "right", "right": "left"}.get(t, t)

    variants = {"perfect": None, "shifted": shift, "no_jersey": no_jersey, "swapped_team": swap_team}
    dirs = {}
    for name, fn in variants.items():
        d = tmp_path / "pred" / name
        d.mkdir(parents=True)
        (d / "SNGS-021.json").write_text(json.dumps(_preds(gt, fn)))
        dirs[name] = str(d)

    res = ge.evaluate(str(tmp_path / "gt" / "valid"), dirs)
    h = {n: {r: v["HOTA"] for r, v in rows.items()} for n, rows in res.items()}

    assert all(v == pytest.approx(100.0) for v in h["perfect"].values())
    # shifted: image-space tracking is perfect, every pitch-space rung is lower
    assert h["shifted"]["image_hota"] == pytest.approx(100.0)
    assert h["shifted"]["pitch"] < 99.0
    # no jersey numbers: only the last rung drops (2 of 5 people carry a player jersey)
    assert all(h["no_jersey"][r] == pytest.approx(100.0) for r in ("image_hota", "pitch", "+role", "+team"))
    assert h["no_jersey"]["gs_hota"] < 80.0
    # wrong team side: +team and gs_hota drop, +role does not
    assert h["swapped_team"]["+role"] == pytest.approx(100.0)
    assert h["swapped_team"]["+team"] < 60.0
    assert "gs_hota" in ge.format_table(res)
