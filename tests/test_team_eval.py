"""Team classification scoring against per-track team labels."""
from __future__ import annotations

import pandas as pd

from tools.team_eval import team_scores


def _box(frame, tid, x, team=-1, cls=2):
    return {"frame": frame, "track_id": tid, "display_track_id": tid, "class_id": cls,
            "x1": x, "y1": 0, "x2": x + 20, "y2": 50, "team_id": team}


def test_team_scores_mapping_coverage_and_flips():
    gt = pd.DataFrame([_box(f, 1, 0) for f in range(4)] + [_box(f, 2, 100) for f in range(4)])
    teams = {"team0": [1], "team1": [2]}
    # the pipeline calls team 0 "1" and team 1 "0" (arbitrary numbering); id 7 flips once, id 8 has a gap
    pred = pd.DataFrame([_box(0, 7, 1, 1), _box(1, 7, 1, 1), _box(2, 7, 1, 0), _box(3, 7, 1, 1)] +
                        [_box(0, 8, 101, 0), _box(1, 8, 101, -1), _box(2, 8, 101, 0), _box(3, 8, 101, 0)])
    s = team_scores(pred, gt, teams)
    assert s["matched_boxes"] == 8 and s["coverage"] == 0.875
    assert s["wrong_boxes"] == 1 and abs(s["accuracy"] - 6 / 7) < 1e-3
    assert s["flip_ids"] == 1 and s["worst_gt"] == {1: 0.75}


def test_goalkeeper_nearest_team():
    import numpy as np

    from vision.goalkeeper_team import nearest_team

    players = np.array([[10, 0], [20, 0], [80, 0], [90, 0]], float)
    teams = np.array([0, 0, 1, 1])
    assert nearest_team(np.array([[0, 0], [100, 0], [np.nan, 0]]), players, teams).tolist() == [0, 1, -1]
    assert nearest_team(np.array([[0, 0]]), players[:3], teams[:3]).tolist() == [-1]  # team 1 has 1 player
