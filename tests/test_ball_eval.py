"""tools/ball_eval: hit / boot / miss classification."""
from __future__ import annotations

import pandas as pd

from tools.ball_eval import ball_eval


def _row(f, c, x1, y1, x2, y2, **kw):
    return {"frame": f, "class_id": c, "x1": x1, "y1": y1, "x2": x2, "y2": y2, **kw}


def test_kinds():
    player = dict(x1=100, y1=100, x2=140, y2=200)  # feet zone: y 170..210
    gt = pd.DataFrame([_row(f, 2, **player) for f in range(5)]
                      + [_row(f, 0, 300, 300, 310, 310) for f in range(4)])  # frame 4: no ball
    pred = pd.DataFrame([
        _row(0, 0, 301, 301, 311, 311),        # hit
        _row(1, 0, 115, 185, 125, 195),        # boot
        _row(2, 0, 600, 50, 610, 60),          # wrong elsewhere
        # frame 3: miss
        _row(4, 0, 115, 185, 125, 195),        # false ball at feet
    ])
    s, d = ball_eval(pred, gt)
    assert list(d.kind) == ["hit", "wrong_feet", "wrong_other", "miss", "false_feet"]
    assert s["frames_ball_visible"] == 4 and s["hit_rate"] == 0.25 and s["wrong_at_feet_rate"] == 0.25
    assert s["false_balls_at_feet"] == 1 and s["precision"] == 0.25
