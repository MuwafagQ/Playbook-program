"""Pass proposals from possession changes."""
from __future__ import annotations

import pandas as pd

from tools.pass_proposals import propose, score


def _tracks():
    # player 1 (team 0) at x=100, player 2 (team 0) at x=500; ball at 1's feet frames 0-9, flies
    # frames 10-19, at 2's feet frames 20-29
    rows = []
    for f in range(30):
        for pid, x in ((1, 100), (2, 500)):
            rows.append({"frame": f, "raw_tracker_id": pid, "class_id": 2, "team_id": 0,
                         "x1": x - 10, "y1": 200, "x2": x + 10, "y2": 260, "x_m": x * 10.0, "y_m": 0.0})
        bx = 100 if f < 10 else 500 if f >= 20 else 100 + (f - 9) * 40
        by = 255 if (f < 10 or f >= 20) else 150
        rows.append({"frame": f, "raw_tracker_id": -1, "class_id": 0, "team_id": -1,
                     "x1": bx - 3, "y1": by - 3, "x2": bx + 3, "y2": by + 3, "x_m": None, "y_m": None})
    return pd.DataFrame(rows)


def test_one_pass_is_proposed_at_kick_and_touch():
    p = propose(_tracks())
    assert len(p) == 1
    r = p.iloc[0]
    assert (r.passer, r.receiver, r.kick_frame, r.touch_frame, r.same_team) == (1, 2, 9, 20, True)
    tagged = pd.DataFrame({"departure_frame": [8], "arrival_frame": [22]})
    assert score(p, tagged)["found"] == 1


def test_split_pieces_of_one_player_are_not_a_pass():
    t = _tracks()
    t = t[t.raw_tracker_id != 2]
    # player 1's piece ends at frame 9; piece 3 continues at the same spot from frame 11
    t.loc[(t.raw_tracker_id == 1) & (t.frame > 9), "raw_tracker_id"] = 3
    t.loc[t.class_id == 0, ["x1", "x2"]] = [97.0, 103.0]
    t.loc[t.class_id == 0, ["y1", "y2"]] = [252.0, 258.0]
    assert len(propose(t)) == 0
