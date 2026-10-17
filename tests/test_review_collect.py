"""Collecting Pass Review answers into pass rows and proposal quality."""
from __future__ import annotations

import pandas as pd

from tools.review_collect import collect


def _tracks():
    rows = []
    for f in (10, 40):
        for jid, x, y in ((1, 100, 460), (2, 600, 300), (3, 900, 700), (4, 1200, 500), (5, 400, 650)):
            rows.append({"frame": f, "joined_id": jid, "class_id": 2, "x1": x - 10, "y1": y - 60, "x2": x + 10, "y2": y,
                         "x_m": x * 5.0, "y_m": y * 5.0})
    return pd.DataFrame(rows)


def test_collect_rows_and_quality():
    index = {"segments": [{"name": "s1", "seg_start": 0, "frames": 100, "proposals": [
        {"id": "p0", "kick": 10, "touch": 40, "passer": {"jid": 1}, "receiver": {"jid": 2}},
        {"id": "p1", "kick": 50, "touch": 60, "passer": {"jid": 2}, "receiver": {"jid": 3}},
        {"id": "p2", "kick": 70, "touch": 80, "passer": {"jid": 3}, "receiver": {"jid": 4}}]}]}
    reviews = {
        "p0": {"status": "pass", "kick": 10, "touch": 40, "outcome": "complete",
               "passer": {"jid": 1, "name": "B7"}, "receiver": {"jid": 2, "name": "B9"}},
        "p1": {"status": "not_pass", "kick": 50, "touch": 60},
        "p2": {"status": "pass", "kick": 70, "touch": 80, "outcome": "incomplete",
               "passer": {"jid": 4, "name": "AGK"}, "receiver": {"jid": 3, "name": "B9"}},
        "add-x": {"status": "pass", "kick": 10, "touch": 40, "added": True, "source": "added", "outcome": "complete",
                  "passer": {"jid": None, "x": 700, "y": 400, "name": "A5"}, "receiver": {"jid": 4, "name": "A6"}},
    }
    passes, q = collect(reviews, _tracks(), index)
    assert len(passes) == 3 and list(passes.review_id) == ["p0", "add-x", "p2"]
    p0 = passes.set_index("review_id").loc["p0"]
    assert (p0.passer_team, p0.passer_x_m, p0.receiver_x_m) == ("B", 5.0, 30.0)
    added = passes.set_index("review_id").loc["add-x"]
    assert abs(added.passer_x_m - 35.0) < 0.5 and abs(added.passer_y_m - 20.0) < 0.5           # hand-added player placed by the frame's players
    assert q["accepted_as_proposed"] == 1 and q["accepted_after_fix"] == 1 and q["rejected"] == 1
    assert q["added_by_hand"] == 1 and q["proposal_precision"] == 0.667 and q["proposal_recall"] == 0.667
