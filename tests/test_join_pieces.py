"""Joining tracker pieces into players by appearance, team and time."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tools.join_pieces import cluster, joined_ids, piece_table


def _tracks():
    # player X: pieces 1 (frames 0-9) and 3 (20-29); player Y: pieces 2 (0-9) and 4 (20-29); same team.
    # piece 5 (frames 5-14) looks like X but overlaps piece 1 -> must not join it; piece 6 other team.
    rows = []
    for piece, frames, team in ((1, range(0, 10), 0), (2, range(0, 10), 0), (3, range(20, 30), 0),
                                (4, range(20, 30), 0), (5, range(5, 15), 0), (6, range(40, 50), 1)):
        rows += [{"frame": f, "raw_tracker_id": piece, "class_id": 2, "team_id": team} for f in frames]
    rows += [{"frame": f, "raw_tracker_id": -1, "class_id": 0, "team_id": -1} for f in range(50)]  # ball
    return pd.DataFrame(rows)


def test_joins_by_appearance_but_not_overlapping_or_other_team():
    t = _tracks()
    x, y = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    look = {1: x, 2: y, 3: x, 4: y, 5: x, 6: x}
    crop_piece = np.repeat(list(look), 3)
    emb = np.stack([look[p] for p in crop_piece])
    lab = cluster(piece_table(t), emb, crop_piece, thr=0.3)
    assert lab[1] == lab[3] and lab[2] == lab[4]
    assert lab[1] != lab[2]
    assert lab[5] != lab[1]                      # on screen at the same time as piece 1
    assert lab[6] not in (lab[1], lab[5])        # other team
    j = joined_ids(t, lab)
    assert (j[t.class_id == 0] == -1).all()
    assert sorted(j[j > 0].unique()) == list(range(1, len(set(lab.values())) + 1))
    first = t.assign(j=j)[t.raw_tracker_id == 1].j.iloc[0]
    assert first == 1                            # numbered by first appearance


def test_an_id_is_never_on_two_boxes_in_one_frame():
    # piece 2 starts 2 frames before piece 1 ends (allowed overlap): same player, joined
    rows = [{"frame": f, "raw_tracker_id": 1, "class_id": 2, "team_id": 0} for f in range(0, 10)]
    rows += [{"frame": f, "raw_tracker_id": 2, "class_id": 2, "team_id": 0} for f in range(8, 30)]
    t = pd.DataFrame(rows)
    lab = cluster(piece_table(t), np.array([[1.0, 0], [1.0, 0]]), [1, 2], thr=0.3)
    assert lab[1] == lab[2]
    j = t.assign(j=joined_ids(t, lab))
    assert not j.duplicated(["frame", "j"]).any()
    assert (j[j.raw_tracker_id == 2].j == j[j.raw_tracker_id == 2].j.iloc[0]).all()   # the longer piece keeps it
    assert (j[(j.raw_tracker_id == 1) & (j.frame < 8)].j == j[j.raw_tracker_id == 2].j.iloc[0]).all()


def test_below_threshold_stays_apart():
    t = _tracks()
    crop_piece = np.array([1, 2, 3, 4, 5, 6])
    emb = np.eye(6)                              # nobody looks like anybody
    lab = cluster(piece_table(t), emb, crop_piece, thr=0.3)
    assert len(set(lab.values())) == 6
