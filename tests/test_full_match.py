"""Full match: segments on one timeline, teams labelled the same in every segment."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tools.full_match import ID_STRIDE, apply_team_mapping, merge_segments, team_mapping


def _seg(n, team_of_piece):
    rows = [{"frame": f, "raw_tracker_id": p, "track_id": p, "display_track_id": p, "class_id": 2, "team_id": tm}
            for f in range(n) for p, tm in team_of_piece.items()]
    rows += [{"frame": f, "raw_tracker_id": -1, "track_id": -1, "display_track_id": -1, "class_id": 0, "team_id": -1}
             for f in range(n)]
    return pd.DataFrame(rows)


def test_merge_puts_segments_back_to_back_with_unique_ids():
    m = merge_segments([("a", _seg(5, {1: 0, 2: 1}), 5), ("b", _seg(4, {1: 1, 2: 0}), 4)])
    assert m.frame.max() == 8 and (m[m.segment == "b"].frame.min(), m[m.segment == "b"].segment_frame.min()) == (5, 0)
    assert set(m[m.segment == "b"].raw_tracker_id.unique()) == {-1, ID_STRIDE + 1, ID_STRIDE + 2}
    assert (m[m.class_id == 0].raw_tracker_id == -1).all()
    assert m[m.segment == "b"].display_track_id_segment.max() == 2


def test_team_labels_are_aligned_across_segments():
    rng = np.random.default_rng(0)
    kit = {"blue": np.array([1.0, 0, 0, 0]), "yellow": np.array([0, 1.0, 0, 0])}
    # segment a: team 0 = blue; segment b: its k-means happened to call blue "1"
    team = {1: 0, 2: 1, ID_STRIDE + 1: 1, ID_STRIDE + 2: 0}
    clip = {1: "a", 2: "a", ID_STRIDE + 1: "b", ID_STRIDE + 2: "b"}
    colour = {1: "blue", 2: "yellow", ID_STRIDE + 1: "blue", ID_STRIDE + 2: "yellow"}
    crop_piece = np.repeat(list(team), 5)
    emb = np.stack([kit[colour[p]] + 0.1 * rng.normal(size=4) for p in crop_piece])
    maps = team_mapping(emb, crop_piece, team, clip, ["a", "b"])
    assert maps["a"][0] == 0 and maps["a"][1] == 1
    assert maps["b"][0] == 1 and maps["b"][1] == 0 and maps["b"]["margin"] > 1
    m = merge_segments([("a", _seg(3, {1: 0, 2: 1}), 3), ("b", _seg(3, {1: 1, 2: 0}), 3)])
    out = apply_team_mapping(m, maps)
    blue = out[out.raw_tracker_id.isin([1, ID_STRIDE + 1])]
    assert (blue.team_id == 0).all()                 # blue is team 0 in both segments now
    assert (out[out.class_id == 0].team_id == -1).all()


def test_piece_summary_and_coverage():
    from tools.full_match import id_coverage, piece_summary
    m = merge_segments([("a", _seg(10, {1: 0, 2: 1}), 10), ("b", _seg(4, {1: 1, 2: 0}), 4)])
    m["joined_id"] = np.where(m.raw_tracker_id == 1, 1, np.where(m.raw_tracker_id >= 0, 2, -1))
    p = piece_summary(m)
    assert len(p) == 4 and set(p.columns) >= {"piece", "segment", "team_id", "rows", "joined_id"}
    assert p.set_index("piece").loc[ID_STRIDE + 1, "rows"] == 4
    c = id_coverage(p)
    assert c["ids"] == 2 and c["ids_for_50pct"] == 1 and c["ids_for_90pct"] == 2
