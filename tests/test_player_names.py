"""Player names typed while tagging passes."""
from __future__ import annotations

import pandas as pd

from tools.player_names import NameBook, apply_names, coverage, label, piece_column


def _tracks():
    rows = []
    for f in range(10):
        rows.append({"frame": f, "raw_tracker_id": 5, "track_id": 1, "class_id": 2, "team_id": 1})
        rows.append({"frame": f, "raw_tracker_id": 6, "track_id": 2, "class_id": 2, "team_id": 0})
        rows.append({"frame": f, "raw_tracker_id": -1, "track_id": -1, "class_id": 0, "team_id": -1})  # ball
    return pd.DataFrame(rows)


def test_label_adds_team_letter_to_numbers():
    assert label("10", 1) == "B10" and label(" 7 ", 0) == "A7" and label("07", 0) == "A7"
    assert label("9", -1) == "?9" and label("9", None) == "?9"
    assert label("gk", 0) == "GK" and label("b 4", 0) == "B4" and label("", 0) is None


def test_name_holds_from_its_frame_and_backwards_for_the_first(tmp_path):
    b = NameBook(str(tmp_path / "names.csv"))
    b.set(5, 4, "B10")
    assert b.name_at(5, 0) == "B10" and b.name_at(5, 9) == "B10"
    b.set(5, 7, "B3")                       # tracker swapped at frame 7
    assert b.name_at(5, 6) == "B10" and b.name_at(5, 7) == "B3"
    b.set(5, 8, "B3")                       # same name again: no new entry
    assert b.entries[5] == [(4, "B10"), (7, "B3")]
    b.save()
    assert NameBook(str(tmp_path / "names.csv")).entries == b.entries


def test_apply_names_only_people_rows():
    t = _tracks()
    assert piece_column(t) == "raw_tracker_id"
    b = NameBook()
    b.set(5, 2, "B10"); b.set(5, 6, "B3")
    out = apply_names(t, b)
    p5 = out[out.raw_tracker_id == 5].set_index("frame").player_name
    assert p5[0] == "B10" and p5[5] == "B10" and p5[6] == "B3"
    assert out[out.raw_tracker_id == 6].player_name.isna().all()
    assert out[out.class_id == 0].player_name.isna().all()
    assert coverage(t, b) == {"named_rows": 0.5, "names": 2, "pieces_named": 1}


def test_old_runs_without_raw_ids_use_track_id():
    t = _tracks().drop(columns="raw_tracker_id")
    assert piece_column(t) == "track_id"
    b = NameBook(); b.set(2, 0, "A7")
    assert (apply_names(t, b).query("track_id == 2").player_name == "A7").all()
