"""Tests for tools/benchmark.py on a synthetic run with known defects."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

from tools import benchmark as bm

FPS = 25.0


def _row(frame, tid, cls=2, x=100.0, y=500.0, h=80.0, xm=1000.0, ym=1000.0, team=0, interp=0):
    return {
        "frame": frame, "raw_tracker_id": tid, "track_id": tid, "display_track_id": tid,
        "class_id": cls, "conf": 0.9,
        "x1": x - 15, "y1": y - h, "x2": x + 15, "y2": y,
        "x_m": xm, "y_m": ym, "team_id": team,
        "homography_ok": frame % 10 != 0, "kp_used": 6,
        "homography_state": "ok", "ball_interpolated": interp,
    }


def _synthetic(tmp_path):
    rows = []
    for f in range(250):
        # id 1: smooth, but a 20 m pitch jump at frame 150 with smooth image motion
        # (homography glitch), and its team label flips 0 -> 1 from frame 200.
        xm = 1000 + 10 * f + (2000 if f >= 150 else 0)
        rows.append(_row(f, 1, x=100 + f, xm=xm, team=0 if f < 200 else 1))
        # id 2: jumps 500 px in the image at frame 100 (identity swap signature).
        rows.append(_row(f, 2, x=600 + (500 if f >= 100 else 0) + f, xm=3000 + 10 * f + (1500 if f >= 100 else 0)))
        # id 3: present 0-50 and 60-100 -> two segments (gap of 9 frames > gap_tol 5).
        if f <= 50 or 60 <= f <= 100:
            rows.append(_row(f, 3, x=900 + f, xm=5000 + 10 * f, team=1))
        if f % 2 == 0:
            rows.append(_row(f, -1, cls=0, x=400, y=300, h=8, xm=6000, ym=3500, team=-1, interp=int(f % 4 == 0)))
    path = tmp_path / "tracks.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _args(csv, out):
    return SimpleNamespace(
        csv=str(csv), kpi=None, out_dir=str(out), fps=FPS, id_col="display_track_id",
        expected_players=20, gap_tol=5, long_seconds=4.0, max_speed_mps=12.0,
        max_body_heights_per_frame=1.0, max_gap=5, spotcheck=3, window_seconds=2.0, seed=0,
    )


def test_run_detects_known_defects(tmp_path):
    csv = _synthetic(tmp_path)
    rep = bm.build_report(_args(csv, tmp_path / "out"))

    ident = rep["identity"]["player"]
    assert ident["unique_ids"] == 3
    assert ident["segments"] == 4  # ids 1 and 2 whole, id 3 split in two

    jumps = rep["jumps"]
    assert jumps["image_jumps"] == 1  # id 2 at frame 100
    assert jumps["pitch_jumps"] == 1  # id 1 at frame 150

    suspects = pd.read_csv(tmp_path / "out" / "suspects.csv")
    img = suspects[suspects.kind == "image_jump"].iloc[0]
    assert (img.frame, img.display_track_id) == (100, 2)
    pit = suspects[suspects.kind == "pitch_jump"].iloc[0]
    assert (pit.frame, pit.display_track_id) == (150, 1)

    assert rep["team"]["ids_with_team_flip"] == 1
    assert rep["duplicates"]["duplicate_id_rows_in_same_frame"] == 0
    assert abs(rep["ball"]["frames_with_ball"] - 0.5) < 1e-6
    assert abs(rep["homography"]["homography_ok_rate"] - 0.9) < 1e-6
    assert "identity_raw_tracker" in rep

    assert (tmp_path / "out" / "benchmark.md").read_text().startswith("# Pipeline benchmark")
    assert len(pd.read_csv(tmp_path / "out" / "spotcheck.csv")) == 3


def test_score_spotcheck(tmp_path):
    sc = pd.DataFrame([
        {"window_id": 1, "start_frame": 0, "end_frame": 1499, "players_checked": 10, "id_switches_found": 2},
        {"window_id": 2, "start_frame": 1500, "end_frame": 2999, "players_checked": 10, "id_switches_found": 1},
        {"window_id": 3, "start_frame": 3000, "end_frame": 4499, "players_checked": "", "id_switches_found": ""},
    ])
    path = tmp_path / "spotcheck.csv"
    sc.to_csv(path, index=False)
    res = bm.score_spotcheck(str(path), FPS)
    assert res["windows_reviewed"] == 2
    assert abs(res["player_minutes_reviewed"] - 20.0) < 1e-6  # 2 windows x 1 min x 10 players
    assert abs(res["id_switches_per_player_minute"] - 0.15) < 1e-6
    assert res["ci95_low"] < 0.15 < res["ci95_high"]


def test_compare(tmp_path):
    a = tmp_path / "a" / "benchmark.json"
    b = tmp_path / "b" / "benchmark.json"
    a.parent.mkdir()
    b.parent.mkdir()
    a.write_text(json.dumps({"jumps": {"image_jumps": 10}}))
    b.write_text(json.dumps({"jumps": {"image_jumps": 4}}))
    out = bm.compare(str(a), str(b))
    assert "| jumps.image_jumps | 10 | 4 | -6 |" in out
