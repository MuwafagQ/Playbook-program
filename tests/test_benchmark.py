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

    assert rep["jumps"]["image_jumps"] == 1  # id 2 at frame 100; id 1's pitch-only jump is not one
    suspects = pd.read_csv(tmp_path / "out" / "suspects.csv")
    assert len(suspects) == 1 and (suspects.frame[0], suspects.display_track_id[0]) == (100, 2)

    jit = rep["pitch_jitter"]  # everyone moves 10 cm/frame = 2.5 m/s at 25 fps
    assert abs(jit["speed_frame_to_frame_median_mps"] - 2.5) < 1e-6
    assert abs(jit["jitter_ratio"] - 1.0) < 0.05

    assert rep["team"]["ids_with_team_flip"] == 1
    assert rep["duplicates"]["duplicate_id_rows_in_same_frame"] == 0
    assert abs(rep["ball"]["frames_with_ball"] - 0.5) < 1e-6
    assert abs(rep["homography"]["homography_ok_rate"] - 0.9) < 1e-6
    assert "identity_raw_tracker" in rep

    assert (tmp_path / "out" / "benchmark.md").read_text().startswith("# Pipeline benchmark")
    assert len(pd.read_csv(tmp_path / "out" / "spotcheck.csv")) == 3


def _box(frame, tid, x, notes=""):
    return {"frame": frame, "display_track_id": tid, "class_id": 2,
            "x1": x - 15, "y1": 420, "x2": x + 15, "y2": 500, "notes": notes}


def test_score_gt_known_errors():
    X = {"A": 100, "B": 400, "C": 700}
    gt, pred = [], []
    for f in range(100):
        gt.append(_box(f, 1, X["A"]))
        gt.append(_box(f, 2, X["B"]))
        # C's last row was fabricated by gap interpolation -> dropped from scoring.
        gt.append(_box(f, 3, X["C"], notes="idfix_interp" if f == 99 else ""))
        # A and C swap labels from frame 60 (mid-track swap, the raw14 bug class).
        # No pred box on C at frame 99 (after the swap C is labelled 10).
        if f < 99:
            pred.append(_box(f, 10, X["A"] if f < 60 else X["C"]))
        pred.append(_box(f, 13, X["C"] if f < 60 else X["A"]))
        # B fragments into a new id at frame 50.
        pred.append(_box(f, 11 if f < 50 else 12, X["B"]))
    pred.append(_box(5, 99, 1000))  # one false positive

    res, per_player, switches = bm.score_against_gt(pd.DataFrame(pred), pd.DataFrame(gt), offset_search=2)

    assert res["gt_rows_dropped_as_fabricated"] == 1
    assert (res["TP"], res["FP"], res["FN"]) == (299, 1, 0)
    assert res["id_switches"] == 3  # B fragment + A and C swapping
    assert res["pred_ids_covering_2plus_players"] == 2
    assert abs(res["IDF1"] - 2 * 170 / (299 + 300)) < 1e-3
    assert abs(res["MOTA"] - (1 - 4 / 299)) < 1e-3
    assert res["frame_offset_best_fit"] == 0
    assert res["warnings"] == []
    acc = dict(zip(per_player.gt_id, per_player.id_accuracy))
    assert abs(acc["2:1"] - 0.6) < 1e-6 and abs(acc["2:2"] - 0.5) < 1e-6
    assert sorted(switches.frame.tolist()) == [50, 60, 60]


def test_score_gt_flags_frame_misalignment():
    gt = [_box(f, 1, 100 + 12 * f) for f in range(100)]
    pred = [_box(f - 2, 7, 100 + 12 * f) for f in range(100)]  # pred numbered 2 frames early
    res, _, _ = bm.score_against_gt(pd.DataFrame(pred), pd.DataFrame(gt), offset_search=3)
    assert res["frame_offset_best_fit"] == 2
    assert any("misaligned" in w for w in res["warnings"])


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
