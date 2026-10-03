"""Tests for tools/frame_sampler.py on synthetic match videos."""
from __future__ import annotations

import json
import zipfile

import cv2
import numpy as np

from tools import frame_sampler as fs

FPS = 10


def _video(path, scenes):
    """scenes: list of (seconds, kind, tint). kind 'pitch' = green game view, 'crowd' = no pitch."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), FPS, (320, 180))
    for secs, kind, tint in scenes:
        for i in range(secs * FPS):
            if kind == "pitch":
                img = np.zeros((180, 320, 3), np.uint8)
                img[:] = (40, 150 + tint, 40)
                img[:40] = (120 + tint, 120, 110)  # stands strip, tinted per stadium
                for k in range(5):
                    x = 20 + 60 * k + (i % 20)
                    cv2.rectangle(img, (x, 90), (x + 8, 120), (255 - 3 * tint, 255, 255), -1)
            else:
                img = np.full((180, 320, 3), (90, 80, 120), np.uint8)
                cv2.circle(img, (160, 90), 50 + (i % 10), (30, 30, 200), -1)
            vw.write(img)
    vw.release()


def test_scan_filter_and_diverse_selection(tmp_path):
    a, b = tmp_path / "matchA.avi", tmp_path / "matchB.avi"
    _video(a, [(20, "pitch", 0), (10, "crowd", 0), (20, "pitch", 60)])   # two stadium looks
    _video(b, [(30, "pitch", 30), (10, "crowd", 0)])
    cands = fs.scan_video(str(a), "A", every_s=1.0) + fs.scan_video(str(b), "B", every_s=1.0)
    assert len(cands) == 50 + 40

    kept = fs.filter_broadcast(cands, min_green=0.35, min_sharp_pct=0)
    assert all(c.green >= 0.35 for c in kept)
    crowd_a = [c for c in kept if c.match == "A" and 20 <= c.time_s < 30]
    assert crowd_a == []  # close-up / crowd shots dropped

    picks = fs.select_diverse(kept, per_match=2, min_gap_s=3.0)
    by = {m: sorted(c.time_s for c in picks if c.match == m) for m in ("A", "B")}
    assert len(by["A"]) == 2 and len(by["B"]) == 2
    # match A's two picks come from its two different-looking scenes
    assert by["A"][0] < 20 and by["A"][1] >= 30
    for m in by.values():
        assert all(t2 - t1 >= 3.0 for t1, t2 in zip(m, m[1:]))


def test_hardness_prefers_uncertain_small_crowded():
    easy = [(0, 0, 40, 100, 0.95, 2), (200, 0, 240, 100, 0.9, 2)]
    hard = [(0, 0, 40, 100, 0.3, 2), (10, 0, 50, 100, 0.35, 2), (300, 500, 306, 520, 0.8, 2)]
    assert fs.hardness(easy, 1080) == 0.0
    assert fs.hardness(hard, 1080) > 0.3
    assert fs.hardness([(0, 0, 5, 5, 0.2, 0)], 1080) == 0.0  # the ball does not count


def test_select_diverse_uses_hardness():
    base = np.zeros(4, np.float32)
    mk = lambda t, h, f: fs.Candidate("M", "v", int(t * 10), t, 0.9, 10.0, f, np.zeros((2, 2, 3), np.uint8), hardness=h)
    cands = [mk(0, 0.0, base), mk(10, 0.0, base + 0.01), mk(20, 2.0, base + 0.01)]
    # with an existing pick at 'base', both others are equally far; hardness breaks the tie
    picks = fs.select_diverse(cands[1:], per_match=1, hardness_weight=1.0, already=[cands[0]])
    assert picks[0].time_s == 20


def test_export_coco_split_by_match(tmp_path):
    a, b = tmp_path / "A.avi", tmp_path / "B.avi"
    _video(a, [(5, "pitch", 0)])
    _video(b, [(5, "pitch", 30)])
    picks = fs.scan_video(str(a), "A")[:2] + fs.scan_video(str(b), "B")[:1]
    for c in picks:
        c.detections = [(10, 20, 30, 80, 0.9, 2), (50, 20, 70, 80, 0.2, 3)]  # 2nd below label conf
    splits = fs.split_matches(["A", "B"], valid_frac=0.5, seed=1)
    assert sorted(splits.values()) == ["train", "valid"]
    zips = fs.export_coco(picks, str(tmp_path / "out"), splits, min_label_conf=0.3)
    total = 0
    for split, zp in zips.items():
        with zipfile.ZipFile(zp) as z:
            coco = json.loads(z.read("_annotations.coco.json"))
            names = set(z.namelist()) - {"_annotations.coco.json"}
        assert {im["file_name"] for im in coco["images"]} == names
        assert {im["extra"]["match"] for im in coco["images"]} == {m for m, s in splits.items() if s == split}
        assert [c["name"] for c in coco["categories"]] == fs.CLASSES
        assert len(coco["annotations"]) == len(coco["images"])  # low-confidence box left out
        total += len(coco["images"])
    assert total == 3
    fs.contact_sheet(picks, str(tmp_path / "sheet.jpg"))
    assert (tmp_path / "sheet.jpg").exists()


def test_split_matches_fixed_validation():
    s = fs.split_matches(["m_2", "m_3", "m_4", "m_9"], valid_matches=["m_2", "m_4"])
    assert s == {"m_2": "valid", "m_3": "train", "m_4": "valid", "m_9": "train"}
    import pytest
    with pytest.raises(ValueError):
        fs.split_matches(["m_2"], valid_matches=["m_7"])


def test_zoomed_out_play_is_kept():
    """A wide shot with stands filling most of the frame and a thin strip of pitch."""
    wide = np.full((180, 320, 3), (110, 100, 120), np.uint8)
    wide[140:] = (40, 150, 40)  # ~22% pitch
    crowd = np.full((180, 320, 3), (110, 100, 120), np.uint8)
    mk = lambda img, t: fs.Candidate("M", "v", t, float(t), fs.pitch_green_ratio(img), 50.0,
                                     fs.appearance_feature(img), img)
    kept = fs.filter_broadcast([mk(wide, 0), mk(crowd, 1)], min_sharp_pct=0)
    assert [c.time_s for c in kept] == [0.0]
