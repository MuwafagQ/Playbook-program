import json

import cv2
import numpy as np
import pandas as pd
import supervision as sv

from tools.id_cards import CARD_H, CARD_W, STRIPS_PER_ROW, ball_candidates, build_sheet, card_plan, sheet_layout


def _video(path, n=40, w=320, h=240):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (w, h))
    for i in range(n):
        f = np.zeros((h, w, 3), np.uint8)
        f[:, :, 1] = i * 5
        vw.write(f)
    vw.release()


def _tracks():
    rows = []
    for f in range(40):
        rows.append(dict(segment_frame=f, raw_tracker_id=7, class_id=2, x1=10, y1=10, x2=40, y2=90))
        if f < 10:   # too short for a card
            rows.append(dict(segment_frame=f, raw_tracker_id=8, class_id=2, x1=100, y1=10, x2=130, y2=90))
        rows.append(dict(segment_frame=f, raw_tracker_id=9, class_id=0, x1=5, y1=5, x2=9, y2=9))  # the ball
    return pd.DataFrame(rows)


def test_plan_spreads_crops_over_long_pieces_only():
    p = card_plan(_tracks(), min_frames=30, per_piece=4)
    assert set(p.piece) == {7}
    assert list(p.slot) == [0, 1, 2, 3]
    assert p.frame.min() >= 3 and p.frame.max() <= 36 and p.frame.is_monotonic_increasing


def test_layout_packs_strips_in_rows():
    lay = sheet_layout(list(range(STRIPS_PER_ROW + 1)), per_piece=4)
    assert lay[0] == (0, 0) and lay[1] == (4 * CARD_W, 0)
    assert lay[STRIPS_PER_ROW] == (0, CARD_H)


def test_sheet_has_one_strip_per_piece(tmp_path):
    v = tmp_path / "v.avi"
    _video(v)
    idx = build_sheet(str(v), card_plan(_tracks()), str(tmp_path / "cards.jpg"))
    img = cv2.imread(str(tmp_path / "cards.jpg"))
    assert img.shape[0] == CARD_H and img.shape[1] == STRIPS_PER_ROW * 4 * CARD_W
    assert list(idx["pieces"]) == ["7"] and len(idx["pieces"]["7"][2]) == 4
    json.dumps(idx)


def test_ball_candidates_every_other_frame(tmp_path):
    v = tmp_path / "v.avi"
    _video(v, n=6)
    calls = []

    def fn(crops, conf):
        calls.append(len(crops))
        return [sv.Detections(xyxy=np.array([[10, 10, 16, 16]], np.float32), confidence=np.array([0.9], np.float32),
                              class_id=np.zeros(1, np.int32))] + [None] * (len(crops) - 1)

    c = ball_candidates(str(v), fn, every=2, preprocess=False)
    assert list(c.frame) == [0, 2, 4]
    assert c.x.iloc[0] == 13 and c.conf.iloc[0] == 0.9
