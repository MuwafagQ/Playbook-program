import cv2
import numpy as np
import pandas as pd

from tools.jersey import best_views, clean_number, enhance, fuse, torso_crop, vote


def test_best_views_takes_big_boxes_spread_in_time():
    rows = pd.DataFrame({"segment_frame": range(100), "x1": 0, "y1": 0, "x2": 30,
                         "y2": [50 + f for f in range(100)]})
    v = best_views(rows, k=3, min_gap=15)
    assert list(v.segment_frame) == [99, 84, 69]


def test_torso_crop_is_the_shirt_area():
    frame = np.zeros((200, 200, 3), np.uint8)
    c = torso_crop(frame, (50, 20, 90, 120))
    assert c.shape[0] == 46 and 30 <= c.shape[1] <= 40


def test_enhance_and_fuse_keep_the_number_shape():
    img = np.full((40, 30, 3), 40, np.uint8)
    cv2.putText(img, "7", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2)
    noisy = [np.clip(img + np.random.default_rng(i).normal(0, 25, img.shape), 0, 255).astype(np.uint8) for i in range(5)]
    f = fuse(noisy)
    assert np.abs(f.astype(int) - img).mean() < np.abs(noisy[0].astype(int) - img).mean()   # fusion removes noise
    assert enhance(f).shape == (120, 90, 3)


def test_number_cleaning_and_voting():
    assert clean_number("14") == "14" and clean_number(" 7.") == "7"
    assert clean_number("07") is None and clean_number("123") is None and clean_number("") is None
    reads = pd.DataFrame({"piece": [1, 1, 1, 2], "number": ["14", "14", "11", None], "conf": [0.9, 0.8, 0.95, 0.9]})
    v = vote(reads).set_index("piece")
    assert v.number[1] == "14" and v.views[1] == 2 and 2 not in v.index
