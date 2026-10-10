import numpy as np
import pandas as pd

from tools.ball_path import fill, solve


def _cands():
    rows = []
    rng = np.random.default_rng(1)
    for f in range(0, 200, 2):
        x = 100 + 5 * f                          # the real ball: steady flight
        if not 80 <= f < 100:                    # hidden for 20 frames
            rows.append((f, x, 300.0, 0.6))
        rows.append((f, 1500.0, 900.0, 0.7))     # a static false blob (a boot, a line), more confident
        rows.append((f, rng.uniform(0, 1900), rng.uniform(0, 1000), 0.2))   # noise
    return pd.DataFrame(rows, columns=["frame", "x", "y", "conf"])


def test_path_follows_the_moving_ball_not_the_brighter_static_blob():
    p = solve(_cands())
    on_ball = (p.y == 300.0).mean()
    assert on_ball > 0.9
    assert 80 not in set(p.frame)                # nothing invented while hidden


def test_fill_bridges_short_gaps_only():
    path = pd.DataFrame({"frame": [0, 10, 100], "x": [0.0, 100.0, 1000.0], "y": [0.0, 0.0, 0.0]})
    b = fill(path, np.arange(0, 101), max_fill=30).set_index("frame")
    assert b.filled[5] == 1 and b.x[5] == 50
    assert np.isnan(b.x[50])                     # 90-frame gap: no ball rather than a guess
    assert b.filled[100] == 0
