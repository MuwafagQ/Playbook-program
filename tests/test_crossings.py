import numpy as np
import pandas as pd

from tools.crossings import find_crossings, red_share, sample_frames, swap_score


def _two_players_crossing(n=60, meet=30):
    rows = []
    for f in range(n):
        xa = 100 + (f - meet) * 3          # A walks right, B walks left; they meet at frame `meet`
        xb = 100 - (f - meet) * 3
        rows.append((f, 1, xa, 100, xa + 30, 180))
        rows.append((f, 2, xb, 100, xb + 30, 180))
    return pd.DataFrame(rows, columns=["frame", "raw_tracker_id", "x1", "y1", "x2", "y2"])


def test_finds_one_crossing_with_both_tracks_going_on():
    X = find_crossings(_two_players_crossing())
    assert len(X) == 1
    r = X.iloc[0]
    assert (r.p, r.q) == (1, 2) and r.f0 < 30 < r.f1 and r.p_after and r.q_after


def test_crossing_needs_history_before():
    t = _two_players_crossing()
    t = t[~((t.raw_tracker_id == 2) & (t.frame < 25))]           # track 2 starts right at the crossing
    assert len(find_crossings(t)) == 0


def test_sample_frames_before_and_after():
    t = _two_players_crossing()
    X = find_crossings(t)
    s = sample_frames(t, X)
    assert set(s.when) == {"b", "a"} and set(s.side) == {"p", "q"}
    assert (s[s.when == "b"].frame < X.f0[0]).all() and (s[s.when == "a"].frame > X.f1[0]).all()


def test_red_share_of_a_red_shirt():
    red = np.zeros((60, 30, 3), np.uint8); red[..., 2] = 200
    assert red_share(red) > 0.9 and red_share(np.full((60, 30, 3), 90, np.uint8)) == 0.0


def test_swap_score_by_colour_and_by_appearance():
    a, b = np.eye(4)[0], np.eye(4)[1]
    red, green = {"emb": [a], "red": [0.7]}, {"emb": [b], "red": [0.0]}
    s, basis = swap_score(red, green, green, red)                  # after the crossing the colours swapped
    assert basis == "colour" and s > 1.5
    s, _ = swap_score(red, green, red, green)
    assert s < -1.5
    p, q = {"emb": [a], "red": [0.0]}, {"emb": [b], "red": [0.0]}   # same team: appearance decides
    s, basis = swap_score(p, q, q, p)
    assert basis == "appearance" and s > 0.5
    s, _ = swap_score(p, q, p, None)                                # only p goes on, still p
    assert s < -0.5
