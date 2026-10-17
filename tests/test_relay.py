import pandas as pd

from tools.relay import clear, relay_events


def _rows(piece, frames, x):
    return [dict(frame=f, raw_tracker_id=piece, x1=x, y1=100, x2=x + 20, y2=160) for f in frames]


def test_new_piece_lists_the_piece_that_just_ended_nearby_first():
    t = pd.DataFrame(_rows(1, range(0, 50), 100) + _rows(2, range(55, 100), 104)      # 1 -> 2 hand-over
                     + _rows(3, range(0, 100), 600) + _rows(4, range(40, 80), 900))   # 3 runs on; 4 far away
    ev = {e["piece"]: e for e in relay_events(t, [1, 2, 3, 4])}
    assert ev[2]["cands"][0][0] == 1 and ev[2]["cands"][0][2] == 6
    assert all(c[0] != 3 for c in ev[2]["cands"])        # still on screen: cannot be the same player
    assert clear(ev[2])


def test_two_close_predecessors_are_not_clear():
    t = pd.DataFrame(_rows(1, range(0, 50), 100) + _rows(5, range(0, 48), 112) + _rows(2, range(55, 100), 104))
    ev = {e["piece"]: e for e in relay_events(t, [1, 2, 5])}
    assert len(ev[2]["cands"]) == 2 and not clear(ev[2])
