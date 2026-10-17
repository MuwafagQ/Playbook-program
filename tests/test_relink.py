"""Re-linking of track pieces by pitch position, team and timing."""
from __future__ import annotations

import pandas as pd

from tools.relink import candidates, link, pieces, apply_links


def _rows(tid, frames, x, team, cls=2):
    return [{"frame": f, "track_id": tid, "display_track_id": tid, "class_id": cls, "x_m": x * 100.0,
             "y_m": 3000.0, "team_id": team} for f in frames]


def test_relink_joins_only_feasible_same_team_pieces():
    t = pd.DataFrame(_rows(1, range(0, 30), 10, 0)              # ends at x=10 m, frame 29
                     + _rows(2, range(60, 90), 12, 0)            # 1 s later, 2 m away, same team: feasible
                     + _rows(3, range(60, 90), 11, 1)            # other team: never
                     + _rows(4, range(40, 70), 80, 0))           # 70 m away after 0.4 s: not reachable
    p = pieces(t)
    c = candidates(p, fps=30)
    assert set(zip(c["from"], c["to"])) == {(1, 2)}
    applied, sugg = link(c)                                     # automatic joining off by default
    assert applied == [] and sugg[0]["from"] == 1 and sugg[0]["options"][0]["to"] == 2
    applied, _ = link(c, auto_score=0.5)
    out = apply_links(t, applied)
    assert out[out.track_id == 1].display_track_id.iloc[0] == out[out.track_id == 2].display_track_id.iloc[0]
    assert out[out.track_id == 3].display_track_id.iloc[0] != out[out.track_id == 1].display_track_id.iloc[0]
    assert (out.piece_id == out.track_id).all()
