import numpy as np
import pandas as pd

from tools.name_proposals import clashes, labels_from_reviews, piece_table, propose


def _tracks():
    rows = []
    for f in range(100):
        rows.append(dict(frame=f, raw_tracker_id=1, joined_id=10, class_id=2, team_id=0))
        rows.append(dict(frame=f, raw_tracker_id=2, joined_id=20, class_id=2, team_id=1))
    for f in range(100, 200):
        rows.append(dict(frame=f, raw_tracker_id=3, joined_id=10, class_id=2, team_id=0))
        rows.append(dict(frame=f, raw_tracker_id=4, joined_id=20, class_id=2, team_id=1))
    return pd.DataFrame(rows)


def test_labels_come_from_the_piece_under_the_tagged_box():
    rev = {"a": {"passer": {"jid": 10, "frame": 5, "name": "A7"}, "receiver": {"jid": 20, "frame": 20, "name": "B9"}},
           "b": {"passer": {"jid": 10, "frame": 150, "name": "A8", "piece": 3}}}
    lab = labels_from_reviews(rev, _tracks())
    assert lab.to_dict() == {1: "A7", 2: "B9", 3: "A8"}


def test_proposals_follow_appearance_and_team():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=16), rng.normal(size=16)
    emb, piece = [], []
    for p, base in ((1, a), (2, b), (3, a), (4, b), (5, a), (6, b)):
        for _ in range(6):
            emb.append(base + 0.05 * rng.normal(size=16))
            piece.append(p)
    pieces = piece_table(_tracks())
    pieces.loc[5] = [0, 1, 2, 0]
    pieces.loc[6] = [0, 1, 2, 1]
    labels = pd.Series({1: "A7", 2: "B9", 5: "A7", 6: "B9"})
    pr = propose(np.array(emb), np.array(piece), labels, pieces).set_index("piece")
    assert pr.name[3] == "A7" and pr.name[4] == "B9"
    assert pr.margin[3] > 0


def test_clash_is_same_name_on_screen_together():
    pieces = piece_table(_tracks())
    assert clashes(pd.Series({1: "A7", 2: "A7", 3: "A7"}), pieces) == [(1, 2, "A7")]


def test_exclusive_assignment_keeps_one_piece_per_name_at_a_time():
    from tools.name_proposals import assign_exclusive
    pieces = piece_table(_tracks())          # 1 and 2 together, then 3 and 4 together
    props = pd.DataFrame({"piece": [2, 3, 4], "scores": [[("A7", 0.95), ("B9", 0.9)], [("A7", 0.9)], [("A7", 0.8), ("B9", 0.75)]]})
    out = assign_exclusive(props, pd.Series({1: "A7"}), pieces)
    assert out[1] == ("A7", 1.0)
    assert out[2][0] == "B9"                 # A7 is taken by piece 1 at that time
    assert out[3][0] == "A7" and out[4][0] == "B9"


def test_kit_rule_keeps_dark_kits_for_the_dark_keeper():
    from tools.name_proposals import apply_kit
    props = pd.DataFrame({"piece": [1, 2, 3], "scores": [[("BGK", 0.9), ("AGK", 0.8)]] * 3})
    out = apply_kit(props, {1: 0.4, 2: 0.02, 3: 0.17}).set_index("piece").scores
    assert out[1] == [("AGK", 0.8)]                  # black kit: only the black keeper
    assert out[2] == [("BGK", 0.9)]                  # light kit: never the black keeper
    assert out[3] == [("BGK", 0.9), ("AGK", 0.8)]    # in between: no rule
