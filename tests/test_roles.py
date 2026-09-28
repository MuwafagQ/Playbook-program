"""vision/roles.RoleVoter: a track's role follows its majority label, not single frames."""
import numpy as np
import supervision as sv

from vision.roles import RoleVoter


def _t(cls, tid=7, conf=0.8):
    d = sv.Detections(xyxy=np.array([[0, 0, 10, 20]], np.float32), confidence=np.array([conf], np.float32),
                      class_id=np.array([cls], np.int32))
    d.tracker_id = np.array([tid])
    return d


def test_flicker_is_voted_out():
    v = RoleVoter()
    seq = [2, 2, 2, 3, 2, 2, 3, 2, 2, 2]  # a player sometimes labelled referee
    out = [int(v.update(_t(c))[0]) for c in seq]
    assert out[0] == 2 and all(o == 2 for o in out)


def test_true_referee_stays_referee_and_tracks_are_separate():
    v = RoleVoter()
    for _ in range(5):
        v.update(_t(3, tid=1))
    assert int(v.update(_t(2, tid=1))[0]) == 3      # one "player" frame does not flip him
    assert int(v.update(_t(2, tid=2))[0]) == 2      # another track has its own votes


def test_ball_rows_untouched():
    v = RoleVoter()
    assert int(v.update(_t(0))[0]) == 0


def test_role_is_sticky_near_a_tie():
    v = RoleVoter()
    out = [int(v.update(_t(c))[0]) for c in [2, 2, 2, 3, 3, 3, 2, 3]]
    assert all(o == 2 for o in out)  # 3 vs 3-4 votes is not enough to flip a set role
