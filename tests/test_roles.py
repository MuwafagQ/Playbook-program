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


def test_ball_challenger_switch():
    """A clearly more confident ball outside the gate for 3 frames takes over the track."""
    from vision.ball import BallSmoother

    def dets(*items):
        return sv.Detections(xyxy=np.array([[x - 4, y - 4, x + 4, y + 4] for x, y, _ in items], np.float32),
                             confidence=np.array([c for _, _, c in items], np.float32),
                             class_id=np.zeros(len(items), np.int32))
    b = BallSmoother(ball_id=0, switch_frames=3)
    b.update(dets((100, 100, 0.55)))                    # locks on a logo
    for _ in range(2):
        out, _ = b.update(dets((100, 100, 0.55), (900, 500, 0.9)))
        assert abs(out.xyxy[0, 0] + 4 - 100) < 1        # still on the logo
    out, _ = b.update(dets((100, 100, 0.55), (903, 502, 0.9)))
    assert abs(out.xyxy[0, 0] + 4 - 903) < 1 and b.switches == 1
    off = BallSmoother(ball_id=0)                        # off by default
    off.update(dets((100, 100, 0.55)))
    for _ in range(5):
        out, _ = off.update(dets((100, 100, 0.55), (900, 500, 0.9)))
    assert abs(out.xyxy[0, 0] + 4 - 100) < 1
