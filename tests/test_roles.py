import pandas as pd

from tools.roles import roles


def _frame(f, team, xs_ys):
    return [dict(frame=f, team_id=team, class_id=2, x_m=x * 100, y_m=y * 100) for x, y in xs_ys]


def test_roles_follow_the_attacking_direction():
    # team 0 stands in the left half (attacks +x) before half time, in the right half after
    shape = [(20, 34), (35, 20), (35, 48), (50, 34)]       # back, two wide mids, forward
    rows = _frame(0, 0, shape) + _frame(100, 0, [(105 - x, 68 - y) for x, y in shape])
    t = roles(pd.DataFrame(rows), half_start=50)
    first, second = t[t.frame == 0].role.tolist(), t[t.frame == 100].role.tolist()
    assert first[0] == "A centre defence" and first[3] == "A centre attack"
    assert first == second                                   # same shape, mirrored pitch: same roles
    assert {first[1], first[2]} == {"A left midfield", "A right midfield"}


def test_no_role_with_too_few_teammates():
    t = roles(pd.DataFrame(_frame(0, 1, [(30, 30), (60, 30)])), half_start=50)
    assert t.role.isna().all()
