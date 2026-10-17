"""Every regression moment (`tools/moments.py`, the JSON files beside this one) against every
strategy it concerns. A moment with a `known_failure` is a strict xfail, so its fix shows as
XPASS: then delete the `known_failure`."""

import pytest

from tools import moments


def _cases():
    for path in sorted(moments.MOMENTS_DIR.glob("*.json")):
        m = moments.load(path)
        marks = [pytest.mark.xfail(strict=True, reason=m.known_failure)] if m.known_failure else []
        for strategy in moments.strategies_for(m):
            yield pytest.param(path, strategy, id=f"{m.name}-{moments.short_name(strategy)}", marks=marks)


@pytest.mark.parametrize("path, strategy", list(_cases()))
def test_moment(path, strategy):
    m = moments.load(path)
    failures = moments.judge(m, moments.play(m, strategy))
    assert not failures, f"{m.name}: {m.note}\n" + "\n".join(failures)


def _tick(t, command, ball, ours):
    return moments.Tick(t=t, command=command, ball=ball, ours=ours, our_speeds={}, our_foul=None)


def test_keeps_clear_lets_a_robot_inside_leave_and_flags_one_driving_in():
    from utama_core.entities.referee.referee_command import RefereeCommand

    stop, dt = RefereeCommand.STOP, 1 / moments.TICKS_PER_SECOND
    leaving = [_tick(i * dt, stop, (0.0, 0.0), {1: (0.2 + 0.01 * i, 0.0)}) for i in range(60)]
    assert moments.keeps_clear(leaving) == []
    driving_in = [_tick(i * dt, stop, (0.0, 0.0), {1: (0.45 - 0.004 * i, 0.0)}) for i in range(60)]
    assert "drove in" in moments.keeps_clear(driving_in)[0]


def test_keeps_clear_does_not_blame_a_robot_standing_still_for_the_ball_coming_to_it():
    """placement_corner_robot_drives_onto_ball: the placement moved the ball 0.2 m onto a robot."""
    from utama_core.entities.referee.referee_command import RefereeCommand

    stop, dt = RefereeCommand.STOP, 1 / moments.TICKS_PER_SECOND
    ball_comes = [_tick(i * dt, stop, (0.004 * i, 0.0), {1: (0.45, 0.0)}) for i in range(60)]
    assert moments.keeps_clear(ball_comes) == []


def test_keeps_clear_flags_a_robot_still_inside_after_the_grace_and_ignores_our_own_restart():
    from utama_core.entities.referee.referee_command import RefereeCommand

    dt = 1 / moments.TICKS_PER_SECOND
    n = int((moments.LEAVE_WITHIN_S + 0.1) / dt)
    stuck = [_tick(i * dt, RefereeCommand.DIRECT_FREE_BLUE, (0.0, 0.0), {2: (0.3, 0.0)}) for i in range(n)]
    assert "inside the keep-out circle" in moments.keeps_clear(stuck)[0]
    ours = [_tick(i * dt, RefereeCommand.DIRECT_FREE_YELLOW, (0.0, 0.0), {2: (0.3, 0.0)}) for i in range(n)]
    assert moments.keeps_clear(ours) == []


def test_a_tactic_that_is_never_applicable_fails_its_moment(monkeypatch):
    """Its robots would hold position for the whole moment, so the moment must not pass."""
    import dataclasses

    from utama_core.tactics.give_and_go import GiveAndGoTactic

    monkeypatch.setattr(GiveAndGoTactic, "applicable", lambda self, game: False)
    m = moments.template("never_ran_probe", "a note", "NORMAL_START", ["keeps_clear"])
    m = dataclasses.replace(m, seconds=0.2, tactic={"class": "GiveAndGoTactic", "robots": [1, 2], "mem": None})
    failures = moments.judge(m, moments.play(m, moments.strategies_for(m)[0]))
    assert "GiveAndGoTactic never ran on robots [1, 2]" in failures
