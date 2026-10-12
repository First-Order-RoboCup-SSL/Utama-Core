from types import SimpleNamespace

import numpy as np
import pytest

from utama_core.analysis.goal_ledger import (
    SUSTAINED,
    attribute,
    find_goals,
    find_restarts,
    fold_restart_takes,
    live_mask,
    possession_spells,
    situations,
    tables,
)
from utama_core.entities.referee.referee_command import RefereeCommand as C

_HZ = 60.0


def _commands(*spans):
    """(command, n_ticks) spans -> per-tick command values."""
    return np.array([cmd.value for cmd, n in spans for _ in range(n)], dtype=np.int64)


def _spell(side, t, third="middle"):
    return {"side": side, "t": t, "t_end": t + 2.0, "third": third, "takes_restart": False}


def _restart(t, kind="free kick", side="friendly", third="attacking"):
    return {"t": t, "kind": kind, "side": side, "third": third}


def _goal(t, side="friendly"):
    return {"t": t, "side": side}


def test_a_goal_exactly_the_window_after_a_start_is_sustained_possession():
    goals = [_goal(14.99), _goal(15.0)]
    attribute(goals, [{**_spell("friendly", 5.0), "kind": "ball won"}])
    assert goals[0]["situation"]["t"] == 5.0
    assert goals[0]["after_s"] == pytest.approx(9.99)
    assert goals[1]["situation"] is None and goals[1]["after_s"] is None


def test_the_kicking_sides_first_spell_within_the_fold_window_is_the_restart_being_taken():
    on_time, late = [_spell("friendly", 3.5)], [_spell("friendly", 3.51)]
    fold_restart_takes([_restart(2.0)], on_time)
    fold_restart_takes([_restart(2.0)], late)
    assert on_time[0]["takes_restart"] and not late[0]["takes_restart"]


def test_an_interception_before_the_taker_keeps_both_spells_as_situations():
    spells = [_spell("enemy", 2.5), _spell("friendly", 3.0)]
    fold_restart_takes([_restart(2.0)], spells)
    assert not any(s["takes_restart"] for s in spells)


def test_a_goal_after_a_folded_take_goes_to_the_restart():
    match = {"restarts": [_restart(2.0)], "spells": [_spell("friendly", 2.5)]}
    fold_restart_takes(match["restarts"], match["spells"])
    goals = [_goal(9.0)]
    attribute(goals, situations(match))
    assert goals[0]["situation"]["kind"] == "free kick"


def test_a_restart_wins_a_tie_with_a_spell_starting_on_the_same_tick():
    match = {"restarts": [_restart(5.0)], "spells": [_spell("enemy", 5.0)]}
    fold_restart_takes(match["restarts"], match["spells"])
    goals = [_goal(7.0)]
    attribute(goals, situations(match))
    assert goals[0]["situation"]["kind"] == "free kick"


def _flicker(enemy_ticks):
    """Friendly on the ball for 2 s, then the enemy for `enemy_ticks`, then friendly to 5 s."""
    n = int(5 * _HZ)
    ts = np.arange(n) / _HZ
    enemy = np.zeros(n, dtype=bool)
    enemy[120 : 120 + enemy_ticks] = True
    return possession_spells(ts, np.ones(n, dtype=bool), ~enemy, enemy, np.zeros(n), np.ones(n))


def test_a_spell_under_a_second_is_merged_into_the_holder_before_it():
    spells = _flicker(60)  # 59 ticks from its first to its last: 0.98 s
    assert [(s["side"], s["t"]) for s in spells] == [("friendly", 0.0)]
    assert spells[0]["t_end"] == pytest.approx(299 / _HZ)


def test_a_spell_of_a_full_second_is_a_change_of_possession():
    spells = _flicker(61)  # 60 ticks: 1.0 s
    assert [(s["side"], s["t"]) for s in spells] == [("friendly", 0.0), ("enemy", 2.0), ("friendly", 181 / _HZ)]


def test_spell_thirds_are_in_the_winning_sides_attack_frame():
    n = 240
    ts = np.arange(n) / _HZ
    enemy = np.arange(n) >= 120
    # Friendly attacks -x: it wins the ball at x = 2, its defensive third; the enemy at x = -2, its own.
    ball_x = np.where(enemy, -2.0, 2.0)
    spells = possession_spells(ts, np.ones(n, dtype=bool), ~enemy, enemy, ball_x, np.full(n, -1.0))
    assert [(s["side"], s["third"]) for s in spells] == [("friendly", "defensive"), ("enemy", "defensive")]


def test_our_referee_holds_a_free_kick_until_normal_start():
    cmd = _commands(
        (C.STOP, 5), (C.DIRECT_FREE_YELLOW, 5), (C.NORMAL_START, 5), (C.STOP, 5), (C.DIRECT_FREE_BLUE, 5), (C.STOP, 5)
    )
    live = live_mask(cmd)
    assert np.flatnonzero(live).tolist() == list(range(10, 15))  # the voided free kick is not play
    ts = np.arange(len(cmd)) / _HZ
    restarts = find_restarts(ts, cmd, live, np.full(len(cmd), 2.0), np.ones(len(cmd)), my_team_is_yellow=True)
    assert restarts == [{"t": ts[10], "kind": "free kick", "side": "friendly", "third": "attacking"}]


def test_the_official_controllers_free_kick_is_play_from_its_first_tick():
    cmd = _commands(
        (C.STOP, 5), (C.DIRECT_FREE_BLUE, 5), (C.STOP, 5), (C.FORCE_START, 5), (C.STOP, 5), (C.DIRECT_FREE_YELLOW, 5)
    )
    live = live_mask(cmd)
    ts = np.arange(len(cmd)) / _HZ
    restarts = find_restarts(ts, cmd, live, np.full(len(cmd), 2.0), np.ones(len(cmd)), my_team_is_yellow=True)
    # Blue is the enemy and attacks -x: the ball at x = 2 is in its defensive third.
    assert restarts == [
        {"t": ts[5], "kind": "free kick", "side": "enemy", "third": "defensive"},
        {"t": ts[15], "kind": "force start", "side": None, "third": None},  # not the free kick again
        {"t": ts[25], "kind": "free kick", "side": "friendly", "third": "attacking"},
    ]


def _referee(yellow, blue):
    return SimpleNamespace(yellow_team=SimpleNamespace(score=yellow), blue_team=SimpleNamespace(score=blue))


def test_a_goal_is_timed_when_play_stopped_for_it():
    live = np.array([True] * 10 + [False] * 300)
    ts = np.arange(len(live)) / _HZ
    ours = find_goals(ts, live, {0: _referee(0, 0), 10: _referee(1, 0)}, my_team_is_yellow=True)
    official = find_goals(ts, live, {0: _referee(0, 0), 250: _referee(0, 1)}, my_team_is_yellow=True)
    assert ours == [{"t": ts[10], "side": "friendly"}]
    assert official == [{"t": ts[10], "side": "enemy"}]


def test_a_lowered_score_withdraws_that_sides_latest_goal():
    live = np.ones(100, dtype=bool)
    ts = np.arange(100) / _HZ
    referee = {0: _referee(0, 0), 10: _referee(1, 0), 20: _referee(2, 1), 30: _referee(1, 1)}
    assert find_goals(ts, live, referee, my_team_is_yellow=False) == [
        {"t": ts[10], "side": "enemy"},
        {"t": ts[20], "side": "friendly"},
    ]


def test_goals_per_match_is_the_sum_over_situations():
    def match(friendly, enemy, restarts, spells, goals):
        m = {"labels": {"friendly": friendly, "enemy": enemy}, "restarts": restarts, "spells": spells}
        fold_restart_takes(restarts, spells)
        attribute(goals, situations(m))
        return {**m, "goals": goals}

    matches = [
        match(
            "a",
            "b",
            [_restart(0.0, "kickoff", "friendly", "middle"), _restart(30.0, "free kick", "enemy", "attacking")],
            [_spell("friendly", 0.5), _spell("enemy", 12.0, "attacking"), _spell("friendly", 40.0, "defensive")],
            [_goal(15.0, "enemy"), _goal(35.0, "enemy"), _goal(60.0, "friendly")],
        ),
        match("b", "a", [_restart(0.0, "penalty", "enemy", "attacking")], [], [_goal(3.0, "enemy")]),
    ]
    strategies = tables(matches)
    for s in strategies.values():
        types = s["types"].values()
        assert sum(r["scored_per_match"] for r in types) == pytest.approx(s["goals_for"] / s["matches"])
        assert sum(r["conceded_per_match"] for r in types) == pytest.approx(s["goals_against"] / s["matches"])
    a, b = strategies["a"], strategies["b"]
    assert (a["goals_for"], a["goals_against"], b["goals_for"], b["goals_against"]) == (2, 2, 2, 2)
    assert a["types"]["own kickoff"]["situations_per_match"] == 0.5  # the taker's spell is the kickoff
    assert a["types"]["own penalty"]["p_scored"] == 1.0
    assert b["types"]["own ball won, attacking third"]["p_scored"] == 1.0
    assert b["types"]["own free kick, attacking third"]["p_scored"] == 1.0
    assert a["types"][SUSTAINED]["scored_per_match"] == 0.5  # 20 s after the last start
    assert a["types"][SUSTAINED]["situations_per_match"] is None
