"""Flat OpenJev engine, model-free: cells, each robot's menu, the scripted rule, the chained
decision with a fake client (the real one needs a FOR-Engine server), and a short rsim match."""

from __future__ import annotations

import pytest

from utama_core.entities.data.vector import Vector2D
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.strategy.openjev import flat_state as fs
from utama_core.strategy.openjev.client import Decision
from utama_core.strategy.openjev.flat import FlatJevTactic, FlatMem, scripted_choice
from utama_core.tests.fixtures.game_builder import build_game, make_ball, make_robot

OUTFIELD = (1, 2, 3, 4, 5)


@pytest.mark.parametrize("is_right", [True, False])
@pytest.mark.parametrize("cell", fs.CELLS)
def test_every_cell_centre_maps_back_to_its_cell(is_right, cell):
    game = build_game({1: make_robot(1, is_friendly=True)}, {}, my_team_is_right=is_right)
    x_lo, x_hi, y_lo, y_hi = fs.cell_bounds(game, cell)
    assert fs.cell_of(game, Vector2D((x_lo + x_hi) / 2, (y_lo + y_hi) / 2)) == cell


def test_cells_are_named_from_our_point_of_view():
    # A team on the left attacks +x, where its left is +y; a team on the right attacks -x, left is -y.
    left_team = build_game({1: make_robot(1, is_friendly=True)}, {}, my_team_is_right=False)
    right_team = build_game({1: make_robot(1, is_friendly=True)}, {}, my_team_is_right=True)
    assert fs.cell_of(left_team, Vector2D(4.0, 2.0)) == "attacking_left"
    assert fs.cell_of(right_team, Vector2D(-4.0, -2.0)) == "attacking_left"
    assert fs.cell_of(right_team, Vector2D(-4.0, 2.0)) == "attacking_right"
    assert fs.cell_of(right_team, Vector2D(4.0, 0.0)) == "defensive_centre"


def test_adjacent_cells_include_the_cell_itself():
    assert fs.adjacent_cells("defensive_left") == ["defensive_left", "defensive_centre", "middle_left", "middle_centre"]
    assert len(fs.adjacent_cells("middle_centre")) == 9
    assert len(fs.adjacent_cells("attacking_centre")) == 6


def _game(carrier_xy=None, mates=None, enemies=None, ball_xy=(0.0, 0.0)):
    """Our team on the left (attacking +x), referee live so the enemy keeper id (0) is known."""
    friendly = {0: make_robot(0, is_friendly=True, x=-4.4, y=0.0)}
    for rid, (x, y) in (mates or {}).items():
        friendly[rid] = make_robot(rid, is_friendly=True, x=x, y=y)
    has_ball_robot = None
    if carrier_xy is not None:
        rid, x, y = carrier_xy
        friendly[rid] = make_robot(rid, is_friendly=True, x=x, y=y)
        ball_xy = (x + 0.09, y)
        has_ball_robot = ("friendly", rid)
    enemy = {eid: make_robot(eid, is_friendly=False, x=x, y=y) for eid, (x, y) in (enemies or {0: (4.4, 0.0)}).items()}
    return build_game(
        friendly,
        enemy,
        make_ball(*ball_xy),
        my_team_is_right=False,
        referee_command=RefereeCommand.NORMAL_START,
        has_ball_robot=has_ball_robot,
    )


def _view(game, rid, carry_left_m=None, marked_by=None, blockers=0):
    return fs.robot_view(game, rid, OUTFIELD, carry_left_m=carry_left_m, marked_by=marked_by or {}, blockers=blockers)


WALL = {0: (4.4, 0.0), 1: (2.6, -0.15), 2: (2.6, 0.0), 3: (2.6, 0.15)}  # three robots across the carrier's shot


def test_carrier_menu_has_ball_actions_and_no_dribble_once_its_carry_is_used():
    game = _game(carrier_xy=(2, 0.0, 0.0), mates={3: (1.0, 2.0), 4: (-1.0, 0.0)})
    view = _view(game, 2, carry_left_m=0.5)
    assert view.has_ball
    assert {"shoot", "pass:3", "pass:4", "clear"} <= set(view.options)
    assert "pass:0" not in view.options  # the goalkeeper plays by itself
    assert any(k.startswith("dribble:") for k in view.options)
    assert not any(k.startswith("dribble:") for k in _view(game, 2, carry_left_m=0.2).options)


def test_off_ball_menu_marks_every_opponent_but_their_goalkeeper():
    game = _game(mates={1: (0.0, 0.0)}, enemies={0: (4.4, 0.0), 3: (1.0, 1.0), 5: (-2.0, -1.0)}, ball_xy=(1.0, 0.0))
    view = _view(game, 1, marked_by={3: [4]}, blockers=1)
    assert not view.has_ball
    assert {"get_ball", "block_shot", "hold", "mark:3", "mark:5"} <= set(view.options)
    assert "mark:0" not in view.options
    assert len([k for k in view.options if k.startswith("move:")]) == 9
    assert view.options["mark:3"].endswith("marked by 4")
    assert view.options["mark:5"].endswith("unmarked")


def test_scripted_carrier_shoots_on_sight_else_passes_to_a_free_teammate():
    open_shot = _game(carrier_xy=(2, 2.0, 0.0), mates={3: (2.0, 2.0)}, enemies={0: (4.4, 2.5)})
    assert scripted_choice(open_shot, _view(open_shot, 2), {}, True) == "shoot"

    walled = _game(carrier_xy=(2, 2.0, 0.0), mates={3: (2.0, 2.0)}, enemies=WALL)
    view = _view(walled, 2)
    assert view.options["shoot"].startswith("shot blocked")  # the precondition the rule reads
    assert "lane open" in view.options["pass:3"] and ", free," in view.options["pass:3"]
    assert scripted_choice(walled, view, {}, True) == "pass:3"


def test_scripted_off_ball_two_chasers_two_blockers_then_an_outlet():
    game = _game(mates={1: (0.0, 0.0)}, enemies={0: (4.4, 0.0), 3: (1.0, 1.0), 5: (-2.0, -1.0)}, ball_xy=(1.0, 0.0))
    decided = {}
    for expected in ("get_ball", "get_ball", "block_shot", "block_shot", "move:attacking_centre"):
        view = _view(game, 1, blockers=sum(a == "block_shot" for a in decided.values()))
        assert scripted_choice(game, view, decided, False) == expected
        decided[len(decided) + 2] = expected


def test_block_shot_is_not_offered_to_a_third_robot():
    game = _game(mates={1: (0.0, 0.0)}, ball_xy=(1.0, 0.0))
    assert "block_shot" in _view(game, 1, blockers=1).options
    assert "block_shot" not in _view(game, 1, blockers=2).options


class FakeClient:
    """Answers each robot from `answers[robot]` (else the first option); records every request."""

    def __init__(self, answers=None, fail=False):
        self.answers = answers or {}
        self.fail = fail
        self.requests: list[tuple[str, dict]] = []
        self.prewarmed: list[str] = []

    def prewarm(self, state):
        self.prewarmed.append(state)
        return {"ok": True}

    def decide(self, state, questions):
        if self.fail:
            raise ConnectionError("no server")
        self.requests.append((state, questions))
        q = questions[fs.QUESTION_KEY]
        robot = int(q["instructions"].rsplit("robot ", 1)[1].split()[0])
        choice = self.answers.get(robot, next(iter(q["criteria"])))
        return Decision(
            answers={fs.QUESTION_KEY: {"choice": choice, "probabilities": {choice: 1.0}}}, latency_ms=1.0, cached=False
        )


def _carrier_game():
    # Robot 2 has the ball; 4 is nearest the ball after it, then 5, then 1.
    return _game(
        carrier_xy=(2, 0.0, 0.0),
        mates={1: (-3.0, 0.0), 3: (1.5, 2.0), 4: (0.5, -0.5), 5: (-1.0, 1.0)},
        enemies={0: (4.4, 0.0), 1: (2.0, 0.0)},
    )


def test_a_decision_asks_the_carrier_first_and_chains_every_earlier_choice():
    client = FakeClient(answers={2: "pass:3", 4: "move:attacking_left", 5: "block_shot", 1: "hold"})
    tactic = FlatJevTactic(client)
    mem = FlatMem()
    game = _carrier_game()
    tactic._decide(game, 0.0, OUTFIELD, 2, mem)

    asked = [int(q[fs.QUESTION_KEY]["instructions"].rsplit("robot ", 1)[1].split()[0]) for _, q in client.requests]
    assert asked == [2, 4, 5, 1]  # carrier, then nearest the ball; 3 receives and is not asked
    states = {s for s, _ in client.requests}
    assert len(states) == 1 and next(iter(states)).startswith(fs.STATIC_TEXT)  # one cacheable state per moment
    assert client.prewarmed == list(states)  # warmed once, so each question computes only its own tail
    second = client.requests[1][1][fs.QUESTION_KEY]["instructions"]
    assert "robot 2 -> pass to robot 3" in second and "robot 3 -> receive the pass from robot 2" in second
    last = client.requests[-1][1][fs.QUESTION_KEY]["instructions"]
    assert "robot 4 -> move:attacking_left" in last and "robot 5 -> block_shot" in last
    assert mem.plans[2].action == "pass:3" and mem.plans[3].action == "receive"
    assert mem.plans[4].target is not None and fs.cell_of(game, mem.plans[4].target) == "attacking_left"


def test_robots_in_a_pass_are_not_asked_again_until_it_ends():
    client = FakeClient(answers={2: "pass:3"})
    tactic = FlatJevTactic(client)
    mem = FlatMem()
    game = _carrier_game()
    tactic._decide(game, 0.0, OUTFIELD, 2, mem)
    client.requests.clear()
    tactic._decide(game, 0.2, OUTFIELD, 2, mem)
    asked = {int(q[fs.QUESTION_KEY]["instructions"].rsplit("robot ", 1)[1].split()[0]) for _, q in client.requests}
    assert asked == {1, 4, 5}
    assert "robot 2 -> passing to robot 3" in client.requests[0][1][fs.QUESTION_KEY]["instructions"]


def test_a_failing_server_falls_back_to_the_scripted_rule_per_robot():
    tactic = FlatJevTactic(FakeClient(fail=True))
    mem = FlatMem()
    tactic._decide(_carrier_game(), 0.0, OUTFIELD, 2, mem)
    assert tactic.n_fallback == tactic.n_questions > 0
    assert set(mem.plans) == set(OUTFIELD)


def test_a_blocker_with_a_tackling_teammate_holds_the_line_in_front_of_the_ball():
    from utama_core.strategy.openjev.flat import (
        BLOCK_HOLD_M,
        enemy_holder,
        shot_line_point,
    )

    game = _game(mates={1: (0.0, 0.0)}, enemies={0: (4.4, 0.0), 3: (-1.0, 0.0)}, ball_xy=(-1.12, 0.0))
    assert enemy_holder(game) == 3
    p = shot_line_point(game, BLOCK_HOLD_M)
    assert abs(p.distance_to(Vector2D(-1.12, 0.0)) - BLOCK_HOLD_M) < 1e-6
    assert p.x < -1.12  # between the ball and our goal (x = -4.5)
    assert (
        enemy_holder(_game(mates={1: (-1.2, 0.0)}, enemies={0: (4.4, 0.0), 3: (-1.0, 0.0)}, ball_xy=(-1.12, 0.0)))
        is None
    )
