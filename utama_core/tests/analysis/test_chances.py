from types import SimpleNamespace

import pytest

from utama_core.analysis.chances import (
    CHANCE_WINDOW_S,
    REGAIN_MIN_HOLD_S,
    SHOT_GOAL_WINDOW_S,
    ChanceTracker,
    open_goal,
    rates,
    side_totals,
)
from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.entities.referee.referee_command import RefereeCommand as C

HALF_LENGTH = STANDARD_FIELD_DIMS.full_field_half_length
# Friendly (yellow) plays on the right, so it attacks the goal at -HALF_LENGTH.
ATTACKED_GOAL_X = -HALF_LENGTH


def test_open_goal_is_the_share_of_the_mouth_no_blocker_covers():
    assert open_goal((-2.5, 0.0), ATTACKED_GOAL_X, []) == 1.0
    # A robot on the line to the goal centre covers the middle of the mouth, not its posts.
    middle = open_goal((-2.5, 0.0), ATTACKED_GOAL_X, [(-3.5, 0.0)])
    assert 0.0 < middle < 1.0
    # One just in front of the shooter covers all of it.
    assert open_goal((-2.5, 0.0), ATTACKED_GOAL_X, [(-2.65, 0.0)]) == 0.0


class _Match:
    """Drives a `ChanceTracker` tick by tick with a stand-in accumulator."""

    def __init__(self):
        self.tracker = ChanceTracker()
        self.t = 0.0
        self.cmd = C.NORMAL_START
        self.live_since = -100.0
        self.poss = (None, None)  # side, robot id
        self.shots = {"friendly": 0, "enemy": 0}
        self.score = [0, 0]  # yellow (friendly), blue
        self.ball = (0.0, 0.0)

    def tick(self, seconds: float = 1 / 60, shot: str = None):
        ticks = max(1, round(seconds * 60))
        for i in range(ticks):
            self.t += 1 / 60
            before = dict(self.shots)
            if shot is not None and i == 0:
                self.shots[shot] += 1
            v2 = lambda x, y: SimpleNamespace(x=x, y=y)  # noqa: E731
            frame = SimpleNamespace(
                ts=self.t,
                my_team_is_right=True,
                my_team_is_yellow=True,
                ball=SimpleNamespace(p=v2(*self.ball), v=v2(0.0, 0.0)),
                friendly_robots={0: SimpleNamespace(p=v2(*self.ball))},
                enemy_robots={0: SimpleNamespace(p=v2(4.0, 2.0))},
                referee=SimpleNamespace(
                    referee_command=self.cmd,
                    yellow_team=SimpleNamespace(score=self.score[0]),
                    blue_team=SimpleNamespace(score=self.score[1]),
                ),
            )
            acc = SimpleNamespace(_poss_side=self.poss[0], _poss_robot_id=self.poss[1], _shots=dict(self.shots))
            self.tracker.step(frame, acc, self.cmd, self.live_since, before)

    def result(self) -> dict:
        return self.tracker.result()


def test_a_shot_scores_only_if_its_side_scores_within_the_window():
    m = _Match()
    m.poss, m.ball = ("friendly", 0), (-2.5, 0.0)
    m.tick(shot="friendly")
    m.tick(SHOT_GOAL_WINDOW_S - 0.5)
    m.score[0] += 1
    m.tick()
    m.tick(shot="friendly")
    m.tick(SHOT_GOAL_WINDOW_S + 0.5)
    m.score[0] += 1
    m.tick()

    r = m.result()

    assert [s["scored"] for s in r["shots"]] == [True, False]
    assert r["shots"][0]["distance_m"] == pytest.approx(HALF_LENGTH - 2.5, abs=0.01)
    assert r["shots"][0]["open_goal"] == 1.0
    assert r["unshot_goals"] == {"friendly": 1, "enemy": 0}


def test_a_regain_counts_once_held_for_its_minimum_and_times_the_next_shot():
    m = _Match()
    m.poss = ("enemy", 0)
    m.tick(2.0)
    m.poss = ("friendly", 0)  # a flicker: back to the enemy before the minimum hold
    m.tick(REGAIN_MIN_HOLD_S - 0.5)
    m.poss = ("enemy", 0)
    m.tick(2.0)
    m.poss = ("friendly", 0)  # a real regain, shot from 3 s later
    m.tick(3.0)
    m.tick(shot="friendly")
    m.tick(2.0)
    m.poss = ("enemy", 0)
    m.tick()

    friendly = [g for g in m.result()["regains"] if g["side"] == "friendly"]

    assert len(friendly) == 1
    assert friendly[0]["shot_after_s"] == pytest.approx(3.0 + 1 / 60, abs=0.02)


def test_a_change_of_possession_just_after_a_restart_is_not_a_regain():
    m = _Match()
    m.poss = ("enemy", 0)
    m.tick(2.0)
    m.live_since = m.t  # play restarted; the friendly side takes the kick
    m.poss = ("friendly", 0)
    m.tick(3.0)
    m.poss = ("enemy", 0)
    m.tick()

    assert [g for g in m.result()["regains"] if g["side"] == "friendly"] == []


def test_danger_is_live_time_the_opponent_holds_the_ball_in_the_defensive_third():
    m = _Match()
    m.poss, m.ball = ("enemy", 0), (HALF_LENGTH - 1.0, 0.0)  # near friendly's own goal
    m.tick(2.0)
    m.cmd = C.STOP
    m.tick(5.0)  # stopped: not danger
    m.cmd = C.NORMAL_START
    m.tick(1.0)

    danger = m.result()["danger"]

    assert danger["friendly"]["s"] == pytest.approx(3.0, abs=0.05)
    assert danger["friendly"]["spells"] == 2
    assert danger["enemy"] == {"s": 0.0, "spells": 0}


def test_a_free_kick_converts_when_its_side_shoots_within_the_window():
    m = _Match()
    m.ball = (-3.5, 2.5)  # a corner the friendly side attacks
    for shoot_after in (CHANCE_WINDOW_S - 1.0, CHANCE_WINDOW_S + 1.0):
        m.cmd = C.STOP
        m.tick()
        m.cmd = C.DIRECT_FREE_YELLOW
        m.tick()
        m.cmd = C.NORMAL_START
        m.tick()
        m.poss = ("friendly", 0)
        m.tick(shoot_after)
        m.tick(shot="friendly")

    kicks = m.result()["free_kicks"]

    assert [(k["side"], k["attacking_third"]) for k in kicks] == [("friendly", True)] * 2
    assert [k["shot_after_s"] is not None for k in kicks] == [True, False]


def test_save_rate_is_from_the_shots_the_other_side_took():
    record = {
        "shots": [
            {"side": "enemy", "distance_m": 2.0, "open_goal": 0.5, "scored": True},
            {"side": "enemy", "distance_m": 3.0, "open_goal": 0.1, "scored": False},
            {"side": "enemy", "distance_m": 3.0, "open_goal": 0.3, "scored": False},
            {"side": "friendly", "distance_m": 1.0, "open_goal": 1.0, "scored": True},
        ],
        "unshot_goals": {"friendly": 0, "enemy": 1},
        "regains": [],
        "danger": {"friendly": {"s": 10.0, "spells": 2}, "enemy": {"s": 0.0, "spells": 0}},
        "free_kicks": [],
    }

    friendly = rates(side_totals(record, "friendly"))

    assert (friendly["shots_faced"], friendly["save_rate"]) == (3, 0.67)
    assert friendly["faced_open_goal"] == 0.3
    assert (friendly["shots"], friendly["conversion"]) == (1, 1.0)
    assert friendly["regain_to_shot"] is None
