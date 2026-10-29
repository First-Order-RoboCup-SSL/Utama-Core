"""`ScriptedMovesTactic` plays its moves in order, restarts a move a pause interrupted, then stands still."""

from __future__ import annotations

from types import SimpleNamespace

from utama_core.tactics.scripted_moves import Move, ScriptedMovesTactic

SCRIPT = (Move("go", 0.5, forward=0.3), Move("turn", 0.5, turn=1.0))


def _tick(tactic: ScriptedMovesTactic, ts: float, robot_ids=(4, 7)):
    commands, _ = tactic.tick(SimpleNamespace(ts=ts), None, robot_ids, None)
    return commands


def _run(tactic, t0: float, seconds: float) -> dict:
    commands = None
    for k in range(round(seconds * 60)):
        commands = _tick(tactic, t0 + k / 60)
    return commands


def test_moves_play_in_order_on_the_first_robot_and_the_others_stand_still():
    tactic = ScriptedMovesTactic(SCRIPT)
    first = _run(tactic, 0.0, 0.25)
    assert first[4].local_forward_vel == 0.3 and first[7].local_forward_vel == 0.0
    assert _run(tactic, 0.25, 0.5)[4].angular_vel == 1.0


def test_a_pause_restarts_the_interrupted_move_from_its_start():
    tactic = ScriptedMovesTactic(SCRIPT)
    _run(tactic, 0.0, 0.4)  # 0.4 s into the 0.5 s "go"
    # HALT for 3 s: no ticks. Resumed, "go" runs a full 0.5 s again, not the 0.1 s left.
    assert _run(tactic, 3.4, 0.45)[4].local_forward_vel == 0.3
    assert _run(tactic, 3.85, 0.1)[4].angular_vel == 1.0


def test_once_done_the_robot_stands_still():
    tactic = ScriptedMovesTactic(SCRIPT)
    last = _run(tactic, 0.0, 1.2)
    assert tactic.done and last[4] == last[7] and last[4].local_forward_vel == 0.0 and last[4].angular_vel == 0.0


def test_a_prompt_holds_the_robot_still_until_the_operator_proceeds():
    script = (Move("go", 0.2, forward=0.3), Move("place ball", 0.0, prompt="ball on"), Move("kick", 0.2, kick=True))
    tactic = ScriptedMovesTactic(script)
    _run(tactic, 0.0, 0.3)
    assert tactic.waiting_for == "ball on"
    held = _run(tactic, 0.3, 5.0)  # however long it waits
    assert held[4].local_forward_vel == 0.0 and not held[4].kick
    tactic.proceed()
    kicked = _tick(tactic, 5.3 + 1 / 60)
    assert tactic.waiting_for is None and kicked[4].kick
    # The wait counts for nothing: "kick" plays its whole 0.2 s.
    assert _run(tactic, 5.3 + 2 / 60, 0.15)[4].kick


def test_dribble_and_kick_moves_reach_the_robot():
    tactic = ScriptedMovesTactic((Move("dribble", 0.5, turn=1.0, dribble=True),))
    command = _tick(tactic, 0.0)[4]
    assert command.dribble and command.angular_vel == 1.0 and not command.kick


# A there-and-back as the calibration script has it: go, rest, back, rest.
PAIR = (
    Move("go", 1.0, forward=0.5, home=True),
    Move("rest", 0.5),
    Move("go back", 1.0, forward=-0.5),
    Move("rest", 0.5),
    Move("turn", 0.5, turn=1.0),
)


class _Robot:
    """A robot that does what it is told at once, along the x axis facing +x."""

    def __init__(self, x: float = 0.0):
        self.x, self.heading = x, 0.0

    def game(self, ts: float):
        robot = SimpleNamespace(p=SimpleNamespace(x=self.x, y=0.0), orientation=self.heading)
        return SimpleNamespace(ts=ts, friendly_robots={4: robot})

    def play(self, tactic, t0: float, seconds: float) -> dict:
        commands = None
        for k in range(round(seconds * 60)):
            commands, _ = tactic.tick(self.game(t0 + k / 60), None, (4,), None)
            self.x += commands[4].local_forward_vel / 60
            self.heading += commands[4].angular_vel / 60
        return commands


def test_a_pause_in_the_back_half_drives_the_robot_home_and_replays_the_pair():
    # 2026-10-29: each auto-HALT restarted the "back" move where the robot stood; three of them
    # drove it past the goal line. Now it drives back to where "go" began and replays the pair.
    robot, tactic = _Robot(x=0.0), ScriptedMovesTactic(PAIR)
    robot.play(tactic, 0.0, 1.5)  # go 0.5 m, rest
    robot.play(tactic, 1.5, 0.6)  # 0.6 s into "go back": x ~0.2
    assert 0.15 < robot.x < 0.25
    robot.play(tactic, 5.0, 3.0)  # paused 2.4 s, then on: home, rest, "go" again
    assert tactic.step <= 1  # replaying the pair
    robot.play(tactic, 8.0, 3.0)
    assert tactic.step == 4 and abs(robot.x) < 0.06  # the pair done, back where it began


def test_repeated_pauses_never_take_the_robot_further_than_the_pair_does():
    robot, tactic = _Robot(x=0.0), ScriptedMovesTactic(PAIR)
    t = 0.0
    for _ in range(6):  # a pause every 1.7 s, as on 2026-10-29's feed
        robot.play(tactic, t, 1.7)
        assert -0.06 < robot.x < 0.56
        t += 2.0
    robot.play(tactic, t, 20.0)
    assert tactic.done and abs(robot.x) < 0.06


def test_the_pairs_halves_are_named_apart():
    from utama_core.tactics.scripted_moves import CALIBRATION_SCRIPT

    names = [m.name for m in CALIBRATION_SCRIPT if m.name != "rest"]
    assert "forward 0.3 m/s back" in names and names.count("forward 0.3 m/s") == 1
