"""`ScriptedMovesTactic` plays its moves in order, restarts a move a pause interrupted, then stands still."""

from __future__ import annotations

from types import SimpleNamespace

from utama_core.tactics.scripted_moves import FULL_SCRIPT, Move, ScriptedMovesTactic

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


def test_the_full_script_fires_the_kicker_before_the_ball_goes_on():
    """A kicker left unused runs flat (field day 2026-10-28): one kick with no ball charges it,
    before the operator is first asked to put the ball on the dribbler."""
    first_kick = next(i for i, m in enumerate(FULL_SCRIPT) if m.kick)
    first_ball = next(i for i, m in enumerate(FULL_SCRIPT) if m.prompt is not None and "dribbler" in m.prompt)
    assert first_kick < first_ball
