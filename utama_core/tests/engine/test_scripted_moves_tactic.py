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
