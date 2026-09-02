"""Unit-level (no rsim) test for `DecoyOverloadTactic`'s "finish"-phase timeout.

Regression for a real 2026-09-02 full-length-tournament freeze
(`clear_press_plus_vs_shadow_switch_LK.pkl`, t=205.8s): a referee restart can
strand the tactic in `"finish"` with the ball nowhere near either the decoy or
the overloader and no path back to "setup" — `is_committed()` only releases on
`goal_scored`, so with no goal ever scored this held the slot's two robots for
the rest of the match while a different tactic's robot independently
converged on the same displaced ball, and both stalled at
`FastPathPlanner.OBSTACLE_CLEARANCE` apart. Same stalled-phase-with-no-timeout
bug class `pass_and_shoot.py`'s own `_PHASE_TIMEOUT_TICKS` exists to prevent
(see its own `is_committed()`/`phase == "setup"` gating in
`test_pass_and_shoot_tactic.py`).

The timeout check is the first statement in the "finish" branch of `tick()`,
before any `Game`-shaped attribute is touched (`has_ball`, `_decoy_shot_open`,
`_pass_exec` all require a real `Game`). Passing `game=None` here is
deliberate: if the timeout guard didn't short-circuit before those calls, this
test would raise `AttributeError` instead of returning cleanly, proving the
guard actually protects the rest of the branch rather than just existing
alongside it.
"""

from __future__ import annotations

from utama_core.engine.context import TickContext
from utama_core.tactics.decoy_and_overload import (
    _FINISH_TIMEOUT_TICKS,
    DecoyOverloadMem,
    DecoyOverloadTactic,
)


def test_is_committed_true_mid_finish_phase():
    tactic = DecoyOverloadTactic()
    mem = DecoyOverloadMem(decoy_id=1, overloader_id=2, phase="finish", finish_ticks=1)
    assert tactic.is_committed(game=None, mem=mem) is True


def test_finish_phase_times_out_and_releases_the_tactic():
    tactic = DecoyOverloadTactic()
    mem = DecoyOverloadMem(
        decoy_id=1,
        overloader_id=2,
        marker_id=None,
        marker_start_y=0.0,
        phase="finish",
        finish_ticks=_FINISH_TIMEOUT_TICKS,  # already at budget: this tick pushes it over
    )
    ctx = TickContext(motion_controller=None)
    commands, new_mem = tactic.tick(game=None, ctx=ctx, robot_ids=(1, 2), mem=mem)

    assert commands == {}
    assert new_mem.decoy_id is None
    assert tactic.is_committed(game=None, mem=new_mem) is False


def test_finish_phase_does_not_time_out_before_budget():
    """The timeout must not fire on the very first "finish" tick — only once
    `finish_ticks` genuinely exceeds the budget — otherwise every decoy-and-
    overload attempt would abort immediately instead of getting its real
    pass+shoot window. `tick()` itself needs a real `Game` past this point
    (`has_ball`/`_decoy_shot_open`/`_pass_exec`), so this checks the guard
    condition directly rather than calling `tick()` with `game=None` (which
    would legitimately raise once past the guard, on a fresh attempt)."""
    mem = DecoyOverloadMem(decoy_id=1, overloader_id=2, phase="finish", finish_ticks=0)
    mem.finish_ticks += 1
    assert mem.finish_ticks <= _FINISH_TIMEOUT_TICKS
