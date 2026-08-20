"""probe_default_vs_lowblock.py — instrumented reproduction of the tournament's
default_vs_lowblock match, for investigating the 0-0 / passer-tangle pattern.

Same match configuration as `demo_tournament.run_match` (6v6, headless rsim,
yellow = default config attacking the left goal, blue = low_block), but every
tick dumps a JSON row with:

  - sim time, referee command
  - ball position/velocity
  - all 12 robots' positions + orientations (pitch frame / "my" perspective)
  - per-side kernel slot state: tactic id, assigned robots, `is_committed()`,
    and each tactic's `mem` fields (phase, assigned pair, setup targets,
    phase ticks, setup_ticks_without_ball, goal_scored)

Yellow's own-frame targets are mirrored into the pitch frame (yellow is the
right team), blue's are not (blue is the left team), so all coordinates in
the dump are comparable in the same frame.

Usage:  pixi run python probe_default_vs_lowblock.py [duration_seconds] [initial_command]

`initial_command` is any RefereeCommand name (e.g. PREPARE_KICKOFF_YELLOW to
start with a proper kickoff ceremony); defaults to FORCE_START (the
StrategyRunner sim default that the tournament uses).

Output: /tmp/opencode/probe_default_vs_lowblock.jsonl (one row per tick).
Replays (my + opp perspective) and match log / stats are also written via the
standard StrategyRunner machinery.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict

from utama_core.custom_referee import CustomReferee
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.kernel import kernel_strategy
from utama_core.replay.replay_writer import ReplayWriterConfig
from utama_core.run import StrategyRunner
from utama_core.strategy.common.abstract_strategy import AbstractStrategy

N_OUTFIELD = 5
OUTFIELD_ROBOT_IDS = tuple(range(1, N_OUTFIELD + 1))
DURATION_SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
INITIAL_COMMAND = RefereeCommand[sys.argv[2]] if len(sys.argv) > 2 else RefereeCommand.FORCE_START
TICKS_PER_SECOND = 60

OUT_PATH = "/tmp/opencode/probe_default_vs_lowblock.jsonl"

_MEM_ATTRS = (
    "phase",
    "assigned_pair",
    "locked_assignment",
    "passer_position",
    "receiver_position",
    "phase_ticks",
    "setup_ticks_without_ball",
    "goal_scored",
)


def _slot_state(game, tactic_id, slot):
    """Compact, serializable view of one kernel tactic slot (my frame dims)."""
    state = {
        "tactic": tactic_id,
        "robots": sorted(slot.assigned_robots),
        "committed": bool(slot.tactic.is_committed(game, slot.mem)),
    }
    mem = slot.mem
    if mem is not None and mem is not getattr(mem, "EMPTY", None) and not getattr(mem, "__empty__", False):
        if hasattr(mem, "pass_and_score") and mem.pass_and_score is not None:
            # PassAndShootMem wraps PassAndScoreMem; phase/targets live inside.
            outer, inner = mem, mem.pass_and_score
        else:
            outer, inner = mem, mem
        for attr in _MEM_ATTRS:
            if not hasattr(inner, attr):
                continue
            value = getattr(inner, attr)
            if attr in ("passer_position", "receiver_position") and value is not None:
                value = (round(value.x, 2), round(value.y, 2))
            if isinstance(value, tuple):
                value = list(value)
            state[attr] = value
        if hasattr(outer, "assigned_pair") and outer.assigned_pair is not None:
            state["assigned_pair"] = list(outer.assigned_pair)
    return state


def _sides_state(runner):
    """Per-tick, per-team tactic slot states, mirrored into the pitch frame."""

    def _kernel_of(strategy):
        return getattr(strategy, "_kernel_strategy", None)

    my_kernel = _kernel_of(runner.my.strategy)
    opp_kernel = _kernel_of(runner.opp.strategy)

    my_slots = (
        {tid: _slot_state(runner.my.game, tid, slot) for tid, slot in my_kernel._slots.items()}
        if my_kernel is not None
        else None
    )
    opp_slots = (
        {tid: _slot_state(runner.opp.game, tid, slot) for tid, slot in opp_kernel._slots.items()}
        if opp_kernel is not None
        else None
    )
    # Yellow (right team) targets are stored in yellow's own frame: mirror x
    # into the pitch frame so blue/yellow targets are directly comparable.
    for tid, st in (my_slots or {}).items():
        for attr in ("passer_position", "receiver_position"):
            if st.get(attr) is not None:
                st[attr] = [round(-st[attr][0], 2), st[attr][1]]
    return my_slots, opp_slots


def _row(runner, tick):
    frame = runner.my.current_game_frame
    referee = getattr(runner.my.game, "referee", None)
    command = getattr(referee, "referee_command", None)
    command = str(command) if command is not None else None
    score = None
    if referee is not None:
        score = {
            "yellow": getattr(referee.yellow_team, "score", None),
            "blue": getattr(referee.blue_team, "score", None),
        }
    my_slots, opp_slots = _sides_state(runner)

    def _robots(robots):
        out = {}
        for rid, robot in sorted((robots or {}).items()):
            out[str(rid)] = [round(robot.p.x, 2), round(robot.p.y, 2), round(robot.orientation, 3)]
        return out

    ball = frame.ball
    violation = None
    custom = getattr(runner, "referee", None)
    if custom is not None and getattr(custom, "last_violation", None) is not None:
        v = custom.last_violation
        violation = {
            "rule": v.rule_name,
            "suggested": str(v.suggested_command),
            "message": v.status_message,
        }
    return {
        "tick": tick,
        "t": round(tick / TICKS_PER_SECOND, 2),
        "cmd": command,
        "score": score,
        "violation": violation,
        "y_has": {str(rid): bool(rb.has_ball) for rid, rb in sorted((frame.friendly_robots or {}).items())},
        "b_has": {str(rid): bool(rb.has_ball) for rid, rb in sorted((frame.enemy_robots or {}).items())},
        "ball": (
            [round(ball.p.x, 3), round(ball.p.y, 3), round(ball.v.x, 3), round(ball.v.y, 3)]
            if ball is not None
            else None
        ),
        "yellow": _robots(frame.friendly_robots),
        "blue": _robots(frame.enemy_robots),
        "my_slots": my_slots,
        "opp_slots": opp_slots,
    }


def main() -> None:
    build_default = getattr(kernel_strategy, "build_default_kernel_strategy")
    build_low_block = getattr(kernel_strategy, "build_low_block_kernel_strategy")

    strategy_a = AbstractStrategy(build_kernel_strategy=build_default(OUTFIELD_ROBOT_IDS))
    strategy_b = AbstractStrategy(build_kernel_strategy=build_low_block(OUTFIELD_ROBOT_IDS))

    referee = CustomReferee.from_profile_name(
        "simulation", n_robots_yellow=N_OUTFIELD + 1, n_robots_blue=N_OUTFIELD + 1
    )

    runner = StrategyRunner(
        strategy=strategy_a,
        opp_strategy=strategy_b,
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=N_OUTFIELD + 1,
        exp_enemy=N_OUTFIELD + 1,
        exp_ball=True,
        referee=referee,
        referee_initial_command=INITIAL_COMMAND,
        enable_vision_stream=False,
        replay_writer_config=ReplayWriterConfig(replay_name="probe_default_vs_lowblock", overwrite_existing=True),
        match_log_path="/tmp/opencode/probe_matchlog.jsonl",
        stats_path="/tmp/opencode/probe_stats.json",
    )

    total_ticks = int(DURATION_SECONDS * TICKS_PER_SECOND)
    out_file = open(OUT_PATH, "w")
    start_wall = time.monotonic()
    try:
        for tick in range(1, total_ticks + 1):
            runner.step_once()
            if tick % 600 == 0:
                print(f"tick {tick} t={tick / TICKS_PER_SECOND:.0f}s", flush=True)
            out_file.write(json.dumps(_row(runner, tick)) + "\n")
    finally:
        out_file.close()
        runner.close()

    wall = time.monotonic() - start_wall
    ref_data = runner.my.game.referee
    print(
        f"\nFinal: yellow(default) {ref_data.yellow_team.score} - "
        f"{ref_data.blue_team.score} blue(low_block)  ({wall:.1f}s wall for {DURATION_SECONDS:.0f}s sim)"
    )
    print(f"Probe rows: {OUT_PATH}")


if __name__ == "__main__":
    main()
