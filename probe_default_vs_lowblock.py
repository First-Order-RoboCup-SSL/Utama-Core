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

Usage:  pixi run python probe_default_vs_lowblock.py [my_strategy] [opp_strategy] [duration_seconds] [initial_command]

`my_strategy` / `opp_strategy` are any `build_*_kernel_strategy` factory names
(short form, e.g. `tiki_taka`, `low_block`, `default`); defaults:
`default` vs `low_block` — the original investigation matchup. Both sides are
always constructed from kernel factories, so any two configs can be compared
in the same fixture.

`initial_command` is any RefereeCommand name (e.g. PREPARE_KICKOFF_YELLOW to
start with a proper kickoff ceremony); defaults to FORCE_START (the
StrategyRunner sim default that the tournament uses).

Output: /tmp/opencode/probe_default_vs_lowblock.jsonl (one row per tick).
Replays (my + opp perspective) and match log / stats are also written via the
standard StrategyRunner machinery; a compact boxscore is printed at the end.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from typing import Optional

from utama_core.custom_referee import CustomReferee
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.kernel import kernel_strategy
from utama_core.replay.replay_writer import ReplayWriterConfig
from utama_core.run import StrategyRunner
from utama_core.strategy.common.abstract_strategy import AbstractStrategy

N_OUTFIELD = 5
OUTFIELD_ROBOT_IDS = tuple(range(1, N_OUTFIELD + 1))
TICKS_PER_SECOND = 60

OUT_PATH = "/tmp/opencode/probe_default_vs_lowblock.jsonl"
STATS_PATH = "/tmp/opencode/probe_stats.json"
MATCHLOG_PATH = "/tmp/opencode/probe_matchlog.jsonl"


def _parse_args(argv: list[str]):
    """Parse [my] [opp] [duration] [command] with backward-compatible defaults."""
    duration = 60.0
    command_name = None
    strategy_names: list[str] = []
    for token in argv:
        try:
            duration = float(token)
            continue
        except ValueError:
            pass
        if token in RefereeCommand.__members__:
            command_name = token
            continue
        if len(strategy_names) < 2:
            strategy_names.append(token)
    my_name = strategy_names[0] if strategy_names else "default"
    opp_name = strategy_names[1] if len(strategy_names) > 1 else "low_block"
    initial_command = RefereeCommand[command_name] if command_name else RefereeCommand.FORCE_START
    return my_name, opp_name, duration, initial_command


def _resolve_builder(name: str):
    """Find a build_*_kernel_strategy by short or full factory name."""
    candidates = (f"build_{name}_kernel_strategy", name)
    for candidate in candidates:
        builder = getattr(kernel_strategy, candidate, None)
        if callable(builder):
            return builder
    available = sorted(
        n for n in dir(kernel_strategy) if n.startswith("build_") and callable(getattr(kernel_strategy, n))
    )
    raise SystemExit(
        f"Unknown strategy {name!r}. Available: {[n.removeprefix('build_').removesuffix('_kernel_strategy') for n in available]}"
    )


_MEM_ATTRS = (
    "phase",
    "assigned_pair",
    "locked_assignment",
    "passer_position",
    "receiver_position",
    "phase_ticks",
    "setup_ticks_without_ball",
    "goal_scored",
    # GiveAndGoTactic mem (carrier/receiver/hop cycle).
    "carrier_id",
    "receiver_id",
    "hop_count",
)


def _slot_state(game, tactic_id, slot):
    """Compact, serializable view of one kernel tactic slot (my frame dims)."""
    state = {
        "tactic": tactic_id,
        "robots": sorted(slot.assigned_robots),
        # Unassigned slots have mem=None; only ticked slots can commit.
        "committed": bool(slot.assigned_robots and slot.tactic.is_committed(game, slot.mem)),
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
        "tm": dict(_TARGETS),
        # What the arena pickers actually see (same game object, same
        # proximity lookup they use) — diagnostic for possession-edge bugs.
        "edge": _picker_edge(runner.my.game),
    }


def _picker_edge(game):
    """The pickers' possession-edge verdict + the raw proximity distances."""
    from utama_core.entities.data.object import TeamType
    from utama_core.kernel.kernel_strategy import _friendly_closer_to_ball

    _, friendly_dist = game.proximity_lookup.closest_to_ball(team_type_filter=TeamType.FRIENDLY)
    _, enemy_dist = game.proximity_lookup.closest_to_ball(team_type_filter=TeamType.ENEMY)
    verdict = None
    if (
        friendly_dist is not None
        and enemy_dist is not None
        and friendly_dist != float("inf")
        and enemy_dist != float("inf")
    ):
        verdict = bool(friendly_dist < enemy_dist)
    return {
        "verdict": verdict,
        # The helper the pickers themselves call on the same game object.
        "helper": _friendly_closer_to_ball(game),
        "friendly_m": None if friendly_dist == float("inf") else round(friendly_dist, 3),
        "enemy_m": None if enemy_dist == float("inf") else round(enemy_dist, 3),
    }


# Every position target funnels through MotionController.calculate — record
# them per tick so the dump shows what each robot was *commanded to do*, not
# just where physics put it (positions alone cannot distinguish a bad target
# from a transient body collision).
_TARGETS: dict[int, tuple[float, float]] = {}


class _TargetRecorder:
    """Delegating MotionController wrapper that records calculate() targets."""

    def __init__(self, inner):
        self._inner = inner

    def calculate(self, game, robot_id, target_pos, target_oren, **kwargs):
        _TARGETS[robot_id] = (round(float(target_pos.x), 3), round(float(target_pos.y), 3))
        return self._inner.calculate(
            game=game, robot_id=robot_id, target_pos=target_pos, target_oren=target_oren, **kwargs
        )

    def reset(self, robot_id: int):
        return self._inner.reset(robot_id)


def _install_target_recorders(runner) -> None:
    """Wrap both kernels' motion controllers so every tick's targets land in `_TARGETS`."""
    for side in (runner.my, runner.opp):
        kernel = getattr(side.strategy, "_kernel_strategy", None)
        if kernel is not None:
            kernel._ctx.motion_controller = _TargetRecorder(kernel._ctx.motion_controller)


def main() -> None:
    my_name, opp_name, duration_seconds, initial_command = _parse_args(sys.argv[1:])
    build_my = _resolve_builder(my_name)
    build_opp = _resolve_builder(opp_name)

    strategy_a = AbstractStrategy(build_kernel_strategy=build_my(OUTFIELD_ROBOT_IDS))
    strategy_b = AbstractStrategy(build_kernel_strategy=build_opp(OUTFIELD_ROBOT_IDS))

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
        referee_initial_command=initial_command,
        enable_vision_stream=False,
        replay_writer_config=ReplayWriterConfig(replay_name="probe_default_vs_lowblock", overwrite_existing=True),
        match_log_path=MATCHLOG_PATH,
        stats_path=STATS_PATH,
    )

    total_ticks = int(duration_seconds * TICKS_PER_SECOND)
    out_file = open(OUT_PATH, "w")
    start_wall = time.monotonic()
    _install_target_recorders(runner)
    # A foul freezes the match at STOP. CustomReferee only auto-advances when
    # the STOP carries a placement position (out-of-bounds); a defense-area
    # foul (e.g. the opponent's own Defenders converging in its box) suggests
    # a bare STOP and the match would sit frozen forever. Real SSL restarts
    # after a foul, so the harness resumes play if STOP persists.
    stop_ticks = 0
    try:
        for tick in range(1, total_ticks + 1):
            _TARGETS.clear()
            runner.step_once()
            if tick % 600 == 0:
                print(f"tick {tick} t={tick / TICKS_PER_SECOND:.0f}s", flush=True)
            out_file.write(json.dumps(_row(runner, tick)) + "\n")
            cmd = _current_command(runner)
            if cmd == RefereeCommand.STOP:
                stop_ticks += 1
                if stop_ticks >= 120:
                    runner.referee.force_command(RefereeCommand.FORCE_START, runner.my.current_game_frame.ts)
                    print(f"auto-resume: STOP held {stop_ticks} ticks -> FORCE_START", flush=True)
                    stop_ticks = 0
            else:
                stop_ticks = 0
    finally:
        out_file.close()
        runner.close()

    wall = time.monotonic() - start_wall
    _print_boxscore(my_name, opp_name, duration_seconds, wall, initial_command.name)


def _current_command(runner) -> Optional[RefereeCommand]:
    referee = getattr(runner.my.game, "referee", None)
    command = getattr(referee, "referee_command", None)
    if isinstance(command, RefereeCommand):
        return command
    return None


def _print_boxscore(my_name: str, opp_name: str, duration_seconds: float, wall: float, initial_cmd: str) -> None:
    """Compact post-match summary from the stats JSON + the per-tick dump."""
    rows = [json.loads(line) for line in open(OUT_PATH)]
    score = rows[-1]["score"]
    stats = {}
    try:
        stats = json.load(open(STATS_PATH))
    except (OSError, ValueError):
        pass

    print()
    print(
        f"Match: yellow({my_name}) vs blue({opp_name})  |  {initial_cmd} start, {duration_seconds:.0f}s sim "
        f"({wall:.1f}s wall)"
    )
    print(f"Score:            {score['yellow']} - {score['blue']}")
    events = stats.get("rule_event_counts", {})
    if events:
        print(f"Rule events:      {events}")
    print(
        f"Shots:            friendly {stats.get('shots', {}).get('friendly', 0)} | "
        f"enemy {stats.get('shots', {}).get('enemy', 0)}"
    )
    print(
        f"Possession:       friendly {stats.get('possession_pct', {}).get('friendly', 0):.0%} | "
        f"enemy {stats.get('possession_pct', {}).get('enemy', 0):.0%}"
    )
    print(f"Ball travel:      {stats.get('ball_travel_m', 0):.1f} m")

    motion = stats.get("robot_motion_pct", {})
    my_motion = [f"r{i}:{motion.get(f'friendly_{i}', 0):.0%}" for i in range(1, 6)]
    opp_motion = [f"r{i}:{motion.get(f'enemy_{i}', 0):.0%}" for i in range(1, 6)]
    print("Motion share my:  " + "  ".join(my_motion))
    print("Motion share opp: " + "  ".join(opp_motion))
    for i in range(1, 6):
        if motion.get(f"friendly_{i}", 1.0) < 0.05 and motion.get(f"friendly_{i}") is not None:
            print(f"  !! zombie: my robot {i} moved <5% of ticks")
        if motion.get(f"enemy_{i}", 1.0) < 0.05 and motion.get(f"enemy_{i}") is not None:
            print(f"  !! zombie: opp robot {i} moved <5% of ticks")

    # Slot phase summary per side from the per-tick dump.
    for side_key, side_name in (("my_slots", "my"), ("opp_slots", "opp")):
        phases: dict[str, dict[str, int]] = {}
        for r in rows:
            for slot_name, slot in r[side_key].items():
                bucket = phases.setdefault(slot_name, {})
                phase = slot.get("phase")
                bucket[phase] = bucket.get(phase, 0) + 1
        summary = ", ".join(
            f"{slot}:"
            + ";".join(
                f"{p or 'n/a'}={n / TICKS_PER_SECOND:.1f}s"
                for p, n in sorted(phases[slot].items(), key=lambda kv: str(kv[0]))
            )
            for slot in sorted(phases)
        )
        print(f"{side_name} slots:      {summary}")

    print(f"Probe rows: {OUT_PATH}")


if __name__ == "__main__":
    main()
