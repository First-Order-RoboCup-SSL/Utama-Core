"""arena_tournament.py — headless kernel-strategy tournament / matchup runner.

Born as `probe_default_vs_lowblock.py`, the instrumented reproduction of the
tournament's default_vs_lowblock 0-0 passer-tangle investigation; generalized
into a runner for *any* two `build_*_kernel_strategy` factories, with a
round-robin mode for the arena strategies. Same match configuration as
`tournament.run_match` (6v6, headless rsim), but every tick dumps a JSON
row with:

  - sim time, referee command
  - ball position/velocity
  - all 12 robots' positions + orientations (pitch frame / "my" perspective)
  - per-side kernel slot state: tactic id, assigned robots, `is_committed()`,
    and each tactic's `mem` fields (phase, assigned pair, setup targets,
    phase ticks, setup_ticks_without_ball, goal_scored)
  - each robot's commanded motion target this tick (via a recorder wrapped
    around both kernels' motion controllers — what tactics wanted, as opposed
    to where physics put them; keys `y<id>` / `b<id>` because both teams'
    robot ids collide)
  - the arena pickers' possession-edge verdict + raw proximity distances

Yellow's own-frame targets are mirrored into the pitch frame (yellow is the
right team), blue's are not (blue is the left team), so all coordinates in
the dump are comparable in the same frame.

Usage:
  pixi run python arena_tournament.py [my_strategy] [opp_strategy] [duration_seconds] [initial_command]
  pixi run python arena_tournament.py round_robin [duration_seconds] [initial_command]

`my_strategy` / `opp_strategy` are any `build_*_kernel_strategy` factory names
(short form, e.g. `tiki_taka`, `low_block`, `default`); defaults: `default`
vs `low_block` — the original investigation matchup. Both sides are always
constructed from kernel factories, so any two configs can be compared in the
same fixture.

`round_robin` plays every pairing of the three arena strategies (tiki_taka,
counter_press, zone_fluid) in both orientations — six matches — and prints a
results table. A stale boxscore is noted for matches the sim wedged (kickoff
passer tangle) or the referee froze.

`initial_command` is any RefereeCommand name (e.g. PREPARE_KICKOFF_YELLOW to
start with a proper kickoff ceremony, which breaks the center-circle passer
tangle for matchups that otherwise wedge); defaults to FORCE_START (the
StrategyRunner sim default that the tournament uses).

Output: /tmp/opencode/arena_tournament.jsonl (one row per tick); the last
match's rows also land in /tmp/opencode/probe_default_vs_lowblock.jsonl when
the first token is a back-compat positional form. Replays (my + opp
perspective) and match log / stats are also written via the standard
StrategyRunner machinery; a compact boxscore is printed at the end.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from typing import Optional

from utama_core.custom_referee import CustomReferee
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.kernel.abstract_strategy import AbstractStrategy
from utama_core.replay.replay_writer import ReplayWriterConfig
from utama_core.run import StrategyRunner
from utama_core.strategy import kernel_strategy

N_OUTFIELD = 5
OUTFIELD_ROBOT_IDS = tuple(range(1, N_OUTFIELD + 1))
TICKS_PER_SECOND = 60

# Arena strategies for the round-robin tournament.
_ARENA_STRATEGIES = ("tiki_taka", "counter_press", "zone_fluid")

OUT_PATH = "/tmp/opencode/arena_tournament.jsonl"
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
    from utama_core.strategy.kernel_strategy import _friendly_closer_to_ball

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
_TARGETS: dict[str, tuple[float, float]] = {}


class _TargetRecorder:
    """Delegating MotionController wrapper that records calculate() targets.

    Both teams' robot ids collide (both are 0..5), so keys are
    team-qualified: ``y<id>`` for the my-side kernel, ``b<id>`` for opp.
    """

    def __init__(self, inner, side: str):
        self._inner = inner
        self._prefix = side

    def calculate(self, game, robot_id, target_pos, target_oren, **kwargs):
        # go_to_point accepts Vector2D or plain (x, y) tuples — record either.
        if hasattr(target_pos, "x"):
            x, y = target_pos.x, target_pos.y
        else:
            x, y = target_pos[0], target_pos[1]
        _TARGETS[f"{self._prefix}{robot_id}"] = (round(float(x), 3), round(float(y), 3))
        return self._inner.calculate(
            game=game, robot_id=robot_id, target_pos=target_pos, target_oren=target_oren, **kwargs
        )

    def reset(self, robot_id: int):
        return self._inner.reset(robot_id)


def _install_target_recorders(runner) -> None:
    """Wrap both kernels' motion controllers so every tick's targets land in `_TARGETS`."""
    for side, prefix in ((runner.my, "y"), (runner.opp, "b")):
        kernel = getattr(side.strategy, "_kernel_strategy", None)
        if kernel is not None:
            kernel._ctx.motion_controller = _TargetRecorder(kernel._ctx.motion_controller, prefix)


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "round_robin":
        _run_round_robin(argv[1:])
        return
    my_name, opp_name, duration_seconds, initial_command = _parse_args(argv)
    _run_match(my_name, opp_name, duration_seconds, initial_command)


def _run_round_robin(rest: list[str]) -> None:
    """Play every arena pairing in both orientations and print a results table."""
    duration = 60.0
    command_name = None
    for token in rest:
        try:
            duration = float(token)
            continue
        except ValueError:
            pass
        if token in RefereeCommand.__members__:
            command_name = token
    initial_command = RefereeCommand[command_name] if command_name else RefereeCommand.FORCE_START

    results: list[dict] = []
    done: set[tuple[str, str]] = set()
    for a in _ARENA_STRATEGIES:
        for b in _ARENA_STRATEGIES:
            if a == b or (a, b) in done:
                continue
            done.add((b, a))
            print(f"\n==================== {a} (yellow) vs {b} (blue) ====================")
            result = _run_match(a, b, duration, initial_command)
            if result is not None:
                results.append(result)

    print("\n==================== TOURNAMENT TABLE ====================")
    print(f"format: {initial_command.name}, {duration:.0f}s sim per match")
    print(f"{'yellow':<14}{'blue':<14}{'score':<9}{'shots':<10}{'poss%':<12}{'travel':<9}{'events':<24}note")
    for r in results:
        score = f"{r['score_y']}-{r['score_b']}"
        shots = f"{r['shots_y']}-{r['shots_b']}"
        poss = f"{r['pos_y'] * 100:.0f}-{r['pos_b'] * 100:.0f}"
        print(
            f"{r['my']:<14}{r['opp']:<14}{score:<9}{shots:<10}{poss:<12}"
            f"{r['travel']:<9.1f}{str(r['events']):<24}{r['note']}"
        )


def _run_match(my_name: str, opp_name: str, duration_seconds: float, initial_command: RefereeCommand) -> Optional[dict]:
    """Run one matchup under the same deterministic fixture; return its boxscore fields."""
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
        replay_writer_config=ReplayWriterConfig(replay_name="arena_tournament", overwrite_existing=True),
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
    return _print_boxscore(my_name, opp_name, duration_seconds, wall, initial_command.name)


def _current_command(runner) -> Optional[RefereeCommand]:
    referee = getattr(runner.my.game, "referee", None)
    command = getattr(referee, "referee_command", None)
    if isinstance(command, RefereeCommand):
        return command
    return None


def _print_boxscore(my_name: str, opp_name: str, duration_seconds: float, wall: float, initial_cmd: str) -> dict:
    """Compact post-match summary from the stats JSON + the per-tick dump.

    Returns the boxscore fields so the round-robin table can reuse them.
    """
    rows = [json.loads(line) for line in open(OUT_PATH)]
    score = rows[-1]["score"]
    stats = {}
    try:
        stats = json.load(open(STATS_PATH))
    except (OSError, ValueError):
        pass

    events = stats.get("rule_event_counts", {})
    travel = stats.get("ball_travel_m", 0)
    # Flag matches the deterministic sim wedged (kickoff passer tangle) or
    # froze into referee foul-replay loops — their scorelines are physics
    # artifacts, not strategy outcomes.
    note = ""
    if travel < 8.0:
        note = "sim wedge (low ball travel)"
    elif any(n >= 10 for n in events.values()):
        note = "foul-frozen replay"

    print()
    print(
        f"Match: yellow({my_name}) vs blue({opp_name})  |  {initial_cmd} start, {duration_seconds:.0f}s sim "
        f"({wall:.1f}s wall)"
    )
    print(f"Score:            {score['yellow']} - {score['blue']}")
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
    print(f"Ball travel:      {travel:.1f} m")

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

    return {
        "my": my_name,
        "opp": opp_name,
        "score_y": score["yellow"],
        "score_b": score["blue"],
        "shots_y": stats.get("shots", {}).get("friendly", 0),
        "shots_b": stats.get("shots", {}).get("enemy", 0),
        "pos_y": stats.get("possession_pct", {}).get("friendly", 0),
        "pos_b": stats.get("possession_pct", {}).get("enemy", 0),
        "travel": travel,
        "events": events,
        "note": note,
    }


if __name__ == "__main__":
    main()
