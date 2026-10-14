"""Reconstruct a live match state from a replay, for reproducing a stall.

Finding a stall today means running a round-robin, spotting a suspicious
match from `summary.json`/`stuck_detector.py`, then tracing the replay to
find the window (`render_window`/`load_frames_in_range`). Once the window is
known ("ball frozen from t=264s"), reproducing it previously meant replaying
the whole match from t=0 in a fresh headless run just to get back to that
one tick. `scenario_from_replay()` + `apply_scenario()` skip that: read the
single nearest frame from the replay, then teleport a freshly-constructed
`StrategyRunner` (same two strategies, same profile — see
`repro_from_replay.py`) directly into that field state, so the investigator
can start ticking (with `match_log` tracing on) from just before the stall.

Coordinate frame: `GameFrame.friendly_robots`/`enemy_robots`/`ball` store
positions in plain pitch-frame metres — the same frame
`AbstractSimController.teleport_ball`/`teleport_robot` take directly (see
`utama_core/rsoccer_simulator/src/ssl/ssl_gym_base.py`'s `teleport_ball`/
`teleport_robot`, which do the y-flip into rsim's internal frame themselves;
callers, including the teleport accuracy test in
`utama_core/tests/strategy_runner/teleport_position_accuracy_test.py`, pass
plain (x, y) metres and get them back unchanged on the next frame). There is
no additional per-team-color mirroring at the `GameFrame` layer — that only
exists one level up, in kernel-tactic target *storage* (see
the kernel tactics' target storage, which mirrors *tactic slot* targets,
not `GameFrame` positions). So `scenario_from_replay` can read a frame's
`p.x`/`p.y`/`orientation` and hand them to `apply_scenario` unchanged; this
is verified by `apply_scenario` itself reading back the first post-teleport
`GameFrame` and asserting positions match within a few cm (mirroring
`teleport_position_accuracy_test.py`'s `POSITION_TOLERANCE`).

A replay only ever records one side's perspective (`ReplayWriterConfig`/
`ColumnarReplayWriterConfig.is_my_perspective`, and
`tournament.run_match`/`repro_from_replay.py` both always construct the
runner with `my_team_is_yellow=True, my_team_is_right=True` — config_a is
always yellow/friendly, and right in the first half; `scenario_from_replay` turns a
second-half frame so it is right here too) — so `friendly_robots` in the replay is
config_a's robots and `enemy_robots` is config_b's, keyed by `Robot.id`
(the same id `sim_controller.teleport_robot(is_team_yellow, robot_id, ...)`
expects).

Known limitations, stated here once rather than re-derived by every caller:
- **Tactic `mem` is not in the replay.** Every `Tactic`'s internal per-slot
  state (phase, committed targets, timers — see `docs/STRATEGY_DEVELOPMENT.
  md`'s "Writing a Tactic") lives only in the live kernel `Strategy`
  process, never serialized to the replay. A repro therefore always starts
  with **fresh tactic state** — the kernel re-partitions robots from
  scratch on the first tick after the scenario is applied, which may not
  immediately reproduce a bug that depended on a specific `mem`/phase
  history leading into the stall. It's usually still enough to see whether
  the *same field geometry* re-stalls, which is what most stall bugs are
  (a referee-restart deadlock, a geometric dead end) — but a bug that's
  purely about accumulated tactic state (e.g. a phase counter that only
  wedges after N timeouts) will not reproduce from a scenario alone.
- **Robot velocities**: both replay formats (`ReplayWriter`'s per-frame
  pickle and `ColumnarReplayWriter`'s columnar arrays) record `Robot.v`
  alongside position, so velocities *are* available and `Scenario` carries
  them — but `AbstractSimController.teleport_robot` has no velocity
  parameter (only `teleport_ball` does), so `apply_scenario` cannot apply a
  robot's replayed velocity; robots always resume from rest. Ball velocity
  *is* applied (`teleport_ball(x, y, vx, vy)`).
- **Referee command**: most tournament `.pkl` replays *do* carry a live
  `RefereeData` on every frame (`_step_game` attaches whatever the
  `CustomReferee` returned that tick before writing it out) — confirmed by
  inspecting a real tournament replay. But this isn't guaranteed for every
  replay `scenario_from_replay` might be pointed at (a hand-built replay, a
  different writer path, or any future frame that legitimately has no
  referee attached), so when a frame's `referee` is `None`,
  `scenario_from_replay` falls back to the `<name>.intentions.jsonl`
  sidecar next to the replay, which independently logs every referee
  transition as `{"event": "referee", "sim_time": ..., "command": ...}`
  rows (see `custom_referee.py`'s referee-transition logging) — the last
  such row at or before `t_seconds` is taken as the command in effect at
  that instant. Both `Scenario.referee_command` and the "if the fallback
  found nothing" case resolve to `None` without raising, either way.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, GameFrame, Robot
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.replay.replay_player import _load_replay

logger = logging.getLogger(__name__)

_POSITION_TOLERANCE_M = 0.15  # mirrors teleport_position_accuracy_test.py's POSITION_TOLERANCE
_VERIFY_MAX_TICKS = 10  # see apply_scenario's verify=True docstring for why more than 1 tick is needed


@dataclass(frozen=True)
class RobotState:
    id: int
    x: float
    y: float
    orientation: float
    vx: float
    vy: float


@dataclass(frozen=True)
class Scenario:
    """A single field state extracted from a replay, ready to apply to a
    fresh `StrategyRunner`.

    `friendly_robots`/`enemy_robots` follow the replay's own perspective
    (see module docstring): friendly is config_a (yellow/right in every
    `tournament.run_match`/`repro_from_replay.py` construction), enemy is
    config_b.
    """

    sim_time: float
    ball_x: float
    ball_y: float
    ball_vx: float
    ball_vy: float
    friendly_robots: tuple[RobotState, ...]
    enemy_robots: tuple[RobotState, ...]
    referee_command: Optional[RefereeCommand]
    source_replay: Path
    config_a_name: Optional[str] = None  # e.g. "build_high_line_zone_kernel_strategy" (friendly)
    config_b_name: Optional[str] = None  # enemy
    frame_ts: float = field(default=0.0)  # the actual frame's `ts`, may differ slightly from requested sim_time


def _short_name(config_name: str) -> str:
    """Mirror `tournament._short_name` without importing the CLI script as a
    module (it's a root-level script, not a package)."""
    return config_name.removeprefix("build_").removesuffix("_kernel_strategy")


def _robot_states(robots: dict, turn: bool = False) -> tuple[RobotState, ...]:
    """`turn`: rotate the pitch half a turn, as `scenario_from_replay` does for a frame where the
    recorded team defends the left goal."""
    k = -1.0 if turn else 1.0
    return tuple(
        RobotState(
            id=r.id,
            x=k * r.p.x,
            y=k * r.p.y,
            orientation=math.remainder(r.orientation + math.pi, 2 * math.pi) if turn else r.orientation,
            vx=k * r.v.x,
            vy=k * r.v.y,
        )
        for r in robots.values()
    )


def _nearest_frame(replay_path: Path, t_seconds: float) -> GameFrame:
    """Load every frame and pick the one whose `ts` is closest to `t_seconds`.

    Dispatches on extension like `replay_player.load_frames_in_range`: `.npz`
    uses the columnar reader (no full-match object reconstruction), `.pkl`
    scans the pickle stream. Either way this is a one-off lookup for a single
    timestamp, not a bulk sweep, so a linear scan is the smallest correct
    approach — no index/seek machinery to maintain.
    """
    if replay_path.suffix == ".npz":
        from utama_core.replay.columnar_reader import load_columnar_replay

        columnar = load_columnar_replay(replay_path)
        if columnar.n_ticks == 0:
            raise ValueError(f"Replay {replay_path} contains no frames.")
        diffs = abs(columnar.ts - t_seconds)
        tick = int(diffs.argmin())
        return columnar.frame_at(tick)

    best: Optional[GameFrame] = None
    best_diff = float("inf")
    for obj in _load_replay(replay_path):
        if not isinstance(obj, GameFrame):
            continue
        diff = abs(obj.ts - t_seconds)
        if diff < best_diff:
            best_diff = diff
            best = obj
    if best is None:
        raise ValueError(f"Replay {replay_path} contains no frames.")
    return best


def _sidecar_path(replay_path: Path) -> Path:
    """`<name>.pkl`/`<name>.npz` -> `<name>.intentions.jsonl`, same directory."""
    return replay_path.with_name(f"{replay_path.stem}.intentions.jsonl")


def _referee_command_at(sidecar_path: Path, t_seconds: float) -> Optional[RefereeCommand]:
    """Last `event: referee` row at or before `t_seconds`; None if the
    sidecar is missing, empty, or has no referee row at/before that time."""
    if not sidecar_path.exists():
        return None

    best_command: Optional[str] = None
    best_ts = float("-inf")
    with open(sidecar_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") != "referee":
                continue
            ts = row.get("sim_time")
            if ts is None or ts > t_seconds:
                continue
            if ts >= best_ts:
                best_ts = ts
                best_command = row.get("command")

    if best_command is None:
        return None
    try:
        return RefereeCommand[best_command]
    except KeyError:
        logger.warning("Unrecognized referee command %r in %s; ignoring.", best_command, sidecar_path)
        return None


def _config_names_from_summary(replay_path: Path) -> tuple[Optional[str], Optional[str]]:
    """Derive (config_a_name, config_b_name) from the tournament directory's
    `summary.json`, when `replay_path` sits inside one.

    Matches by the same `{short(a)}_vs_{short(b)}` tag `tournament.run_match`
    uses to name replay files, applied to `replay_path.stem` (stripping any
    writer-added `_N` disambiguation suffix is not attempted — a colliding
    tag is rare enough, and ambiguous enough, that reporting "not found" is
    more honest than guessing).
    """
    summary_path = replay_path.parent / "summary.json"
    if not summary_path.exists():
        return None, None
    try:
        summary = json.loads(summary_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None, None

    stem = replay_path.stem
    for result in summary.get("results", []):
        config_a = result.get("config_a")
        config_b = result.get("config_b")
        if not config_a or not config_b:
            continue
        if f"{_short_name(config_a)}_vs_{_short_name(config_b)}" == stem:
            return config_a, config_b
    return None, None


def scenario_from_replay(replay_path, t_seconds: float) -> Scenario:
    """Read the frame nearest `t_seconds` from `replay_path` and build a
    `Scenario` from it, merging in the referee command from the
    `.intentions.jsonl` sidecar (frames themselves carry no referee data —
    see module docstring) and the two strategy config names from the
    tournament directory's `summary.json`, when available.
    """
    replay_path = Path(replay_path)
    frame = _nearest_frame(replay_path, t_seconds)

    referee_command = None
    if frame.referee is not None:
        referee_command = frame.referee.referee_command
    else:
        referee_command = _referee_command_at(_sidecar_path(replay_path), t_seconds)

    config_a_name, config_b_name = _config_names_from_summary(replay_path)

    # A scenario is played with config_a defending the right goal. After the teams change ends
    # at half-time it defends the left one, so the pitch is turned half a turn: the same
    # situation, seen from the other end.
    turn = not frame.my_team_is_right
    k = -1.0 if turn else 1.0
    ball_x = ball_y = ball_vx = ball_vy = 0.0
    if frame.ball is not None:
        ball_x, ball_y = k * frame.ball.p.x, k * frame.ball.p.y
        ball_vx, ball_vy = k * frame.ball.v.x, k * frame.ball.v.y

    return Scenario(
        sim_time=t_seconds,
        ball_x=ball_x,
        ball_y=ball_y,
        ball_vx=ball_vx,
        ball_vy=ball_vy,
        friendly_robots=_robot_states(frame.friendly_robots, turn),
        enemy_robots=_robot_states(frame.enemy_robots, turn),
        referee_command=referee_command,
        source_replay=replay_path,
        config_a_name=config_a_name,
        config_b_name=config_b_name,
        frame_ts=frame.ts,
    )


def _overwrite_current_game_frame(side, scenario: Scenario, *, my_is_friendly: bool) -> None:
    """Rewrite `side.current_game_frame` (a `SideRuntime`'s public,
    mutable field — see `apply_scenario`'s comment above this call) so its
    `friendly_robots`/`enemy_robots`/`ball` already hold `scenario`'s
    positions, in `side`'s own perspective.

    `scenario.friendly_robots`/`enemy_robots` are always in the replay's
    (config_a/config_b) perspective (see module docstring); `side` is
    `runner.my` when `my_is_friendly=True` (config_a's own perspective, no
    swap needed) or `runner.opp` when `my_is_friendly=False` (config_b's
    perspective, where the replay's "enemy" is *this* side's "friendly"
    and vice versa) — the same friendly/enemy swap
    `map_friendly_enemy_to_colors` performs elsewhere for the opponent's
    side of a PVP match.

    Robots are written at rest (`v=a=Vector2D(0, 0)`): `teleport_robot` has
    no velocity parameter (see module docstring's Known limitations), so
    the actual post-teleport rsim state a moment later *is* at-rest — this
    just makes the seed match that reality instead of carrying over a
    stale replayed velocity the teleport never applied. The ball keeps
    `scenario`'s velocity, since `teleport_ball` does apply it.

    Only called from `apply_scenario`, immediately followed by
    `side.position_refiner.reset()` + `start_filtering()` there — this
    function alone does not reset the refiner, so calling it without that
    follow-up would leave stale Kalman filters seeded from the old frame.
    """
    scenario_friendly, scenario_enemy = (
        (scenario.friendly_robots, scenario.enemy_robots)
        if my_is_friendly
        else (scenario.enemy_robots, scenario.friendly_robots)
    )

    def _make_robot(rs: RobotState, is_friendly: bool) -> Robot:
        return Robot(
            id=rs.id,
            is_friendly=is_friendly,
            has_ball=False,
            p=Vector2D(rs.x, rs.y),
            v=Vector2D(0.0, 0.0),
            a=Vector2D(0.0, 0.0),
            orientation=rs.orientation,
        )

    friendly_robots = {rs.id: _make_robot(rs, True) for rs in scenario_friendly}
    enemy_robots = {rs.id: _make_robot(rs, False) for rs in scenario_enemy}
    ball = Ball(
        p=Vector3D(scenario.ball_x, scenario.ball_y, 0.0),
        v=Vector3D(scenario.ball_vx, scenario.ball_vy, 0.0),
        a=Vector3D(0.0, 0.0, 0.0),
    )

    side.current_game_frame = dataclasses.replace(
        side.current_game_frame,
        friendly_robots=friendly_robots,
        enemy_robots=enemy_robots,
        ball=ball,
    )


def apply_scenario(runner, scenario: Scenario, *, verify: bool = True) -> None:
    """Teleport `runner`'s ball and every robot into `scenario`'s field
    state, through `runner.sim_controller` (the same public attribute
    `teleport_position_accuracy_test.py` drives directly).

    Friendly robots are teleported as `runner.my_team_is_yellow`'s color,
    enemy robots as the opposite color — matching how the replay was
    recorded (see module docstring: config_a/friendly is always yellow/right
    in `tournament.run_match`/`repro_from_replay.py`, but `apply_scenario`
    itself just follows whatever `runner.my_team_is_yellow` says, so it also
    works for a runner built with the opposite orientation).

    Robot velocities are not applied — `AbstractSimController.teleport_robot`
    has no velocity parameter; robots resume from rest (see module
    docstring's Known limitations). Ball velocity is applied.

    Sets the referee command via the runner's `CustomReferee.set_command`
    when both `scenario.referee_command` is known and `runner.referee` is a
    `CustomReferee` — except when the command is `NORMAL_START`, which is
    left alone (the referee already reaches/holds `NORMAL_START` on its own
    via `seed_clock`/normal play, and calling `set_command` would reset its
    internal state-machine timers for no behavioural gain).

    When `verify` is True (the default), steps the runner (up to
    `_VERIFY_MAX_TICKS` times, stopping early once every position is
    within `_POSITION_TOLERANCE_M`) and asserts every applied position
    round-trips within that tolerance — raising `AssertionError` if not, so
    a coordinate-frame regression fails loudly instead of silently
    producing a wrong repro. More than one tick is needed in practice, not
    just the "one frame of physics settling"
    `teleport_position_accuracy_test.py`'s `POSITION_TOLERANCE` was sized
    for: a robot/ball teleported far from wherever the runner's initial
    formation placed it (the normal case here — a real match's positions
    are essentially always far from a fresh runner's starting formation,
    unlike that test's small, deliberately-nearby teleports) can take
    several ticks for rsim's own vision/physics pipeline to settle,
    especially near a field boundary (observed: up to ~6-7 ticks for a
    robot teleported near the goal line, confirmed against a real replay,
    2026-09-03).
    """
    from utama_core.custom_referee import CustomReferee

    sim_controller = runner.sim_controller
    if sim_controller is None:
        raise RuntimeError("apply_scenario requires a runner with a sim_controller (rsim/grsim mode).")

    sim_controller.teleport_ball(scenario.ball_x, scenario.ball_y, scenario.ball_vx, scenario.ball_vy)

    for robot in scenario.friendly_robots:
        sim_controller.teleport_robot(runner.my_team_is_yellow, robot.id, robot.x, robot.y, robot.orientation)
    for robot in scenario.enemy_robots:
        sim_controller.teleport_robot(not runner.my_team_is_yellow, robot.id, robot.x, robot.y, robot.orientation)

    # `PositionRefiner` runs a per-robot Kalman filter (see
    # `utama_core/data_processing/refiners/position.py`) that seeds its
    # internal state from `last_robot.p` — i.e. `SideRuntime.
    # current_game_frame`, still holding the runner's *initial formation*
    # positions from construction (`StrategyRunner._load_game()`) — the
    # first time it sees a given robot id. Teleporting via `sim_controller`
    # changes rsim's raw frame but neither that seed nor the filter's
    # velocity estimate, so the very next predict/update cycle blends a
    # multi-metre "measurement jump" (old formation -> scenario position)
    # with Kalman gain ~0.75, landing well past the teleport target in the
    # direction of travel (observed: robot teleported to x=0.68 landed at
    # x=1.35 one tick later — reproduces exactly from the filter's own
    # closed-form update, not noise or a coordinate-frame bug).
    # `teleport_position_accuracy_test.py` never hits this because it
    # teleports inside `reset_field()`, which `StrategyRunner._reset_game()`
    # always follows with `position_refiner.reset()` *and* a full
    # `GameGater`-driven `_load_game()` reload — which is what actually
    # refreshes `current_game_frame` to the post-teleport position before
    # any Kalman filter runs. `apply_scenario` has no equivalent public
    # "reset and reload" hook to call (there is no `StrategyRunner.
    # reset_game()`/`reload_game()` exposed outside the `run_test` harness —
    # see the module docstring's wishlist), so it reproduces just the
    # `current_game_frame`-refresh half of that sequence directly: overwrite
    # each side's `current_game_frame` (a public, mutable `SideRuntime`
    # field, same publicness tier as `runner.sim_controller`) with the
    # scenario's positions via `dataclasses.replace`, THEN reset the
    # refiners — so the filter's first-ever seed for each robot is already
    # the teleport target, and the first blend is target-into-target.
    _overwrite_current_game_frame(runner.my, scenario, my_is_friendly=True)
    runner.my.position_refiner.reset()
    runner.my.position_refiner.start_filtering()
    if runner.opp is not None:
        _overwrite_current_game_frame(runner.opp, scenario, my_is_friendly=False)
        runner.opp.position_refiner.reset()
        runner.opp.position_refiner.start_filtering()

    referee = getattr(runner, "referee", None)
    if (
        scenario.referee_command is not None
        and isinstance(referee, CustomReferee)
        and scenario.referee_command != RefereeCommand.NORMAL_START
    ):
        referee.set_command(scenario.referee_command, runner.my.current_game_frame.ts)

    if not verify:
        return

    def _max_deviation(frame: GameFrame) -> tuple[float, str]:
        """Largest (deviation, label) over the ball and every scenario
        robot present in `frame` — used both to decide whether the
        polling loop below can stop early and to build the final
        assertion message."""
        worst = 0.0
        worst_label = ""

        def _consider(label: str, actual_x: float, actual_y: float, expected_x: float, expected_y: float) -> None:
            nonlocal worst, worst_label
            d = max(abs(actual_x - expected_x), abs(actual_y - expected_y))
            if d > worst:
                worst = d
                worst_label = (
                    f"{label} landed at ({actual_x:.3f}, {actual_y:.3f}), expected ({expected_x:.3f}, {expected_y:.3f})"
                )

        if frame.ball is not None:
            _consider("ball", frame.ball.p.x, frame.ball.p.y, scenario.ball_x, scenario.ball_y)
        for robot in scenario.friendly_robots:
            actual = frame.friendly_robots.get(robot.id)
            if actual is not None:
                _consider(f"friendly robot {robot.id}", actual.p.x, actual.p.y, robot.x, robot.y)
        for robot in scenario.enemy_robots:
            actual = frame.enemy_robots.get(robot.id)
            if actual is not None:
                _consider(f"enemy robot {robot.id}", actual.p.x, actual.p.y, robot.x, robot.y)
        return worst, worst_label

    deviation, label = float("inf"), ""
    for _ in range(_VERIFY_MAX_TICKS):
        runner.step_once()
        deviation, label = _max_deviation(runner.my.current_game_frame)
        if deviation <= _POSITION_TOLERANCE_M:
            break

    if deviation > _POSITION_TOLERANCE_M:
        raise AssertionError(
            f"apply_scenario: {label} (tolerance={_POSITION_TOLERANCE_M}m) after "
            f"{_VERIFY_MAX_TICKS} ticks. Coordinate-frame mismatch?"
        )
