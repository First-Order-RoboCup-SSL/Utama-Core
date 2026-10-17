"""Picture of where the referee override sends every friendly robot for one restart (`tools/restart_picture.py`).

For a replay state whose referee command is a restart (STOP, timeouts, kick-offs, penalties, free kicks,
ball placement), this runs `RefereeOverride.tick` exactly as the strategy does and records the target
each robot is commanded to. The `*Step` classes only produce velocity commands through `move` and
`turn_on_spot`, so `_recording_steps` swaps those two names in `custom_referee.actions` for recorders for
the duration of the call (and restores them): the commanded target is what is captured, and no engine
or planner code changes. A detour waypoint is captured as the target, as commanded.

The picture: the full pitch, the ball with its keep-out circle, the placement or free-kick spot, each of
our robots at its real radius with its id, an arrow to its commanded target and the role the step gives
it (kicker, placer, keeper), and the other team faded. A robot whose target is inside the keep-out circle,
inside the opponent defence area (with its keep distance), or inside our own defence area beyond the one
allowed defender is flagged in red. Read-only: nothing is sent to a robot or a referee.

Coordinates are the replay's pitch frame (metres). `side="enemy"` gives the other team's override: its
robots are "ours", and the team flags are swapped, so the pitch is not mirrored.
"""

from __future__ import annotations

import dataclasses
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Optional

import numpy as np  # type: ignore
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from utama_core.analysis.plan_snapshot import (
    AGAINST,
    DPI,
    FOR,
    GRID,
    INK,
    INK_2,
    INK_3,
    SURFACE,
    draw_pitch,
    draw_robot,
    game_from_scenario,
    style_axes,
)
from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.config.referee_constants import (
    BALL_KEEP_OUT_DISTANCE,
    OPPONENT_DEFENSE_AREA_KEEP_DISTANCE,
)
from utama_core.custom_referee import actions as _actions
from utama_core.engine.referee_override import RefereeOverride, is_override_command
from utama_core.entities.data.referee import RefereeData
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.entities.game.team_info import TeamInfo
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage
from utama_core.replay.columnar_reader import load_columnar_replay
from utama_core.replay.scenario import Scenario

# A red that is not in the categorical palette, reserved for the defects this picture exists to show.
ALERT = "#b3261e"
_KICKOFF = ("PREPARE_KICKOFF_YELLOW", "PREPARE_KICKOFF_BLUE")
_CLEARS_OWN_AREA = ("STOP", "TIMEOUT_YELLOW", "TIMEOUT_BLUE") + _KICKOFF
_FREE_KICKS = ("DIRECT_FREE_YELLOW", "DIRECT_FREE_BLUE", "INDIRECT_FREE_YELLOW", "INDIRECT_FREE_BLUE")
_PENALTIES = ("PREPARE_PENALTY_YELLOW", "PREPARE_PENALTY_BLUE")


@dataclass(frozen=True)
class CommandedTarget:
    kind: str  # "move" (a `move` call) or "turn" (a `turn_on_spot` call; the target is the robot's own spot)
    x: float
    y: float


@dataclass(frozen=True)
class RobotPlan:
    robot_id: int
    pos: tuple[float, float]
    kind: str  # "move" | "turn" | "hold" (an empty command: target = own spot) | "none" (no override command)
    target: Optional[tuple[float, float]]
    role: Optional[str]  # "kicker", "placer", "keeper" or None
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class RestartPlan:
    command: RefereeCommand
    side: str  # "ours", "theirs" or "both" (STOP)
    sim_time: Optional[float]
    ball: Optional[tuple[float, float]]
    designated: Optional[tuple[float, float]]
    robots: tuple[RobotPlan, ...]  # the planned side, by id
    enemy: tuple[tuple[int, float, float], ...]
    notes: tuple[str, ...] = ()

    @property
    def flagged(self) -> tuple[RobotPlan, ...]:
        return tuple(r for r in self.robots if r.flags)


@contextmanager
def _recording_steps() -> Iterator[dict[int, CommandedTarget]]:
    """Swap `move` and `turn_on_spot` in `custom_referee.actions` for recorders, then restore them."""
    recorded: dict[int, CommandedTarget] = {}
    real_move, real_turn = _actions.move, _actions.turn_on_spot

    def move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False):
        recorded[robot_id] = CommandedTarget("move", float(target_coords.x), float(target_coords.y))
        return _actions.empty_command(dribbling)

    def turn_on_spot(game, motion_controller, robot_id, target_oren, dribbling=False):
        p = game.friendly_robots[robot_id].p
        recorded[robot_id] = CommandedTarget("turn", float(p.x), float(p.y))
        return _actions.empty_command(dribbling)

    _actions.move, _actions.turn_on_spot = move, turn_on_spot
    try:
        yield recorded
    finally:
        _actions.move, _actions.turn_on_spot = real_move, real_turn


def _team_is_ours(command: RefereeCommand, my_team_is_yellow: bool) -> Optional[bool]:
    """True when the command is aimed at our team, False when at theirs, None when at both (STOP)."""
    name = command.name
    if name.endswith("_YELLOW"):
        return my_team_is_yellow
    if name.endswith("_BLUE"):
        return not my_team_is_yellow
    return None


def _keeper_id(game: Game) -> Optional[int]:
    ref = game.referee
    if ref is None:
        return None
    team = ref.yellow_team if game.my_team_is_yellow else ref.blue_team
    return None if team is None else team.goalkeeper


def _roles(game: Game, command: RefereeCommand, ours: Optional[bool]) -> dict[int, str]:
    """The role each step gives a robot: the lowest-id non-keeper, or the robot nearest the ball."""
    roles: dict[int, str] = {}
    keeper = _keeper_id(game)
    if keeper in game.friendly_robots:
        roles[keeper] = "keeper"
    if not ours or not game.friendly_robots:
        return roles
    name = command.name
    others = sorted(rid for rid in game.friendly_robots if rid != keeper)
    if name in _KICKOFF or name in _PENALTIES:
        if others:
            roles[others[0]] = "kicker"
    elif name in _FREE_KICKS or name.startswith("BALL_PLACEMENT"):
        if game.ball is not None:
            nearest = min(game.friendly_robots, key=lambda rid: game.friendly_robots[rid].p.distance_to(game.ball.p))
            roles[nearest] = "placer" if name.startswith("BALL_PLACEMENT") else "kicker"
    return roles


def _flags(game: Game, rp: RobotPlan, ball: Optional[Vector2D], own_area_rank: Optional[int]) -> tuple[str, ...]:
    if rp.target is None or rp.kind == "none":
        return ()
    flags: list[str] = []
    target = Vector2D(rp.target[0], rp.target[1])
    if rp.role in ("kicker", "placer"):
        return ()  # the robot that must reach the ball is allowed inside the keep-out
    if ball is not None:
        d = (target - ball).mag()
        if d < BALL_KEEP_OUT_DISTANCE - 1e-9:
            # The clearing step holds a robot already within 1 cm of its edge (dx^2 + dy^2 < 1e-4), so a
            # target 1 cm inside is that deadband, not a robot sent into the circle.
            deadband = " (1 cm stop deadband)" if d > BALL_KEEP_OUT_DISTANCE - 0.02 else ""
            flags.append(f"target inside keep-out ({d:.2f} m < {BALL_KEEP_OUT_DISTANCE:.2f} m){deadband}")
    projected = _actions._project_outside_opp_defense_area(game, target, OPPONENT_DEFENSE_AREA_KEEP_DISTANCE)
    if projected != target:
        flags.append(f"target inside opponent defence area (+{OPPONENT_DEFENSE_AREA_KEEP_DISTANCE:.2f} m)")
    if own_area_rank is not None and own_area_rank > 0:
        flags.append("target inside own defence area beyond the one allowed defender")
    return tuple(flags)


def plan_restart(game: Game, command: RefereeCommand, *, sim_time: Optional[float] = None) -> RestartPlan:
    """What the referee override commands for every friendly robot, for this restart, on this `game`."""
    if not is_override_command(command):
        raise ValueError(f"{command.name} is not a restart the referee override handles")
    with _recording_steps() as recorded:
        cmd_map = RefereeOverride().tick(game, None, command)

    ours = _team_is_ours(command, game.my_team_is_yellow)
    roles = _roles(game, command, ours)
    ball = None if game.ball is None else Vector2D(game.ball.p.x, game.ball.p.y)
    notes: list[str] = []
    team = None
    if game.referee is not None:
        team = game.referee.yellow_team if game.my_team_is_yellow else game.referee.blue_team
    if team is None:
        notes.append("no team data in the replay at this tick: goalkeeper taken as 0")
    # The referee keeps its last designated position after a placement or free kick ends; it is only
    # the spot the restart means when the command is one of those.
    spot_command = command.name.startswith("BALL_PLACEMENT") or command.name in _FREE_KICKS
    designated = game.referee.designated_position if game.referee is not None and spot_command else None

    robots: list[RobotPlan] = []
    for rid, robot in sorted(game.friendly_robots.items()):
        pos = (robot.p.x, robot.p.y)
        if rid in recorded:
            c = recorded[rid]
            kind, target = c.kind, (c.x, c.y)
        elif rid in cmd_map:
            kind, target = "hold", pos
        else:
            kind, target = "none", None
        robots.append(RobotPlan(rid, pos, kind, target, roles.get(rid)))

    # Our robots whose target is inside our own defence area, beyond the one allowed defender (ranked by
    # id; STOP and kick-off clear the area, a kick-off leaves its kicker and keeper out of the count).
    own_rank: dict[int, int] = {}
    if command.name in _CLEARS_OWN_AREA:
        keeper = _keeper_id(game)
        in_own = sorted(
            rp.robot_id
            for rp in robots
            if rp.target is not None
            and rp.role not in ("kicker", "placer")
            and not (command.name in _KICKOFF and rp.robot_id == keeper)
            and _actions._is_in_own_defense_area(game, rp.target[0], rp.target[1])
        )
        own_rank = {rid: rank for rank, rid in enumerate(in_own)}

    flagged = tuple(dataclasses.replace(rp, flags=_flags(game, rp, ball, own_rank.get(rp.robot_id))) for rp in robots)
    enemy = tuple((r.id, float(r.p.x), float(r.p.y)) for r in sorted(game.enemy_robots.values(), key=lambda r: r.id))
    return RestartPlan(
        command=command,
        side="both" if ours is None else ("ours" if ours else "theirs"),
        sim_time=sim_time,
        ball=None if ball is None else (ball.x, ball.y),
        designated=None if designated is None else (float(designated[0]), float(designated[1])),
        robots=flagged,
        enemy=enemy,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Replay state for a restart
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _columnar(replay: str):
    return load_columnar_replay(Path(replay))


def restart_referee(replay: Path, scenario: Scenario) -> RefereeData:
    """The replay's own team and designated-position data at the scenario's frame, as the override reads it.

    `Scenario` carries only the command, so this reads the columnar replay at the same frame. The scenario
    turns the replay's coordinates half a turn when its team defends the left goal; the designated position
    is turned the same way here."""
    rep = _columnar(str(Path(replay)))
    tick = int(np.argmin(np.abs(rep.ts - scenario.frame_ts)))
    ref = rep.frame_at(tick).referee
    designated = None if ref is None else ref.designated_position
    if designated is not None and not rep.is_right_at(tick):
        designated = (-float(designated[0]), -float(designated[1]))
    empty = TeamInfo(name="", score=0, goalkeeper=0)
    blue = ref.blue_team if ref is not None and ref.blue_team is not None else empty
    yellow = ref.yellow_team if ref is not None and ref.yellow_team is not None else empty
    return RefereeData(
        source_identifier=None,
        time_sent=0.0,
        time_received=0.0,
        referee_command=scenario.referee_command,
        referee_command_timestamp=0.0,
        stage=ref.stage if ref is not None else Stage.NORMAL_FIRST_HALF,
        stage_time_left=0.0,
        blue_team=blue,
        yellow_team=yellow,
        designated_position=designated,
    )


def restart_game(replay: Path, scenario: Scenario, side: str = "friendly") -> Game:
    """The `Game` one side's override sees for the scenario's restart."""
    return game_from_scenario(scenario, side, referee=restart_referee(replay, scenario))


def restart_commands(replay: Path) -> list[tuple[float, RefereeCommand]]:
    """The first occurrence of each distinct restart command in the replay's referee sidecar
    (`<name>.intentions.jsonl`), in time order."""
    sidecar = Path(replay).with_name(f"{Path(replay).stem}.intentions.jsonl")
    rows: list[tuple[float, RefereeCommand]] = []
    if sidecar.exists():
        for line in sidecar.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") != "referee" or row.get("sim_time") is None:
                continue
            try:
                command = RefereeCommand[row["command"]]
            except KeyError:
                continue
            rows.append((float(row["sim_time"]), command))
    seen: set[RefereeCommand] = set()
    first: list[tuple[float, RefereeCommand]] = []
    for t, command in sorted(rows, key=lambda r: r[0]):
        if command in seen or not is_override_command(command):
            continue
        seen.add(command)
        first.append((t, command))
    return first


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _draw_plan(ax, plan: RestartPlan, *, compact: bool) -> None:
    field_dims = STANDARD_FIELD_DIMS
    draw_pitch(ax, field_dims)
    half_l, half_w = field_dims.full_field_half_length, field_dims.full_field_half_width
    ax.set_xlim(-half_l - 0.2, half_l + 0.2)
    ax.set_ylim(-half_w - 0.2, half_w + 0.2)

    if plan.ball is not None:
        ax.add_patch(
            Circle(
                plan.ball,
                BALL_KEEP_OUT_DISTANCE,
                fill=False,
                edgecolor=INK_2,
                linewidth=0.8,
                linestyle="--",
                zorder=2,
            )
        )
        if plan.designated is not None:
            ax.plot(
                [plan.ball[0], plan.designated[0]],
                [plan.ball[1], plan.designated[1]],
                color=INK_3,
                linewidth=0.7,
                linestyle=":",
                zorder=3,
            )
            ax.plot(plan.designated[0], plan.designated[1], "x", color=INK, markersize=6, markeredgewidth=1.3, zorder=8)
        ax.plot(plan.ball[0], plan.ball[1], "o", color=INK, markersize=3.5 if compact else 4.5, zorder=6)

    for rid, x, y in plan.enemy:
        draw_robot(ax, x, y, rid, AGAINST, alpha=0.3, zorder=4)

    for rp in plan.robots:
        colour = ALERT if rp.flags else INK_2
        if rp.target is not None and rp.kind in ("move", "turn") and _dist(rp.pos, rp.target) > 0.02:
            _arrow(ax, rp.pos, rp.target, colour)
            ax.plot(rp.target[0], rp.target[1], "x", color=colour, markersize=5, markeredgewidth=1.2, zorder=7)
        draw_robot(
            ax,
            rp.pos[0],
            rp.pos[1],
            rp.robot_id,
            FOR,
            edge=ALERT if rp.flags else None,
        )
        if rp.role:
            ax.text(
                rp.pos[0],
                rp.pos[1] - ROBOT_RADIUS - 0.05,
                rp.role,
                ha="center",
                va="top",
                fontsize=5.5 if compact else 6.5,
                color=INK_2,
                fontweight="bold",
                zorder=12,
            )


def _arrow(ax, start, end, colour) -> None:
    """An arrow from the edge of the robot at `start` to `end` (a target)."""
    d = np.array(end, dtype=float) - np.array(start, dtype=float)
    u = d / float(np.hypot(*d))
    begin = np.array(start, dtype=float) + u * ROBOT_RADIUS
    ax.annotate(
        "",
        xy=(end[0], end[1]),
        xytext=(begin[0], begin[1]),
        arrowprops=dict(arrowstyle="-|>", color=colour, lw=0.9, shrinkA=0, shrinkB=0, mutation_scale=8),
        zorder=4,
    )


def _text_lines(plan: RestartPlan) -> list[str]:
    side = {
        "ours": "ours (a restart for the planned side's team)",
        "theirs": "theirs (a restart for the other team, against the planned side)",
    }.get(plan.side, "both teams")
    when = f", t = {plan.sim_time:g} s" if plan.sim_time is not None else ""
    lines = [f"command: {plan.command.name}{when}", f"for: {side}"]
    if plan.designated is not None:
        lines.append(f"designated spot: ({plan.designated[0]:.2f}, {plan.designated[1]:.2f})")
    for rp in plan.robots:
        role = rp.role or ""
        if rp.target is None:
            lines.append(f"  {rp.robot_id:>2} {role:<8} no override command")
            continue
        d = "" if plan.ball is None else f"  {_dist(rp.target, plan.ball):.2f} m from ball"
        lines.append(f"  {rp.robot_id:>2} {role:<8} {rp.kind:<4} -> ({rp.target[0]:+.2f}, {rp.target[1]:+.2f}){d}")
        for flag in rp.flags:
            lines.append(f"       FLAG: {flag}")
    if not plan.flagged:
        lines.append("no robot target flagged")
    lines.extend(plan.notes)
    return lines


def _legend_handles() -> list[Line2D]:
    return [
        Line2D([], [], marker="o", linestyle="", markerfacecolor=FOR, markeredgecolor="none", label="our robot"),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor=AGAINST,
            alpha=0.4,
            markeredgecolor="none",
            label="other team",
        ),
        Line2D([], [], marker="x", linestyle="", color=INK_2, label="commanded target"),
        Line2D([], [], marker="x", linestyle="", color=ALERT, label="flagged target"),
        Line2D([], [], color=INK_2, linestyle="--", linewidth=0.8, label="ball keep-out 0.8 m"),
        Line2D([], [], marker="x", linestyle="", color=INK, label="designated spot"),
    ]


def render_restart(plan: RestartPlan, out_path: Path, *, title: Optional[str] = None) -> Path:
    """One restart as a single picture: the full pitch, the text box beneath it and a legend."""
    fig = Figure(figsize=(6.6, 7.4), dpi=DPI, facecolor=SURFACE)
    ax = style_axes(fig.add_axes([0.1, 0.42, 0.86, 0.52]))
    _draw_plan(ax, plan, compact=False)
    ax.set_xlabel("x (m)", fontsize=8, color=INK_2)
    ax.set_ylabel("y (m)", fontsize=8, color=INK_2)
    if title:
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    fig.text(
        0.1,
        0.37,
        "\n".join(_text_lines(plan)),
        ha="left",
        va="top",
        fontsize=6.8,
        color=INK,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=SURFACE, edgecolor=GRID, linewidth=0.6),
    )
    fig.legend(
        handles=_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        fontsize=6.5,
        frameon=False,
        labelcolor=INK_2,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, facecolor=SURFACE)
    return out_path


def render_restart_grid(
    plans: list[RestartPlan], out_path: Path, *, ncols: int = 4, title: Optional[str] = None
) -> Path:
    """Small multiples: one compact pitch per restart, titled with its command, side, time and flag count."""
    ncols = max(1, min(ncols, len(plans)))
    nrows = math.ceil(len(plans) / ncols)
    fig = Figure(figsize=(4.3 * ncols, 3.4 * nrows + 0.5), dpi=DPI, facecolor=SURFACE)
    for i, plan in enumerate(plans):
        ax = style_axes(fig.add_subplot(nrows, ncols, i + 1))
        _draw_plan(ax, plan, compact=True)
        ax.tick_params(labelsize=5)
        when = f"  t={plan.sim_time:.1f}s" if plan.sim_time is not None else ""
        n = len(plan.flagged)
        ax.set_title(
            f"{plan.command.name}\n{plan.side}{when}  flags: {n}",
            fontsize=6.5,
            color=ALERT if n else INK,
            loc="left",
        )
    if title:
        fig.suptitle(title, fontsize=9, color=INK, x=0.02, ha="left")
    fig.subplots_adjust(left=0.03, right=0.98, top=0.86 if title else 0.9, bottom=0.03, wspace=0.12, hspace=0.45)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, facecolor=SURFACE)
    return out_path
