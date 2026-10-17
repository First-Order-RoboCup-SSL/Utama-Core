"""A picture of what `FastPathPlanner` decided for one robot on one tick (`tools/plan_snapshot.py`).

Builds the `Game` the planner saw (from a replay frame via `replay.scenario`, or from a hand-made
state via `build_game`), runs the planner's own `_path_to`, and draws the pitch around the robot:
every robot at its real radius with its id, the ball and the ball keep-out circle when the planner
keeps out of it, each robot's clearance ring as `_get_obstacles` computes it, the target, and the
carrot `_path_to` returned. Read-only: the planner is not changed; obstacles and clearance come from
its own `_get_obstacles`, called after `_path_to`.

Coordinates are the replay's pitch frame (metres). `side="enemy"` plans the other strategy's robot
by swapping the two robot sets and turning the team flags to match, so the pitch is not mirrored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np  # type: ignore
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon

from utama_core.config.field_params import STANDARD_FIELD_DIMS, FieldDimensions
from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.config.referee_constants import BALL_KEEP_OUT_DISTANCE
from utama_core.entities.data.referee import RefereeData
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, Game, GameFrame, GameHistory, Robot
from utama_core.entities.game.field import Field
from utama_core.entities.game.team_info import TeamInfo
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage
from utama_core.motion_planning.src.fastpathplanning.planner import (
    FastPathPlanner,
    _keeps_off_ball,
)
from utama_core.replay.scenario import Scenario, scenario_from_replay

# Same ink and palette as tools/signal_report.py (FOR = friendly, AGAINST = enemy).
INK, INK_2, INK_3, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e4e3df"
FOR, AGAINST = "#2a78d6", "#eb6834"
CARROT = "#1baf7a"  # categorical slot 3: the planner's output, distinct from the team colours
DPI = 130


@dataclass(frozen=True)
class PlanSnapshot:
    carrot: np.ndarray  # what `_path_to` returned
    inside_keep_out: bool  # the robot is strictly inside the ball keep-out circle
    keeps_off_ball: bool  # the planner keeps out of the ball's circle on this tick
    robot_to_carrot: float  # metres
    robot_to_ball: Optional[float]  # metres, None without a ball
    path: Path


def _referee(command: Optional[RefereeCommand]) -> Optional[RefereeData]:
    if command is None:
        return None
    empty = TeamInfo(name="", score=0, goalkeeper=0)
    return RefereeData(
        source_identifier=None,
        time_sent=0.0,
        time_received=0.0,
        referee_command=command,
        referee_command_timestamp=0.0,
        stage=Stage.NORMAL_FIRST_HALF,
        stage_time_left=0.0,
        blue_team=empty,
        yellow_team=empty,
    )


def build_game(
    friendly: dict,
    enemy: dict,
    ball: Optional[Ball],
    *,
    my_team_is_right: bool = True,
    my_team_is_yellow: bool = True,
    referee_command: Optional[RefereeCommand] = None,
    field_dims: FieldDimensions = STANDARD_FIELD_DIMS,
    ts: float = 0.0,
) -> Game:
    """A real `Game` (not a mock) from robot dicts keyed by id, as the planner reads it."""
    frame = GameFrame(
        ts=ts,
        my_team_is_yellow=my_team_is_yellow,
        my_team_is_right=my_team_is_right,
        friendly_robots=dict(friendly),
        enemy_robots=dict(enemy),
        ball=ball,
        referee=_referee(referee_command),
    )
    field = Field(my_team_is_right=my_team_is_right, field_dims=field_dims, field_bounds=field_dims.full_field_bounds)
    return Game(GameHistory(10), frame, field=field)


def _robot(state, is_friendly: bool) -> Robot:
    return Robot(
        id=state.id,
        is_friendly=is_friendly,
        has_ball=False,
        p=Vector2D(state.x, state.y),
        v=Vector2D(state.vx, state.vy),
        a=Vector2D(0.0, 0.0),
        orientation=state.orientation,
    )


def game_from_scenario(scenario: Scenario, side: str = "friendly") -> Game:
    """The `Game` one side's planner sees in a replay scenario.

    The replay's friendly side is config_a: yellow, defending the right goal. `side="enemy"` makes
    config_b the planned side: blue, defending the left goal, same coordinates."""
    if side not in ("friendly", "enemy"):
        raise ValueError(f"side must be 'friendly' or 'enemy', not {side!r}")
    planned_is_replay_friendly = side == "friendly"
    mine, theirs = (
        (scenario.friendly_robots, scenario.enemy_robots)
        if planned_is_replay_friendly
        else (scenario.enemy_robots, scenario.friendly_robots)
    )
    ball = Ball(
        p=Vector3D(scenario.ball_x, scenario.ball_y, 0.0),
        v=Vector3D(scenario.ball_vx, scenario.ball_vy, 0.0),
        a=Vector3D(0.0, 0.0, 0.0),
    )
    return build_game(
        {s.id: _robot(s, True) for s in mine},
        {s.id: _robot(s, False) for s in theirs},
        ball,
        my_team_is_right=planned_is_replay_friendly,
        my_team_is_yellow=planned_is_replay_friendly,
        referee_command=scenario.referee_command,
        ts=scenario.frame_ts,
    )


def scenario_robot_position(replay: Path, t_seconds: float, side: str, robot_id: int) -> Tuple[float, float]:
    """Where `robot_id` of `side` is at `t_seconds` in the replay (replay pitch frame)."""
    scenario = scenario_from_replay(replay, t_seconds)
    for state in scenario.friendly_robots if side == "friendly" else scenario.enemy_robots:
        if state.id == robot_id:
            return state.x, state.y
    raise ValueError(f"no {side} robot {robot_id} in {replay} at t={t_seconds}s")


def _stadium(a: np.ndarray, b: np.ndarray, r: float, n: int = 24) -> np.ndarray:
    """Outline of every point within `r` of the segment `a`-`b` (a rounded rectangle)."""
    d = b - a
    length = float(np.hypot(d[0], d[1]))
    if length < 1e-9:
        t = np.linspace(0.0, 2 * math.pi, 2 * n)
        return a + r * np.stack([np.cos(t), np.sin(t)], axis=1)
    theta = math.atan2(d[1], d[0])
    cap_b = np.linspace(theta + math.pi / 2, theta - math.pi / 2, n)
    cap_a = np.linspace(theta - math.pi / 2, theta - 3 * math.pi / 2, n)
    pts = np.concatenate(
        [
            (b + r * np.array([math.cos(theta + math.pi / 2), math.sin(theta + math.pi / 2)]))[None],
            b + r * np.stack([np.cos(cap_b), np.sin(cap_b)], axis=1),
            (b + r * np.array([math.cos(theta - math.pi / 2), math.sin(theta - math.pi / 2)]))[None],
            (a + r * np.array([math.cos(theta - math.pi / 2), math.sin(theta - math.pi / 2)]))[None],
            a + r * np.stack([np.cos(cap_a), np.sin(cap_a)], axis=1),
        ]
    )
    return pts


def _short(config_name: Optional[str]) -> str:
    if not config_name:
        return "?"
    return config_name.removeprefix("build_").removesuffix("_kernel_strategy")


def plan_snapshot(
    game: Game,
    robot_id: int,
    target: Sequence[float],
    out_path: Path,
    *,
    half_width: float = 1.5,
    full_pitch: bool = False,
    field_dims: FieldDimensions = STANDARD_FIELD_DIMS,
    title: Optional[str] = None,
    target_note: Optional[str] = None,
) -> PlanSnapshot:
    """Run `FastPathPlanner._path_to` for `robot_id` toward `target` on `game` and draw the result.

    The window is `half_width` metres either side of the robot (`full_pitch=True` shows the whole
    pitch). `target_note` is printed in the text box, e.g. to say the target is a stand-in."""
    field_bounds = field_dims.full_field_bounds
    planner = FastPathPlanner(env=None)
    robot = game.friendly_robots[robot_id]
    our_pos = np.array([robot.p.x, robot.p.y], dtype=float)
    target_xy = np.array(target, dtype=float)

    carrot = np.asarray(planner._path_to(game, robot_id, (target_xy[0], target_xy[1]), field_bounds), dtype=float)

    # Obstacles as the planner sees them: every robot within its look-ahead range (a ghost segment
    # from its position to where it will be), then the static field-bound and defence-area edges.
    obstacles, clearance, _ = planner._get_obstacles(game, robot_id, our_pos, field_bounds)
    n_static = len(planner._obstacle_cache_static)
    robot_obstacles = obstacles[: len(obstacles) - n_static]

    ball_pos = np.array([game.ball.p.x, game.ball.p.y], dtype=float) if game.ball is not None else None
    keeps_off = _keeps_off_ball(game)
    robot_to_ball = float(np.linalg.norm(our_pos - ball_pos)) if ball_pos is not None else None
    inside = robot_to_ball is not None and robot_to_ball < BALL_KEEP_OUT_DISTANCE
    robot_to_carrot = float(np.linalg.norm(carrot - our_pos))

    fig = Figure(figsize=(6.6, 7.2), dpi=DPI, facecolor=SURFACE)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.92, bottom=0.16)
    ax = fig.add_subplot(111)
    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal")
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color(INK_3)
        spine.set_linewidth(0.6)
    ax.tick_params(colors=INK_2, labelsize=7, length=2)

    # Pitch: outline, halfway line, centre circle, defence areas, goal mouths.
    half_l, half_w = field_dims.full_field_half_length, field_dims.full_field_half_width
    ax.add_patch(Polygon(field_dims.full_field, closed=True, fill=False, edgecolor=INK_3, linewidth=0.7))
    ax.plot([0, 0], [-half_w, half_w], color=INK_3, linewidth=0.5)
    ax.add_patch(Circle((0, 0), field_dims.center_circle_radius, fill=False, edgecolor=INK_3, linewidth=0.5))
    for area in (field_dims.left_defense_area, field_dims.right_defense_area):
        ax.add_patch(Polygon(area, closed=True, fill=False, edgecolor=INK_3, linewidth=0.5))
    for goal in (field_dims.left_goal_line, field_dims.right_goal_line):
        ax.plot(goal[:, 0], goal[:, 1], color=INK_2, linewidth=1.6, solid_capstyle="butt")

    # Each robot obstacle: its predicted run (thin), and the clearance ring around that run.
    friendly_xy = {(float(r.p.x), float(r.p.y)) for r in game.friendly_robots.values()}
    for a, b in robot_obstacles:
        colour = FOR if (float(a[0]), float(a[1])) in friendly_xy else AGAINST
        ring = _stadium(np.asarray(a, dtype=float), np.asarray(b, dtype=float), clearance)
        ax.add_patch(Polygon(ring, closed=True, fill=False, edgecolor=colour, linewidth=0.6, linestyle=":", alpha=0.8))
        ax.plot([a[0], b[0]], [a[1], b[1]], color=colour, linewidth=0.8, alpha=0.6)

    # The ball and, while the planner keeps out of it, the keep-out circle.
    if ball_pos is not None:
        if keeps_off:
            ax.add_patch(
                Circle(ball_pos, BALL_KEEP_OUT_DISTANCE, fill=False, edgecolor=INK_2, linewidth=0.8, linestyle="--")
            )
        ax.plot(ball_pos[0], ball_pos[1], "o", color=INK, markersize=3.5, zorder=6)

    # Robots at their real radius, with ids. The planned robot gets a heavier outline.
    for is_friendly, robots in ((True, game.friendly_robots), (False, game.enemy_robots)):
        colour = FOR if is_friendly else AGAINST
        for r in robots.values():
            planned = is_friendly and r.id == robot_id
            ax.add_patch(
                Circle(
                    (r.p.x, r.p.y),
                    ROBOT_RADIUS,
                    facecolor=colour,
                    edgecolor=INK if planned else SURFACE,
                    linewidth=1.8 if planned else 0.6,
                    alpha=0.92,
                    zorder=5,
                )
            )
            ax.text(
                r.p.x,
                r.p.y,
                str(r.id),
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                fontweight="bold",
                zorder=11,
            )

    # Target (x), the carrot the planner returned (filled dot), and the two runs from the robot.
    ax.plot([our_pos[0], carrot[0]], [our_pos[1], carrot[1]], color=INK, linewidth=0.9, zorder=4)
    ax.plot(
        [our_pos[0], target_xy[0]], [our_pos[1], target_xy[1]], color=INK_3, linewidth=0.7, linestyle="--", zorder=3
    )
    ax.plot(target_xy[0], target_xy[1], "x", color=INK, markersize=7, markeredgewidth=1.4, zorder=8)
    ax.plot(carrot[0], carrot[1], "o", color=CARROT, markeredgecolor=INK, markeredgewidth=0.6, markersize=6, zorder=9)

    if full_pitch:
        ax.set_xlim(-half_l - 0.2, half_l + 0.2)
        ax.set_ylim(-half_w - 0.2, half_w + 0.2)
    else:
        ax.set_xlim(our_pos[0] - half_width, our_pos[0] + half_width)
        ax.set_ylim(our_pos[1] - half_width, our_pos[1] + half_width)
    ax.set_xlabel("x (m)", fontsize=8, color=INK_2)
    ax.set_ylabel("y (m)", fontsize=8, color=INK_2)

    ref = game.referee.referee_command.name if game.referee is not None else "none"
    lines = [
        f"referee: {ref}",
        f"keep-out {'active' if keeps_off else 'not active'} ({BALL_KEEP_OUT_DISTANCE:.2f} m)",
        f"robot -> ball: {robot_to_ball:.3f} m" if robot_to_ball is not None else "robot -> ball: no ball",
        f"robot -> carrot: {robot_to_carrot:.3f} m",
        f"inside keep-out circle: {'YES' if inside else 'no'}",
        f"clearance: {clearance:.3f} m",
    ]
    if not full_pitch and not _in_window(target_xy, our_pos, half_width):
        lines.append(f"target outside window: ({target_xy[0]:.2f}, {target_xy[1]:.2f})")
    if target_note:
        lines.append(target_note)
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        color=INK,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=SURFACE, edgecolor=GRID, linewidth=0.6),
        zorder=10,
    )

    handles = [
        Line2D([], [], marker="o", linestyle="", markerfacecolor=FOR, markeredgecolor="none", label="planned side"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor=AGAINST, markeredgecolor="none", label="other side"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor=CARROT, markeredgecolor=INK, label="carrot"),
        Line2D([], [], marker="x", linestyle="", color=INK, label="target"),
        Line2D([], [], color=INK_2, linestyle="--", linewidth=0.8, label="ball keep-out"),
        Line2D([], [], color=INK_3, linestyle=":", linewidth=0.8, label="clearance ring"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.55, 0.0),
        ncol=3,
        fontsize=7,
        frameon=False,
        labelcolor=INK_2,
    )
    if title:
        ax.set_title(title, fontsize=9, color=INK, loc="left")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # No bbox_inches="tight": it would grow the image to take in the clipped pitch outline.
    fig.savefig(out_path, dpi=DPI, facecolor=SURFACE)

    return PlanSnapshot(
        carrot=carrot,
        inside_keep_out=bool(inside),
        keeps_off_ball=bool(keeps_off),
        robot_to_carrot=robot_to_carrot,
        robot_to_ball=robot_to_ball,
        path=out_path,
    )


def title_for(scenario: Scenario, side: str, robot_id: int) -> str:
    config = scenario.config_a_name if side == "friendly" else scenario.config_b_name
    return f"planned {side} {_short(config)}, robot {robot_id}, t = {scenario.sim_time:g} s"


def _in_window(point: np.ndarray, centre: np.ndarray, half_width: float) -> bool:
    return bool(np.all(np.abs(np.asarray(point) - np.asarray(centre)) <= half_width))
