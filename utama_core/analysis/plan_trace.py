"""The fast-path planner over a time window: how one robot's plan unfolded (`tools/plan_trace.py`).

`plan_snapshot` shows one tick. This runs the same `FastPathPlanner._path_to` on every sampled
tick of a window, builds the `Game` each tick saw from a callback (so tests can feed synthetic
games and the replay loader stays separate), and records per tick the robot's position, the
carrot the planner returned, the referee command, the robot's distance to the ball, its
clearance to the nearest obstacle and its speed. Read-only: the planner is not changed.

One `FastPathPlanner` serves the whole window, as it does in a live run, so anything it keeps
between ticks is carried along. The target is fixed when given; otherwise each tick's target is
the robot's own position `lookahead` seconds later (the stand-in `plan_snapshot` uses).

The picture: a pitch panel (the robot's path coloured by time, each sample's carrot joined to the
robot, the ball's path, robots at the window's start faded and at its end solid, the keep-out
circle and clearance rings at the end) above time panels that share one x axis: robot-to-ball
distance, robot-to-carrot length, speed, and the referee command as a band.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np  # type: ignore
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from utama_core.analysis.plan_snapshot import (
    AGAINST,
    CARROT,
    DPI,
    FOR,
    GRID,
    INK,
    INK_2,
    INK_3,
    SURFACE,
    draw_ball,
    draw_obstacles,
    draw_pitch,
    draw_robot,
    robot_obstacles_for,
    style_axes,
)
from utama_core.config.field_params import STANDARD_FIELD_DIMS, FieldDimensions
from utama_core.config.referee_constants import BALL_KEEP_OUT_DISTANCE
from utama_core.entities.game import Game
from utama_core.motion_planning.src.fastpathplanning.planner import (
    FastPathPlanner,
    _keeps_off_ball,
)

GameAt = Callable[[float], Game]

# Single-hue ramp (light to dark FOR blue) for time; the carrot dots keep the carrot colour.
TIME_CMAP = mcolors.LinearSegmentedColormap.from_list("time", ["#d6e6fa", FOR, "#0f3d7a"])
# Referee bands: categorical slots from tools/signal_report.py, STOP grey, BALL_PLACEMENT ink.
REFEREE_COLOURS = {
    "HALT": "#e4e3df",
    "STOP": "#e4e3df",
    "NORMAL_START": "#1baf7a",
    "FORCE_START": "#008300",
    "PREPARE_KICKOFF": "#eda100",
    "PREPARE_PENALTY": "#e87ba4",
    "DIRECT_FREE": "#6250d6",
    "INDIRECT_FREE": "#6250d6",
    "TIMEOUT": "#e4e3df",
    "BALL_PLACEMENT": "#52514e",
}
DARK_REFEREE = {"FORCE_START", "DIRECT_FREE", "INDIRECT_FREE", "BALL_PLACEMENT"}
# Samples whose carrot is this close to the robot, while the target is farther than
# `PARKED_TARGET_M`, are a parked robot (carrot collapsed); annotated when present.
PARKED_CARROT_M = 0.2
PARKED_TARGET_M = 0.5
DETOUR_M = 0.2  # carrot this far off the robot-to-target line is a detour, annotated when present


@dataclass(frozen=True)
class TraceRow:
    t: float  # seconds, on the replay's clock
    robot: Tuple[float, float]  # planned robot's position
    target: Tuple[float, float]  # the target `_path_to` was given this tick
    carrot: Tuple[float, float]  # what `_path_to` returned
    robot_to_carrot: float  # metres
    robot_to_target: float  # metres
    carrot_offset: float  # metres the carrot sits off the robot-to-target line (0 when no distance)
    robot_to_ball: Optional[float]  # metres, None without a ball
    clearance: float  # the planner's clearance for this robot this tick, metres
    speed: Optional[float]  # m/s from the previous sample's position; None on the first sample
    ball: Optional[Tuple[float, float]]
    referee: Optional[str]  # referee command name, None without a referee


@dataclass(frozen=True)
class PlanTrace:
    rows: Tuple[TraceRow, ...]
    path: Path
    target_note: str  # the target as drawn: the given point, or the stand-in description


def sample_times(t0: float, t1: float, step: float) -> List[float]:
    """Every `step` seconds from `t0` to `t1` inclusive (`t1` is kept even off the grid)."""
    if step <= 0:
        raise ValueError(f"step must be positive, not {step}")
    if t1 < t0:
        raise ValueError(f"t1 ({t1}) is before t0 ({t0})")
    n = int(math.floor((t1 - t0) / step + 1e-9))
    times = [round(t0 + i * step, 6) for i in range(n + 1)]
    if t1 - times[-1] > 1e-6:
        times.append(round(t1, 6))
    return times


def _referee_name(game: Game) -> Optional[str]:
    if game.referee is None:
        return None
    return game.referee.referee_command.name


def _xy(p) -> Tuple[float, float]:
    return float(p[0]), float(p[1])


def plan_trace(
    game_at: GameAt,
    robot_id: int,
    t0: float,
    t1: float,
    out_path: Path,
    *,
    step: float = 0.1,
    target: Optional[Sequence[float]] = None,
    lookahead: float = 0.5,
    field_dims: FieldDimensions = STANDARD_FIELD_DIMS,
    title: Optional[str] = None,
) -> PlanTrace:
    """Run `_path_to` for `robot_id` on every sampled tick in [t0, t1] and draw the window.

    `game_at(t)` returns the `Game` the planner sees at time `t` (seconds, the replay's clock).
    `target` is fixed for the window; without it each tick's target is the robot's position at
    `t + lookahead`, which needs `game_at` up to `t1 + lookahead`. The PNG goes to `out_path`."""
    field_bounds = field_dims.full_field_bounds
    games: Dict[float, Game] = {}

    def game(t: float) -> Game:
        key = round(t, 6)
        if key not in games:
            games[key] = game_at(t)
        return games[key]

    planner = FastPathPlanner(env=None)
    rows: List[TraceRow] = []
    times = sample_times(t0, t1, step)
    last = None  # (game, robot obstacles, clearance, carrot) of the window's final tick
    for t in times:
        g = game(t)
        if robot_id not in g.friendly_robots:
            raise ValueError(f"no friendly robot {robot_id} in the game at t={t} s")
        robot = g.friendly_robots[robot_id]
        pos = np.array([robot.p.x, robot.p.y], dtype=float)
        if target is not None:
            tgt = np.array(target, dtype=float)
        else:
            ahead = game(t + lookahead).friendly_robots[robot_id].p
            tgt = np.array([ahead.x, ahead.y], dtype=float)
        carrot = np.asarray(planner._path_to(g, robot_id, (tgt[0], tgt[1]), field_bounds), dtype=float)
        robot_obstacles, clearance = robot_obstacles_for(planner, g, robot_id, field_bounds)

        to_target = tgt - pos
        dist_target = float(np.linalg.norm(to_target))
        to_carrot = carrot - pos
        if dist_target > 1e-9:
            unit = to_target / dist_target
            offset = abs(float(unit[0] * to_carrot[1] - unit[1] * to_carrot[0]))
        else:
            offset = 0.0
        ball = g.ball
        ball_xy = _xy((ball.p.x, ball.p.y)) if ball is not None else None
        robot_to_ball = float(np.linalg.norm(pos - np.array(ball_xy))) if ball_xy is not None else None
        speed = None
        if rows:
            prev = rows[-1]
            dt = t - prev.t
            speed = float(np.linalg.norm(pos - np.array(prev.robot))) / dt if dt > 0 else None
        rows.append(
            TraceRow(
                t=float(t),
                robot=_xy(pos),
                target=_xy(tgt),
                carrot=_xy(carrot),
                robot_to_carrot=float(np.linalg.norm(to_carrot)),
                robot_to_target=dist_target,
                carrot_offset=offset,
                robot_to_ball=robot_to_ball,
                clearance=float(clearance),
                speed=speed,
                ball=ball_xy,
                referee=_referee_name(g),
            )
        )
        last = (g, robot_obstacles, float(clearance), carrot)

    note = (
        f"target: ({target[0]:.2f}, {target[1]:.2f}) fixed"
        if target is not None
        else f"target: stand-in, robot at t+{lookahead:g} s"
    )
    first_game = game(times[0])
    end_game, end_obstacles, end_clearance, _ = last
    _draw(
        rows,
        first_game,
        end_game,
        end_obstacles,
        end_clearance,
        robot_id,
        out_path,
        field_dims=field_dims,
        title=title,
        target_note=note,
        fixed_target=target is not None,
    )
    return PlanTrace(rows=tuple(rows), path=Path(out_path), target_note=note)


def _draw(
    rows: List[TraceRow],
    first: Game,
    end: Game,
    end_obstacles,
    end_clearance: float,
    robot_id: int,
    out_path: Path,
    *,
    field_dims: FieldDimensions,
    title: Optional[str],
    target_note: str,
    fixed_target: bool,
) -> None:
    t = np.array([r.t for r in rows])
    robot_xy = np.array([r.robot for r in rows])
    carrot_xy = np.array([r.carrot for r in rows])
    ball_rows = [r.ball for r in rows]
    t0, t1 = float(t[0]), float(t[-1])
    norm = mcolors.Normalize(vmin=t0, vmax=max(t1, t0 + 1e-6))

    fig = Figure(figsize=(7.0, 10.4), dpi=DPI, facecolor=SURFACE)
    gs = fig.add_gridspec(
        5,
        1,
        height_ratios=[3.3, 1.0, 1.0, 1.0, 0.35],
        left=0.12,
        right=0.95,
        top=0.95,
        bottom=0.1,
        hspace=0.32,
    )
    ax = fig.add_subplot(gs[0])
    style_axes(ax)
    draw_pitch(ax, field_dims)

    # Window the robot moved through, plus its carrots and target, with a margin.
    pts = np.vstack([robot_xy, carrot_xy] + ([np.array([rows[0].target])] if fixed_target else []))
    lo, hi = pts.min(axis=0) - 0.4, pts.max(axis=0) + 0.4
    centre = (lo + hi) / 2
    half = max(float(np.max(hi - lo)) / 2, 0.6)
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)

    # Other robots: at t0 faded, at t1 solid. The planned robot is drawn at t1 only, as the path's end.
    for is_friendly, robots in ((True, first.friendly_robots), (False, first.enemy_robots)):
        colour = FOR if is_friendly else AGAINST
        for r in robots.values():
            if is_friendly and r.id == robot_id:
                continue
            draw_robot(ax, r.p.x, r.p.y, r.id, colour, alpha=0.2, zorder=3)
    for is_friendly, robots in ((True, end.friendly_robots), (False, end.enemy_robots)):
        colour = FOR if is_friendly else AGAINST
        for r in robots.values():
            if is_friendly and r.id == robot_id:
                continue
            draw_robot(ax, r.p.x, r.p.y, r.id, colour)
    draw_obstacles(ax, end, end_obstacles, end_clearance)

    # Ball path (t0 to t1), and the keep-out circle at t1 while the planner keeps out of it.
    ball_pts = np.array([b for b in ball_rows if b is not None])
    if len(ball_pts) > 0:
        ax.plot(ball_pts[:, 0], ball_pts[:, 1], color=INK_2, linewidth=0.8, linestyle="-.", zorder=4)
        ax.plot(
            ball_pts[0, 0], ball_pts[0, 1], "o", markerfacecolor=SURFACE, markeredgecolor=INK, markersize=3, zorder=6
        )
    if end.ball is not None:
        end_ball = np.array([end.ball.p.x, end.ball.p.y], dtype=float)
        draw_ball(ax, end_ball, _keeps_off_ball(end))

    # Carrot per sample: a thin line from the robot to the carrot it returned, and a small dot.
    segs = np.stack([robot_xy, carrot_xy], axis=1)
    ax.add_collection(LineCollection(segs, colors=CARROT, linewidths=0.5, alpha=0.55, zorder=5))
    ax.plot(carrot_xy[:, 0], carrot_xy[:, 1], "o", color=CARROT, markersize=2.2, markeredgewidth=0, zorder=6)

    # The robot's path, each segment coloured by its start time.
    if len(rows) > 1:
        path_segs = np.stack([robot_xy[:-1], robot_xy[1:]], axis=1)
        path_colours = TIME_CMAP(norm(t[:-1]))
        ax.add_collection(LineCollection(path_segs, colors=path_colours, linewidths=2.4, zorder=7, capstyle="round"))
    ax.plot(robot_xy[-1, 0], robot_xy[-1, 1], "s", color=INK, markersize=3, zorder=8)

    if fixed_target:
        ax.plot(rows[0].target[0], rows[0].target[1], "x", color=INK, markersize=7, markeredgewidth=1.4, zorder=9)

    # Annotations: at most two. A parked robot (carrot collapsed, target still far) and a detour.
    annotations = []
    parked = [r for r in rows if r.robot_to_target > PARKED_TARGET_M]
    if parked:
        worst = min(parked, key=lambda r: r.robot_to_carrot)
        if worst.robot_to_carrot < PARKED_CARROT_M:
            annotations.append(
                (
                    worst.t,
                    worst.robot_to_carrot,
                    f"t={worst.t:g} s: parked\ncarrot {worst.robot_to_carrot:.2f} m away, "
                    f"target {worst.robot_to_target:.2f} m",
                )
            )
    detour = max(rows, key=lambda r: r.carrot_offset)
    if detour.carrot_offset > DETOUR_M:
        ax.annotate(
            f"t={detour.t:g} s: carrot {detour.carrot_offset:.2f} m off the\nrobot-to-target line",
            xy=detour.carrot,
            xytext=(0.98, 0.97),
            textcoords="axes fraction",
            ha="right",
            va="top",
            fontsize=6.5,
            color=INK,
            bbox=dict(boxstyle="square,pad=0.2", facecolor=SURFACE, edgecolor="none"),
            arrowprops=dict(arrowstyle="-", linewidth=0.6, color=INK_2),
            zorder=12,
        )

    ref_names = [r.referee for r in rows]
    lines = [
        f"t = {t0:g} to {t1:g} s, {len(rows)} samples",
        target_note,
        f"min robot-to-ball: {min((r.robot_to_ball for r in rows if r.robot_to_ball is not None), default=float('nan')):.2f} m",
        f"max carrot offset: {max(r.carrot_offset for r in rows):.2f} m",
    ]
    if ref_names and ref_names[-1]:
        lines.append(f"referee at t1: {ref_names[-1]}")
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=6.5,
        color=INK,
        bbox=dict(boxstyle="round,pad=0.35", facecolor=SURFACE, edgecolor=GRID, linewidth=0.6),
        zorder=13,
    )
    sm = cm.ScalarMappable(norm=norm, cmap=TIME_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, aspect=18, pad=0.02)
    cbar.set_label("time (s)", fontsize=7, color=INK_2)
    cbar.ax.tick_params(labelsize=6.5, colors=INK_2, length=2)
    cbar.outline.set_edgecolor(INK_3)
    ax.set_xlabel("x (m)", fontsize=8, color=INK_2)
    ax.set_ylabel("y (m)", fontsize=8, color=INK_2)
    if title:
        ax.set_title(title, fontsize=9, color=INK, loc="left")

    # Time panels: one quantity each, sharing the x axis.
    ax_ball = fig.add_subplot(gs[1])
    ax_carrot = fig.add_subplot(gs[2], sharex=ax_ball)
    ax_speed = fig.add_subplot(gs[3], sharex=ax_ball)
    ax_ref = fig.add_subplot(gs[4], sharex=ax_ball)
    for a in (ax_ball, ax_carrot, ax_speed):
        style_axes(a)
        a.set_aspect("auto")
        a.grid(True, axis="y", color=GRID, linewidth=0.5)
        a.tick_params(labelbottom=False)

    ball_d = np.array([np.nan if r.robot_to_ball is None else r.robot_to_ball for r in rows])
    ax_ball.plot(t, ball_d, color=FOR, linewidth=1.1)
    ax_ball.axhline(BALL_KEEP_OUT_DISTANCE, color=INK_2, linewidth=0.8, linestyle="--")
    ax_ball.text(
        t1,
        BALL_KEEP_OUT_DISTANCE,
        f" keep-out {BALL_KEEP_OUT_DISTANCE:.1f} m",
        va="bottom",
        ha="right",
        fontsize=6.5,
        color=INK_2,
    )
    ax_ball.set_ylabel("robot-ball\n(m)", fontsize=7, color=INK_2)

    carrot_len = np.array([r.robot_to_carrot for r in rows])
    ax_carrot.plot(t, carrot_len, color=CARROT, linewidth=1.1)
    ax_carrot.set_ylabel("robot-carrot\n(m)", fontsize=7, color=INK_2)
    ax_carrot.set_ylim(bottom=0)
    if annotations:
        at, val, text = annotations[0]
        ax_carrot.annotate(
            text,
            xy=(at, val),
            xytext=(at + 0.04 * (t1 - t0), 0.55),
            textcoords="data",
            fontsize=6.5,
            color=INK,
            arrowprops=dict(arrowstyle="-", linewidth=0.6, color=INK_2),
            zorder=12,
        )

    speed = np.array([np.nan if r.speed is None else r.speed for r in rows])
    ax_speed.plot(t, speed, color=INK_2, linewidth=1.0)
    ax_speed.set_ylabel("speed\n(m/s)", fontsize=7, color=INK_2)
    ax_speed.set_ylim(bottom=0)

    # Referee band: a span per run of one command, labelled when wide enough.
    style_axes(ax_ref)
    ax_ref.set_aspect("auto")
    ax_ref.set_yticks([])
    ax_ref.set_ylim(0, 1)
    ax_ref.set_ylabel("referee", fontsize=7, color=INK_2, rotation=0, ha="right", va="center")
    spans = _referee_spans(rows, t1)
    for name, start, end_t in spans:
        colour = REFEREE_COLOURS.get(_family(name), "#e4e3df")
        ax_ref.axvspan(start, end_t, color=colour, linewidth=0)
        if end_t - start > 0.12 * max(t1 - t0, 1e-6):
            ax_ref.text(
                (start + end_t) / 2,
                0.5,
                _band_label(name),
                ha="center",
                va="center",
                fontsize=6,
                color="white" if _family(name) in DARK_REFEREE else INK,
                clip_on=True,
            )
    ax_ref.set_xlim(t0, t1 if t1 > t0 else t0 + 1e-6)
    ax_ref.set_xlabel("time (s)", fontsize=8, color=INK_2)
    ax_ref.tick_params(labelbottom=True, labelsize=7)

    handles = [
        Line2D([], [], color=TIME_CMAP(0.6), linewidth=2.4, label="robot path (time)"),
        Line2D([], [], marker="o", linestyle="", color=CARROT, markersize=4, label="carrot"),
        Line2D([], [], color=INK_2, linestyle="-.", linewidth=0.8, label="ball path"),
        Line2D([], [], color=INK_2, linestyle="--", linewidth=0.8, label="ball keep-out (t1)"),
        Line2D([], [], color=INK_3, linestyle=":", linewidth=0.8, label="clearance (t1)"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.0),
        ncol=3,
        fontsize=6.5,
        frameon=False,
        labelcolor=INK_2,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, facecolor=SURFACE)


def _family(name: str) -> str:
    """The referee command's band colour key: PREPARE_KICKOFF_YELLOW -> PREPARE_KICKOFF."""
    for prefix in sorted(REFEREE_COLOURS, key=len, reverse=True):
        if name.startswith(prefix):
            return prefix
    return ""


def _band_label(name: str) -> str:
    """The band's label: the command without its side suffix (PREPARE_KICKOFF_YELLOW -> PREPARE_KICKOFF)."""
    for suffix in ("_YELLOW", "_BLUE"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _referee_spans(rows: List[TraceRow], t1: float) -> List[Tuple[str, float, float]]:
    """(command, start, end) for each run of one referee command over the samples."""
    spans: List[Tuple[str, float, float]] = []
    for i, r in enumerate(rows):
        name = r.referee or "none"
        end_t = rows[i + 1].t if i + 1 < len(rows) else t1
        if spans and spans[-1][0] == name:
            spans[-1] = (name, spans[-1][1], end_t)
        else:
            spans.append((name, r.t, end_t))
    return spans
