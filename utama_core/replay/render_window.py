"""`render_window` — a single PNG showing robot/ball movement over a time window.

The other two observability pieces (`utama_core.engine.match_log.MatchLog`,
`utama_core.engine.match_stats.MatchStats`) answer "why" and "what happened
overall" as text an agent can read directly. Spatial motion over a short
window is the one thing text is bad at — an agent reading a coordinate
series has to do the trajectory integration itself, inconsistently, across
a long context. Matplotlib does that integration instead: each robot/ball's
path over `[t_start, t_end]` is drawn as a trail fading from faint (oldest)
to solid (newest), so direction of travel is visible without arrows, plus
solid markers at the final frame's positions.

Reads from the existing replay `.pkl` (`utama_core.replay.replay_writer`),
not `GameHistory` — `GameHistory` is bounded to `MAX_GAME_HISTORY` (20
frames), far shorter than a useful window, per the same limitation
`match_stats.py` hit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from utama_core.config.field_params import STANDARD_FIELD_DIMS, FieldDimensions
from utama_core.engine.match_log import load_jsonl
from utama_core.entities.game import GameFrame
from utama_core.replay.replay_player import load_frames_in_range
from utama_core.rsoccer_simulator.src.Render.utils import COLORS

_FRIENDLY_COLOR = tuple(c / 255 for c in COLORS["YELLOW"])
_ENEMY_COLOR = tuple(c / 255 for c in COLORS["BLUE"])
_BALL_COLOR = tuple(c / 255 for c in COLORS["ORANGE"])
_PITCH_COLOR = tuple(c / 255 for c in COLORS["BG_GREEN"])
_LINE_COLOR = tuple(c / 255 for c in COLORS["WHITE"])


def _draw_pitch(ax, field_dims: FieldDimensions) -> None:
    length_margin = 0.5
    width_margin = 0.5
    half_l = field_dims.full_field_half_length
    half_w = field_dims.full_field_half_width

    ax.add_patch(
        Rectangle(
            (-half_l - length_margin, -half_w - width_margin),
            2 * (half_l + length_margin),
            2 * (half_w + width_margin),
            facecolor=_PITCH_COLOR,
            zorder=0,
        )
    )
    ax.add_patch(
        Rectangle(
            (-half_l, -half_w), 2 * half_l, 2 * half_w, fill=False, edgecolor=_LINE_COLOR, linewidth=1.5, zorder=1
        )
    )
    ax.add_patch(
        Circle((0, 0), field_dims.center_circle_radius, fill=False, edgecolor=_LINE_COLOR, linewidth=1, zorder=1)
    )
    ax.axvline(0, color=_LINE_COLOR, linewidth=1, zorder=1)

    ax.set_xlim(-half_l - length_margin, half_l + length_margin)
    ax.set_ylim(-half_w - width_margin, half_w + width_margin)
    ax.set_aspect("equal")
    ax.set_facecolor(_PITCH_COLOR)
    ax.set_xticks([])
    ax.set_yticks([])


def _trail(ax, xs: list[float], ys: list[float], color, label: Optional[str] = None) -> None:
    n = len(xs)
    if n == 0:
        return
    if n == 1:
        ax.scatter(xs, ys, color=color, s=40, zorder=3)
        return
    for i in range(n - 1):
        alpha = 0.15 + 0.75 * (i / (n - 2)) if n > 2 else 0.5
        ax.plot(xs[i : i + 2], ys[i : i + 2], color=color, alpha=alpha, linewidth=2, zorder=2)
    ax.scatter([xs[-1]], [ys[-1]], color=color, s=60, zorder=3, edgecolors=_LINE_COLOR, linewidths=0.8, label=label)


def _robot_trails(frames: list[GameFrame], side: str) -> dict[int, tuple[list[float], list[float]]]:
    trails: dict[int, tuple[list[float], list[float]]] = {}
    for frame in frames:
        robots = frame.friendly_robots if side == "friendly" else frame.enemy_robots
        for rid, robot in robots.items():
            xs, ys = trails.setdefault(rid, ([], []))
            xs.append(robot.p.x)
            ys.append(robot.p.y)
    return trails


def render_window(
    replay_path: Union[str, Path],
    t_start: float,
    t_end: float,
    out_path: Union[str, Path],
    *,
    field_dims: FieldDimensions = STANDARD_FIELD_DIMS,
) -> str:
    """Render robot/ball trails over `[t_start, t_end]` plus final positions to `out_path` (PNG).

    Returns `out_path` (as `str`) for convenience chaining. Raises
    `ValueError` if no frames fall in the requested window.
    """
    frames = load_frames_in_range(replay_path, t_start, t_end)
    if not frames:
        raise ValueError(f"No frames found in [{t_start}, {t_end}] for replay {replay_path}")

    fig, ax = plt.subplots(figsize=(10, 7))
    try:
        _draw_pitch(ax, field_dims)

        friendly_trails = _robot_trails(frames, "friendly")
        for rid, (xs, ys) in sorted(friendly_trails.items()):
            _trail(ax, xs, ys, _FRIENDLY_COLOR)
            ax.annotate(str(rid), (xs[-1], ys[-1]), fontsize=8, ha="center", va="center", zorder=4)

        enemy_trails = _robot_trails(frames, "enemy")
        for rid, (xs, ys) in sorted(enemy_trails.items()):
            _trail(ax, xs, ys, _ENEMY_COLOR)
            ax.annotate(str(rid), (xs[-1], ys[-1]), fontsize=8, ha="center", va="center", zorder=4)

        ball_xs = [f.ball.p.x for f in frames if f.ball is not None]
        ball_ys = [f.ball.p.y for f in frames if f.ball is not None]
        _trail(ax, ball_xs, ball_ys, _BALL_COLOR)

        ax.set_title(f"t={t_start:.2f}s → {t_end:.2f}s  ({len(frames)} frames)", color=_LINE_COLOR, fontsize=10)
        fig.patch.set_facecolor("black")
        fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)

    return str(out_path)


def render_around_event(
    replay_path: Union[str, Path],
    match_log_path: Union[str, Path],
    event_index: int,
    pad_seconds: float,
    out_path: Union[str, Path],
    *,
    field_dims: FieldDimensions = STANDARD_FIELD_DIMS,
) -> str:
    """`render_window` anchored on one `IntentionEvent` from a `MatchLog` JSONL trace.

    Lets an agent say "show me around the moment tactic X was assigned"
    instead of guessing a raw timestamp.
    """
    events = load_jsonl(match_log_path)
    if not (0 <= event_index < len(events)):
        raise IndexError(f"event_index {event_index} out of range for {len(events)} events in {match_log_path}")
    center = events[event_index].sim_time
    return render_window(replay_path, center - pad_seconds, center + pad_seconds, out_path, field_dims=field_dims)
