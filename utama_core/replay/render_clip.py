"""`render_clip` — an MP4 of a replay window, for people to watch.

`render_window` (the PNG sibling) is an observability tool: one still image an
agent reads to understand spatial motion. This is its video counterpart for
humans — highlight clips, recruiting/demo material, showing a teammate what a
tactic actually does. Two things make a short clip read well that a still
doesn't need:

- a **following camera** (`camera="follow"`, the default): a fixed-size
  world-space window that pans after an exponentially smoothed ball position,
  clamped so it never shows outside the field. On the full pitch at 1280px
  the ball is a few pixels and a pass barely registers; zoomed in, it reads.
  `camera="full"` gives the static full-pitch view instead.
- a **fading ball trail**, so a pass visibly connects its two endpoints.

Sim ball teleports are not shown. In rsim the referee repositions the ball
for a restart by teleporting it (see `StrategyRunner`'s
`_TELEPORT_SETTLE_TICKS`), which the replay records as the ball sliding metres
across the pitch in a handful of ticks — correct for the simulation, but it
reads as a glitch on video. `displayed_ball` detects teleports from the ball's
own implied speed (the replay has no teleport marker). A teleport during
`BALL_PLACEMENT_*` — nearly all of them — leaves the ball drawn where it went
out until the placement ends, which is when a real robot would have finished
carrying it; it then appears at the placement spot as a cut. Any other
teleport hides the ball through the slide. Either way the trail is cleared and
the camera snaps to the ball's new position instead of panning across.

Drawn in the dashboard's visual style (dark pitch, jersey colours, id labels,
heading ticks, scoreboard header). This is a pygame re-implementation of
`utama_core/dashboard/static/field_canvas.js`'s drawing, not a screenshot of
it — the repo has no headless browser — so the two can drift; if the
dashboard's look changes, update the colours/shapes here to match.

Frames are piped straight into an `ffmpeg` subprocess (H.264, yuv420p), so
`ffmpeg` must be on PATH. Playback rate is derived from the replay's own
frame timestamps, so a clip always plays in real time whatever rate the
replay was recorded at.

CLI:
    pixi run python -m utama_core.replay.render_clip \\
        --replay replays/<name>.npz --t-start 97 --t-end 105.5 --out clip.mp4 [--camera full]
"""

from __future__ import annotations

import argparse
import math
import shutil
import statistics
import subprocess
from pathlib import Path
from typing import Literal, Optional, Sequence, Union

import pygame

from utama_core.config.field_params import STANDARD_FIELD_DIMS, FieldDimensions
from utama_core.entities.game import GameFrame
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.replay.replay_player import load_frames_in_range

# Matches FIELD_COLORS in utama_core/dashboard/static/field_canvas.js.
_PITCH = (0x24, 0x28, 0x32)
_LINE = (0x4B, 0x51, 0x62)
_YELLOW = (0xE0, 0xA5, 0x27)
_BLUE = (0x3D, 0x7F, 0xD1)
_BALL = (0xC9, 0x8A, 0x3F)
_MARKER = (0xE8, 0xEA, 0xF0)
_BACKGROUND = (0x14, 0x16, 0x1C)
_HEADER = (0x1A, 0x1C, 0x24)
_CLOCK = (0x8A, 0x8F, 0x9C)

_ROBOT_RADIUS_M = 0.09

# Frame-to-frame ball speed above which a jump is a teleport, not play. The
# SSL rules cap kicks at 6.5 m/s and real play in rsim replays peaks around
# 6.1 m/s; every teleport sampled (three replays, ~40 teleports) opened with
# a spike of 10.5 m/s or more, so 8 m/s sits in the gap with margin both ways.
_TELEPORT_SPEED_MPS = 8.0
# How long the ball stays hidden after a teleport spike outside ball placement. The replayed position
# doesn't step, it slides — the vision filter smooths the jump into a decay
# that takes ~0.1s to fall below real-play speeds (measured on the same
# teleports), so 0.15s covers the whole slide.
_TELEPORT_HIDE_S = 0.15
_HEADER_H = 48
_MARGIN = 12

Camera = Literal["follow", "full"]

_BALL_PLACEMENT = {RefereeCommand.BALL_PLACEMENT_YELLOW, RefereeCommand.BALL_PLACEMENT_BLUE}

# (center_x, center_y, half_width, half_height) in world meters.
CameraWindow = tuple[float, float, float, float]


def _field_bounds(field_dims: FieldDimensions) -> tuple[float, float]:
    """Half-extents of the visible world: the pitch plus the goal depth behind each goal line."""
    return field_dims.full_field_half_length + field_dims.goal_depth, field_dims.full_field_half_width


def camera_windows(
    ball_positions: Sequence[Optional[tuple[float, float]]],
    *,
    camera: Camera = "follow",
    half_extent: tuple[float, float] = (3.0, 2.2),
    smoothing: float = 0.15,
    field_dims: FieldDimensions = STANDARD_FIELD_DIMS,
    cuts: Optional[Sequence[bool]] = None,
) -> list[CameraWindow]:
    """One world-space camera window per frame.

    `ball_positions[i]` is the ball's `(x, y)` at frame `i`, or `None` when
    the ball isn't visible — a gap holds the last known position (or the
    first known one, for a leading gap) so the camera doesn't lurch to the
    origin. `smoothing` is the per-frame EMA factor on the camera centre
    (lower = slower, smoother pan). `half_extent` is shrunk to the field's
    own half-extent if larger, so the window always fits; its centre is then
    clamped so the window never extends past the field bounds.

    `cuts[i]` marks frame `i` as inside a cut (an overridden `displayed_ball` span): the
    camera holds through it like a gap, then snaps straight to the first
    visible position after it instead of panning there.
    """
    field_half_x, field_half_y = _field_bounds(field_dims)
    if camera == "full":
        return [(0.0, 0.0, field_half_x, field_half_y)] * len(ball_positions)

    half_w = min(half_extent[0], field_half_x)
    half_h = min(half_extent[1], field_half_y)

    known = [p for p in ball_positions if p is not None]
    last = known[0] if known else (0.0, 0.0)
    cx, cy = last
    windows = []
    in_cut = False
    for i, p in enumerate(ball_positions):
        if cuts is not None and cuts[i]:
            in_cut = True
            p = None
        if p is not None:
            last = p
        if in_cut and p is not None:
            cx, cy = p
            in_cut = False
        else:
            cx += (last[0] - cx) * smoothing
            cy += (last[1] - cy) * smoothing
        clamped_x = max(-field_half_x + half_w, min(field_half_x - half_w, cx))
        clamped_y = max(-field_half_y + half_h, min(field_half_y - half_h, cy))
        windows.append((clamped_x, clamped_y, half_w, half_h))
    return windows


def displayed_ball(
    frames: Sequence[GameFrame],
    *,
    max_ball_speed: float = _TELEPORT_SPEED_MPS,
    hide_s: float = _TELEPORT_HIDE_S,
) -> tuple[list[Optional[tuple[float, float]]], list[bool]]:
    """Where to draw the ball in each frame (`None` = hidden), and whether that frame is overridden.

    A teleport is a frame whose ball moved faster than `max_ball_speed` since
    the previous frame that had a ball. During `BALL_PLACEMENT_*` the ball is
    held at its last pre-teleport position until the command leaves ball
    placement; otherwise it is hidden until `hide_s` after the teleport
    (extended by any further spike inside that span). Overridden frames are
    exactly the held or hidden ones.
    """
    positions: list[Optional[tuple[float, float]]] = []
    overridden: list[bool] = []
    prev: Optional[GameFrame] = None
    held: Optional[tuple[float, float]] = None
    hide_until = -math.inf
    for frame in frames:
        in_placement = frame.referee is not None and frame.referee.referee_command in _BALL_PLACEMENT
        if not in_placement:
            held = None
        if frame.ball is not None:
            if prev is not None and frame.ts > prev.ts:
                dist = math.hypot(frame.ball.p.x - prev.ball.p.x, frame.ball.p.y - prev.ball.p.y)
                if dist / (frame.ts - prev.ts) > max_ball_speed:
                    if in_placement:
                        if held is None:
                            held = (prev.ball.p.x, prev.ball.p.y)
                    else:
                        hide_until = frame.ts + hide_s
            prev = frame
        if held is not None:
            positions.append(held)
            overridden.append(True)
        elif frame.ts <= hide_until:
            positions.append(None)
            overridden.append(True)
        else:
            positions.append((frame.ball.p.x, frame.ball.p.y) if frame.ball is not None else None)
            overridden.append(False)
    return positions, overridden


def _clip_fps(frames: list[GameFrame]) -> int:
    """Real-time playback rate from the frames' own timestamps (median spacing, robust to a dropped tick)."""
    dts = [b.ts - a.ts for a, b in zip(frames, frames[1:]) if b.ts > a.ts]
    if not dts:
        return 60
    return max(1, round(1.0 / statistics.median(dts)))


class _FrameDrawer:
    def __init__(self, width: int, height: int, field_dims: FieldDimensions):
        pygame.font.init()
        self.width, self.height = width, height
        self.pitch_h = height - _HEADER_H
        self.g = field_dims
        self.font_id = pygame.font.SysFont("monospace", 15, bold=True)
        self.font_score = pygame.font.SysFont("monospace", 26, bold=True)
        self.font_clock = pygame.font.SysFont("monospace", 15)

    def draw(
        self,
        frame: GameFrame,
        window: CameraWindow,
        trail: list[tuple[float, float]],
        ball: Optional[tuple[float, float]],
    ) -> pygame.Surface:
        surf = pygame.Surface((self.width, self.height))
        surf.fill(_BACKGROUND)
        self._draw_header(surf, frame)

        cx, cy, half_w, half_h = window
        usable_w = self.width - 2 * _MARGIN
        usable_h = self.pitch_h - 2 * _MARGIN
        s = min(usable_w / (2 * half_w), usable_h / (2 * half_h))
        origin_x = _MARGIN + (usable_w - 2 * half_w * s) / 2
        origin_y = _HEADER_H + _MARGIN + (usable_h - 2 * half_h * s) / 2

        def to_px(x: float, y: float) -> tuple[float, float]:
            return origin_x + (x - (cx - half_w)) * s, origin_y + ((cy + half_h) - y) * s

        clip_rect = pygame.Rect(0, _HEADER_H, self.width, self.pitch_h)
        surf.set_clip(clip_rect)
        pygame.draw.rect(surf, _PITCH, clip_rect)
        self._draw_pitch_lines(surf, to_px, s)

        for i, (tx, ty) in enumerate(trail):
            alpha = (i + 1) / len(trail)
            color = tuple(int(_PITCH[c] + (_BALL[c] - _PITCH[c]) * alpha * 0.85) for c in range(3))
            pygame.draw.circle(surf, color, to_px(tx, ty), 2 + 3 * alpha)

        friendly_fill = _YELLOW if frame.my_team_is_yellow else _BLUE
        enemy_fill = _BLUE if frame.my_team_is_yellow else _YELLOW
        for bot in frame.enemy_robots.values():
            self._draw_robot(surf, bot, enemy_fill, to_px, s)
        for bot in frame.friendly_robots.values():
            self._draw_robot(surf, bot, friendly_fill, to_px, s)

        if ball is not None:
            ball_px = to_px(*ball)
            pygame.draw.circle(surf, (0, 0, 0), ball_px, 5)
            pygame.draw.circle(surf, _BALL, ball_px, 4.5)

        surf.set_clip(None)
        pygame.draw.rect(surf, _LINE, clip_rect, 1)
        return surf

    def _draw_header(self, surf: pygame.Surface, frame: GameFrame) -> None:
        pygame.draw.rect(surf, _HEADER, pygame.Rect(0, 0, self.width, _HEADER_H))
        ref = frame.referee
        parts = [
            self.font_score.render(str(ref.yellow_team.score if ref else 0), True, _YELLOW),
            self.font_score.render("-", True, _MARKER),
            self.font_score.render(str(ref.blue_team.score if ref else 0), True, _BLUE),
        ]
        gap = 8
        x = self.width / 2 - (sum(p.get_width() for p in parts) + gap * (len(parts) - 1)) / 2
        for part in parts:
            surf.blit(part, (x, _HEADER_H / 2 - part.get_height() / 2))
            x += part.get_width() + gap

        clock = self.font_clock.render(f"t={frame.ts:6.1f}s", True, _CLOCK)
        surf.blit(clock, (12, _HEADER_H / 2 - clock.get_height() / 2))

    def _draw_pitch_lines(self, surf: pygame.Surface, to_px, s: float) -> None:
        g = self.g
        half_l, half_w = g.full_field_half_length, g.full_field_half_width

        def rect(x1: float, y1: float, x2: float, y2: float) -> None:
            (px1, py1), (px2, py2) = to_px(x1, y1), to_px(x2, y2)
            pygame.draw.rect(surf, _LINE, pygame.Rect(min(px1, px2), min(py1, py2), abs(px2 - px1), abs(py2 - py1)), 2)

        rect(-half_l, -half_w, half_l, half_w)
        pygame.draw.line(surf, _LINE, to_px(0, -half_w), to_px(0, half_w), 2)
        pygame.draw.circle(surf, _LINE, to_px(0, 0), max(1, g.center_circle_radius * s), 2)
        pygame.draw.circle(surf, _LINE, to_px(0, 0), 2)

        defense_depth = 2 * g.half_defense_area_depth
        defense_half_w = g.half_defense_area_width
        rect(-half_l, -defense_half_w, -half_l + defense_depth, defense_half_w)
        rect(half_l - defense_depth, -defense_half_w, half_l, defense_half_w)

        rect(-half_l - g.goal_depth, -g.half_goal_width, -half_l, g.half_goal_width)
        rect(half_l, -g.half_goal_width, half_l + g.goal_depth, g.half_goal_width)

    def _draw_robot(self, surf: pygame.Surface, bot, fill, to_px, s: float) -> None:
        cx, cy = to_px(bot.p.x, bot.p.y)
        r = max(3, _ROBOT_RADIUS_M * s)
        pygame.draw.circle(surf, fill, (cx, cy), r)
        heading_len = r * 1.6
        heading_end = (cx + heading_len * math.cos(bot.orientation), cy - heading_len * math.sin(bot.orientation))
        pygame.draw.line(surf, _PITCH, (cx, cy), heading_end, max(1, int(r * 0.28)))
        label = self.font_id.render(str(bot.id), True, fill)
        surf.blit(label, (cx - label.get_width() / 2, cy - r - 17))


def render_clip(
    replay_path: Union[str, Path],
    t_start: float,
    t_end: float,
    out_path: Union[str, Path],
    *,
    camera: Camera = "follow",
    size: tuple[int, int] = (1280, 800),
    trail_len: int = 18,
    field_dims: FieldDimensions = STANDARD_FIELD_DIMS,
) -> str:
    """Render `[t_start, t_end]` of a replay to an H.264 MP4 at `out_path`.

    Returns `out_path` (as `str`). Raises `ValueError` if no frames fall in
    the window, `RuntimeError` if `ffmpeg` isn't on PATH or fails.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("render_clip needs ffmpeg on PATH to encode the MP4")

    frames = load_frames_in_range(replay_path, t_start, t_end)
    if not frames:
        raise ValueError(f"No frames found in [{t_start}, {t_end}] for replay {replay_path}")

    balls, overridden = displayed_ball(frames)
    windows = camera_windows(balls, camera=camera, field_dims=field_dims, cuts=overridden)
    width, height = size
    drawer = _FrameDrawer(width, height, field_dims)

    proc = subprocess.Popen(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-framerate",
            str(_clip_fps(frames)),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    trail: list[tuple[float, float]] = []
    try:
        was_overridden = False
        for frame, window, ball, is_overridden in zip(frames, windows, balls, overridden):
            if was_overridden and not is_overridden:
                trail.clear()  # the cut back to the real ball: don't draw a trail across it
            if ball is None:
                if is_overridden:
                    trail.clear()
            else:
                trail.append(ball)  # a held ball's trail collapses onto it, like a ball stopping
                del trail[:-trail_len]
            was_overridden = is_overridden
            surface = drawer.draw(frame, window, trail, ball)
            proc.stdin.write(pygame.image.tobytes(surface, "RGB"))
    finally:
        proc.stdin.close()
        stderr = proc.stderr.read()
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {stderr.decode(errors='replace')}")
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a replay window to an MP4.")
    parser.add_argument("--replay", required=True, help="replay file (.npz or .pkl)")
    parser.add_argument("--t-start", type=float, required=True)
    parser.add_argument("--t-end", type=float, required=True)
    parser.add_argument("--out", required=True, help="output .mp4 path")
    parser.add_argument("--camera", choices=["follow", "full"], default="follow")
    args = parser.parse_args()
    print(render_clip(args.replay, args.t_start, args.t_end, args.out, camera=args.camera))


if __name__ == "__main__":
    main()
