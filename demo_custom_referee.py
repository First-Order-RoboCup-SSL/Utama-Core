"""demo_custom_referee.py — visual demonstration of the CustomReferee system.

Runs entirely in-process (no network, no RSim binary needed). A scripted
scenario exercises every referee rule in sequence while the pygame window shows:
  - The field with robots and ball
  - Purple polygons: defense area boundaries
  - Red circle: keep-out zone (during stoppages)
  - Green crosshair: designated_position (ball reset target, e.g. centre after goal)
  - HUD bar: current command (colour-coded), score, next command, designated pos, scene label

Each scene in SCENES declares:
  - What command to force at scene start (simulates operator "play" button)
  - How the ball moves (keyframes interpolated over the scene duration)
  - How robots move (per-robot keyframes)
  - Whether to simulate the ball teleport on STOP (ball_teleports_on_stop=True)

Controls:
    ESC / close window — quit
    SPACE             — pause / unpause
    R                 — restart from scene 0
    → / ←            — skip forward / back one scene
"""

from __future__ import annotations

import dataclasses
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import pygame

from utama_core.custom_referee import CustomReferee, RefereeGeometry
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.robot import Robot
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.rsoccer_simulator.src.Render import COLORS, SSLRenderField
from utama_core.rsoccer_simulator.src.Render.ball import RenderBall
from utama_core.rsoccer_simulator.src.Render.overlay import (
    OverlayObject,
    OverlayType,
    RenderOverlay,
)
from utama_core.rsoccer_simulator.src.Render.robot import RenderSSLRobot

# ---------------------------------------------------------------------------
# Scene definition
# ---------------------------------------------------------------------------


@dataclass
class Vec2KF:
    """A (t, x, y) keyframe for linear interpolation."""

    t: float
    x: float
    y: float


def _interp(kfs: list[Vec2KF], t: float) -> tuple[float, float]:
    """Linearly interpolate a list of Vec2KF keyframes at time t."""
    if not kfs:
        return 0.0, 0.0
    if t <= kfs[0].t:
        return kfs[0].x, kfs[0].y
    if t >= kfs[-1].t:
        return kfs[-1].x, kfs[-1].y
    for i in range(len(kfs) - 1):
        k0, k1 = kfs[i], kfs[i + 1]
        if k0.t <= t <= k1.t:
            a = (t - k0.t) / (k1.t - k0.t)
            return k0.x + a * (k1.x - k0.x), k0.y + a * (k1.y - k0.y)
    return kfs[-1].x, kfs[-1].y


@dataclass
class Scene:
    """One demo scene: sets up initial command, describes motion, and labels it."""

    title: str  # shown large in HUD
    subtitle: str  # shown small in HUD
    duration: float  # seconds
    # Command forced at t=0 of this scene (simulates operator input)
    force_command: Optional[RefereeCommand] = None
    # Ball keyframes relative to scene start
    ball_kfs: list[Vec2KF] = field(default_factory=list)
    # Per-robot overrides: dict robot_index → list[Vec2KF]
    # Robot layout indices: 0,1,2 = yellow; 3,4,5 = blue
    robot_kfs: dict[int, list[Vec2KF]] = field(default_factory=dict)
    # If True, once the referee issues STOP the ball position is overridden to
    # designated_position for the rest of the scene (mirrors StrategyRunner teleport).
    ball_teleports_on_stop: bool = False


# Base robot positions (field coords, metres)
# Index:  0=(Y0), 1=(Y1), 2=(Y2), 3=(B0), 4=(B1), 5=(B2)
_Y0 = Vec2KF(0, 1.5, 0.0)
_Y1 = Vec2KF(0, 3.0, 1.2)
_Y2 = Vec2KF(0, 3.0, -1.2)
_B0 = Vec2KF(0, -1.5, 0.0)
_B1 = Vec2KF(0, -3.0, 1.2)
_B2 = Vec2KF(0, -3.0, -1.2)

BASE_ROBOTS = [_Y0, _Y1, _Y2, _B0, _B1, _B2]  # one keyframe each (static default)


SCENES: list[Scene] = [
    # ------------------------------------------------------------------
    # 0. NORMAL START — play in progress, nothing happening
    # ------------------------------------------------------------------
    Scene(
        title="NORMAL START",
        subtitle="Active play — no violations",
        duration=3.0,
        force_command=RefereeCommand.NORMAL_START,
        ball_kfs=[
            Vec2KF(0.0, 0.0, 0.0),
            Vec2KF(3.0, 2.0, 1.0),
        ],
    ),
    # ------------------------------------------------------------------
    # 1. GOAL — yellow robot dribbles ball into the LEFT goal (blue's goal).
    #    yellow_is_right=True so the left goal is blue's → Yellow scores.
    #    Ball teleports to (0,0) the moment STOP fires (simulates StrategyRunner).
    # ------------------------------------------------------------------
    Scene(
        title="GOAL RULE",
        subtitle="Yellow attacks left goal (Blue's) → Yellow scores, STOP + ball teleport to centre",
        duration=5.0,
        force_command=RefereeCommand.NORMAL_START,
        ball_kfs=[
            Vec2KF(0.0, 0.0, 0.3),
            Vec2KF(2.5, -5.5, 0.0),  # crosses left goal line — triggers GoalRule
            Vec2KF(5.0, -5.5, 0.0),
        ],
        robot_kfs={
            # Yellow robot 1 starts centre-left and charges toward the left goal.
            # Stops just outside the defense area (x=-3.2) to avoid entering it.
            1: [Vec2KF(0.0, -0.5, 0.3), Vec2KF(2.0, -3.2, 0.2), Vec2KF(5.0, -3.2, 0.2)],
        },
        ball_teleports_on_stop=True,
    ),
    # ------------------------------------------------------------------
    # 2. STOP phase after goal — ball already at centre (0,0)
    # ------------------------------------------------------------------
    Scene(
        title="STOP (after goal)",
        subtitle="Ball at centre (designated_position) — keep-out circle active",
        duration=4.0,
        # No force_command here — the state machine already issued STOP from scene 1.
        ball_kfs=[Vec2KF(0.0, 0.0, 0.0), Vec2KF(4.0, 0.0, 0.0)],
        robot_kfs={
            # Yellow robot 0 respects the circle
            0: [Vec2KF(0.0, 1.5, 0.0), Vec2KF(4.0, 1.5, 0.0)],
            # Blue robot 0 also stays back
            3: [Vec2KF(0.0, -1.5, 0.0), Vec2KF(4.0, -1.5, 0.0)],
        },
    ),
    # ------------------------------------------------------------------
    # 3. OUT OF BOUNDS — ball kicked off sideline
    # ------------------------------------------------------------------
    Scene(
        title="OUT OF BOUNDS RULE",
        subtitle="Ball crosses top sideline → STOP + DIRECT_FREE issued",
        duration=5.0,
        force_command=RefereeCommand.NORMAL_START,
        ball_kfs=[
            Vec2KF(0.0, 0.5, 1.0),
            Vec2KF(2.5, 1.5, 3.8),  # crosses top boundary (half_width=3.0)
            Vec2KF(5.0, 1.5, 3.8),
        ],
        robot_kfs={
            # Yellow robot 0 kicked it
            0: [Vec2KF(0.0, 0.5, 1.0), Vec2KF(2.0, 1.0, 2.5), Vec2KF(5.0, 1.0, 2.5)],
        },
    ),
    # ------------------------------------------------------------------
    # 4. DEFENSE AREA — blue attacker walks into yellow's right defense area
    # ------------------------------------------------------------------
    Scene(
        title="DEFENSE AREA RULE",
        subtitle="Enemy attacker enters right defense area → STOP + DIRECT_FREE",
        duration=5.0,
        force_command=RefereeCommand.NORMAL_START,
        ball_kfs=[
            Vec2KF(0.0, 2.0, 0.0),
            Vec2KF(5.0, 3.5, 0.0),
        ],
        robot_kfs={
            # Blue robot 0 drives from midfield into the right defense area
            # Right defense area: x >= 4.5 - 2*0.5 = 3.5, |y| <= 1.0
            3: [Vec2KF(0.0, 0.5, 0.0), Vec2KF(2.5, 3.8, 0.3), Vec2KF(5.0, 3.8, 0.3)],
            # Blue robot 1 stays put
            4: [Vec2KF(0.0, -3.0, 1.2), Vec2KF(5.0, -3.0, 1.2)],
        },
    ),
    # ------------------------------------------------------------------
    # 5. KEEP-OUT — blue robot creeps inside the 0.5 m circle during STOP
    # ------------------------------------------------------------------
    Scene(
        title="KEEP-OUT RULE",
        subtitle="Robot stays < 0.5 m from ball for 30 frames → DIRECT_FREE issued",
        duration=6.0,
        force_command=RefereeCommand.STOP,
        ball_kfs=[Vec2KF(0.0, 0.0, 0.0), Vec2KF(6.0, 0.0, 0.0)],
        robot_kfs={
            # Blue robot 0 slowly creeps inside the keep-out circle
            3: [Vec2KF(0.0, -2.0, 0.0), Vec2KF(1.5, -0.3, 0.0), Vec2KF(6.0, -0.3, 0.0)],
        },
    ),
]

TOTAL_SCENES = len(SCENES)


# ---------------------------------------------------------------------------
# Robot position at time t within a scene
# ---------------------------------------------------------------------------


def _robot_pos(scene: Scene, robot_idx: int, t: float) -> tuple[float, float]:
    """Return (x, y) for robot_idx at scene-relative time t."""
    if robot_idx in scene.robot_kfs:
        return _interp(scene.robot_kfs[robot_idx], t)
    # Fall back to base position (static).
    base = BASE_ROBOTS[robot_idx]
    return base.x, base.y


def _make_frame(scene: Scene, t: float, current_time: float) -> GameFrame:
    bx, by = _interp(scene.ball_kfs, t) if scene.ball_kfs else (0.0, 0.0)
    ball = Ball(p=Vector3D(bx, by, 0.0), v=Vector3D(0, 0, 0), a=Vector3D(0, 0, 0))

    friendly: dict[int, Robot] = {}
    enemy: dict[int, Robot] = {}

    for idx in range(3):  # yellow robots — attack left, so face −x (π radians)
        x, y = _robot_pos(scene, idx, t)
        friendly[idx] = Robot(
            id=idx,
            is_friendly=True,
            has_ball=False,
            p=Vector2D(x, y),
            v=Vector2D(0, 0),
            a=Vector2D(0, 0),
            orientation=math.pi,
        )

    for idx in range(3):  # blue robots — attack right, so face +x (0 radians)
        x, y = _robot_pos(scene, idx + 3, t)
        enemy[idx] = Robot(
            id=idx,
            is_friendly=False,
            has_ball=False,
            p=Vector2D(x, y),
            v=Vector2D(0, 0),
            a=Vector2D(0, 0),
            orientation=0.0,
        )

    return GameFrame(
        ts=current_time,
        my_team_is_yellow=True,
        my_team_is_right=True,  # yellow defends right goal, blue defends left
        friendly_robots=friendly,
        enemy_robots=enemy,
        ball=ball,
        referee=None,
    )


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------


def _field_to_screen(x: float, y: float, fr: SSLRenderField) -> tuple[int, int]:
    return (
        int(x * fr.scale + fr.center_x),
        int(-y * fr.scale + fr.center_y),
    )


_STOPPAGE_CMDS = {
    RefereeCommand.STOP,
    RefereeCommand.DIRECT_FREE_YELLOW,
    RefereeCommand.DIRECT_FREE_BLUE,
    RefereeCommand.PREPARE_KICKOFF_YELLOW,
    RefereeCommand.PREPARE_KICKOFF_BLUE,
}


def _draw_designated_position(
    surface: pygame.Surface,
    dx: float,
    dy: float,
    fr: SSLRenderField,
) -> None:
    """Draw a green crosshair + diamond at the designated ball position."""
    cx, cy = _field_to_screen(dx, dy, fr)
    color = (0, 220, 80)

    arm = int(18 * fr.scale / 100)  # crosshair arm length scaled with field
    arm = max(arm, 10)
    thick = 2

    # Crosshair
    pygame.draw.line(surface, color, (cx - arm, cy), (cx + arm, cy), thick)
    pygame.draw.line(surface, color, (cx, cy - arm), (cx, cy + arm), thick)

    # Diamond outline
    d = int(arm * 0.7)
    pts = [(cx, cy - d), (cx + d, cy), (cx, cy + d), (cx - d, cy)]
    pygame.draw.polygon(surface, color, pts, thick)


def _build_overlays(
    geo: RefereeGeometry,
    command: RefereeCommand,
    bx: float,
    by: float,
    fr: SSLRenderField,
) -> list[OverlayObject]:
    ovs: list[OverlayObject] = []

    # Defense area outlines (purple, always visible)
    rdx = geo.half_length - 2 * geo.half_defense_length  # 3.5
    ldx = -geo.half_length + 2 * geo.half_defense_length  # -3.5

    for pts in [
        # Right defense area (yellow's goal)
        [
            (rdx, geo.half_defense_width),
            (geo.half_length, geo.half_defense_width),
            (geo.half_length, -geo.half_defense_width),
            (rdx, -geo.half_defense_width),
        ],
        # Left defense area (blue's goal)
        [
            (-geo.half_length, geo.half_defense_width),
            (ldx, geo.half_defense_width),
            (ldx, -geo.half_defense_width),
            (-geo.half_length, -geo.half_defense_width),
        ],
    ]:
        ovs.append(
            OverlayObject(
                type=OverlayType.POLYGON,
                color="PURPLE",
                points=[_field_to_screen(px, py, fr) for px, py in pts],
                width=2,
            )
        )

    # Keep-out circle (red ring, during stoppages)
    if command in _STOPPAGE_CMDS:
        cx, cy = _field_to_screen(bx, by, fr)
        r_px = int(geo.center_circle_radius * fr.scale)
        n = 48
        circle_pts = [
            (int(cx + r_px * math.cos(2 * math.pi * i / n)), int(cy + r_px * math.sin(2 * math.pi * i / n)))
            for i in range(n)
        ]
        ovs.append(
            OverlayObject(
                type=OverlayType.POLYGON,
                color="RED",
                points=circle_pts,
                width=2,
            )
        )

    return ovs


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------

_CMD_COLORS: dict[RefereeCommand, tuple] = {
    RefereeCommand.HALT: (200, 50, 50),
    RefereeCommand.STOP: (230, 130, 0),
    RefereeCommand.NORMAL_START: (60, 210, 60),
    RefereeCommand.FORCE_START: (60, 210, 60),
    RefereeCommand.DIRECT_FREE_YELLOW: (240, 210, 50),
    RefereeCommand.DIRECT_FREE_BLUE: (80, 150, 255),
    RefereeCommand.PREPARE_KICKOFF_YELLOW: (240, 210, 50),
    RefereeCommand.PREPARE_KICKOFF_BLUE: (80, 150, 255),
}


def _draw_hud(
    surface: pygame.Surface,
    fonts: dict,
    ref_data,
    scene_idx: int,
    scene_title: str,
    scene_subtitle: str,
    paused: bool,
    scene_t: float,
    scene_dur: float,
) -> None:
    W = surface.get_width()
    panel_h = 120
    panel = pygame.Surface((W, panel_h), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 170))
    surface.blit(panel, (0, 0))

    cmd = ref_data.referee_command
    cmd_color = _CMD_COLORS.get(cmd, (200, 200, 200))
    next_cmd = ref_data.next_command

    # Scene counter (top right)
    sc_surf = fonts["tiny"].render(
        f"Scene {scene_idx + 1}/{TOTAL_SCENES}  [{scene_t:.1f}/{scene_dur:.0f}s]",
        True,
        (150, 150, 150),
    )
    surface.blit(sc_surf, (W - sc_surf.get_width() - 10, 6))

    # Command (large, colour-coded)
    cmd_surf = fonts["large"].render(cmd.name.replace("_", " "), True, cmd_color)
    surface.blit(cmd_surf, (10, 6))

    # Score
    score_str = f"Yellow {ref_data.yellow_team.score}  –  {ref_data.blue_team.score} Blue"
    score_surf = fonts["medium"].render(score_str, True, (230, 230, 230))
    surface.blit(score_surf, (10, 40))

    # next command + designated position (right column)
    right_x = W // 2
    if next_cmd:
        nc_surf = fonts["small"].render(f"next → {next_cmd.name.replace('_', ' ')}", True, (180, 180, 180))
        surface.blit(nc_surf, (right_x, 40))
    desg = ref_data.designated_position
    if desg is not None and cmd == RefereeCommand.STOP:
        dp_surf = fonts["small"].render(f"designated → ({desg[0]:.2f}, {desg[1]:.2f}) m", True, (0, 220, 80))
        surface.blit(dp_surf, (right_x, 58))

    # Scene title / subtitle
    title_surf = fonts["small"].render(f"▶ {scene_title}", True, (200, 255, 200))
    surface.blit(title_surf, (10, 68))

    sub_surf = fonts["tiny"].render(scene_subtitle, True, (170, 200, 170))
    surface.blit(sub_surf, (10, 90))

    # Pause indicator
    if paused:
        p_surf = fonts["small"].render("PAUSED — SPACE to resume", True, (255, 200, 80))
        surface.blit(p_surf, (W // 2, 90))

    # Progress bar
    bar_y = panel_h - 4
    bar_w = int(W * min(scene_t / max(scene_dur, 0.001), 1.0))
    pygame.draw.rect(surface, (80, 80, 80), (0, bar_y, W, 4))
    pygame.draw.rect(surface, cmd_color, (0, bar_y, bar_w, 4))


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _draw_robots(
    surface: pygame.Surface,
    frame: GameFrame,
    fr: SSLRenderField,
) -> None:
    for robot in frame.friendly_robots.values():
        sx, sy = _field_to_screen(robot.p.x, robot.p.y, fr)
        direction_deg = math.degrees(robot.orientation)
        RenderSSLRobot(sx, sy, direction_deg, fr.scale, robot.id, COLORS["YELLOW"]).draw(surface)
    for robot in frame.enemy_robots.values():
        sx, sy = _field_to_screen(robot.p.x, robot.p.y, fr)
        direction_deg = math.degrees(robot.orientation)
        RenderSSLRobot(sx, sy, direction_deg, fr.scale, robot.id, COLORS["BLUE"]).draw(surface)


def _draw_ball(
    surface: pygame.Surface,
    bx: float,
    by: float,
    fr: SSLRenderField,
) -> None:
    sx, sy = _field_to_screen(bx, by, fr)
    RenderBall(sx, sy, fr.scale).draw(surface)


def _draw_frame(
    screen: pygame.Surface,
    fr_renderer: SSLRenderField,
    fonts: dict,
    geo: RefereeGeometry,
    frame: GameFrame,
    ref_data,
    scene_idx: int,
    scene: Scene,
    scene_t: float,
    paused: bool,
) -> None:
    bx, by = frame.ball.p.x, frame.ball.p.y
    fr_renderer.draw(screen)

    ovs = _build_overlays(geo, ref_data.referee_command, bx, by, fr_renderer)
    if ovs:
        RenderOverlay(ovs, fr_renderer.scale).draw(screen)

    _draw_robots(screen, frame, fr_renderer)
    _draw_ball(screen, bx, by, fr_renderer)

    # Designated position marker (green crosshair) — only during STOP,
    # where it signals where the ball should be placed before play resumes.
    if ref_data.designated_position is not None and ref_data.referee_command == RefereeCommand.STOP:
        dx, dy = ref_data.designated_position
        _draw_designated_position(screen, dx, dy, fr_renderer)

    _draw_hud(
        screen,
        fonts,
        ref_data,
        scene_idx,
        scene.title,
        scene.subtitle,
        paused,
        scene_t,
        scene.duration,
    )

    pygame.display.flip()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _make_referee() -> CustomReferee:
    referee = CustomReferee.from_profile_name("strict_ai", n_robots_yellow=3, n_robots_blue=3)
    referee.set_command(RefereeCommand.HALT, timestamp=0.0)
    return referee


def main() -> None:
    pygame.init()
    fr_renderer = SSLRenderField()
    W, H = fr_renderer.window_size
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Custom Referee — Visual Demo  |  SPACE pause  R restart  ←→ scenes")
    clock = pygame.time.Clock()

    fonts = {
        "large": pygame.font.SysFont("monospace", 22, bold=True),
        "medium": pygame.font.SysFont("monospace", 18, bold=True),
        "small": pygame.font.SysFont("monospace", 15),
        "tiny": pygame.font.SysFont("monospace", 12),
    }

    geo = RefereeGeometry.from_standard_div_b()
    referee = _make_referee()

    scene_idx = 0
    scene_start_wall = time.perf_counter()
    scene_forced = False  # whether we've already applied force_command this scene
    paused = False
    pause_wall = 0.0
    pause_acc = 0.0  # accumulated pause time in current scene
    prev_command = RefereeCommand.HALT  # for teleport edge detection
    teleport_pos: Optional[tuple[float, float]] = None  # active teleport override

    # Hold latest data for redraw during pause
    last_frame = _make_frame(SCENES[0], 0.0, 0.0)
    last_ref = referee.step(last_frame, current_time=0.0)

    def reset_scene(idx: int, wall_now: float) -> None:
        nonlocal scene_idx, scene_start_wall, scene_forced, pause_acc, prev_command, teleport_pos
        scene_idx = idx
        scene_start_wall = wall_now
        scene_forced = False
        pause_acc = 0.0
        prev_command = RefereeCommand.HALT
        teleport_pos = None

    running = True
    while running:
        wall_now = time.perf_counter()

        # ---- Events --------------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:
                    if paused:
                        pause_acc += wall_now - pause_wall
                        paused = False
                    else:
                        pause_wall = wall_now
                        paused = True

                elif event.key == pygame.K_r:
                    referee = _make_referee()
                    reset_scene(0, wall_now)
                    paused = False
                    pause_acc = 0.0

                elif event.key == pygame.K_RIGHT:
                    nxt = (scene_idx + 1) % TOTAL_SCENES
                    reset_scene(nxt, wall_now)

                elif event.key == pygame.K_LEFT:
                    prv = (scene_idx - 1) % TOTAL_SCENES
                    reset_scene(prv, wall_now)

        # ---- Scene time ----------------------------------------------------
        if paused:
            _draw_frame(
                screen,
                fr_renderer,
                fonts,
                geo,
                last_frame,
                last_ref,
                scene_idx,
                SCENES[scene_idx],
                # show time as it was when paused
                (pause_wall - scene_start_wall) - pause_acc,
                paused=True,
            )
            clock.tick(30)
            continue

        scene = SCENES[scene_idx]
        scene_t = (wall_now - scene_start_wall) - pause_acc

        # Force command at scene start (once per scene)
        if not scene_forced and scene.force_command is not None:
            referee.set_command(scene.force_command, timestamp=scene_t)
            scene_forced = True

        # Clamp to scene duration
        scene_t_clamped = min(scene_t, scene.duration)
        current_time = wall_now  # absolute time for referee cooldowns

        frame = _make_frame(scene, scene_t_clamped, current_time)
        ref_data = referee.step(frame, current_time=current_time)

        # Simulate StrategyRunner ball teleport: on STOP transition edge with
        # a designated_position, snap the ball there for the rest of the scene.
        if (
            scene.ball_teleports_on_stop
            and ref_data.referee_command == RefereeCommand.STOP
            and ref_data.designated_position is not None
            and prev_command != RefereeCommand.STOP
        ):
            teleport_pos = ref_data.designated_position

        # If a teleport is active, rebuild the frame with the ball at that position.
        if teleport_pos is not None and scene.ball_teleports_on_stop:
            tx, ty = teleport_pos
            teleported_ball = frame.ball.__class__(
                p=frame.ball.p.__class__(tx, ty, 0.0),
                v=frame.ball.v,
                a=frame.ball.a,
            )
            frame = dataclasses.replace(frame, ball=teleported_ball)

        prev_command = ref_data.referee_command
        last_frame = frame
        last_ref = ref_data

        _draw_frame(screen, fr_renderer, fonts, geo, frame, ref_data, scene_idx, scene, scene_t_clamped, paused=False)

        # ---- Auto-advance to next scene ------------------------------------
        if scene_t > scene.duration + 0.8:  # 0.8 s pause between scenes
            next_idx = scene_idx + 1
            if next_idx < TOTAL_SCENES:
                reset_scene(next_idx, wall_now)
            # else: stay on last scene

        clock.tick(60)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
