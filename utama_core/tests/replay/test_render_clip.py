"""Tests for `utama_core.replay.render_clip`."""

from __future__ import annotations

import pickle
import shutil
import subprocess
from typing import Optional

import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.entities.data.referee import RefereeData
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, GameFrame, Robot
from utama_core.entities.game.team_info import TeamInfo
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage
from utama_core.replay.entities import ReplayMetadata
from utama_core.replay.render_clip import (
    _clip_fps,
    camera_windows,
    displayed_ball,
    render_clip,
)

FIELD_HALF_X = STANDARD_FIELD_DIMS.full_field_half_length + STANDARD_FIELD_DIMS.goal_depth
FIELD_HALF_Y = STANDARD_FIELD_DIMS.full_field_half_width


def _referee(command: RefereeCommand) -> RefereeData:
    return RefereeData(
        source_identifier=None,
        time_sent=0.0,
        time_received=0.0,
        referee_command=command,
        referee_command_timestamp=0.0,
        stage=Stage.NORMAL_FIRST_HALF,
        stage_time_left=0.0,
        blue_team=TeamInfo(name="blue", goalkeeper=0),
        yellow_team=TeamInfo(name="yellow", goalkeeper=0),
    )


def _frame(ts: float, ball_x: float, command: Optional[RefereeCommand] = None) -> GameFrame:
    robot = Robot(
        id=1,
        is_friendly=True,
        has_ball=False,
        p=Vector2D(ball_x, 0.3),
        v=Vector2D(0, 0),
        a=Vector2D(0, 0),
        orientation=0.0,
    )
    return GameFrame(
        ts=ts,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots={1: robot},
        enemy_robots={},
        ball=Ball(p=Vector3D(ball_x, 0, 0), v=Vector3D(0, 0, 0), a=Vector3D(0, 0, 0)),
        referee=_referee(command) if command is not None else None,
    )


def _assert_inside_field(window):
    cx, cy, half_w, half_h = window
    assert cx - half_w >= -FIELD_HALF_X - 1e-9 and cx + half_w <= FIELD_HALF_X + 1e-9
    assert cy - half_h >= -FIELD_HALF_Y - 1e-9 and cy + half_h <= FIELD_HALF_Y + 1e-9


def test_follow_camera_clamps_to_field_edge_when_ball_is_in_the_corner():
    windows = camera_windows([(FIELD_HALF_X, FIELD_HALF_Y)] * 500, smoothing=1.0)
    cx, cy, half_w, half_h = windows[-1]
    # Pinned exactly against the corner, not past it and not short of it.
    assert cx + half_w == pytest.approx(FIELD_HALF_X)
    assert cy + half_h == pytest.approx(FIELD_HALF_Y)
    for w in windows:
        _assert_inside_field(w)


def test_follow_camera_larger_than_field_shrinks_to_fit():
    windows = camera_windows([(3.0, 2.0)], half_extent=(100.0, 100.0))
    assert windows[0] == pytest.approx((0.0, 0.0, FIELD_HALF_X, FIELD_HALF_Y))


def test_follow_camera_holds_last_ball_position_through_gaps():
    windows = camera_windows([None, (1.0, 0.5), None, None], smoothing=1.0)
    # Leading gap takes the first known position, trailing gap holds the last — never the origin.
    assert [w[:2] for w in windows] == [pytest.approx((1.0, 0.5))] * 4


def test_follow_camera_pans_smoothly_instead_of_jumping():
    windows = camera_windows([(0.0, 0.0), (2.0, 0.0)], smoothing=0.25)
    assert windows[1][0] == pytest.approx(0.5)


def test_full_camera_shows_whole_field_regardless_of_ball():
    windows = camera_windows([(4.0, 2.0), None], camera="full")
    assert windows == [(0.0, 0.0, FIELD_HALF_X, FIELD_HALF_Y)] * 2


def _frames_moving_at(
    speeds_mps: list[float], dt: float = 1 / 60, commands: Optional[list[RefereeCommand]] = None
) -> list[GameFrame]:
    """Frames at `dt` spacing whose ball moves `speeds_mps[i] * dt` along x between frame i and i+1.

    `commands[i]`, if given, is frame i's referee command (one more entry than `speeds_mps`).
    """
    commands = commands or [None] * (len(speeds_mps) + 1)
    x, frames = 0.0, [_frame(ts=0.0, ball_x=0.0, command=commands[0])]
    for i, speed in enumerate(speeds_mps, start=1):
        x += speed * dt
        frames.append(_frame(ts=i * dt, ball_x=x, command=commands[i]))
    return frames


def _hidden(frames: list[GameFrame], **kwargs) -> list[bool]:
    positions, overridden = displayed_ball(frames, **kwargs)
    return [o and p is None for p, o in zip(positions, overridden)]


# The decaying slide rsim replays show for a teleport: spike, then the vision filter's smoothing.
_SLIDE = [68.0, 27.4, 11.1, 4.7, 5.5, 3.7, 0.9, 0.4, 0.7, 0.4, 0.1]


def test_teleport_threshold_sits_between_legal_kick_and_teleport():
    assert not any(displayed_ball(_frames_moving_at([7.9]))[1])
    assert displayed_ball(_frames_moving_at([8.1]))[1] == [False, True]


def test_sustained_max_legal_kick_is_never_overridden():
    positions, overridden = displayed_ball(_frames_moving_at([6.5] * 120))
    assert not any(overridden)
    assert all(p is not None for p in positions)


def test_teleport_outside_placement_hides_the_slide_for_hide_window_then_shows_ball():
    frames = _frames_moving_at([0.0] + _SLIDE + [0.0] * 6)
    mask = _hidden(frames, hide_s=0.15)
    # Hidden from the 68 m/s jump (lands on frame 2) until 0.15s after the slide's last
    # >8 m/s tick (11.1 m/s, lands on frame 4) — the window extends, it doesn't restart late.
    first_spike_ts, last_spike_ts = frames[2].ts, frames[4].ts
    assert mask == [first_spike_ts <= f.ts <= last_spike_ts + 0.15 for f in frames]
    assert mask[2] and not mask[-1]


def test_teleport_during_ball_placement_holds_ball_where_it_went_out_until_placement_ends():
    placement, free_kick = RefereeCommand.BALL_PLACEMENT_YELLOW, RefereeCommand.DIRECT_FREE_YELLOW
    speeds = [1.5, 1.5] + _SLIDE + [0.0] * 5
    n_placement = 1 + len(speeds) - 3  # every frame but the last 3 is in placement
    frames = _frames_moving_at(speeds, commands=[placement] * n_placement + [free_kick] * 3)
    positions, overridden = displayed_ball(frames)

    out_x = frames[2].ball.p.x  # last position before the 68 m/s jump lands on frame 3
    assert positions[:3] == [(f.ball.p.x, 0.0) for f in frames[:3]]
    # Held at the out-of-play spot, never drawn at any point of the slide, for all of placement...
    assert positions[3:n_placement] == [(out_x, 0.0)] * (n_placement - 3)
    assert overridden[3:n_placement] == [True] * (n_placement - 3)
    # ...then the real (placed) ball the moment the restart command arrives.
    assert positions[n_placement:] == [(f.ball.p.x, 0.0) for f in frames[n_placement:]]
    assert not any(overridden[n_placement:])


def test_teleport_during_live_play_is_hidden_not_held():
    # rsim also teleports on a STOP -> FORCE_START recovery; holding the ball there would freeze it mid-play.
    frames = _frames_moving_at([0.0] + _SLIDE + [0.0] * 6, commands=[RefereeCommand.FORCE_START] * (len(_SLIDE) + 8))
    positions, overridden = displayed_ball(frames)
    assert all(p is None for p, o in zip(positions, overridden) if o)
    assert overridden[2] and not overridden[-1]


def test_follow_camera_snaps_after_a_cut_but_pans_after_a_plain_gap():
    positions = [(0.0, 0.0), None, (1.0, 0.0)]  # stays inside the clamp range of the default window
    panned = camera_windows(positions, smoothing=0.25)
    snapped = camera_windows(positions, smoothing=0.25, cuts=[False, True, False])
    assert panned[2][0] == pytest.approx(0.25)
    assert snapped[1][0] == pytest.approx(0.0)  # holds through the cut
    assert snapped[2][0] == pytest.approx(1.0)  # then jumps, no pan across the pitch


def test_clip_fps_matches_replay_tick_rate():
    frames = [_frame(ts=i / 60, ball_x=0.0) for i in range(10)]
    assert _clip_fps(frames) == 60
    assert _clip_fps(frames[:1]) == 60  # single frame: fallback, not a divide-by-zero


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
@pytest.mark.parametrize("camera", ["follow", "full"])
def test_render_clip_writes_mp4_with_one_video_frame_per_replay_frame(tmp_path, camera):
    replay = tmp_path / "match.pkl"
    with open(replay, "wb") as f:
        pickle.dump(ReplayMetadata(my_team_is_yellow=True, exp_friendly=1, exp_enemy=0), f)
        for i in range(12):
            pickle.dump(_frame(ts=i * 0.1, ball_x=i * 0.3), f)
    out = tmp_path / "clip.mp4"

    assert render_clip(replay, 0.0, 1.1, out, camera=camera, size=(320, 200)) == str(out)

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames,r_frame_rate",
            "-of",
            "csv=p=0",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe.stdout.strip() == "10/1,12"


def test_render_clip_raises_on_empty_window(tmp_path):
    replay = tmp_path / "match.pkl"
    with open(replay, "wb") as f:
        pickle.dump(ReplayMetadata(my_team_is_yellow=True, exp_friendly=1, exp_enemy=0), f)
        pickle.dump(_frame(ts=0.0, ball_x=0.0), f)
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    with pytest.raises(ValueError):
        render_clip(replay, 5.0, 6.0, tmp_path / "empty.mp4")
