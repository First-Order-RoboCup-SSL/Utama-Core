"""Referee view — attaches `CustomReferee` state/commands to a `DashboardServer`.

This is the direct successor to `custom_referee/gui.py`'s `_RefereeGUIServer`:
same state shape pushed over SSE, same `/command` semantics, but registered
as one channel/route pair on a shared server instead of owning a private
HTTP server. `CustomReferee.step()` still calls `notify()` on every tick —
this module just says what "notify" means now.
"""

from __future__ import annotations

import dataclasses
import json
import time
from typing import TYPE_CHECKING, Optional

from utama_core.dashboard.server import DashboardServer
from utama_core.entities.referee.referee_command import RefereeCommand

if TYPE_CHECKING:
    from utama_core.custom_referee import CustomReferee
    from utama_core.custom_referee.profiles.profile_loader import RefereeProfile

CHANNEL = "referee"


def attach(server: DashboardServer, referee: "CustomReferee", profile: "RefereeProfile") -> None:
    """Wire a `CustomReferee` into the shared dashboard server."""
    static_config = _build_static_config(profile)

    def _config_bytes() -> bytes:
        config = dict(static_config)
        g = referee.geometry
        config["geometry"] = {
            "half_length": g.half_length,
            "half_width": g.half_width,
            "half_goal_width": g.half_goal_width,
            "half_defense_depth": g.half_defense_depth,
            "half_defense_width": g.half_defense_width,
            "center_circle_radius": g.center_circle_radius,
            "goal_depth": g.goal_depth,
        }
        return json.dumps(config).encode()

    def _handle_command(payload: dict) -> None:
        cmd = RefereeCommand[payload["command"]]
        designated = payload.get("designated")
        if designated is not None and hasattr(referee, "force_command"):
            target = (float(designated[0]), float(designated[1]))
            error = placement_target_error(target, referee.geometry, referee.field_check)
            if error:
                raise ValueError(error)  # the page shows it; nothing is sent to the robots
            referee.force_command(cmd, time.time(), ball_placement_target=target)
        else:
            referee.set_command(cmd, time.time())

    server.add_route("/referee/config", _config_bytes)
    server.add_command_handler("/referee/command", _handle_command)

    referee.attach_dashboard_notifier(lambda *args: notify(server, *args))


def placement_target_error(target: tuple[float, float], geometry, field_check: Optional[dict]) -> Optional[str]:
    """Why a ball placement to `target` is refused, or None: it must lie inside the field lines
    and, with a camera, inside the area in play (`FieldCheck.payload`'s bounds), the part the
    camera sees; elsewhere the placing robots would leave vision."""
    x, y = target
    if abs(x) > geometry.half_length or abs(y) > geometry.half_width:
        return f"({x:.2f}, {y:.2f}) is outside the field"
    if field_check:
        x1, y1, x2, y2 = field_check["bounds"]
        if not (x1 <= x <= x2 and y1 <= y <= y2):
            return f"({x:.2f}, {y:.2f}) is outside the area in play"
    return None


def notify(
    server: DashboardServer,
    ref_data,
    game_frame=None,
    tactic_status=None,
    robot_feedback_data=None,
    field_check=None,
) -> None:
    server.notify(CHANNEL, _serialise_state(ref_data, game_frame, tactic_status, robot_feedback_data, field_check))


# Millimetres, mrad and cm/s: finer than anything drawn, and a full float's 17 digits made a
# match's replay ~7x its size on disk.
def _mm(value: float) -> float:
    return round(value, 3)


def _cm(value: float) -> float:
    return round(value, 2)


def _serialise_robots(game_frame) -> dict:
    if game_frame is None:
        return {"friendly": [], "enemy": []}

    def _robot_list(robots_dict):
        return [
            {
                "id": r.id,
                "x": _mm(r.p.x),
                "y": _mm(r.p.y),
                "vx": _cm(r.v.x),
                "vy": _cm(r.v.y),
                "orientation": _mm(r.orientation),
                "has_ball": r.has_ball,
            }
            for r in robots_dict.values()
        ]

    return {
        "friendly": _robot_list(game_frame.friendly_robots),
        "enemy": _robot_list(game_frame.enemy_robots),
    }


def _serialise_ball(game_frame):
    if game_frame is None or game_frame.ball is None:
        return None
    ball = game_frame.ball
    return {"x": _mm(ball.p.x), "y": _mm(ball.p.y), "vx": _cm(ball.v.x), "vy": _cm(ball.v.y)}


def _serialise_state(ref_data, game_frame=None, tactic_status=None, robot_feedback_data=None, field_check=None) -> dict:
    designated = None
    if ref_data.designated_position is not None:
        try:
            designated = list(ref_data.designated_position)
        except TypeError:
            designated = [ref_data.designated_position.x, ref_data.designated_position.y]

    return {
        "command": ref_data.referee_command.name,
        "next_command": (ref_data.next_command.name if ref_data.next_command else None),
        "stage": ref_data.stage.name,
        "stage_time_left": ref_data.stage_time_left or 0.0,
        "yellow_score": ref_data.yellow_team.score,
        "blue_score": ref_data.blue_team.score,
        "designated": designated,
        "status_message": ref_data.status_message,
        "ts": getattr(game_frame, "ts", None),  # game time, for the Live timeline
        "my_team_is_right": getattr(game_frame, "my_team_is_right", False),
        "my_team_is_yellow": getattr(game_frame, "my_team_is_yellow", True),
        "robots": _serialise_robots(game_frame),
        "ball": _serialise_ball(game_frame),
        "tactic_status": tactic_status or {},
        "robot_feedback": robot_feedback_data or [],
        "field_check": field_check,
    }


def _build_static_config(profile: "RefereeProfile") -> dict:
    r = profile.rules
    gm = profile.game
    return {
        "profile_name": profile.profile_name,
        # Whether any rule can stop play and set up a restart; the page hides the restart
        # buttons when none can (goal detection alone only stops play).
        "restarts": any(getattr(r, f.name).enabled for f in dataclasses.fields(r) if f.name != "goal_detection"),
        "rules": {
            "goal_detection": {
                "enabled": r.goal_detection.enabled,
                "cooldown_seconds": r.goal_detection.cooldown_seconds,
            },
            "out_of_bounds": {
                "enabled": r.out_of_bounds.enabled,
                "free_kick_assigner": r.out_of_bounds.free_kick_assigner,
            },
            "defense_area": {
                "enabled": r.defense_area.enabled,
                "max_defenders": r.defense_area.max_defenders,
                "attacker_infringement": r.defense_area.attacker_infringement,
            },
            "keep_out": {
                "enabled": r.keep_out.enabled,
                "radius_meters": r.keep_out.radius_meters,
                "violation_persistence_frames": r.keep_out.violation_persistence_frames,
            },
        },
        "game": {
            "half_duration_seconds": gm.half_duration_seconds,
            "kickoff_team": gm.kickoff_team,
            "force_start_after_goal": gm.force_start_after_goal,
            "stop_duration_seconds": gm.stop_duration_seconds,
            "auto_advance": {
                "stop_to_next_command": gm.auto_advance.stop_to_next_command,
                "prepare_kickoff_to_normal": gm.auto_advance.prepare_kickoff_to_normal,
                "prepare_penalty_to_normal": gm.auto_advance.prepare_penalty_to_normal,
                "direct_free_to_normal": gm.auto_advance.direct_free_to_normal,
                "ball_placement_to_next": gm.auto_advance.ball_placement_to_next,
                "normal_start_to_force": gm.auto_advance.normal_start_to_force,
            },
        },
    }
