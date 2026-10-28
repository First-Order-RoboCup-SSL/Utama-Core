"""Per-side runtime state (`SideRuntime`) and the frame transforms used when the teams change
ends (`_turned`) or when the opponent's view of a frame is needed in sim PVP (`_mirrored`)."""

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Optional

from utama_core.data_processing.refiners import (
    PositionRefiner,
    RobotInfoRefiner,
    VelocityRefiner,
)
from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game, GameFrame, GameHistory, Robot
from utama_core.entities.game.field import FieldBounds
from utama_core.motion_planning.src.common.motion_controller import MotionController


def _turned(robot: Robot, bounds: FieldBounds) -> Robot:
    """`robot` half a turn about the centre spot, at rest: where it stands after changing ends.
    Kept within `bounds`: a robot in the run-off past a line comes back onto it, since the sim
    refuses to teleport a robot outside the field (a match crashed at half-time that way)."""
    (left, top), (right, bottom) = bounds.top_left, bounds.bottom_right
    return dataclasses.replace(
        robot,
        p=Vector2D(min(max(-robot.p.x, left), right), min(max(-robot.p.y, bottom), top)),
        v=Vector2D(0.0, 0.0),
        a=Vector2D(0.0, 0.0),
        orientation=math.remainder(robot.orientation + math.pi, 2 * math.pi),
    )


def _mirrored(frame: GameFrame) -> GameFrame:
    """`frame` as the other team sees it: the same absolute coordinates, friendly and enemy swapped."""

    def flip(robots: dict[int, Robot]) -> dict[int, Robot]:
        return {i: Robot(r.id, not r.is_friendly, r.has_ball, r.p, r.v, r.a, r.orientation) for i, r in robots.items()}

    return GameFrame(
        frame.ts,
        not frame.my_team_is_yellow,
        not frame.my_team_is_right,
        flip(frame.enemy_robots),
        flip(frame.friendly_robots),
        frame.ball,
        frame.referee,
    )


@dataclass(slots=True)
class SideRuntime:
    """Encapsulates all per-side (my team / opponent) runtime state.

    Args:
        strategy (AbstractStrategy): The strategy for this side.
        position_refiner (PositionRefiner): Position refiner for this side.
        velocity_refiner (VelocityRefiner): Velocity refiner for this side.
        robot_info_refiner (RobotInfoRefiner): Robot info refiner for this side.
        motion_controller (type[MotionController]): Motion controller factory for this side.
    """

    strategy: AbstractStrategy
    position_refiner: PositionRefiner
    velocity_refiner: VelocityRefiner
    robot_info_refiner: RobotInfoRefiner
    motion_controller: type[MotionController]
    control_scheme: str = "fpp"

    game: Optional[Game] = field(init=False, default=None)
    game_history: Optional[GameHistory] = field(init=False, default=None)
    current_game_frame: Optional[GameFrame] = field(init=False, default=None)
