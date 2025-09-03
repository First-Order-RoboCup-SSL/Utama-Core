from dataclasses import dataclass

from entities.data.vector import Vector3D


@dataclass(frozen=True)
class Ball:
    p: Vector3D
    v: Vector3D
    a: Vector3D

    def is_ball_in_goal(self, right_goal: bool) -> bool:
        ball_pos = self.p
        return (
            ball_pos.x < -self.field.half_length
            and (ball_pos.y < self.field.half_goal_width and ball_pos.y > -self.field.half_goal_width)
            and not right_goal
            or ball_pos.x > self.field.half_length
            and (ball_pos.y < self.field.half_goal_width and ball_pos.y > -self.field.half_goal_width)
            and right_goal
        )
