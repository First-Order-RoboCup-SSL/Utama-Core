# utama_core/strategy/rookie.py
"""`rookie`: the robot nearest the ball chases it and kicks it forward. The others follow the play."""

from utama_core.engine.context import TickContext
from utama_core.engine.strategy import Strategy
from utama_core.engine.tactic import BaseTactic, TacticTag
from utama_core.entities.data.vector import Vector2D
from utama_core.skills.go_to_ball import go_to_ball
from utama_core.skills.go_to_point import go_to_point
from utama_core.skills.kick_upfield import kick_upfield

LANES = [-2.0, -0.7, 0.7, 2.0]  # where the helpers stand across the pitch (y, metres)


class ChaseAndKick(BaseTactic[None]):
    tag = TacticTag.ATTACK

    def initial_mem(self):
        return None  # this tactic remembers nothing between ticks

    def tick(self, game, ctx, robot_ids, mem):
        if not robot_ids or game.ball is None:
            return {}, mem
        ball = game.ball.p
        robots = game.friendly_robots

        # The robot nearest the ball is the chaser.
        chaser = min(robot_ids, key=lambda i: robots[i].p.distance_to(ball))
        # +1 if our goal is on the right, -1 if it is on the left.
        back = 1 if game.field.my_goal_line[0][0] > 0 else -1

        commands = {}
        if robots[chaser].has_ball:
            commands[chaser] = kick_upfield(game, ctx.motion_controller, chaser)
        else:
            commands[chaser] = go_to_ball(game, ctx.motion_controller, chaser)

        # The helpers 1 m behind the ball, spread across the pitch.
        helpers = [i for i in robot_ids if i != chaser]
        x = ball.x - back * 0.5
        x = max(-3.3, min(3.3, x))  # stay out of the goal areas
        for i, y in zip(helpers, LANES):
            commands[i] = go_to_point(game, ctx.motion_controller, i, Vector2D(x, y))
        return commands, mem


def build(outfield_robot_ids, motion_controller):
    return Strategy(
        tactics={"chase": ChaseAndKick()},
        partitioner=Strategy.single_tactic_picker(lambda game, active: "chase"),
        outfield_robot_ids=outfield_robot_ids,
        ctx=TickContext(motion_controller=motion_controller),
    )
