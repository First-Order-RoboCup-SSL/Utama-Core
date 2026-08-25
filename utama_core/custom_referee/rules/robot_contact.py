"""Colour-blind friendly/enemy robot-pair contact detection shared by
`PushingRule` and `CrashingRule` (SSL rulebook §8.4.1/§8.4.2).

Both rules are defined purely in terms of robot *position and velocity* at
the moment of body contact — neither the rulebook definitions nor this
detector consult `has_ball`. `has_ball` is dribbler-mouth contact (real IR
sensor on hardware); two robots can be pinned together, ball trapped
between their fronts, with neither dribbler ever touching it — exactly the
case that motivated keeping this contact-only, per a live ball-contest
deadlock investigated earlier (see docs/roadmap.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.robot import Robot

# Two robot bodies are "in contact" once their centre-to-centre distance
# closes to about their combined radii. Given a small tolerance for
# simulation/vision jitter around the exact geometric contact distance
# (2 * ROBOT_RADIUS = 0.18m) rather than requiring bodies to be
# interpenetrating or exactly osculating before a foul can ever be
# considered.
_CONTACT_MARGIN = 0.03  # metres
CONTACT_DISTANCE = 2 * ROBOT_RADIUS + _CONTACT_MARGIN


@dataclass(frozen=True)
class RobotPairContact:
    """One friendly/enemy robot pair currently in body contact."""

    friendly: Robot
    enemy: Robot
    distance: float
    # Unit vector from enemy's centre toward friendly's centre.
    direction_enemy_to_friendly: tuple[float, float]

    @property
    def closing_speed_friendly_into_enemy(self) -> float:
        """Component of the friendly robot's velocity directed at the enemy
        robot (metres/second, along the pair's connecting line).

        Positive means the friendly robot is moving *toward* the enemy
        (pushing/closing); negative means moving away.
        """
        ux, uy = self.direction_enemy_to_friendly
        return -(self.friendly.v.x * ux + self.friendly.v.y * uy)

    @property
    def closing_speed_enemy_into_friendly(self) -> float:
        """Same as `closing_speed_friendly_into_enemy`, from the enemy's side."""
        ux, uy = self.direction_enemy_to_friendly
        return self.enemy.v.x * ux + self.enemy.v.y * uy

    @property
    def projected_velocity_difference(self) -> float:
        """(v_enemy - v_friendly) projected onto the line connecting the two
        robots' positions — the exact quantity SSL rulebook §8.4.2's
        "Crashing" rule is defined on: "the difference of the speed vectors
        of both robots is taken and projected onto the line that is defined
        by the position of both robots."

        Sign convention is deliberately not meaningful on its own (the rule
        only cares about magnitude and which robot was faster) — see
        `crashing_rule.py` for how this combines with each robot's own
        closing speed to decide who was "faster."
        """
        ux, uy = self.direction_enemy_to_friendly
        dvx = self.enemy.v.x - self.friendly.v.x
        dvy = self.enemy.v.y - self.friendly.v.y
        return dvx * ux + dvy * uy


def find_robot_pair_contacts(game_frame: GameFrame) -> Iterator[RobotPairContact]:
    """Yield every friendly/enemy robot pair currently within `CONTACT_DISTANCE`
    of each other.

    Deliberately only pairs across teams — same-team contact isn't a foul
    under any rule in this section, and every SSL rulebook rule this
    detector backs (Pushing, Crashing) is specifically "an opponent robot."
    """
    for friendly in game_frame.friendly_robots.values():
        for enemy in game_frame.enemy_robots.values():
            dx = friendly.p.x - enemy.p.x
            dy = friendly.p.y - enemy.p.y
            distance = math.hypot(dx, dy)
            if distance > CONTACT_DISTANCE:
                continue
            if distance == 0.0:
                # Degenerate (perfectly overlapping) positions — no
                # well-defined connecting line to project onto. Should
                # never happen with real collision geometry; skip rather
                # than propagate a NaN direction.
                continue
            direction = (dx / distance, dy / distance)
            yield RobotPairContact(
                friendly=friendly,
                enemy=enemy,
                distance=distance,
                direction_enemy_to_friendly=direction,
            )
