from typing import List
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class Action:
    id: int
    name: str
    rating: int = field(init=False, default=0)

    def reset(self):
        self.rating = 0


class Role:
    def __init__(self, id, name):
        self.id: int = id
        self.name: str = name
        self.possible_actions: List[Action] = None

    def update_rating(self, action_id: int, raiting: float):
        if isinstance(action_id, int):
            if 0 <= action_id < len(self.possible_actions):
                self.possible_actions[action_id].rating = raiting
            else:
                raise ValueError(f"Invalid rating index: {action_id}")

    def reset_all_ratings(self):
        for action in self.possible_actions:
            action.reset()

    def get_suggested_action(self):
        max_raiting_action = {"action_id": None, "max_raiting": 0}
        for action in self.possible_actions:
            if action.rating > max_raiting_action["max_raiting"]:
                # TODO: change action.name or action.id both works depending on what is better
                max_raiting_action["action_id"] = action.name
        return max_raiting_action["action_id"]


@dataclass
class Attack(Role):
    def __post_init__(self):
        super().__init__(id=0, name="attacker")
        self.possible_actions = (
            [
                Action(0, name="pass"),
                Action(1, name="shoot_to_goal"),
                Action(2, name="go_to_ball"),
                Action(3, name="cross_ball"),
                Action(4, name="receive_then_score"),
            ],
        )


@dataclass
class Defend(Role):
    def __post_init__(self):
        super().__init__(id=1, name="defender")
        self.possible_actions = [
            Action(0, name="pass"),
            Action(1, name="man_mark"),
            Action(2, name="intercept_ball"),
            Action(3, name="tackle ball"),
            Action(4, name="clear ball"),
            Action(5, name="block"),
            Action(6, name="go_to_ball"),
            Action(7, name="receive_ball"),
        ]


if __name__ == "__main__":
    ROLES: List[Role] = [Defend(), Attack()]
    role = ROLES[0]
    role.possible_actions[2].rating = 5
    logger.debug(f"Role name: {role.name}")
    logger.debug(f"Role suggested action: {role.get_suggested_action()}")
