from typing import Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum


class RoleType(Enum):
    ATTACK = "attack"
    DEFEND = "defend"
    KEEPER = "keeper"
    BALL_PLACEMENT = "ball_placement"

    @classmethod
    def default(cls):
        return cls.ATTACK


@dataclass
class Action:
    action: str
    rating: int = 0

    def update_rating(self, rating: float):
        if rating >= 0:
            self.rating = rating
        else:
            raise ValueError("raiting value mist be non-negative")

    def reset(self):
        self.rating = 0


@dataclass
class Role:
    robot_id: int
    # temporary until we have setup formation
    role: RoleType = RoleType.default()
    suggested_action: Optional[Action] = None
    possible_actions: List[Action] = field(default_factory=list)
    sprt_rbt_ids: List[int] = field(default_factory=list)

    def __post_init__(self):
        self.change_role(self.role)

    def change_role(self, role: Union[RoleType, str]) -> None:
        if isinstance(role, RoleType):
            self.role = role
        elif isinstance(role, str) and role in RoleType._value2member_map_:
            self.role = RoleType(role)
        else:
            raise ValueError(f"Invalid role: {role}")

        self.possible_actions = self._get_actions_for_role()

    def _get_actions_for_role(self) -> List[Action]:
        # TODO: This list is currently not accurate
        role_actions = {
            "attack": [
                Action(action="pass"),
                Action(action="shoot_to_goal"),
                Action(action="go_to_ball"),
                Action(action="cross_ball"),
                Action(action="receive_then_score"),
            ],
            "defend": [
                Action(action="pass"),
                Action(action="man_mark"),
                Action(action="intercept_ball"),
                Action(action="tackle ball"),
                Action(action="clear ball"),
                Action(action="block"),
                Action(action="go_to_ball"),
                Action(action="receive_ball"),
            ],
        }
        return role_actions.get(self.role, [])

    def set_sprt_rbt_ids(self, sprt_rbt_ids: List[int]) -> None:
        self.sprt_rbt_ids = list(set(sprt_rbt_ids))  # Ensure unique IDs

    def update_rating(self, action_id: int, raiting: float):
        if isinstance(action_id, int):
            if 0 <= action_id < len(self.possible_actions):
                self.possible_actions[action_id].change_rating(raiting)
            else:
                raise ValueError(f"Invalid rating index: {action_id}")

    def reset_all_ratings(self):
        for action in self.possible_actions:
            action.reset()
