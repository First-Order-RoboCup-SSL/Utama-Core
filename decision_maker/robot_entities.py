from typing import Optional, List

from decision_maker.role import Attack, Defend, Role

from enum import Enum

class RoleType(Enum):
    ATTACK = 1
    DEFEND = 2

class Friendly():
    def __init__(
        self,
        robot_id: int,
        role_id: Optional[int] = None,

    ):
        self._robot_id: int = robot_id
        self._has_ball: bool = False
        self._aggro_rating: float = 0
        self._sprt_rbt_ids: List[int] = None
        self._role = None

        if role_id != None and self._role == None:
            self.role = role_id

    @property
    def id(self) -> int:
        return self._id

    @property
    def sprt_rbt_ids(self) -> List[int]:
        return self._sprt_rbt_ids

    @sprt_rbt_ids.setter
    def sprt_rbt_ids(self, sprt_rbt_ids: List[int]):
        self._sprt_rbt_ids = list(set(sprt_rbt_ids))  # Ensure unique IDs

    @property
    def role(self) -> Role:
        return self._role

    @role.setter
    def role(self, input: RoleType):
        # TODO: docstring
        if input == RoleType.ATTACK:
            self._role = Attack()
        elif input == RoleType.DEFEND:
            self._role = Defend()

    @property
    def aggro_rating(self) -> float:
        return self._aggro_rating

    @aggro_rating.setter
    def aggro_rating(self, input: float):
        if input >= 0:
            self._aggro_rating = input
        else:
            raise ValueError("agro raiting value must be non-negative")


class Enemy():
    def __init__(self, robot_id: int):
        self._id = robot_id
        # TODO: add properties like danger raiting etc
        
    @property
    def id(self) -> int:
        return self._id

if __name__ == "__main__":
    # inital setup
    friendly = Friendly(0, RoleType.ATTACK)
    
    # atk role demo
    print(friendly.role.possible_actions)
    
    # change aggro rating
    friendly.aggro_rating = 0.5
    print(friendly.aggro_rating)
    
    # change supporting robot ids
    friendly.sprt_rbt_ids = [1, 2, 3, 4, 5, 5, 5]
    print(friendly.sprt_rbt_ids)
    
    # change role
    friendly.role = RoleType.DEFEND
    print(friendly.role.possible_actions)