### UMMMM YESS, IDK WHATS GOING ON LOL ###

# from typing import Tuple
# import py_trees

# from motion_planning.src.pid.pid import PID, TwoDPID
# from robot_control.src.skills import go_to_point
# from strategy.behaviour_trees.behaviours.robocup_behaviour import RobocupBehaviour
# from vector import VectorObject2D

# class StepTowardsPointBehaviour(RobocupBehaviour):
#     def __init__(self):
#         super().__init__(name="StepTowardsPointBehaviour")
#         self.blackboard.register_key(key="GoToPointBehaviour/robot_id", access=py_trees.common.Access.READ)
#         self.blackboard.register_key(key="GoToPointBehaviour/target_coords", access=py_trees.common.Access.READ)
#         self.blackboard.register_key(key="GoToPointBehaviour/target_oren", access=py_trees.common.Access.READ)
#         self.blackboard.register_key(key="GoToPointBehaviour/dribbling", access=py_trees.common.Access.READ)

#     def update(self) -> py_trees.common.Status:
#         command = go_to_point(
#             self.blackboard.present_future_game.current,
#             self.blackboard.pid_oren,
#             self.blackboard.pid_trans,
#             self.blackboard.GoToPointBehaviour.robot_id,
#             self.blackboard.GoToPointBehaviour.target_coords,
#             self.blackboard.GoToPointBehaviour.target_oren,
#             self.blackboard.GoToPointBehaviour.dribbling
#         )

#         self.blackboard.robot_controller.add_robot_commands(command, self.blackboard.robot_id)
#         return py_trees.common.Status.SUCCESS

# class IsAtPointBehaviour(RobocupBehaviour):
#     def __init__(self, robot_id: int, target_coords: VectorObject2D, target_oren: float, tolerance: float=0.05):
#         super().__init__(name="IsAtPointBehaviour")
#         self.blackboard.register_key(key="GoToPointBehaviour/robot_id", access=py_trees.common.Access.READ)
#         self.blackboard.register_key(key="GoToPointBehaviour/target_coords", access=py_trees.common.Access.READ)
#         self.blackboard.register_key(key="GoToPointBehaviour/target_oren", access=py_trees.common.Access.READ)
#         self.blackboard.register_key(key="GoToPointBehaviour/tolerance", access=py_trees.common.Access.READ)

#     def update(self):
#         current_position = self.present_future_game.current.friendly_robots[self.robot_id].p
#         if abs(current_position - self.target_coords) < self.tolerance 
#             return py_trees.common.Status.SUCCESS
#         else:
#             return py_trees.common.Status.FAILURE

# def GoToPointBehaviour(pid_oren: PID, pid_trans: TwoDPID, robot_id: int, target_coords: VectorObject2D, target_oren: float, dribbling: bool = False, tolerance: float=0.05):
#     blackboard = py_trees.blackboard.Client(name="GoToPointBehaviour")

# robot_id: int, target_coords: Tuple[float, float], target_oren: float, dribbling: bool = False


#     set_blackboard_variable = py_trees.behaviours.SetBlackboardVariable(
#         name="Set Nested",
#         variable_name="nested",
#         variable_value=Nested(),
#         overwrite=True,
#     )
    
#     selector = py_trees.composites.Selector(name="GoToPointBehaviour")
#     selector.add_child(IsAtPointBehaviour(robot_id, target_coords, target_oren, tolerance))
#     selector.add_child(StepTowardsPointBehaviour(pid_oren, pid_trans, robot_id, target_coords, target_oren, dribbling))

#     return selector
