### UMMMM YESS, IDK WHATS GOING ON LOL ###

# import py_trees

# from config.starting_formation import LEFT_START_ONE, RIGHT_START_ONE
# from strategy.behaviour_trees.behaviours.skills_behaviours import GoToPointBehaviour

# from motion_planning.src.pid.pid import PID, TwoDPID

# class GoToStartupTargetIfPresent(robot_id: int):
#     START_FORMATION = RIGHT_START_ONE if present_future_game.current.my_team_is_right else LEFT_START_ONE


# def StartupBehaviour(n_robots: int):
#     selector = py_trees.composites.Parallel(name="StartupBehaviour")

#     for robot_id, robot_data in enumerate(present_future_game.current.friendly_robots.items()):
#         target_coords = START_FORMATION[robot_id]
#             commands[robot_id] = self._calculate_robot_velocities(
#                 robot_id, target_coords, present_future_game, face_ball=True
#             )

#     selector.add_child(GoToPointBehaviour())
#     return selector

#    def __init__(self, pid_oren: PID, pid_trans: TwoDPID):
#         self.pid_oren = pid_oren
#         self.pid_trans = pid_trans

#     def done():


#     def step(self, present_future_game: PresentFutureGame) -> Dict[int, RobotCommand]:

#         commands = {}

#         return commands

#     def _calculate_robot_velocities(
#         self,
#         robot_id: int,
#         target_coords: Tuple[float, float],
#         present_future_game: PresentFutureGame
#         face_ball=False,
#     ) -> RobotCommand:

#         ball_p = present_future_game.current.ball.p
#         current_p = present_future_game.current.friendly_robots[robot_id].p
#         target_oren=(ball_p - current_p).phi if face_ball else None
#         return go_to_point(
#             present_future_game.current,
#             self.pid_oren,
#             self.pid_trans,
#             robot_id,
#             target_coords,
#             target_oren
#         )
