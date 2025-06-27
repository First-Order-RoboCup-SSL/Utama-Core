class ProximityMap:
    pass


#     """
#     A proximity map that tracks the distance between robots and the ball.
#     """

#     def __init__(self, List):

#     def update(self, robot_key: ObjectKey, ball_key: ObjectKey, distance: float):
#         """
#         Update the distance between a robot and the ball.
#         """
#         self._distances[(robot_key, ball_key)] = distance

#     def get_distance(self, robot_key: ObjectKey, ball_key: ObjectKey) -> float:
#         """
#         Get the distance between a robot and the ball.
#         """
#         return self._distances.get((robot_key, ball_key), float("inf"))
