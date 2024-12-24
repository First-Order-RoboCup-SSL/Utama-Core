# RSoccer SSL Simulator Environment

An alternative light-weight simulator platform that can be used in conjunction with grSim. Adapted from a RL `gymnasium` format simulator created by RoboCin. We have also added custom functions in the simulator for visualisation and debugging.

This simulator works directly with RSimRobotController and RSimController (similar to GRSimRobotController and GRSimController).

Currently only `SSLStandardEnv` has been tested. More environments can be created for more specific testing.

## Documentation

```python
class SSLStandardEnv(
    field_type: int = 1,
    render_mode: str = "human",
    n_robots_blue: int = 6,
    n_robots_yellow: int = 6,
    time_step: float = TIMESTEP,
    blue_starting_formation: list[tuple] = None,
    yellow_starting_formation: list[tuple] = None
)
```
##### Description:
    Environment stripped to be a lightweight simulator for testing and development.

    args:
        field_type
        Num
        0       Divison A pitch
        1       Division B pitch
        2       HW Challenge

        blue/yellow_starting_formation
        Type: List[Tuple[float, float, float]]
        Description:
            list of (x, y, theta) coords for each robot to spawn in (in meters and radians).
            See the default BLUE_START_ONE/YELLOW_START_ONE for reference.
    Observation:
        Type: Tuple[FrameData, List[RobotInfo], List[RobotInfo]]
        Num     Item
        0       contains position info of ball and robots on the field
        1       contains RobotInfo data (robot.has_ball) for yellow_robots
        2       contains RobotInfo data (robot.has_ball) for blue_robots

    Actions:
        Type: Box(5, )
        Num     Action
        0       id 0 Blue Global X Direction Speed (max set by self.max_v)
        1       id 0 Blue Global Y Direction Speed
        2       id 0 Blue Angular Speed (max set by self.max_w)
        3       id 0 Blue Kick x Speed (max set by self.kick_speed_x)
        4       id 0 Blue Dribbler (true if positive)

### Ball and Robot movement (interfaces with RSimController)

Note that the teleport commands are sent together with the rest of the robot commands ONLY at the end of the frame.

#### Teleport Ball

```python
def teleport_ball(
    x: float
    y: float
    vx: float = 0
    vy: float = 0
)
```
##### Description:
    teleports ball to new position in meters, meters per second.

#### Teleport Robot

```python
def teleport_robot(
    is_team_yellow: bool,
    robot_id: bool,
    x: float,
    y: float,
    theta: float = None,
)
```
##### Description:
    teleport robot to new position in meters, radians.

### Overlay Drawing

Functions used to draw points/lines/polygons of interest. Used for visualisation and debugging. Note that these are also sent together with robot commands ONLY at the end of the frame.

For the list of available colours, see `rsoccer_simulator/src/Render/utils.py`.

#### Draw Point

```python
def draw_point(
    x: float, 
    y: float, 
    color: str = "RED", 
    width: float = 0.05
)
```
##### Description:
    Draws a point as an overlay.

    Parameters:
    -----------
    points : list[tuple[float, float]]
        A list of tuples, where each tuple represents a point (x, y) in 2D space.
    color : str, optional
        The color of the line. Default is "RED". See 
    width : float, optional
        The radius of the point. Default is 1. Cannot be less than 1.

#### Draw Line

```python
def draw_line(
    points: list[tuple[float, float]],
    color: str = "RED", 
    width: float = 1
)
```
##### Description:
    Draws a line as an overlay using the first and last point in a list of points.

    Parameters:
    -----------
    points : list[tuple[float, float]]
        A list of tuples, where each tuple represents a point (x, y) in 2D space.
    color : str, optional
        The color of the line. Default is "RED".
    width : float, optional
        The width of the line. Default is 1. Cannot be less than 1.

#### Draw Polygon

```python
def draw_polygon(
    points: list[tuple[float, float]], 
    color: str = "RED", 
    width: float = 1
)
```
##### Description:
    Draws a polygon as an overlay using a list of points.

    Parameters:
    -----------
    points : list[tuple[float, float]]
        A list of tuples, where each tuple represents a point (x, y) in 2D space.
    color : str, optional
        The color of the line. Default is "RED".
    width : float, optional
        The width of the line. Default is 1. Cannot be less than 1.

## Reference

```
@InProceedings{10.1007/978-3-030-98682-7_14,
author="Martins, Felipe B.
and Machado, Mateus G.
and Bassani, Hansenclever F.
and Braga, Pedro H. M.
and Barros, Edna S.",
editor="Alami, Rachid
and Biswas, Joydeep
and Cakmak, Maya
and Obst, Oliver",
title="rSoccer: A Framework for Studying Reinforcement Learning in Small and Very Small Size Robot Soccer",
booktitle="RoboCup 2021: Robot World Cup XXIV",
year="2022",
publisher="Springer International Publishing",
address="Cham",
pages="165--176",
abstract="Reinforcement learning is an active research area with a vast number of applications in robotics, and the RoboCup competition is an interesting environment for studying and evaluating reinforcement learning methods. A known difficulty in applying reinforcement learning to robotics is the high number of experience samples required, being the use of simulated environments for training the agents followed by transfer learning to real-world (sim-to-real) a viable path. This article introduces an open-source simulator for the IEEE Very Small Size Soccer and the Small Size League optimized for reinforcement learning experiments. We also propose a framework for creating OpenAI Gym environments with a set of benchmarks tasks for evaluating single-agent and multi-agent robot soccer skills. We then demonstrate the learning capabilities of two state-of-the-art reinforcement learning methods as well as their limitations in certain scenarios introduced in this framework. We believe this will make it easier for more teams to compete in these categories using end-to-end reinforcement learning approaches and further develop this research area.",
isbn="978-3-030-98682-7"
}
```