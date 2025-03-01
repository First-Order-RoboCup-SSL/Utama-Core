


from unittest.mock import MagicMock
import vector
from entities.data.raw_vision import RawBallData, RawRobotData, RawVisionData
from entities.data.vision import VisionBallData, VisionRobotData
from entities.game.ball import Ball
from entities.game.game import Game
from entities.game.robot import Robot
from refiners.position import PositionRefiner


def test_combining_single_team_combines_single_robot():
    zv = vector.obj(x=0, y=0)
    game_robots = {
        0: Robot(0, True, False, zv, zv,zv, 0)
    }
    vision_robots = [VisionRobotData(0, 1,2,3)]
    result = PositionRefiner._combine_single_team_positions(game_robots, vision_robots, friendly=True)

    assert len(result) == 1
    rb = result[0]
    assert rb.p.x == 1
    assert rb.p.y == 2
    assert rb.orientation == 3


def test_combining_with_robot_not_in_game_adds():
    zv = vector.obj(x=0, y=0)
    game_robots = {
        0: Robot(0, True, False, zv, zv,zv, 0)
    }
    vision_robots = [VisionRobotData(1, 1,2,3)]
    result = PositionRefiner._combine_single_team_positions(game_robots, vision_robots, friendly=True)

    assert len(result) == 2
    rb = result[1]
    assert rb.p.x == 1
    assert rb.p.y == 2
    assert rb.orientation == 3

def test_get_most_confident_ball():
    balls = [VisionBallData(0,0, 0, 0.2), VisionBallData(1,2,3, 0.8)]
    res = PositionRefiner._get_most_confident_ball(balls)

    assert res.p.x == 1
    assert res.p.y == 2
    assert res.p.z == 3

def test_most_confident_with_empty():
    res = PositionRefiner._get_most_confident_ball([])
    assert res is None

def rfac(id,is_friendly, x, y) -> Robot:
    return Robot(id=id, is_friendly=is_friendly, has_ball=False, p=vector.obj(x=x, y=y),
                 v = vector.obj(x=0, y=0),
                 a = vector.obj(x=0, y=0), orientation=0)

def bfac(x, y) -> Ball:
    return Ball(vector.obj(x=x, y=y), vector.obj(x=0, y=0), vector.obj(x=0, y=0))

def test_refine_for_yellow():
    friendly = {0:rfac(0, True, 0,0)}
    enemy = {}
    raw_yellow = [RawRobotData(0,-1,-10, 0, 1)]
    raw_blue = [RawRobotData(0, -100,-1000, 0, 1)]
    raw_balls = [RawBallData(0,0,0, 0)]
    raw_vision_data_cam1 = RawVisionData(0,raw_yellow, raw_blue, raw_balls, 0)
    raw_vision_data_cam2 = RawVisionData(0,raw_yellow, raw_blue, raw_balls, 1)
    p = PositionRefiner()
    g = Game(0,True, True, friendly, enemy, bfac(0,0))
    result = p.refine(g, [raw_vision_data_cam1, raw_vision_data_cam2])
    fr = result.friendly_robots[0]
    er = result.enemy_robots[0]
    assert fr.p.x == -1
    assert fr.p.y == -10

    assert er.p.x == -100
    assert er.p.y == -1000

def test_refine_for_multiple_yellow():
    friendly = {0:rfac(0, True, 0,0)}
    raw_yellow = [RawRobotData(0,-1,-10, 0, 1), RawRobotData(1,-2,-20, 0,1)]
    raw_balls = [RawBallData(0,0,0, 0)]
    raw_vision_data_cam1 = RawVisionData(0,raw_yellow, [], raw_balls, 0)
    raw_vision_data_cam2 = RawVisionData(0,raw_yellow, [], raw_balls, 1)
    p = PositionRefiner()
    g = Game(0,True, True, friendly, {}, bfac(0,0))
    result = p.refine(g, [raw_vision_data_cam1, raw_vision_data_cam2])

    assert len(result.friendly_robots) == 2
    for i in range(2):
        fr = result.friendly_robots[i]
        assert fr.p.x == raw_yellow[i].x
        assert fr.p.y == raw_yellow[i].y
        assert fr.orientation == raw_yellow[i].orientation



def test_refine_for_blue():
    friendly = {0:rfac(0, False, 0,0)}
    enemy = {}
    raw_yellow = [RawRobotData(0,-1,-10, 0, 1)]
    raw_blue = [RawRobotData(0, -100,-1000, 0, 1)]
    raw_balls = [RawBallData(0,0,0, 0)]
    raw_vision_data_cam1 = RawVisionData(0,raw_yellow, raw_blue, raw_balls, 0)
    raw_vision_data_cam2 = RawVisionData(0,raw_yellow, raw_blue, raw_balls, 1)
    p = PositionRefiner()
    g = Game(0,False, True, friendly, enemy, bfac(0,0))
    result = p.refine(g, [raw_vision_data_cam1, raw_vision_data_cam2])
    fr = result.friendly_robots[0]
    er = result.enemy_robots[0]
    assert fr.p.x == -100
    assert fr.p.y == -1000

    assert er.p.x == -1
    assert er.p.y == -10



if __name__ == "__main__":
    test_combining_single_team_combines_single_robot()
    test_combining_with_robot_not_in_game_adds()
    test_get_most_confident_ball()
    test_most_confident_with_empty()
    test_refine_for_yellow()
    test_refine_for_blue()
    test_refine_for_multiple_yellow()
