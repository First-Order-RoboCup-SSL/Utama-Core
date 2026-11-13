"""In sim PVP `StrategyRunner._run_step` refines each tick's vision once, for our side, and
hands the opponent that frame mirrored. That is only right while the opponent's own position
and velocity refiners would have produced exactly the mirrored frame, which this pins tick by
tick: Kalman smoothing, a robot vanishing and coming back, and acceleration all included."""

import math

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.config.settings import MAX_GAME_HISTORY
from utama_core.data_processing.refiners import PositionRefiner, VelocityRefiner
from utama_core.entities.data.raw_vision import RawBallData, RawRobotData, RawVisionData
from utama_core.entities.game import Game, GameFrame, GameHistory
from utama_core.entities.game.field import Field
from utama_core.run.strategy_runner import _mirrored

DT = 1 / 60


def _vision(tick: int) -> list[RawVisionData]:
    t = tick * DT

    def robots(sign: float) -> list[RawRobotData]:
        out = []
        for i in range(6):
            if sign > 0 and i == 3 and 20 <= tick < 30:
                continue  # vanishes for 10 ticks: the refiner imputes it
            x = sign * (0.5 + 0.6 * i) + 0.4 * math.sin(1.3 * t + i)
            y = -1.5 + 0.6 * i + 0.3 * math.cos(0.9 * t * sign + i) + 0.002 * ((tick * (i + 3)) % 7)
            out.append(RawRobotData(i, x, y, math.remainder(0.7 * t + i, 2 * math.pi), 0.9))
        return out

    ball = RawBallData(1.2 * math.sin(0.8 * t), 0.5 * math.cos(1.1 * t), 0.0, 0.95)
    return [RawVisionData(t, robots(1.0), robots(-1.0), [ball], 0)]


def _side(is_yellow: bool, is_right: bool, first: GameFrame):
    position = PositionRefiner(STANDARD_FIELD_DIMS, filtering=True)
    position.start_filtering()
    history = GameHistory(MAX_GAME_HISTORY)
    field = Field(is_right, STANDARD_FIELD_DIMS, STANDARD_FIELD_DIMS.full_field_bounds)
    return position, VelocityRefiner(), history, Game(history, first, field=field)


def test_the_opponents_own_refinement_is_ours_mirrored():
    # As GameGater does: both starting frames come from our side's refiner, before filtering.
    gater = PositionRefiner(STANDARD_FIELD_DIMS, filtering=True)
    my_frame = gater.refine(GameFrame(0, True, True, {}, {}, None), _vision(0))
    opp_frame = gater.refine(GameFrame(0, False, False, {}, {}, None), _vision(0))
    assert opp_frame == _mirrored(my_frame)

    my_pos, my_vel, my_hist, my_game = _side(True, True, my_frame)
    opp_pos, opp_vel, opp_hist, opp_game = _side(False, False, opp_frame)
    for tick in range(1, 90):
        vision = _vision(tick)
        my_frame = my_vel.refine(my_hist, my_pos.refine(my_frame, vision))
        opp_frame = opp_vel.refine(opp_hist, opp_pos.refine(opp_frame, vision))

        assert opp_frame == _mirrored(my_frame), f"tick {tick}"
        assert list(opp_frame.friendly_robots) == list(_mirrored(my_frame).friendly_robots)
        my_game.add_game_frame(my_frame)
        opp_game.add_game_frame(opp_frame)

    assert any(r.a.x != 0 for r in my_frame.friendly_robots.values())  # acceleration really compared


def test_mirroring_twice_gives_the_frame_back():
    vision = _vision(5)
    frame = PositionRefiner(STANDARD_FIELD_DIMS, filtering=False).refine(GameFrame(0, True, True, {}, {}, None), vision)
    mirrored = _mirrored(frame)
    assert all(not r.is_friendly for r in mirrored.enemy_robots.values())
    assert mirrored.friendly_robots.keys() == frame.enemy_robots.keys()
    assert _mirrored(mirrored) == frame
