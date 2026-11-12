import pytest

from utama_core.data_processing.refiners.robot_info import (
    _BALL_CAPTURE_DIST,
    CONTACT_GRACE_S,
    RobotInfoRefiner,
)
from utama_core.entities.data.command import RobotResponse
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.robot import Robot


def _make_robot(robot_id: int, x: float, y: float, has_ball: bool = False) -> Robot:
    return Robot(
        id=robot_id,
        is_friendly=True,
        has_ball=has_ball,
        p=Vector2D(x, y),
        v=None,
        a=None,
        orientation=0,
    )


def _make_enemy_robot(robot_id: int, x: float, y: float, has_ball: bool = False) -> Robot:
    return Robot(
        id=robot_id,
        is_friendly=False,
        has_ball=has_ball,
        p=Vector2D(x, y),
        v=None,
        a=None,
        orientation=0,
    )


def _make_frame(robots: dict, ball_pos=None) -> GameFrame:
    ball = Ball(Vector3D(*ball_pos, 0), Vector3D(0, 0, 0), None) if ball_pos else None
    return GameFrame(
        ts=0.0,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots=robots,
        enemy_robots={},
        ball=ball,
    )


# ---------------------------------------------------------------------------
# Default behaviour (trusted_ir_robots=None — trust all)
# ---------------------------------------------------------------------------


def test_default_uses_ir_sensor_true():
    refiner = RobotInfoRefiner()
    frame = _make_frame({0: _make_robot(0, 0.0, 0.0, has_ball=False)}, ball_pos=(1.0, 0.0))
    result = refiner.refine(frame, [RobotResponse(id=0, has_ball=True)])
    assert result.friendly_robots[0].has_ball is True


def test_default_uses_ir_sensor_false():
    refiner = RobotInfoRefiner()
    # Robot is right on top of the ball but IR says False — should respect IR.
    frame = _make_frame({0: _make_robot(0, 0.0, 0.0, has_ball=True)}, ball_pos=(0.0, 0.0))
    result = refiner.refine(frame, [RobotResponse(id=0, has_ball=False)])
    assert result.friendly_robots[0].has_ball is False


def test_empty_responses_returns_frame_unchanged():
    refiner = RobotInfoRefiner()
    frame = _make_frame({0: _make_robot(0, 0.0, 0.0, has_ball=True)})
    assert refiner.refine(frame, []) is frame
    assert refiner.refine(frame, None) is frame


def test_untrusted_robot_inferred_even_when_response_dropped():
    # If a broken robot sends no packet at all, has_ball must still be inferred
    # from vision proximity rather than left frozen at the previous value.
    refiner = RobotInfoRefiner(trusted_ir_robots=frozenset())  # no robots trusted
    # Robot is close to ball; previous has_ball=True (stale) — empty responses this frame
    frame = _make_frame({0: _make_robot(0, 0.0, 0.0, has_ball=True)}, ball_pos=(2.0, 0.0))
    result = refiner.refine(frame, [])  # no response arrived
    # Robot is 2 m away — should be inferred as False, not left as stale True
    assert result.friendly_robots[0].has_ball is False


def test_unknown_robot_id_warns(recwarn):
    refiner = RobotInfoRefiner()
    frame = _make_frame({0: _make_robot(0, 0.0, 0.0)})
    refiner.refine(frame, [RobotResponse(id=99, has_ball=True)])
    assert any("99" in str(w.message) for w in recwarn.list)


# ---------------------------------------------------------------------------
# Enemy has_ball: provided via the team-tagged enemy_robot_responses list
# ---------------------------------------------------------------------------


def _make_pvp_frame(enemy_robots: dict) -> GameFrame:
    return GameFrame(
        ts=0.0,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots={0: _make_robot(0, 0.0, 0.0, has_ball=False)},
        enemy_robots=enemy_robots,
        ball=None,
    )


def test_enemy_has_ball_merged_from_enemy_responses():
    """The referee's frame must see the opponent's touches (DoubleTouchRule
    relies on an intervening enemy touch closing its restart window)."""
    refiner = RobotInfoRefiner()
    frame = _make_pvp_frame(enemy_robots={5: _make_enemy_robot(5, 0.0, 0.0, has_ball=False)})
    result = refiner.refine(frame, [], [RobotResponse(id=5, has_ball=True)])
    assert result.enemy_robots[5].has_ball is True
    assert result.friendly_robots[0].has_ball is False


def test_identical_ids_across_teams_updated_via_matching_list():
    """Response ids collide across teams (both sides run 0..N). Each list must
    only affect its own team — friendly responses must not leak onto the
    enemy robot with the same id, and vice versa."""
    refiner = RobotInfoRefiner()
    frame = _make_pvp_frame(enemy_robots={0: _make_enemy_robot(0, 0.0, 0.0, has_ball=False)})
    result = refiner.refine(
        frame,
        [RobotResponse(id=0, has_ball=True)],  # friendly: IR says yes
        [RobotResponse(id=0, has_ball=False)],  # enemy: no contact
    )
    assert result.friendly_robots[0].has_ball is True
    assert result.enemy_robots[0].has_ball is False


def test_enemy_untouched_without_enemy_responses():
    refiner = RobotInfoRefiner()
    frame = _make_pvp_frame(enemy_robots={0: _make_enemy_robot(0, 0.0, 0.0, has_ball=True)})
    assert refiner.refine(frame, []).enemy_robots[0].has_ball is True
    assert refiner.refine(frame, None, None) is frame


# ---------------------------------------------------------------------------
# Trusted-IR allowlist: trusted robot uses IR, untrusted uses vision proximity
# ---------------------------------------------------------------------------


def test_trusted_robot_uses_ir_even_when_far_from_ball():
    refiner = RobotInfoRefiner(trusted_ir_robots=frozenset({0}))
    # Robot is 2 m from ball — no proximity, but IR says True.
    frame = _make_frame({0: _make_robot(0, 0.0, 0.0)}, ball_pos=(2.0, 0.0))
    result = refiner.refine(frame, [RobotResponse(id=0, has_ball=True)])
    assert result.friendly_robots[0].has_ball is True


def test_trusted_robot_respects_ir_false():
    refiner = RobotInfoRefiner(trusted_ir_robots=frozenset({0}))
    # Robot is touching the ball but IR says False — should respect IR.
    frame = _make_frame({0: _make_robot(0, 0.0, 0.0)}, ball_pos=(0.0, 0.0))
    result = refiner.refine(frame, [RobotResponse(id=0, has_ball=False)])
    assert result.friendly_robots[0].has_ball is False


def test_untrusted_robot_infers_true_when_close():
    refiner = RobotInfoRefiner(trusted_ir_robots=frozenset())  # no robots trusted
    # Place robot within capture distance; IR reports False (broken sensor).
    frame = _make_frame({0: _make_robot(0, 0.0, 0.0)}, ball_pos=(0.05, 0.0))
    result = refiner.refine(frame, [RobotResponse(id=0, has_ball=False)])
    assert result.friendly_robots[0].has_ball is True


def test_untrusted_robot_infers_false_when_far():
    refiner = RobotInfoRefiner(trusted_ir_robots=frozenset())
    # Robot is far from ball; IR reports True (broken sensor firing randomly).
    frame = _make_frame({0: _make_robot(0, 0.0, 0.0)}, ball_pos=(1.0, 0.0))
    result = refiner.refine(frame, [RobotResponse(id=0, has_ball=True)])
    assert result.friendly_robots[0].has_ball is False


def test_untrusted_robot_at_exact_capture_boundary():
    refiner = RobotInfoRefiner(trusted_ir_robots=frozenset())
    just_inside = _BALL_CAPTURE_DIST - 0.001
    just_outside = _BALL_CAPTURE_DIST + 0.001

    frame_in = _make_frame({0: _make_robot(0, 0.0, 0.0)}, ball_pos=(just_inside, 0.0))
    assert refiner.refine(frame_in, [RobotResponse(id=0, has_ball=False)]).friendly_robots[0].has_ball is True

    frame_out = _make_frame({0: _make_robot(0, 0.0, 0.0)}, ball_pos=(just_outside, 0.0))
    assert refiner.refine(frame_out, [RobotResponse(id=0, has_ball=False)]).friendly_robots[0].has_ball is False


def test_untrusted_robot_no_ball_in_frame_returns_false():
    refiner = RobotInfoRefiner(trusted_ir_robots=frozenset())
    frame = _make_frame({0: _make_robot(0, 0.0, 0.0)}, ball_pos=None)
    result = refiner.refine(frame, [RobotResponse(id=0, has_ball=True)])
    assert result.friendly_robots[0].has_ball is False


# ---------------------------------------------------------------------------
# Mixed team: one robot trusted, one not
# ---------------------------------------------------------------------------


def test_mixed_team_trusted_uses_ir_untrusted_uses_vision():
    refiner = RobotInfoRefiner(trusted_ir_robots=frozenset({0}))
    robots = {
        0: _make_robot(0, 0.0, 0.0),  # trusted — far from ball
        1: _make_robot(1, 0.05, 0.0),  # untrusted — close to ball
    }
    frame = _make_frame(robots, ball_pos=(0.0, 0.0))
    responses = [
        RobotResponse(id=0, has_ball=True),  # IR says yes → trust it even though far
        RobotResponse(id=1, has_ball=False),  # IR says no → ignore, use proximity
    ]
    result = refiner.refine(frame, responses)
    assert result.friendly_robots[0].has_ball is True  # from IR
    assert result.friendly_robots[1].has_ball is True  # from vision (close enough)


def test_an_untrusted_sensor_stuck_on_is_ignored_away_from_the_ball():
    """2026-10-29: radio 1's IR read 'has ball' all the time; its profile's `ir: false` must
    leave has_ball to vision, false while the ball is far."""
    refiner = RobotInfoRefiner(trusted_ir_robots=frozenset())
    frame = _make_frame({1: _make_robot(1, 1.0, 0.0)}, ball_pos=(0.0, 0.0))
    result = refiner.refine(frame, [RobotResponse(id=1, has_ball=True)])
    assert result.friendly_robots[1].has_ball is False


def test_untrusted_robot_missing_from_responses_uses_vision():
    """Untrusted robot absent from responses this tick must not keep stale has_ball."""
    refiner = RobotInfoRefiner(trusted_ir_robots=frozenset({0}))
    robots = {
        0: _make_robot(0, 2.0, 0.0, has_ball=False),  # trusted
        1: _make_robot(1, 0.05, 0.0, has_ball=True),  # untrusted, stale True
    }
    frame = _make_frame(robots, ball_pos=(0.0, 0.0))
    # Robot 1 did not respond this tick (non-blocking poll returned nothing for it).
    responses = [RobotResponse(id=0, has_ball=False)]
    result = refiner.refine(frame, responses)
    # Robot 1 is close to ball → vision inference gives True (not stale, but correct)
    assert result.friendly_robots[1].has_ball is True

    # Now robot 1 is far from ball; stale value is True but inference should correct it.
    robots_far = {
        0: _make_robot(0, 2.0, 0.0, has_ball=False),
        1: _make_robot(1, 1.0, 0.0, has_ball=True),  # stale True, but now far
    }
    frame_far = _make_frame(robots_far, ball_pos=(0.0, 0.0))
    result_far = refiner.refine(frame_far, [RobotResponse(id=0, has_ball=False)])
    assert result_far.friendly_robots[1].has_ball is False  # stale cleared by vision


# ---------------------------------------------------------------------------
# contact_grace_s: real ball sensors drop out with the ball still on the dribbler (every mode)
# ---------------------------------------------------------------------------


def _at(ts: float, ball_pos=(0.09, 0.0)) -> GameFrame:
    frame = _make_frame({0: _make_robot(0, 0.0, 0.0)}, ball_pos=ball_pos)
    return GameFrame(
        ts=ts,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots=frame.friendly_robots,
        enemy_robots={},
        ball=frame.ball,
    )


@pytest.mark.parametrize("gap", [-0.01, 0.01])
def test_a_sensor_dropout_with_the_ball_on_the_dribbler_holds_for_the_grace(gap):
    """2026-10-27 field run, placement at 77 s: the sensor went off mid-carry, the ball 0.09 m ahead.
    The default refiner, so the simulators get it too."""
    refiner = RobotInfoRefiner()
    refiner.refine(_at(0.0), [RobotResponse(id=0, has_ball=True)])
    result = refiner.refine(_at(CONTACT_GRACE_S + gap), [RobotResponse(id=0, has_ball=False)])
    assert result.friendly_robots[0].has_ball is (gap < 0)


@pytest.mark.parametrize(
    "ball_pos", [(_BALL_CAPTURE_DIST + 0.01, 0.0), (-0.09, 0.0)], ids=["out of reach", "behind the robot"]
)
def test_a_dropout_is_believed_once_vision_shows_the_ball_gone(ball_pos):
    refiner = RobotInfoRefiner(contact_grace_s=0.3)
    refiner.refine(_at(0.0), [RobotResponse(id=0, has_ball=True)])
    assert (
        refiner.refine(_at(0.1, ball_pos), [RobotResponse(id=0, has_ball=False)]).friendly_robots[0].has_ball is False
    )


def test_without_a_grace_every_reading_is_taken_as_it_comes():
    refiner = RobotInfoRefiner(contact_grace_s=0.0)
    refiner.refine(_at(0.0), [RobotResponse(id=0, has_ball=True)])
    assert refiner.refine(_at(0.01), [RobotResponse(id=0, has_ball=False)]).friendly_robots[0].has_ball is False


def test_a_ball_on_the_dribbler_never_sensed_is_not_held():
    refiner = RobotInfoRefiner(contact_grace_s=0.3)
    assert refiner.refine(_at(0.0), [RobotResponse(id=0, has_ball=False)]).friendly_robots[0].has_ball is False
