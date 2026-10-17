"""`tools/pitch_maps.py` on tiny synthetic replays: the side normalisation, possession and the
goalkeeper exclusion, and that each figure is written."""

import numpy as np
import pytest

from tools import pitch_maps as pm
from utama_core.engine.match_stats import _POSSESSION_RADIUS_M
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.replay.columnar_reader import ColumnarReplay

NORMAL_START = RefereeCommand.NORMAL_START.value


def _replay(friendly_xy, enemy_xy, ball_xy, defends_right=True, command=NORMAL_START, ball_speed=0.0):
    """One tick. Friendly ids are 0..k-1, enemy ids start at 10; `defends_right` is the friendly
    team's goal (True: it defends +x and attacks toward -x)."""
    f = np.array([friendly_xy], dtype=float)
    e = np.array([enemy_xy], dtype=float)
    k, m = f.shape[1], e.shape[1]
    ball_v = np.array([[ball_speed, 0.0, 0.0]])
    return ColumnarReplay(
        my_team_is_yellow=True,
        my_team_is_right=defends_right,
        friendly_ids=np.arange(k),
        enemy_ids=np.arange(10, 10 + m),
        ts=np.array([0.0]),
        friendly_p=f,
        friendly_v=np.zeros((1, k, 2)),
        friendly_a=np.zeros((1, k, 2)),
        friendly_orientation=np.zeros((1, k)),
        friendly_has_ball=np.zeros((1, k), dtype=bool),
        enemy_p=e,
        enemy_v=np.zeros((1, m, 2)),
        enemy_a=np.zeros((1, m, 2)),
        enemy_orientation=np.zeros((1, m)),
        enemy_has_ball=np.zeros((1, m), dtype=bool),
        ball_p=np.array([[ball_xy[0], ball_xy[1], 0.0]]),
        ball_v=ball_v,
        ball_a=np.zeros((1, 3)),
        has_referee=np.ones(1, dtype=bool),
        referee_command=np.array([command], dtype=np.int8),
        stage=np.zeros(1, dtype=np.int8),
        designated_position=np.full((1, 2), np.nan),
        sparse_referee={},
        side_is_right=np.array([defends_right]),
    )


def test_side_normalisation_maps_the_attacking_team_to_plus_x():
    # A robot at x=+3 for a team attacking right -> left maps to x=-3 (and y is rotated too).
    assert pm.to_attacking_frame(3.0, 1.0, defends_right=True) == (-3.0, -1.0)
    # A team attacking left -> right is already in the frame.
    assert pm.to_attacking_frame(3.0, 1.0, defends_right=False) == (3.0, 1.0)
    assert pm.attack_sign(True) == -1.0 and pm.attack_sign(False) == 1.0


def test_possession_needs_the_ball_near_a_robot_and_slow():
    near = np.array([[0.0, 0.0]])
    ball = np.array([[_POSSESSION_RADIUS_M * 0.5, 0.0, 0.0]])
    slow = np.zeros((1, 3))
    fast = np.array([[3.0, 0.0, 0.0]])
    live = np.array([True])
    assert pm.possession_codes(near, np.empty((1, 0, 2)), ball, slow, live).tolist() == [1]
    assert pm.possession_codes(np.empty((1, 0, 2)), near, ball, slow, live).tolist() == [-1]
    assert pm.possession_codes(near, np.empty((1, 0, 2)), ball, fast, live).tolist() == [0]
    assert pm.possession_codes(near, np.empty((1, 0, 2)), ball, slow, ~live).tolist() == [0]
    far = np.array([[_POSSESSION_RADIUS_M * 2, 0.0, 0.0]])
    assert pm.possession_codes(near, np.empty((1, 0, 2)), far, slow, live).tolist() == [0]


def test_keeper_is_the_robot_in_its_own_box():
    # Own goal is at -x in the common frame; robot 1 is the one inside the box.
    norm = np.array([[[-1.0, 0.0], [-4.2, 0.1], [2.0, 0.0]]])
    assert pm.keeper_slot(norm) == 1


def test_build_views_normalises_positions_shots_and_fouls():
    # Friendly (config_a) defends +x, so it attacks toward -x and its frame is rotated.
    replay = _replay(
        friendly_xy=[(4.0, 0.0), (3.0, 1.0), (-1.0, 0.0)],  # keeper at +4, outfield at (3,1) and (-1,0)
        enemy_xy=[(0.5, 0.0)],
        ball_xy=(3.0, 1.0),
    )
    shots = [
        {"side": "friendly", "t": 0.0, "xy": [-3.0, 1.0], "scored": True},
        {"side": "enemy", "t": 0.0, "xy": [0.5, 0.0], "scored": False},
    ]
    fouls = [
        {"side": "friendly", "rule": "crashing", "sim_time": 0.0, "ball_xy": [3.0, 1.0]},
        {"side": "friendly", "rule": "out_of_bounds", "sim_time": 0.0, "ball_xy": [4.5, 0.0]},
        {"side": "enemy", "rule": "crashing", "sim_time": 0.0, "ball_xy": [0.5, 0.0]},
    ]
    friendly, enemy = pm.build_views(replay, {"friendly": "A", "enemy": "B"}, shots, fouls)

    assert friendly.strategy == "A" and enemy.strategy == "B"
    # The keeper (robot 0) is left out; the others are rotated into the common frame.
    assert sorted(friendly.pos_robot.tolist()) == [1, 2]
    by_robot = dict(zip(friendly.pos_robot.tolist(), friendly.pos_x.tolist()))
    assert by_robot == pytest.approx({1: -3.0, 2: 1.0})
    assert friendly.pos_in.tolist() == [True, True]  # the team has the ball, whichever robot holds it

    own = [s for s in friendly.shots if s["own"]]
    faced = [s for s in friendly.shots if not s["own"]]
    assert len(own) == 1 and len(faced) == 1
    assert (own[0]["x"], own[0]["y"], own[0]["scored"]) == (3.0, -1.0, True)
    assert (faced[0]["x"], faced[0]["y"], faced[0]["scored"]) == (-0.5, 0.0, False)

    # Out of bounds is not a foul; the friendly foul is in the common frame, and enemy fouls stay with enemy.
    assert friendly.fouls == [{"x": -3.0, "y": -1.0, "rule": "crashing"}]
    assert enemy.fouls == [{"x": 0.5, "y": 0.0, "rule": "crashing"}]


def test_enemy_view_uses_the_opposite_frame():
    # The enemy defends -x (friendly defends +x), so it attacks +x and is not rotated. Its robot
    # at (-4, 0) is inside its own box (the keeper); its outfield robot at (-3, 1) holds the ball.
    replay = _replay(friendly_xy=[(4.0, 0.0)], enemy_xy=[(-4.0, 0.0), (-3.0, 1.0)], ball_xy=(-3.0, 1.0))
    _, enemy = pm.build_views(replay, {"friendly": "A", "enemy": "B"}, [], [])
    assert enemy.pos_robot.tolist() == [11]
    assert enemy.pos_x.tolist() == pytest.approx([-3.0])
    assert enemy.pos_y.tolist() == pytest.approx([1.0])
    assert enemy.pos_in.tolist() == [True]


def test_position_grid_shares_sum_to_one_and_means_are_per_robot():
    views = [
        pm.MatchView(
            strategy="A",
            pos_x=np.array([0.5, 0.5, -2.0]),
            pos_y=np.array([0.0, 0.0, 1.0]),
            pos_in=np.array([True, True, False]),
            pos_robot=np.array([1, 1, 2]),
        )
    ]
    grid = pm.position_grid(views, "A", in_possession=True)
    assert grid["share"].sum() == pytest.approx(1.0)
    assert grid["means"] == [(0.5, 0.0)]
    assert pm.position_grid(views, "A", in_possession=False)["means"] == [(-2.0, 1.0)]


def _views():
    names = ["build_a_kernel_strategy", "build_b_kernel_strategy"]
    shots = [{"x": 2.0, "y": 0.5, "scored": True, "own": True}, {"x": -1.0, "y": 0.0, "scored": False, "own": False}]
    fouls = [{"x": 1.0, "y": -1.0, "rule": "crashing"}, {"x": -2.0, "y": 1.0, "rule": "double_touch"}]
    views = []
    for name in names:
        views.append(
            pm.MatchView(
                strategy=name,
                pos_x=np.array([0.0, 1.0, -1.5]),
                pos_y=np.array([0.0, 0.5, -1.0]),
                pos_in=np.array([True, False, True]),
                pos_robot=np.array([1, 1, 2]),
                shots=list(shots),
                fouls=list(fouls),
            )
        )
    return names, {n: n.removeprefix("build_") for n in names}, views


@pytest.mark.parametrize("which", ["positions_attack", "positions_defence", "shots", "fouls_map"])
def test_each_figure_is_written(tmp_path, which):
    order, labels, views = _views()
    path = tmp_path / f"{which}.png"
    if which == "shots":
        pm.plot_shots(views, order, labels, path)
    elif which == "fouls_map":
        pm.plot_fouls(views, order, labels, path)
    else:
        pm.plot_positions(views, order, labels, path, in_possession=which == "positions_attack")
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_rule_colours_take_fixed_slots_in_rank_order():
    assert pm.rule_colours(["crashing", "double_touch"]) == {"crashing": pm.sr.SLOTS[0], "double_touch": pm.sr.SLOTS[1]}


def test_zone_findings_are_six_descriptive_shares_per_strategy_in_the_pitch_zone_names():
    # Common frame: the strategy attacks +x, thirds at x = -1.5 and +1.5 m (docs/pitch_zones.md).
    view = pm.MatchView(
        strategy="build_a_kernel_strategy",
        pos_x=np.array([-2.0, 0.0, 2.0, 2.5]),
        pos_y=np.zeros(4),
        pos_in=np.array([True, True, True, False]),  # three in possession, one out
        pos_robot=np.array([1, 2, 3, 1]),
    )
    findings = pm.zone_findings([view], ["build_a_kernel_strategy"], {"in possession": "a.png"})

    assert len(findings) == 6
    assert all(f.kind == "zone_share" and f.threshold is None and f.replay is None for f in findings)

    def find(label: str, zone: str):
        return next(f for f in findings if f"{label} samples in the {zone}" in f.message)

    for zone in ("defensive third", "middle third", "attacking third"):
        assert find("in possession", zone).value == pytest.approx(1 / 3)
    assert find("in possession", "attacking third").image == "a.png"
    assert find("out of possession", "attacking third").value == 1.0
    assert find("out of possession", "defensive third").value == 0.0
    assert pm.top_rules(_views()[2], k=1) == ["crashing"]
