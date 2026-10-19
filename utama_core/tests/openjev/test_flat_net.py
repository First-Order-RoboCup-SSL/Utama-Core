"""Flat engine's policy network (`flat_net.py`) on the CPU: its view of the game, the menu as actions
and the forward pass, with random weights."""

from __future__ import annotations

import functools
import json
import math

import numpy as np
import pytest

from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.shared.pass_and_score_geometry import reset_possession_state
from utama_core.tactics import flat_net
from utama_core.tactics import flat_state as fs
from utama_core.tests.fixtures.game_builder import build_game, make_ball, make_robot
from utama_core.tests.openjev.test_flat import OUTFIELD, _carrier_game, _view


def random_weights(seed: int = 0, d: int = 16, layers: int = 1, heads: int = 2, ff: int = 32, dk: int = 8) -> dict:
    """Weights of the shape `tools/flat_net_train.py` exports."""
    rng = np.random.default_rng(seed)
    w = {"config": np.array(json.dumps({"layers": layers, "heads": heads}))}

    def lin(name, i, o, bias=True):
        w[f"{name}_w"] = rng.normal(0.0, 1.0 / math.sqrt(i), (i, o)).astype(np.float32)
        if bias:
            w[f"{name}_b"] = rng.normal(0.0, 0.1, o).astype(np.float32)

    for name, f in (("glob", flat_net.F_GLOB), ("ours", flat_net.F_OURS), ("theirs", flat_net.F_THEIRS)):
        lin(f"in_{name}", f, d)
    lin("in_ball", flat_net.F_BALL, d)
    w["pos"] = rng.normal(0.0, 0.1, (2 + 2 * flat_net.SLOTS, d)).astype(np.float32)
    for i in range(layers):
        for ln in ("ln1", "ln2"):
            w[f"l{i}_{ln}_g"], w[f"l{i}_{ln}_b"] = np.ones(d, np.float32), np.zeros(d, np.float32)
        lin(f"l{i}_qkv", d, 3 * d)
        lin(f"l{i}_out", d, d)
        lin(f"l{i}_ff1", d, ff)
        lin(f"l{i}_ff2", ff, d)
    w["ln_f_g"], w["ln_f_b"] = np.ones(d, np.float32), np.zeros(d, np.float32)
    lin("fixed", d, flat_net.PASS0)
    for name in ("pass_q", "pass_k", "mark_q", "mark_k"):
        lin(name, d, dk, bias=False)
    return w


def _moment(game, holder=None):
    return flat_net.moment(game, OUTFIELD, {}, set(), {}, holder, None)


def test_every_menu_option_is_one_action_and_back():
    game = _carrier_game()
    m = _moment(game, holder=2)
    for rid in (2, 4):  # the carrier's menu and an off-ball one
        options = _view(game, rid, carry_left_m=0.5).options
        assert {flat_net.action_key(flat_net.action_index(k, m), m) for k in options} == set(options)
        assert flat_net.menu_mask(options, m).sum() == len(options)


def test_the_game_reads_the_same_from_either_side():
    # One position, played from the left and mirrored from the right: our frame makes them identical.
    def game(is_right):
        s = -1.0 if is_right else 1.0
        turn = math.pi if is_right else 0.0  # mirroring the pitch turns every heading half a circle
        rob = functools.partial(make_robot, is_friendly=True)
        friendly = {
            0: rob(0, x=-4.4 * s, y=0.0, orientation=turn),
            1: rob(1, x=-2.0 * s, y=1.0 * s, vx=0.5 * s, orientation=0.3 + turn),
            2: rob(2, x=0.5 * s, y=-1.2 * s, vy=-0.4 * s, orientation=-1.0 + turn),
        }
        enemy = {
            0: make_robot(0, is_friendly=False, x=4.4 * s, y=0.0, orientation=turn),
            3: make_robot(3, is_friendly=False, x=1.5 * s, y=0.2 * s, orientation=2.0 + turn),
        }
        return build_game(
            friendly,
            enemy,
            make_ball(-1.0 * s, 0.5 * s, vx=1.0 * s),
            my_team_is_right=is_right,
            referee_command=RefereeCommand.NORMAL_START,
        )

    feats = []
    for is_right in (False, True):
        reset_possession_state()
        g = game(is_right)
        m = flat_net.moment(g, (1, 2), {1: "move:middle_left"}, set(), {}, None, None)
        view = fs.robot_view(g, 2, (1, 2), carry_left_m=None, marked_by={}, blockers=0)
        feats.append(flat_net.question(m, view, {1: "get_ball"}, 1))
    for left, right in zip(*feats):
        np.testing.assert_allclose(left, right, atol=1e-5)


def test_the_network_reads_only_present_robots_and_answers_from_the_menu():
    net = flat_net.FlatNet(random_weights())
    game = _carrier_game()
    m = _moment(game, holder=2)
    view = _view(game, 4)
    ours, theirs, ball, glob, ego = flat_net.question(m, view, {2: "pass:3"}, 1)
    assert not theirs[5, flat_net.T_PRESENT]  # only their 0 and 1 are on the pitch
    logits = net.logits(ours, theirs, ball, glob, ego)
    assert logits.shape == (flat_net.N_ACTIONS,)
    mask = flat_net.menu_mask(view.options, m)
    ghost = theirs.copy()
    ghost[5, 1:] = 7.0  # an absent robot changes nothing on the menu (the menu never offers to mark it)
    np.testing.assert_allclose(net.logits(ours, ghost, ball, glob, ego)[mask], logits[mask], atol=1e-5)
    act, p = net.choose((ours, theirs, ball, glob, ego), mask)
    assert mask[act] and abs(p.sum() - 1.0) < 1e-6 and not p[~mask].any()


def test_weights_whose_numpy_pass_differs_from_the_trainers_are_refused(tmp_path):
    w = random_weights()
    game = _carrier_game()
    feats = flat_net.question(_moment(game, holder=2), _view(game, 4), {}, 0)
    check = {f"check_{k}": np.stack([v]) for k, v in zip(("ours", "theirs", "ball", "glob"), feats[:4])}
    check["check_ego"] = np.array([feats[4]])
    check["check_logits"] = np.stack([flat_net.FlatNet(w).logits(*feats)])
    np.savez(tmp_path / "ok.npz", **w, **check)
    flat_net.FlatNet.load(tmp_path / "ok.npz")
    w["fixed_b"] = w["fixed_b"] + 0.1
    np.savez(tmp_path / "bad.npz", **w, **check)
    with pytest.raises(ValueError, match="differs from the trainer"):
        flat_net.FlatNet.load(tmp_path / "bad.npz")
