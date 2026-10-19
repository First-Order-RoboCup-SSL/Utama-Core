"""Flat engine's policy network: what it reads at a decision and at one robot's question, and the
network itself, run in numpy on the CPU (well under a millisecond a question).

Like the scripted rule and OpenJev, it picks one robot's action from that robot's menu
(`flat_state.robot_view`), in the same chain (the carrier first, then nearest the ball first). It reads
the match as tokens, all in our frame (we attack +x and our left is +y, whichever side we play): one
global token, our 6 robots and their 6 in id order, and the ball. Beside raw positions, velocities and
headings, each of our robots carries the facts its menu states (free, lane open from the ball, shot
open / narrow / blocked), so the network can learn the rule exactly, and later more than the rule. A
transformer mixes the tokens; the asked robot's token then scores the fixed actions and points at a
teammate (pass) or an opponent (mark).

`FlatJevTactic(record_path=...)` records every question (these arrays, the menu as a mask, the rule's
pick and the action taken); `tools/flat_net_train.py` trains on them with torch and exports the weights
`FlatNet.load` reads.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.shared.pass_and_score_geometry import (
    CARRY_LIMIT_M,
    ball_in_enemy_defense_area,
    ball_in_own_defense_area,
    enemy_goal_line,
    has_ball,
)
from utama_core.tactics import flat_state as fs

SLOTS = 6  # robots a team (6v6); our tokens and theirs, in id order
KINDS = ("none", "shoot", "pass", "receive", "dribble", "clear", "get_ball", "move", "mark", "block_shot", "hold")
PHASES = ("carrier", "them", "contested", "deep", "us")
SIDES = ("us", "them", "contested")
QUALITIES = ("blocked", "narrow", "open")

# The action space: the fixed actions, then pass to our slot k, then mark their slot k.
FIXED = (
    ("shoot", "clear", "get_ball", "block_shot", "hold")
    + tuple(f"dribble:{c}" for c in fs.CELLS)
    + tuple(f"move:{c}" for c in fs.CELLS)
)
PASS0 = len(FIXED)
MARK0 = PASS0 + SLOTS
N_ACTIONS = MARK0 + SLOTS

# Our robots' columns.
O_PRESENT, O_KEEPER, O_EGO, O_HAS_BALL, O_ON_BALL = 0, 1, 2, 3, 4
O_POS, O_VEL, O_HEADING = 5, 7, 9  # two columns each
O_BALL_DIST, O_RANK, O_FREE, O_ENEMY_DIST = 11, 12, 13, 14
O_SHOT = 15  # QUALITIES one-hot
O_GAP, O_GOAL_DIST, O_LANE = 18, 19, 20
O_THIRD, O_STRIP = 21, 24  # one-hot, three columns each
O_CARRY, O_LOCKED = 27, 28
O_DOING = 29  # KINDS one-hot: the action it is carrying out
O_DECIDED = O_DOING + len(KINDS)  # KINDS one-hot: chosen earlier in this decision ("none" if not yet)
O_DECIDED_CELL = O_DECIDED + len(KINDS)  # its move / dribble cell
F_OURS = O_DECIDED_CELL + len(fs.CELLS)

# Their robots' columns.
T_PRESENT, T_KEEPER, T_HOLDER = 0, 1, 2
T_POS, T_VEL, T_HEADING = 3, 5, 7
T_BALL_DIST, T_GOAL_DIST, T_MARKED = 9, 10, 11
T_THIRD, T_STRIP = 12, 15
F_THEIRS = 18

# The ball.
B_POS, B_VEL, B_SPEED, B_THIRD, B_STRIP, B_OWN_BOX, B_THEIR_BOX = 0, 2, 4, 5, 8, 11, 12
F_BALL = 13

# The global token: the moment, then this question.
G_SCORE, G_TIME, G_SIDE, G_HOLDER = 0, 2, 3, 6
G_PHASE = 7  # PHASES one-hot
G_CHAIN, G_N_GET, G_N_BLOCK, G_N_FORWARD = 12, 13, 14, 15
G_CELLS = 16  # move cells already chosen in this decision
F_GLOB = G_CELLS + len(fs.CELLS)


@dataclass
class Moment:
    """The match at one decision, as the network reads it, before any robot is asked."""

    ours: np.ndarray  # (SLOTS, F_OURS)
    theirs: np.ndarray  # (SLOTS, F_THEIRS)
    ball: np.ndarray  # (F_BALL,)
    glob: np.ndarray  # (F_GLOB,)
    our_ids: tuple[int, ...]  # slot -> robot id
    their_ids: tuple[int, ...]
    score: tuple[int, int]  # ours, theirs


def _kind(action: Optional[str]) -> int:
    return KINDS.index(action.split(":")[0]) if action else 0


def moment(
    game: Game,
    robot_ids: tuple[int, ...],
    plans: dict[int, str],
    locked: set[int],
    carry_left: dict[int, Optional[float]],
    holder: Optional[int],
    enemy_holder: Optional[int],
) -> Moment:
    """`plans`: each robot's current action; `holder`: our carrier, `enemy_holder`: theirs (or None)."""
    a = fs.attack_dir(game)
    hl, hw = game.field.half_length, game.field.half_width
    ball = game.ball.p.to_2d()
    goal_x, y1, y2 = enemy_goal_line(game)
    their_goal = Vector2D(goal_x, (y1 + y2) / 2.0)
    our_goal = Vector2D(game.field.my_goal_line[0][0], 0.0)

    def frame(row: np.ndarray, col: int, p, scale_x: float, scale_y: float) -> None:
        row[col], row[col + 1] = p.x * a / scale_x, p.y * a / scale_y

    def cell(row: np.ndarray, third_col: int, strip_col: int, p: Vector2D) -> None:
        third, strip = fs.cell_coords(game, p)
        row[third_col + third] = row[strip_col + strip] = 1.0

    our_ids = tuple(sorted(game.friendly_robots))[:SLOTS]
    their_ids = tuple(sorted(game.enemy_robots))[:SLOTS]
    outfield = sorted(
        (rid for rid in robot_ids if rid in game.friendly_robots),
        key=lambda rid: game.friendly_robots[rid].p.distance_to(ball),
    )

    ours = np.zeros((SLOTS, F_OURS), np.float32)
    for s, rid in enumerate(our_ids):
        r, row = game.friendly_robots[rid], ours[s]
        row[O_PRESENT] = 1.0
        row[O_KEEPER] = rid not in robot_ids
        # The visual read has hysteresis state: only for robots the engine already reads every tick.
        row[O_HAS_BALL] = has_ball(game, rid, visual=True) if rid in robot_ids else has_ball(game, rid)
        row[O_ON_BALL] = has_ball(game, rid)
        frame(row, O_POS, r.p, hl, hw)
        frame(row, O_VEL, r.v, 3.0, 3.0)
        row[O_HEADING], row[O_HEADING + 1] = a * math.cos(r.orientation), a * math.sin(r.orientation)
        row[O_BALL_DIST] = r.p.distance_to(ball) / 9.0
        row[O_RANK] = (outfield.index(rid) + 1) / 5.0 if rid in outfield else 0.0
        enemy_d = fs.nearest_enemy_dist(game, r.p)
        row[O_FREE] = enemy_d >= fs.FREE_RADIUS_M
        row[O_ENEMY_DIST] = min(enemy_d, 3.0) / 3.0
        quality, width = fs.shot_quality(game, r.p)
        row[O_SHOT + QUALITIES.index(quality)] = 1.0
        row[O_GAP] = min(width, 1.0)
        row[O_GOAL_DIST] = r.p.distance_to(their_goal) / 9.0
        row[O_LANE] = fs.lane_open(game, ball, r.p)
        cell(row, O_THIRD, O_STRIP, r.p)
        left = carry_left.get(rid)
        row[O_CARRY] = 1.0 if left is None else max(0.0, left) / CARRY_LIMIT_M
        row[O_LOCKED] = rid in locked
        row[O_DOING + _kind(plans.get(rid))] = 1.0
        row[O_DECIDED] = 1.0  # "none" until this decision chooses for it

    theirs = np.zeros((SLOTS, F_THEIRS), np.float32)
    keeper = fs.enemy_keeper_id(game)
    for s, eid in enumerate(their_ids):
        r, row = game.enemy_robots[eid], theirs[s]
        row[T_PRESENT] = 1.0
        row[T_KEEPER] = eid == keeper
        row[T_HOLDER] = eid == enemy_holder
        frame(row, T_POS, r.p, hl, hw)
        frame(row, T_VEL, r.v, 3.0, 3.0)
        row[T_HEADING], row[T_HEADING + 1] = a * math.cos(r.orientation), a * math.sin(r.orientation)
        row[T_BALL_DIST] = r.p.distance_to(ball) / 9.0
        row[T_GOAL_DIST] = r.p.distance_to(our_goal) / 9.0
        cell(row, T_THIRD, T_STRIP, r.p)

    b = np.zeros(F_BALL, np.float32)
    frame(b, B_POS, ball, hl, hw)
    frame(b, B_VEL, game.ball.v, 3.0, 3.0)
    b[B_SPEED] = math.hypot(game.ball.v.x, game.ball.v.y) / 5.0
    cell(b, B_THIRD, B_STRIP, ball)
    b[B_OWN_BOX] = ball_in_own_defense_area(game)
    b[B_THEIR_BOX] = ball_in_enemy_defense_area(game)

    g = np.zeros(F_GLOB, np.float32)
    score = (0, 0)
    ref = game.referee
    if ref is not None:
        us, them = (ref.yellow_team, ref.blue_team) if game.my_team_is_yellow else (ref.blue_team, ref.yellow_team)
        score = (int(us.score), int(them.score))
        g[G_TIME] = min(max(float(ref.stage_time_left), 0.0), 600.0) / 300.0
    g[G_SCORE], g[G_SCORE + 1] = score[0] / 10.0, score[1] / 10.0
    g[G_SIDE + SIDES.index(fs.side_of(game, robot_ids))] = 1.0
    g[G_HOLDER] = holder is not None
    return Moment(ours, theirs, b, g, our_ids, their_ids, score)


def question(
    m: Moment, view: fs.RobotView, decided: dict[int, str], chain: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """(ours, theirs, ball, glob, ego slot) for asking `view.robot_id`, after `decided` this decision."""
    ours, theirs, glob = m.ours.copy(), m.theirs.copy(), m.glob.copy()
    ego = m.our_ids.index(view.robot_id)
    ours[ego, O_EGO] = 1.0
    for rid, action in decided.items():
        if rid in m.our_ids:
            row = ours[m.our_ids.index(rid)]
            row[O_DECIDED : O_DECIDED + len(KINDS)] = 0.0
            row[O_DECIDED + _kind(action)] = 1.0
            if action.startswith(("move:", "dribble:")):
                row[O_DECIDED_CELL + fs.CELLS.index(action.split(":", 1)[1])] = 1.0
        if action.startswith("mark:") and int(action[5:]) in m.their_ids:
            theirs[m.their_ids.index(int(action[5:])), T_MARKED] += 0.5
    chosen = list(decided.values())
    glob[G_PHASE + PHASES.index(view.phase)] = 1.0
    glob[G_CHAIN] = chain / 5.0
    glob[G_N_GET] = chosen.count("get_ball") / 3.0
    glob[G_N_BLOCK] = chosen.count("block_shot") / 2.0
    glob[G_N_FORWARD] = sum(c.startswith("move:") and not c.startswith("move:defensive") for c in chosen) / 3.0
    for c in chosen:
        if c.startswith("move:"):
            glob[G_CELLS + fs.CELLS.index(c[5:])] = 1.0
    return ours, theirs, m.ball, glob, ego


def action_index(key: str, m: Moment) -> int:
    if key.startswith("pass:"):
        return PASS0 + m.our_ids.index(int(key[5:]))
    if key.startswith("mark:"):
        return MARK0 + m.their_ids.index(int(key[5:]))
    return FIXED.index(key)


def action_key(index: int, m: Moment) -> str:
    if index >= MARK0:
        return f"mark:{m.their_ids[index - MARK0]}"
    if index >= PASS0:
        return f"pass:{m.our_ids[index - PASS0]}"
    return FIXED[index]


def menu_mask(options, m: Moment) -> np.ndarray:
    mask = np.zeros(N_ACTIONS, bool)
    for key in options:
        mask[action_index(key, m)] = True
    return mask


# --- the network ----------------------------------------------------------------------------------


def _layer_norm(x: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + 1e-5) * g + b


class FlatNet:
    """The policy network's forward pass; `tools/flat_net_train.py` defines the same network in torch
    and exports its weights (names as below, matrices as (in, out))."""

    def __init__(self, weights: dict[str, np.ndarray]):
        cfg = json.loads(str(weights["config"]))
        self.layers, self.heads = int(cfg["layers"]), int(cfg["heads"])
        self.w = {k: np.asarray(v, np.float32) for k, v in weights.items() if k != "config"}

    @classmethod
    def load(cls, path: str | Path) -> FlatNet:
        """The weights file; when it carries the trainer's own outputs for a few questions (`check_*`),
        the numpy forward pass must reproduce them."""
        with np.load(path) as z:
            data = {k: z[k] for k in z.files}
        net = cls({k: v for k, v in data.items() if not k.startswith("check_")})
        if "check_logits" in data:
            for k in range(len(data["check_logits"])):
                feats = [data[f"check_{f}"][k] for f in ("ours", "theirs", "ball", "glob")]
                got = net.logits(*feats, int(data["check_ego"][k]))
                err = float(np.abs(got - data["check_logits"][k]).max())
                if err > 1e-3:
                    raise ValueError(f"{path}: numpy forward differs from the trainer's by {err:.2g}")
        return net

    def logits(self, ours: np.ndarray, theirs: np.ndarray, ball: np.ndarray, glob: np.ndarray, ego: int) -> np.ndarray:
        w = self.w
        x = np.concatenate(
            [
                glob[None].astype(np.float32) @ w["in_glob_w"] + w["in_glob_b"],
                ours.astype(np.float32) @ w["in_ours_w"] + w["in_ours_b"],
                theirs.astype(np.float32) @ w["in_theirs_w"] + w["in_theirs_b"],
                ball[None].astype(np.float32) @ w["in_ball_w"] + w["in_ball_b"],
            ]
        )
        x = x + w["pos"]
        n, d = x.shape
        dh = d // self.heads
        present = np.concatenate([[1.0], ours[:, O_PRESENT], theirs[:, T_PRESENT], [1.0]]) > 0.5
        key_bias = np.where(present, 0.0, -1e9).astype(np.float32)
        for i in range(self.layers):
            h = _layer_norm(x, w[f"l{i}_ln1_g"], w[f"l{i}_ln1_b"])
            q, k, v = np.split(h @ w[f"l{i}_qkv_w"] + w[f"l{i}_qkv_b"], 3, axis=-1)
            q, k, v = (t.reshape(n, self.heads, dh).transpose(1, 0, 2) for t in (q, k, v))
            s = q @ k.transpose(0, 2, 1) / math.sqrt(dh) + key_bias
            s = np.exp(s - s.max(-1, keepdims=True))
            s /= s.sum(-1, keepdims=True)
            x = x + (s @ v).transpose(1, 0, 2).reshape(n, d) @ w[f"l{i}_out_w"] + w[f"l{i}_out_b"]
            h = _layer_norm(x, w[f"l{i}_ln2_g"], w[f"l{i}_ln2_b"])
            x = x + np.maximum(h @ w[f"l{i}_ff1_w"] + w[f"l{i}_ff1_b"], 0.0) @ w[f"l{i}_ff2_w"] + w[f"l{i}_ff2_b"]
        x = _layer_norm(x, w["ln_f_g"], w["ln_f_b"])
        me = x[1 + ego]
        dk = w["pass_q_w"].shape[1]
        fixed = me @ w["fixed_w"] + w["fixed_b"]
        passes = (x[1 : 1 + SLOTS] @ w["pass_k_w"]) @ (me @ w["pass_q_w"]) / math.sqrt(dk)
        marks = (x[1 + SLOTS : 1 + 2 * SLOTS] @ w["mark_k_w"]) @ (me @ w["mark_q_w"]) / math.sqrt(dk)
        return np.concatenate([fixed, passes, marks])

    def choose(self, feats: tuple, mask: np.ndarray) -> tuple[int, np.ndarray]:
        """The best action on the menu, and the probabilities over the menu."""
        z = np.where(mask, self.logits(*feats), -np.inf)
        p = np.exp(z - z.max())
        p /= p.sum()
        return int(np.argmax(p)), p
