"""striker_e2e.py — the 1v1 duel (`striker_duel.py`) with an end-to-end attacker: a network that drives the
robot itself, at `1 / ACTION_REPEAT` of the 60 Hz tick (20 Hz), with no planner or skill in between. Each
action is the robot's own command: forward and sideways speed and turn rate (in [-1, 1] of `RSIM_PARAMS`'
`MAX_VEL`/`MAX_ANGULAR_VEL`, a Gaussian), kick and dribbler (each a coin), held for `ACTION_REPEAT` ticks.
It sees `N_OBS` numbers, most in its own frame (`observe`): the ball, its own motion and last action,
the goal and the keeper, and `striker_duel.goal_view`'s goal mouth.

  pixi run python -m tools.striker_e2e record --teacher w.npz --episodes 6000 --workers 16 --out bc.npz
  pixi run python -m tools.striker_e2e play --weights e.npz --episodes 1500 --workers 16 --out it0.npz
  pixi run python -m tools.striker_e2e eval [--weights e.npz | --teacher w.npz] --episodes 4000 --workers 16

`record` plays `striker_duel`'s high-level network (`--teacher`, or the rule without it) held at the same
20 Hz and writes what it saw and the command it gave, for imitation (`striker_e2e_rl.py`); with --weights
the network drives and the teacher only labels (DAgger). `play`
samples the network's actions and writes each decision with `phi`, the shaping potential
(`striker_e2e_rl.py` turns it into a reward). `eval` plays greedily and prints the goal rate per keeper;
the same --seed gives the same starts as `striker_duel eval`.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from tools.striker_duel import (
    EPISODE_S,
    Duel,
    StrikerTactic,
    goal_rates,
    goal_view,
    load_params,
    net_chooser,
)
from utama_core.config.robot_params import RSIM_PARAMS
from utama_core.engine.context import TickContext
from utama_core.engine.tactic import BaseTactic, RobotId
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.shared.pass_and_score_geometry import (
    CARRY_LIMIT_M,
    carry_origin,
    enemy_goal_line,
    has_ball,
)

ACTION_REPEAT = 3  # ticks each action is held: 20 Hz
N_OBS = 42
N_ACT = 5  # forward, left, turn (Gaussian, [-1, 1]); kick, dribble (Bernoulli)
OBS_CLIP = 5.0
PHI_BALL, PHI_GOAL = 0.1, 0.1  # shaping potential per metre: robot to ball, ball to the goal's centre

Actor = Callable[[np.ndarray], tuple[np.ndarray, float]]


def to_command(a: np.ndarray) -> RobotCommand:
    v = np.clip(a[:3], -1.0, 1.0)
    return RobotCommand(
        local_forward_vel=float(v[0]) * RSIM_PARAMS.MAX_VEL,
        local_left_vel=float(v[1]) * RSIM_PARAMS.MAX_VEL,
        angular_vel=float(v[2]) * RSIM_PARAMS.MAX_ANGULAR_VEL,
        kick=int(a[3] > 0.5),
        chip=0,
        dribble=bool(a[4] > 0.5),
    )


def from_command(cmd: RobotCommand) -> np.ndarray:
    return np.array(
        [
            cmd.local_forward_vel / RSIM_PARAMS.MAX_VEL,
            cmd.local_left_vel / RSIM_PARAMS.MAX_VEL,
            cmd.angular_vel / RSIM_PARAMS.MAX_ANGULAR_VEL,
            float(bool(cmd.kick)),
            float(bool(cmd.dribble)),
        ],
        dtype=np.float32,
    ).clip(-1.0, 1.0)


def _local(v: Vector2D, heading: float) -> tuple[float, float]:
    c, s = math.cos(heading), math.sin(heading)
    return v.x * c + v.y * s, -v.x * s + v.y * c


def observe(game: Game, rid: int, t0: float, prev: np.ndarray, origin: Optional[Vector2D]) -> np.ndarray:
    me, ball = game.friendly_robots[rid], game.ball
    keeper = next(iter(game.enemy_robots.values()))
    h, hx, hy = me.orientation, game.field.half_length, game.field.half_width
    goal_x, y1, y2 = enemy_goal_line(game)
    b = ball.p.to_2d() - me.p
    bx, by = _local(b, h)
    dist = max(b.mag(), 1e-6)
    carry = 0.0 if origin is None else ball.p.to_2d().distance_to(origin)
    points = [Vector2D(goal_x, (y1 + y2) / 2.0), Vector2D(goal_x, y1), Vector2D(goal_x, y2), keeper.p]
    rel = [c / hx for p in points for c in _local(p - me.p, h)]
    # Clipped: the goal mouth's fractions blow up with the ball on the goal line.
    return np.array(
        [
            bx,
            by,
            min(dist, 3.0),
            bx / dist,
            by / dist,
            *(c / 4.0 for c in _local(ball.v, h)),
            *(c / 2.0 for c in _local(me.v, h)),
            *(c / 2.0 for c in _local(keeper.v, h)),
            *prev,
            float(has_ball(game, rid)),
            carry / CARRY_LIMIT_M,
            *rel,
            me.p.x / hx,
            me.p.y / hy,
            math.cos(h),
            math.sin(h),
            ball.p.x / hx,
            ball.p.y / hy,
            (game.ts - t0) / EPISODE_S,
            *goal_view(game, rid),
        ],
        dtype=np.float32,
    ).clip(-OBS_CLIP, OBS_CLIP)


def phi(game: Game, rid: int) -> float:
    """The shaping potential: nearer the ball, and the ball nearer their goal, is better."""
    me, ball = game.friendly_robots[rid].p, game.ball.p.to_2d()
    goal_x, y1, y2 = enemy_goal_line(game)
    return -PHI_BALL * me.distance_to(ball) - PHI_GOAL * ball.distance_to(Vector2D(goal_x, (y1 + y2) / 2.0))


@dataclass
class E2EMem:
    tick: int = 0
    command: Optional[RobotCommand] = None
    prev: np.ndarray = field(default_factory=lambda: np.zeros(N_ACT, dtype=np.float32))
    origin: Optional[Vector2D] = None
    teacher_mem: object = None


class E2ETactic(BaseTactic[E2EMem]):
    """Robot `robot_id` alone, driven every `ACTION_REPEAT` ticks by `act(obs) -> (action, logp)`, or by
    `teacher` (a `striker_duel.StrikerTactic`) held the same way. Each decision is appended to
    `trajectory` as (obs, action, logp, phi); with both, `act` drives and the action recorded is the
    teacher's, ticked alongside: what it would have done where the network took the robot (DAgger)."""

    def __init__(self, act: Optional[Actor] = None, teacher: Optional[StrikerTactic] = None, robot_id: int = 0):
        self.robot_id = robot_id
        self.act = act
        self.teacher = teacher
        self.t0 = 0.0
        self.trajectory: list[tuple[np.ndarray, np.ndarray, float, float]] = []

    def initial_mem(self) -> E2EMem:
        return E2EMem(teacher_mem=self.teacher.initial_mem() if self.teacher is not None else None)

    def highlights(self, mem: E2EMem) -> dict[RobotId, str]:
        return {self.robot_id: "e2e"}

    def tick(
        self, game: Game, ctx: TickContext, robot_ids: tuple[RobotId, ...], mem: E2EMem
    ) -> tuple[dict[RobotId, RobotCommand], E2EMem]:
        rid = self.robot_id
        if game.ball is None or rid not in game.friendly_robots:
            return {}, mem
        mem.origin = carry_origin(game, rid, mem.origin)
        teacher_cmd = None
        if self.teacher is not None:
            self.teacher.t0 = self.t0
            cmds, mem.teacher_mem = self.teacher.tick(game, ctx, robot_ids, mem.teacher_mem)
            teacher_cmd = cmds.get(rid)
        if mem.tick % ACTION_REPEAT == 0 or mem.command is None:
            obs = observe(game, rid, self.t0, mem.prev, mem.origin)
            label = None
            if self.teacher is not None:
                label = from_command(teacher_cmd) if teacher_cmd is not None else np.zeros(N_ACT, np.float32)
            a, logp = self.act(obs) if self.act is not None else (label, 0.0)
            self.trajectory.append((obs, a if label is None else label, logp, phi(game, rid)))
            mem.command, mem.prev = to_command(a), np.concatenate([np.clip(a[:3], -1, 1), a[3:]]).astype(np.float32)
        mem.tick += 1
        return {rid: mem.command}, mem


# --- the policy ------------------------------------------------------------------------------------


def forward(params: dict[str, np.ndarray], obs: np.ndarray) -> np.ndarray:
    """`striker_e2e_rl.Policy`: two tanh layers; the speeds' means are tanh of the first three outputs,
    kick and dribbler logits the last two."""
    h = np.tanh(obs @ params["w0"] + params["b0"])
    h = np.tanh(h @ params["w1"] + params["b1"])
    return h @ params["w2"] + params["b2"]


def actor(params: dict[str, np.ndarray], greedy: bool, rng: np.random.Generator) -> Actor:
    std = np.exp(params["log_std"]).astype(np.float64)

    def act(obs: np.ndarray) -> tuple[np.ndarray, float]:
        out = forward(params, obs).astype(np.float64)
        mean, p = np.tanh(out[:3]), 1.0 / (1.0 + np.exp(-out[3:]))
        if greedy:
            return np.concatenate([mean, (p > 0.5).astype(np.float64)]).astype(np.float32), 0.0
        v = mean + std * rng.standard_normal(3)
        bits = (rng.random(2) < p).astype(np.float64)
        logp = -0.5 * (((v - mean) / std) ** 2).sum() - np.log(std).sum() - 1.5 * math.log(2 * math.pi)
        logp += float(np.log(np.where(bits > 0, p, 1 - p) + 1e-12).sum())
        return np.concatenate([v, bits]).astype(np.float32), float(logp)

    return act


# --- running ---------------------------------------------------------------------------------------

_DUEL: Optional[Duel] = None
_PARAMS: dict[str, dict[str, np.ndarray]] = {}


def _params(path: str) -> dict[str, np.ndarray]:
    if path not in _PARAMS:
        _PARAMS.clear()
        _PARAMS[path] = load_params(Path(path))
    return _PARAMS[path]


def _episode(task: tuple[int, str, Optional[str], Optional[str], bool, bool]) -> dict:
    global _DUEL
    seed, keeper, weights, teacher, greedy, label = task
    if _DUEL is None:
        _DUEL = Duel()
    rng = np.random.default_rng(seed)
    act = actor(_params(weights), greedy, rng) if weights is not None else None
    tutor = None
    if label:
        tutor = StrikerTactic(net_chooser(_params(teacher), True, rng) if teacher is not None else None)
    tactic = E2ETactic(act=act, teacher=tutor)
    outcome = _DUEL.episode(random.Random(seed), tactic, keeper)
    return {"seed": seed, "keeper": keeper, "outcome": outcome, "trajectory": tactic.trajectory}


def run(
    seed: int,
    episodes: int,
    keepers: list[str],
    weights: Optional[str],
    teacher: Optional[str],
    greedy: bool,
    workers: int,
    label: bool = False,
) -> list[dict]:
    """Episode i: start seed `seed * 1_000_000 + i`, keeper `keepers[i % len(keepers)]` (as `striker_duel.run`).
    `label`: the teacher labels every decision (it drives too, unless `weights`)."""
    tasks = [
        (seed * 1_000_000 + i, keepers[i % len(keepers)], weights, teacher, greedy, label) for i in range(episodes)
    ]
    with mp.get_context("fork").Pool(workers) as pool:
        return list(pool.imap(_episode, tasks, chunksize=4))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("record", "play", "eval"))
    parser.add_argument("--weights", help="the end-to-end network (striker_e2e_rl's e<k>.npz)")
    parser.add_argument("--teacher", help="record/eval: striker_duel's high-level network; none: its rule")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--keepers", default="default,flat")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path)
    a = parser.parse_args()
    record = a.mode == "record"
    # Without --weights the teacher plays; recording with them, the network drives (greedily) and the
    # teacher labels.
    label = record or a.weights is None
    greedy = a.mode == "eval" or record
    results = run(a.seed, a.episodes, a.keepers.split(","), a.weights, a.teacher, greedy, a.workers, label)
    name = a.weights or (f"teacher:{a.teacher}" if a.teacher else "teacher:rule")
    summary = {"policy": name, "seed": a.seed, **goal_rates(results)}
    print(json.dumps(summary, indent=1))
    if a.out is None:
        return
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.mode == "eval":
        a.out.write_text(json.dumps(summary, indent=1))
        return
    steps = [(i, *step) for i, r in enumerate(results) for step in r["trajectory"]]
    np.savez_compressed(
        a.out,
        obs=np.array([s[1] for s in steps], dtype=np.float32).reshape(-1, N_OBS),
        act=np.array([s[2] for s in steps], dtype=np.float32).reshape(-1, N_ACT),
        logp=np.array([s[3] for s in steps], dtype=np.float32),
        phi=np.array([s[4] for s in steps], dtype=np.float32),
        episode=np.array([s[0] for s in steps], dtype=np.int64),
        outcome=np.array([r["outcome"] for r in results]),
        keeper=np.array([r["keeper"] for r in results]),
    )


if __name__ == "__main__":
    main()
