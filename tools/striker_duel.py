"""striker_duel.py — one attacker against one goalkeeper: the attacker's choices from a network trained by
reinforcement learning (`tools/striker_rl.py`) or, as the control, from the rule (flat_rule's shot).

Each episode puts the ball, the attacker (yellow, attacking +x) and the keeper (blue) at a random start
(`sample_start`) in live play and runs until a goal (reward 1), the keeper holding the ball or the ball at
rest in its area ("saved"), a stoppage (out of play or a foul) or `EPISODE_S` seconds (reward 0). The
attacker gets to the ball by itself; with the ball on its dribbler it chooses, every `DECIDE_EVERY_S`,
one of `ACTIONS`: a shot at one of `N_SHOTS` points across the goal, a kick where it faces, or a short
carry in one of six directions (masked near the 1 m dribbling limit). A shot is kept until it is kicked.
The keeper is `--keepers`: "default" (`GoalkeeperTactic`, every pool strategy's) or "flat" (flat_rule's).

  pixi run python -m tools.striker_duel play --weights w.npz --episodes 2000 --workers 16 --out it0.npz
  pixi run python -m tools.striker_duel eval [--weights w.npz] --episodes 2000 --workers 16

`play` samples the network's actions and writes every decision (obs, mask, act, logp, episode) and each
episode's reward, outcome and keeper to --out. `eval` plays greedily (the rule without --weights) and
prints the goal rate per keeper; the same --seed gives every policy the same starts.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import multiprocessing as mp
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.custom_referee import CustomReferee
from utama_core.custom_referee.profiles.profile_loader import load_profile
from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.engine.context import TickContext
from utama_core.engine.tactic import BaseTactic, RobotId
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.replay.scenario import RobotState, Scenario, apply_scenario
from utama_core.rsoccer_simulator.src.Simulators.robosim.robosim_wrapper import (
    enable_sim_reuse,
)
from utama_core.run import StrategyRunner
from utama_core.shared.pass_and_score_geometry import (
    CARRY_LIMIT_M,
    ball_in_own_defense_area,
    carry_exhausted,
    carry_origin,
    clamp_outside_enemy_defense_area,
    clamp_to_field,
    enemy_goal_line,
    find_best_shot,
    has_ball,
    oriented_towards,
)
from utama_core.skills.src.go_to_ball import go_to_ball
from utama_core.skills.src.utils.move_utils import kick, move, turn_on_spot
from utama_core.strategy.default import build_default_kernel_strategy
from utama_core.tactics._pass_and_score import _SHOT_SWITCH_MARGIN
from utama_core.tactics.flat import FlatKeeperTactic
from utama_core.tactics.goalkeeper import GoalkeeperTactic

TICKS_PER_SECOND = 60
EPISODE_S = 8.0
DECIDE_EVERY_S = 0.1
SHOT_TIMEOUT_S = 1.5  # a shot not kicked by then is decided again
N_SHOTS = 9
SHOT_SPAN = 0.9  # the shots' points span this fraction of the goal's width
DRIBBLE_DIRS_DEG = (0, 60, -60, 120, -120, 180)  # from the line to the goal's centre
DRIBBLE_STEP_M = 0.35
ACTIONS = (*(f"shoot:{i}" for i in range(N_SHOTS)), "kick_now", *(f"dribble:{d}" for d in DRIBBLE_DIRS_DEG))
N_OBS = 28
SAVED_BALL_SPEED = 0.3  # a ball slower than this in the keeper's area is the keeper's
PINNED_M, PINNED_RAD = 0.2, 0.3  # this close to the ball and this far from facing it: pinned beside it
KEEPERS = {"default": GoalkeeperTactic, "flat": FlatKeeperTactic}
LIVE = (RefereeCommand.FORCE_START, RefereeCommand.NORMAL_START)
GRSIM_SETTLE_TICKS = 30  # grSim: how long a start is held before it goes live

Chooser = Callable[[np.ndarray, np.ndarray], tuple[int, float]]


# --- the attacker ----------------------------------------------------------------------------------


@dataclass
class StrikerMem:
    action: Optional[int] = None
    decided_at: float = -math.inf
    carry_origin: Optional[Vector2D] = None
    shot_y: Optional[float] = None  # the rule's last gap, for `find_best_shot`'s hysteresis


def shot_point(game: Game, i: int) -> Vector2D:
    goal_x, y1, y2 = enemy_goal_line(game)
    mid, half = (y1 + y2) / 2.0, abs(y2 - y1) / 2.0 * SHOT_SPAN
    return Vector2D(goal_x, mid - half + 2.0 * half * i / (N_SHOTS - 1))


def dribble_target(game: Game, rid: int, deg: float) -> Vector2D:
    me = game.friendly_robots[rid].p
    goal_x, y1, y2 = enemy_goal_line(game)
    a = math.atan2((y1 + y2) / 2.0 - me.y, goal_x - me.x) + math.radians(deg)
    p = Vector2D(me.x + DRIBBLE_STEP_M * math.cos(a), me.y + DRIBBLE_STEP_M * math.sin(a))
    return clamp_outside_enemy_defense_area(game, clamp_to_field(p, game))


def approach(game: Game, ctx: TickContext, rid: int) -> RobotCommand:
    """`go_to_ball`, except beside the ball: there a robot pressing in cannot turn to face it (pinned
    against it for seconds, `docs/investigation_ball_contact_orientation_divergence.md`), so it backs
    off a few cm turning towards it first."""
    me, ball = game.friendly_robots[rid], game.ball.p.to_2d()
    facing = me.p.angle_to(ball)
    error = abs((facing - me.orientation + math.pi) % (2 * math.pi) - math.pi)
    if me.p.distance_to(ball) < PINNED_M and error > PINNED_RAD:
        back = ball + (me.p - ball).norm() * (PINNED_M + 0.02)
        return move(game, ctx.motion_controller, rid, back, facing, dribbling=True)
    return go_to_ball(game=game, motion_controller=ctx.motion_controller, robot_id=rid, ctx=ctx)


def goal_view(game: Game, rid: int) -> list[float]:
    """The goal's mouth as the ball sees it: its angular width; the keeper's place across it and the
    gaps either side of it (fractions of the width); the keeper's distance off its line; the ball's
    distance to the goal; the rule's gap (`find_best_shot`, as a fraction of the half-width, and
    whether there is none); the attacker's heading off the goal's centre."""
    me, ball = game.friendly_robots[rid], game.ball.p.to_2d()
    keeper = next(iter(game.enemy_robots.values())).p
    goal_x, y1, y2 = enemy_goal_line(game)
    lo, hi, mid = min(y1, y2), max(y1, y2), (y1 + y2) / 2.0
    a_lo, a_hi = math.atan2(lo - ball.y, goal_x - ball.x), math.atan2(hi - ball.y, goal_x - ball.x)
    width = max(a_hi - a_lo, 1e-3)
    a_k = math.atan2(keeper.y - ball.y, keeper.x - ball.x)
    half_k = math.atan2(ROBOT_RADIUS, max(ball.distance_to(keeper), ROBOT_RADIUS))
    best_y, _ = find_best_shot(me.p, list(game.enemy_robots.values()), goal_x, y1, y2)
    heading = (me.orientation - math.atan2(mid - me.p.y, goal_x - me.p.x) + math.pi) % (2 * math.pi) - math.pi
    return [
        width,
        (a_k - a_lo) / width,
        (a_k - half_k - a_lo) / width,
        (a_hi - a_k - half_k) / width,
        abs(goal_x - keeper.x),
        ball.distance_to(Vector2D(goal_x, mid)) / game.field.half_length,
        0.0 if best_y is None else (best_y - mid) / ((hi - lo) / 2.0),
        float(best_y is None),
        heading / math.pi,
    ]


def observe(game: Game, rid: int, mem: StrikerMem, t0: float) -> np.ndarray:
    me, ball = game.friendly_robots[rid], game.ball
    keeper = next(iter(game.enemy_robots.values()))
    hx, hy = game.field.half_length, game.field.half_width
    carry = 0.0 if mem.carry_origin is None else ball.p.to_2d().distance_to(mem.carry_origin)
    return np.array(
        [
            ball.p.x / hx,
            ball.p.y / hy,
            ball.v.x / 4.0,
            ball.v.y / 4.0,
            me.p.x / hx,
            me.p.y / hy,
            me.v.x / 2.0,
            me.v.y / 2.0,
            math.cos(me.orientation),
            math.sin(me.orientation),
            keeper.p.x / hx,
            keeper.p.y / hy,
            keeper.v.x / 2.0,
            keeper.v.y / 2.0,
            (ball.p.x - me.p.x) / 0.2,
            (ball.p.y - me.p.y) / 0.2,
            carry / CARRY_LIMIT_M,
            float(has_ball(game, rid)),
            (game.ts - t0) / EPISODE_S,
            *goal_view(game, rid),
        ],
        dtype=np.float32,
    )


def action_mask(game: Game, rid: int, mem: StrikerMem) -> np.ndarray:
    mask = np.ones(len(ACTIONS), dtype=bool)
    if carry_exhausted(game, mem.carry_origin, stops_after=True):
        mask[N_SHOTS + 1 :] = False
        return mask
    me = game.friendly_robots[rid].p
    for j, deg in enumerate(DRIBBLE_DIRS_DEG):
        # A carry the field or their area cuts to under half a step goes nowhere.
        mask[N_SHOTS + 1 + j] = dribble_target(game, rid, deg).distance_to(me) > DRIBBLE_STEP_M / 2
    return mask


def rule_choice(game: Game, rid: int, mem: StrikerMem, mask: np.ndarray) -> int:
    """The rule: flat_rule's shot (`_score_goal`), at the widest gap past the keeper, as one of the
    `N_SHOTS` points; with no gap, a carry towards the goal while it lasts, then the goal's centre."""
    goal_x, y1, y2 = enemy_goal_line(game)
    best_y, _ = find_best_shot(
        game.friendly_robots[rid].p,
        list(game.enemy_robots.values()),
        goal_x,
        y1,
        y2,
        prev_best_shot_y=mem.shot_y,
        switch_margin=_SHOT_SWITCH_MARGIN,
    )
    mem.shot_y = best_y
    if best_y is None:
        return N_SHOTS + 1 if mask[N_SHOTS + 1] else N_SHOTS // 2
    return min(range(N_SHOTS), key=lambda i: abs(shot_point(game, i).y - best_y))


class StrikerTactic(BaseTactic[StrikerMem]):
    """Robot `robot_id` alone, attacking: `choose(obs, mask) -> (action, logp)` picks from `ACTIONS` with
    the ball on the dribbler; None plays `rule_choice`. Both run through the same execution, so the two
    differ only in what they choose. Every decision is appended to `trajectory` as (obs, mask, action,
    logp)."""

    def __init__(self, choose: Optional[Chooser] = None, robot_id: int = 0):
        self.robot_id = robot_id
        self.choose = choose
        self.t0 = 0.0
        self.trajectory: list[tuple[np.ndarray, np.ndarray, int, float]] = []

    def initial_mem(self) -> StrikerMem:
        return StrikerMem()

    def highlights(self, mem: StrikerMem) -> dict[RobotId, str]:
        return {self.robot_id: ACTIONS[mem.action] if mem.action is not None else "get_ball"}

    def tick(
        self, game: Game, ctx: TickContext, robot_ids: tuple[RobotId, ...], mem: StrikerMem
    ) -> tuple[dict[RobotId, RobotCommand], StrikerMem]:
        rid = self.robot_id
        if game.ball is None or rid not in game.friendly_robots:
            return {}, mem
        mem.carry_origin = carry_origin(game, rid, mem.carry_origin)
        if not has_ball(game, rid, visual=True):
            mem.action = None
            return {rid: approach(game, ctx, rid)}, mem
        on_ball = has_ball(game, rid)
        t = game.ts
        kicking = mem.action is not None and mem.action <= N_SHOTS
        if on_ball and (mem.action is None or t - mem.decided_at >= (SHOT_TIMEOUT_S if kicking else DECIDE_EVERY_S)):
            obs, mask = observe(game, rid, mem, self.t0), action_mask(game, rid, mem)
            a, logp = self.choose(obs, mask) if self.choose is not None else (rule_choice(game, rid, mem, mask), 0.0)
            self.trajectory.append((obs, mask, a, logp))
            mem.action, mem.decided_at = a, t
        if mem.action is None or not on_ball:
            # Not yet on the dribbler, or slipped off it while aiming: turning now would leave the
            # ball behind (and an aimed kick hit nothing), so close in on it first.
            return {rid: approach(game, ctx, rid)}, mem
        return {rid: self._execute(game, ctx.motion_controller, rid, mem.action)}, mem

    @staticmethod
    def _execute(game: Game, mc: MotionController, rid: int, action: int) -> RobotCommand:
        me = game.friendly_robots[rid]
        name = ACTIONS[action]
        if name.startswith("dribble:"):
            goal_x, y1, y2 = enemy_goal_line(game)
            target = dribble_target(game, rid, float(name.split(":")[1]))
            return move(game, mc, rid, target, me.p.angle_to(Vector2D(goal_x, (y1 + y2) / 2.0)), dribbling=True)
        if name.startswith("shoot:"):
            oren = me.p.angle_to(shot_point(game, int(name.split(":")[1])))
            if not oriented_towards(game, rid, oren):
                return turn_on_spot(game=game, motion_controller=mc, robot_id=rid, target_oren=oren, dribbling=True)
        return kick()


# --- the policy ------------------------------------------------------------------------------------


def load_params(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k].astype(np.float32) for k in z.files}


def logits(params: dict[str, np.ndarray], obs: np.ndarray) -> np.ndarray:
    """The policy network (`striker_rl.mlp`): two tanh layers and a linear head. A network trained on
    fewer features (rl1's 19) reads the first ones."""
    h = np.tanh(obs[: len(params["w0"])] @ params["w0"] + params["b0"])
    h = np.tanh(h @ params["w1"] + params["b1"])
    return h @ params["w2"] + params["b2"]


def net_chooser(params: dict[str, np.ndarray], greedy: bool, rng: np.random.Generator) -> Chooser:
    def choose(obs: np.ndarray, mask: np.ndarray) -> tuple[int, float]:
        z = np.where(mask, logits(params, obs).astype(np.float64), -np.inf)
        p = np.exp(z - z.max())
        p /= p.sum()
        a = int(np.argmax(p)) if greedy else int(rng.choice(len(p), p=p))
        return a, float(np.log(p[a]))

    return choose


# --- the duel --------------------------------------------------------------------------------------


def sample_start(rng: random.Random) -> Scenario:
    """Ball anywhere from just inside our half to the edge of their area, at rest or rolling; the
    attacker 0.25-2 m from it on any side; the keeper near its goal's centre."""
    while True:
        bx, by = rng.uniform(-0.5, 3.3), rng.uniform(-2.7, 2.7)
        if bx > 3.2 and abs(by) < 1.25:
            continue
        d, a = rng.uniform(0.25, 2.0), rng.uniform(-math.pi, math.pi)
        rx, ry = bx + d * math.cos(a), by + d * math.sin(a)
        if abs(rx) < 4.3 and abs(ry) < 2.8 and not (rx > 3.3 and abs(ry) < 1.3):
            break
    speed, heading = (0.0 if rng.random() < 0.4 else rng.uniform(0.3, 1.5)), rng.uniform(-math.pi, math.pi)
    return Scenario(
        sim_time=0.0,
        ball_x=bx,
        ball_y=by,
        ball_vx=speed * math.cos(heading),
        ball_vy=speed * math.sin(heading),
        friendly_robots=(RobotState(0, rx, ry, rng.uniform(-math.pi, math.pi), 0.0, 0.0),),
        enemy_robots=(RobotState(0, rng.uniform(4.1, 4.4), rng.uniform(-0.5, 0.5), math.pi, 0.0, 0.0),),
        referee_command=None,
        source_replay=Path("-"),
    )


class Duel:
    """One match, one robot a side, replayed from a fresh start every episode: in rsim, or in an
    external grSim (`mode="grsim"`, real time, `tools/striker_grsim.py`)."""

    def __init__(self, mode: str = "rsim") -> None:
        self.mode = mode
        enable_sim_reuse()
        self.attacker = AbstractStrategy(build_kernel_strategy=build_default_kernel_strategy(()))
        self.keeper = AbstractStrategy(build_kernel_strategy=build_default_kernel_strategy(()))
        profile = load_profile("simulation")
        # One endless half: the teams never change ends.
        profile = dataclasses.replace(profile, game=dataclasses.replace(profile.game, half_duration_seconds=1e9))
        self.referee = CustomReferee(profile, n_robots_yellow=1, n_robots_blue=1)
        self.runner = StrategyRunner(
            strategy=self.attacker,
            opp_strategy=self.keeper,
            my_team_is_yellow=True,
            my_team_is_right=False,
            mode=mode,
            exp_friendly=1,
            exp_enemy=1,
            exp_ball=True,
            referee=self.referee,
            enable_vision_stream=False,
            referee_initial_command=RefereeCommand.HALT,
            control_scheme="fpp",
        )

    def episode(self, rng: random.Random, striker: StrikerTactic, keeper: str) -> str:
        # Each side's only robot is its pinned slot: the attacker's is the striker.
        self.attacker._kernel_strategy.set_goalkeeper(0, striker)
        self.keeper._kernel_strategy.set_goalkeeper(0, KEEPERS[keeper](0))
        self.referee.reset()  # in HALT, so nobody moves while the start settles
        # The last episode's stoppage left the runner re-pinning the ball at its restart spot.
        self.runner._teleport_settle_target = None
        self.runner._pending_placement_teleport = None
        start = sample_start(rng)
        if self.mode == "rsim":
            apply_scenario(self.runner, start, verify=True)
        else:
            # grSim's teleports arrive over the network and come back through its vision a few frames
            # later: place everyone with the ball at rest, hold it (HALT), then set the ball rolling.
            apply_scenario(self.runner, dataclasses.replace(start, ball_vx=0.0, ball_vy=0.0), verify=False)
            for _ in range(GRSIM_SETTLE_TICKS):
                self.runner.step_once()
            self.runner.sim_controller.teleport_ball(start.ball_x, start.ball_y, start.ball_vx, start.ball_vy)
        striker.t0 = self.runner.my.current_game_frame.ts
        self.referee.seed_clock(striker.t0, RefereeCommand.FORCE_START)
        for tick in itertools.count(1):
            self.runner.step_once()
            outcome = self._outcome()
            if outcome is not None:
                return outcome
            # rsim steps game time a tick at a time; grSim runs in real time, so read its clock.
            ts = self.runner.my.current_game_frame.ts
            if (tick / TICKS_PER_SECOND if self.mode == "rsim" else ts - striker.t0) >= EPISODE_S:
                return "timeout"

    def _outcome(self) -> Optional[str]:
        ref = self.runner.my.game.referee
        if ref.yellow_team.score > 0:
            return "goal"
        if ref.referee_command not in LIVE:
            v = self.referee.last_violation
            return f"stopped:{v.rule_name if v is not None else ref.referee_command.name}"
        keeper_view = self.runner.opp.game
        keeper = keeper_view.friendly_robots.get(0)
        ball = keeper_view.ball
        if keeper is not None and keeper.has_ball:
            return "saved"
        if ball is not None and ball_in_own_defense_area(keeper_view) and ball.v.mag() < SAVED_BALL_SPEED:
            return "saved"
        return None


_DUEL: Optional[Duel] = None
_PARAMS: dict[str, dict[str, np.ndarray]] = {}


def _episode(task: tuple[int, str, Optional[str], bool]) -> dict:
    global _DUEL
    seed, keeper, weights, greedy = task
    if _DUEL is None:
        _DUEL = Duel()
    choose = None
    if weights is not None:
        if weights not in _PARAMS:
            _PARAMS.clear()
            _PARAMS[weights] = load_params(Path(weights))
        choose = net_chooser(_PARAMS[weights], greedy, np.random.default_rng(seed))
    striker = StrikerTactic(choose)
    outcome = _DUEL.episode(random.Random(seed), striker, keeper)
    return {"seed": seed, "keeper": keeper, "outcome": outcome, "trajectory": striker.trajectory}


def run(seed: int, episodes: int, keepers: list[str], weights: Optional[str], greedy: bool, workers: int) -> list[dict]:
    """Episode i: start seed `seed * 1_000_000 + i`, keeper `keepers[i % len(keepers)]`."""
    tasks = [(seed * 1_000_000 + i, keepers[i % len(keepers)], weights, greedy) for i in range(episodes)]
    with mp.get_context("fork").Pool(workers) as pool:
        results = list(pool.imap(_episode, tasks, chunksize=4))
    return results


def goal_rates(results: list[dict]) -> dict[str, dict]:
    out = {}
    for keeper in sorted({r["keeper"] for r in results}):
        rs = [r for r in results if r["keeper"] == keeper]
        goals = sum(r["outcome"] == "goal" for r in rs)
        p = goals / len(rs)
        out[keeper] = {
            "episodes": len(rs),
            "goals": goals,
            "goal_rate": round(p, 4),
            "se": round(math.sqrt(p * (1 - p) / len(rs)), 4),
            "outcomes": dict(Counter(r["outcome"] for r in rs).most_common()),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("play", "eval"))
    parser.add_argument("--weights", help="the network (striker_rl's w<k>.npz); none: the rule")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--keepers", default="default,flat")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, help="play: the decisions (.npz); eval: the summary (.json)")
    a = parser.parse_args()
    keepers = a.keepers.split(",")
    results = run(a.seed, a.episodes, keepers, a.weights, a.mode == "eval", a.workers)
    rates = goal_rates(results)
    print(json.dumps({"policy": a.weights or "rule", "seed": a.seed, **rates}, indent=1))
    if a.out is None:
        return
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.mode == "eval":
        a.out.write_text(json.dumps({"policy": a.weights or "rule", "seed": a.seed, **rates}, indent=1))
        return
    steps = [(i, *step) for i, r in enumerate(results) for step in r["trajectory"]]
    np.savez_compressed(
        a.out,
        obs=np.array([s[1] for s in steps], dtype=np.float32).reshape(-1, N_OBS),
        mask=np.array([s[2] for s in steps], dtype=bool).reshape(-1, len(ACTIONS)),
        act=np.array([s[3] for s in steps], dtype=np.int64),
        logp=np.array([s[4] for s in steps], dtype=np.float32),
        episode=np.array([s[0] for s in steps], dtype=np.int64),
        reward=np.array([float(r["outcome"] == "goal") for r in results], dtype=np.float32),
        outcome=np.array([r["outcome"] for r in results]),
        keeper=np.array([r["keeper"] for r in results]),
    )


if __name__ == "__main__":
    main()
