"""chase_e2e.py — chasing the ball, end to end: one robot alone on the field gets the ball onto its
dribbler, driven by a network at 20 Hz with no planner or skill in between (`striker_e2e`'s action:
forward, sideways and turn speeds, kick, dribbler). Each episode puts the robot and the ball anywhere on
the field (`sample_start`), the ball at rest or rolling up to 3 m/s, in live play, and runs until the
robot holds the ball (`CATCH_TICKS` ticks of dribbler contact in a row: "caught"), play stops (the ball
out, a foul) or `EPISODE_S` seconds pass. The control is `--baseline`: the `go_to_ball` skill every
strategy chases with, or `approach` (`striker_duel`'s, which backs off a robot pinned beside the ball).

  pixi run python -m tools.chase_e2e play --weights c.npz --episodes 1500 --workers 16 --out it0.npz
  pixi run python -m tools.chase_e2e eval [--weights c.npz | --baseline go_to_ball] --episodes 2000 --workers 16
  pixi run python -m tools.chase_e2e live --weights c.npz --sim grsim --episodes 20 [--record NAME --out r.json]

`play` samples the network's actions and writes each decision with its shaping potential (`phi`: nearer
the ball, and facing it, is better) for `striker_e2e_rl.py --task chase`. `eval` plays greedily and
prints the catch rate and the time to the catch; the same --seed gives every chaser the same starts.
`live` plays greedily one episode after another, in rsim or an external grSim (running, set up as
`docs/setup_external.md` says), printing each; --record writes a replay for `render_clip`.
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
from pathlib import Path
from typing import Optional

import numpy as np

from tools.striker_duel import GRSIM_SETTLE_TICKS, LIVE, TICKS_PER_SECOND, approach, load_params
from tools.striker_e2e import N_ACT, OBS_CLIP, E2EMem, E2ETactic, _local, actor
from utama_core.custom_referee import CustomReferee
from utama_core.custom_referee.profiles.profile_loader import load_profile
from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.engine.context import TickContext
from utama_core.engine.tactic import BaseTactic, RobotId
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.game import Game
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.replay.columnar_writer import ColumnarReplayWriterConfig
from utama_core.replay.scenario import RobotState, Scenario, apply_scenario
from utama_core.rsoccer_simulator.src.Simulators.robosim.robosim_wrapper import (
    enable_sim_reuse,
)
from utama_core.run import StrategyRunner
from utama_core.skills.src.go_to_ball import go_to_ball
from utama_core.strategy.default import build_default_kernel_strategy

EPISODE_S = 6.0
CATCH_TICKS = 6  # 0.1 s of dribbler contact in a row
N_OBS = 23
PHI_DIST, PHI_FACING = 0.2, 0.05  # shaping potential per metre to the ball, per radian off facing it
MAX_BALL_SPEED = 3.0


def facing_error(game: Game, rid: int) -> float:
    me, ball = game.friendly_robots[rid], game.ball.p.to_2d()
    return abs((me.p.angle_to(ball) - me.orientation + math.pi) % (2 * math.pi) - math.pi)


class ChaseTactic(E2ETactic):
    """`E2ETactic` seeing the chase (`observe`): the ball in its own frame, its motion, its own, its last
    action, the contact streak, where it is on the field."""

    def __init__(self, act, robot_id: int = 0):
        super().__init__(act=act, robot_id=robot_id)
        self.streak = 0

    def tick(self, game, ctx, robot_ids, mem):
        me = game.friendly_robots.get(self.robot_id)
        self.streak = self.streak + 1 if me is not None and me.has_ball else 0
        return super().tick(game, ctx, robot_ids, mem)

    def observe(self, game: Game, rid: int, mem: E2EMem) -> np.ndarray:
        me, ball = game.friendly_robots[rid], game.ball
        h, hx, hy = me.orientation, game.field.half_length, game.field.half_width
        bx, by = _local(ball.p.to_2d() - me.p, h)
        dist = max(math.hypot(bx, by), 1e-6)
        return np.array(
            [
                bx,
                by,
                min(dist, 6.0),
                bx / dist,
                by / dist,
                *(c / 4.0 for c in _local(ball.v, h)),
                *(c / 2.0 for c in _local(me.v, h)),
                *mem.prev,
                float(me.has_ball),
                min(self.streak, CATCH_TICKS) / CATCH_TICKS,
                me.p.x / hx,
                me.p.y / hy,
                math.cos(h),
                math.sin(h),
                ball.p.x / hx,
                ball.p.y / hy,
                (game.ts - self.t0) / EPISODE_S,
            ],
            dtype=np.float32,
        ).clip(-OBS_CLIP, OBS_CLIP)

    def potential(self, game: Game, rid: int) -> float:
        dist = game.friendly_robots[rid].p.distance_to(game.ball.p.to_2d())
        return -PHI_DIST * dist - PHI_FACING * facing_error(game, rid)


class SkillChase(BaseTactic[None]):
    """The control: a skill every tick (60 Hz)."""

    def __init__(self, skill: str, robot_id: int = 0):
        self.robot_id, self.skill, self.t0 = robot_id, skill, 0.0
        self.trajectory: list = []

    def initial_mem(self) -> None:
        return None

    def highlights(self, mem) -> dict[RobotId, str]:
        return {self.robot_id: self.skill}

    def tick(self, game: Game, ctx: TickContext, robot_ids, mem) -> tuple[dict[RobotId, RobotCommand], None]:
        rid = self.robot_id
        if game.ball is None or rid not in game.friendly_robots:
            return {}, mem
        if self.skill == "approach":
            return {rid: approach(game, ctx, rid)}, mem
        return {rid: go_to_ball(game=game, motion_controller=ctx.motion_controller, robot_id=rid, ctx=ctx)}, mem


def sample_start(rng: random.Random) -> Scenario:
    """The robot and the ball anywhere on the field, 0.3-6 m apart, the ball outside both areas, at rest
    or rolling."""

    def in_an_area(x: float, y: float, margin: float) -> bool:
        return abs(x) > 3.5 - margin and abs(y) < 1.0 + margin

    while True:
        rx, ry = rng.uniform(-4.2, 4.2), rng.uniform(-2.7, 2.7)
        bx, by = rng.uniform(-4.0, 4.0), rng.uniform(-2.6, 2.6)
        if in_an_area(bx, by, 0.2) or in_an_area(rx, ry, 0.1):
            continue
        if 0.3 <= math.hypot(bx - rx, by - ry) <= 6.0:
            break
    speed = 0.0 if rng.random() < 0.3 else rng.uniform(0.3, MAX_BALL_SPEED)
    heading = rng.uniform(-math.pi, math.pi)
    return Scenario(
        sim_time=0.0,
        ball_x=bx,
        ball_y=by,
        ball_vx=speed * math.cos(heading),
        ball_vy=speed * math.sin(heading),
        friendly_robots=(RobotState(0, rx, ry, rng.uniform(-math.pi, math.pi), 0.0, 0.0),),
        enemy_robots=(),
        referee_command=None,
        source_replay=Path("-"),
    )


class Chase:
    """One robot alone, replayed from a fresh start every episode: in rsim, or in an external grSim
    (`mode="grsim"`, real time). `record`: a replay name under `replays/` (as `striker_duel.Duel`)."""

    def __init__(self, mode: str = "rsim", record: Optional[str] = None) -> None:
        self.mode = mode
        enable_sim_reuse()
        self.chaser = AbstractStrategy(build_kernel_strategy=build_default_kernel_strategy(()))
        profile = load_profile("simulation")
        profile = dataclasses.replace(profile, game=dataclasses.replace(profile.game, half_duration_seconds=1e9))
        self.referee = CustomReferee(profile, n_robots_yellow=1, n_robots_blue=0)
        self.runner = StrategyRunner(
            strategy=self.chaser,
            my_team_is_yellow=True,
            my_team_is_right=False,
            mode=mode,
            exp_friendly=1,
            exp_enemy=0,
            exp_ball=True,
            referee=self.referee,
            enable_vision_stream=False,
            referee_initial_command=RefereeCommand.HALT,
            control_scheme="fpp",
            **(
                {"replay_writer_config": ColumnarReplayWriterConfig(replay_name=record, overwrite_existing=True)}
                if record is not None
                else {}
            ),
        )

    def episode(self, rng: random.Random, tactic) -> str:
        self.chaser._kernel_strategy.set_goalkeeper(0, tactic)  # its only robot is the pinned slot
        self.referee.reset()
        self.runner._teleport_settle_target = None  # see striker_duel.Duel.episode
        self.runner._pending_placement_teleport = None
        start = sample_start(rng)
        # Placed at rest (a fast ball leaves the verifier's tolerance while it settles), then set rolling.
        # grSim's teleports come back through its vision a few frames later: hold the start (HALT) instead.
        apply_scenario(self.runner, dataclasses.replace(start, ball_vx=0.0, ball_vy=0.0), verify=self.mode == "rsim")
        if self.mode != "rsim":
            for _ in range(GRSIM_SETTLE_TICKS):
                self.runner.step_once()
        if start.ball_vx or start.ball_vy or self.mode != "rsim":
            self.runner.sim_controller.teleport_ball(start.ball_x, start.ball_y, start.ball_vx, start.ball_vy)
        tactic.t0 = self.runner.my.current_game_frame.ts
        self.referee.seed_clock(tactic.t0, RefereeCommand.FORCE_START)
        streak = 0
        for tick in itertools.count(1):
            self.runner.step_once()
            game = self.runner.my.game
            if game.referee.referee_command not in LIVE:
                v = self.referee.last_violation
                return f"stopped:{v.rule_name if v is not None else game.referee.referee_command.name}"
            me = game.friendly_robots.get(0)
            streak = streak + 1 if me is not None and me.has_ball else 0
            if streak >= CATCH_TICKS:
                return "caught"
            elapsed = tick / TICKS_PER_SECOND if self.mode == "rsim" else game.ts - tactic.t0
            if elapsed >= EPISODE_S:
                return "timeout"


_CHASE: Optional[Chase] = None
_PARAMS: dict[str, dict[str, np.ndarray]] = {}


def _episode(task: tuple[int, Optional[str], str, bool]) -> dict:
    global _CHASE
    seed, weights, baseline, greedy = task
    if _CHASE is None:
        _CHASE = Chase()
    if weights is not None:
        if weights not in _PARAMS:
            _PARAMS.clear()
            _PARAMS[weights] = load_params(Path(weights))
        tactic = ChaseTactic(actor(_PARAMS[weights], greedy, np.random.default_rng(seed)))
    else:
        tactic = SkillChase(baseline)
    outcome = _CHASE.episode(random.Random(seed), tactic)
    seconds = _CHASE.runner.my.current_game_frame.ts - tactic.t0
    return {"seed": seed, "keeper": "chase", "outcome": outcome, "seconds": seconds, "trajectory": tactic.trajectory}


def live(
    seed: int, episodes: int, weights: Optional[str], baseline: str, mode: str, record: Optional[str]
) -> list[dict]:
    """Episodes one after another in one `Chase` (grSim needs this: one simulator), greedy, printed as they end."""
    chase = Chase(mode=mode, record=record)
    params = load_params(Path(weights)) if weights is not None else None
    results = []
    try:
        for i in range(episodes):
            seed_i = seed * 1_000_000 + i
            tactic = ChaseTactic(actor(params, True, np.random.default_rng(seed_i))) if params else SkillChase(baseline)
            outcome = chase.episode(random.Random(seed_i), tactic)
            t1 = chase.runner.my.current_game_frame.ts
            results.append({"seed": seed_i, "outcome": outcome, "seconds": t1 - tactic.t0, "t0": tactic.t0, "t1": t1})
            caught = sum(r["outcome"] == "caught" for r in results)
            print(
                f"episode {i + 1}/{episodes}: {outcome} in {t1 - tactic.t0:.2f} s  (caught {caught}/{i + 1})",
                flush=True,
            )
    finally:
        chase.runner.close()
    return results


def run(seed: int, episodes: int, weights: Optional[str], baseline: str, greedy: bool, workers: int) -> list[dict]:
    tasks = [(seed * 1_000_000 + i, weights, baseline, greedy) for i in range(episodes)]
    with mp.get_context("fork").Pool(workers) as pool:
        return list(pool.imap(_episode, tasks, chunksize=4))


def summarize(results: list[dict]) -> dict:
    caught = [r["seconds"] for r in results if r["outcome"] == "caught"]
    p = len(caught) / len(results)
    return {
        "episodes": len(results),
        "success_rate": round(p, 4),
        "se": round(math.sqrt(p * (1 - p) / len(results)), 4),
        "median_s": round(float(np.median(caught)), 3) if caught else None,
        "mean_s": round(float(np.mean(caught)), 3) if caught else None,
        "outcomes": dict(Counter(r["outcome"] for r in results).most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("play", "eval", "live"))
    parser.add_argument("--weights", help="the network (striker_e2e_rl --task chase's c<k>.npz)")
    parser.add_argument("--baseline", choices=("go_to_ball", "approach"), default="go_to_ball")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--sim", choices=("rsim", "grsim"), default="rsim", help="live: rsim or an external grSim")
    parser.add_argument("--record", help="live: a replay name under replays/, every tick (episode t0-t1 in --out)")
    a = parser.parse_args()
    if a.mode == "live":
        results = live(a.seed, a.episodes, a.weights, a.baseline, a.sim, a.record)
    else:
        results = run(a.seed, a.episodes, a.weights, a.baseline, a.mode == "eval", a.workers)
    summary = {"policy": a.weights or a.baseline, "seed": a.seed, "chase": summarize(results)}
    print(json.dumps(summary, indent=1))
    if a.out is None:
        return
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.mode == "live":
        a.out.write_text(json.dumps({**summary, "episodes": results}, indent=1))
        return
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
