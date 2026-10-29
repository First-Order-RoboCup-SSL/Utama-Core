# ruff: noqa: E402
"""examples/planner_ab.py — two robots swap ends again and again, the planner alternating by block.

Run:
    pixi run python examples/planner_ab.py                    # real robots, fpp vs trajsample
    pixi run python examples/planner_ab.py --blocks 6 --legs 12
    pixi run python examples/planner_ab.py --mode rsim         # rehearse it in the simulator

Is one planner safer than the other on our robots? The same drill under each, in alternating
blocks (A B A B ...) so a battery running down or the floor changing over the session falls on
both alike. In a block both robots run one planner (`--schemes`, default fpp then trajsample).

The robots are `pass_receive.py`'s pair (one per colour), on `shooting_drill.py`'s field. Each
leg, both drive to a new spot at once, facing where they go, across each other's way: head on
(`HEAD_ON`), back, then crossing on the diagonals (`CROSS`), back, and so on. A leg ends once
both stand on their spots (within `ARRIVED_M`), or after `LEG_TIMEOUT_S`. Per leg it records how
near the two came (centre to centre) and how long it took; contact is nearer than two robot
radii (`CONTACT_M`), a close call nearer than `CLOSE_M`. At the end it prints them per planner
and writes them to `replays/field/<stamp>_planner_ab.json`; on real robots every block is also
recorded as a game log.

Clear the field of everything but the two robots; no ball needed. Ctrl+C halts both and stops.
About 40 legs per planner tell a difference in contacts of a few in ten; fewer say little.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import math
import statistics
import time
from typing import Optional

from pass_receive import ROBOT_A, ROBOT_A_COLOUR, ROBOT_B
from shooting_drill import (
    CAMERA_HALF,
    FIELD,
    _field_bounds,
    _one_robot_strategy,
    no_rules_referee,
)

from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.engine.context import TickContext
from utama_core.engine.tactic import BaseTactic, RobotId, TacticTag
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.run import StrategyRunner
from utama_core.skills.src.utils.move_utils import move

# (robot A's spot, robot B's spot) per leg, in turn; the first is where they start.
START = (Vector2D(-0.8, 0.0), Vector2D(0.8, 0.0))
HEAD_ON = (Vector2D(0.8, 0.0), Vector2D(-0.8, 0.0))
CROSS = (Vector2D(0.7, -0.5), Vector2D(-0.7, 0.5))
LEGS = (("head_on", HEAD_ON), ("back", START), ("cross", CROSS), ("back", START))

ARRIVED_M = 0.05
STILL_MPS = 0.1
LEG_TIMEOUT_S = 10.0
CONTACT_M = 2 * ROBOT_RADIUS
CLOSE_M = 0.25  # as the 2026-10-28 field-log close-call scan counted them


class Drill:
    """The legs both robots drive, shared by their two tactics; robot A's measures them."""

    def __init__(self, legs: int):
        self.legs = legs
        self.leg = -1  # -1: getting to the start, not measured
        self.results: list[dict] = []
        self._started: Optional[float] = None
        self._nearest = math.inf

    @property
    def done(self) -> bool:
        return self.leg >= self.legs

    def spot(self, is_a: bool) -> Vector2D:
        spots = START if self.leg < 0 else LEGS[self.leg % len(LEGS)][1]
        return spots[0] if is_a else spots[1]

    def update(self, ts: float, a, b) -> None:
        """One tick of robot A at `a` and robot B at `b` (robots, or None while unseen)."""
        if a is None or b is None or self.done:
            return
        if self._started is None:
            self._started = ts
        self._nearest = min(self._nearest, a.p.distance_to(b.p))
        arrived = all(
            r.p.distance_to(self.spot(is_a)) <= ARRIVED_M and r.v.mag() <= STILL_MPS
            for r, is_a in ((a, True), (b, False))
        )
        timed_out = ts - self._started > LEG_TIMEOUT_S
        if not (arrived or timed_out):
            return
        if self.leg >= 0:
            self.results.append(
                {
                    "leg": self.leg,
                    "kind": LEGS[self.leg % len(LEGS)][0],
                    "nearest_m": round(self._nearest, 3),
                    "seconds": round(ts - self._started, 2),
                    "arrived": arrived,
                }
            )
        self.leg += 1
        self._started, self._nearest = ts, math.inf


class SwapTactic(BaseTactic[None]):
    """Drives its robot to its spot in `drill`, facing where it goes."""

    tag = TacticTag.MIXED

    def __init__(self, drill: Drill, is_a: bool):
        self.drill = drill
        self.is_a = is_a

    def initial_mem(self) -> None:
        return None

    def tick(
        self, game: Game, ctx: TickContext, robot_ids: tuple[RobotId, ...], mem: None
    ) -> tuple[dict[RobotId, RobotCommand], None]:
        still = RobotCommand(0.0, 0.0, 0.0, kick=False, chip=False, dribble=False)
        commands = {rid: still for rid in robot_ids}
        if not robot_ids or self.drill.done:
            return commands, mem
        rid = robot_ids[0]
        me = game.friendly_robots.get(rid)
        if self.is_a:
            other = next(iter(game.enemy_robots.values()), None)
            self.drill.update(game.ts, me, other)
            if self.drill.done:
                return commands, mem
        if me is None:
            return commands, mem
        spot = self.drill.spot(self.is_a)
        facing = me.p.angle_to(spot) if me.p.distance_to(spot) > 0.2 else me.orientation
        commands[rid] = move(game, ctx.motion_controller, rid, spot, facing)
        return commands, mem


def run_block(mode: str, scheme: str, legs: int, block: int, stamp: str) -> list[dict]:
    """One block of `legs` legs, both robots on `scheme`; its legs' results."""
    real = mode == "real"
    a_is_yellow = ROBOT_A_COLOUR == "yellow"
    a_id = ROBOT_A.vision_id if real else 0
    b_id = ROBOT_B.vision_id if real else 0
    a_map, b_map = {ROBOT_A.vision_id: ROBOT_A.firmware_id}, {ROBOT_B.vision_id: ROBOT_B.firmware_id}
    drill = Drill(legs)
    game_log = f"replays/field/{stamp}_real_planner_ab_{block}_{scheme}.log" if real else None
    strategies = [
        _one_robot_strategy(a_id, lambda: SwapTactic(drill, is_a=True)),
        _one_robot_strategy(b_id, lambda: SwapTactic(drill, is_a=False)),
    ]
    for strategy in strategies:
        strategy.exp_ball = False  # a ball in the way is avoided by fpp, not trajsample: no ball, a fair test
    runner = StrategyRunner(
        game_log_path=game_log,
        strategy=strategies[0],
        opp_strategy=strategies[1],
        control_scheme=scheme,
        opp_control_scheme=scheme,
        my_team_is_yellow=a_is_yellow,
        my_team_is_right=False,
        mode=mode,
        exp_friendly=1,
        exp_enemy=1,
        exp_ball=False,
        full_field_dims=FIELD,
        field_bounds=_field_bounds(FIELD, CAMERA_HALF),
        referee=no_rules_referee(1, 1),
        referee_initial_command=RefereeCommand.FORCE_START,
        show_live_status=False,
        yellow_vision_to_cmd_mapping=(a_map if a_is_yellow else b_map) if real else None,
        blue_vision_to_cmd_mapping=(b_map if a_is_yellow else a_map) if real else None,
        yellow_robot_profiles={a_id: ROBOT_A.profile} if a_is_yellow else {b_id: ROBOT_B.profile},
        blue_robot_profiles={b_id: ROBOT_B.profile} if a_is_yellow else {a_id: ROBOT_A.profile},
    )
    print(f"block {block + 1}: {scheme}" + (f", recording to {game_log}" if game_log else ""))
    shown = None
    try:
        while not drill.done:
            runner.step_once()
            if drill.results and drill.results[-1] is not shown:
                shown = r = drill.results[-1]
                flag = " CONTACT" if r["nearest_m"] < CONTACT_M else " close" if r["nearest_m"] < CLOSE_M else ""
                late = "" if r["arrived"] else " (timed out)"
                print(
                    f"  leg {r['leg'] + 1:>2} {r['kind']:<8} nearest {r['nearest_m']:.2f} m, {r['seconds']:.1f} s{late}{flag}"
                )
    finally:
        runner.close()
    return [{**r, "scheme": scheme, "block": block} for r in drill.results]


def summary(results: list[dict], schemes: tuple[str, ...]) -> str:
    lines = [
        f"{'planner':<11} {'legs':>4} {'contact':>7} {'close':>5} {'nearest (median)':>17} "
        f"{'time (median)':>13} timed out"
    ]
    for scheme in schemes:
        rows = [r for r in results if r["scheme"] == scheme]
        if not rows:
            continue
        lines.append(
            f"{scheme:<11} {len(rows):>4} {sum(r['nearest_m'] < CONTACT_M for r in rows):>7} "
            f"{sum(r['nearest_m'] < CLOSE_M for r in rows):>5} "
            f"{statistics.median(r['nearest_m'] for r in rows):>15.2f} m "
            f"{statistics.median(r['seconds'] for r in rows):>11.1f} s "
            f"{sum(not r['arrived'] for r in rows)}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("real", "rsim", "grsim"), default="real")
    parser.add_argument(
        "--schemes", nargs=2, default=("fpp", "trajsample"), help="the two planners (default %(default)s)"
    )
    parser.add_argument("--blocks", type=int, default=4, help="blocks, alternating the planners (default 4)")
    parser.add_argument("--legs", type=int, default=10, help="legs per block (default 10)")
    args = parser.parse_args()
    schemes = tuple(args.schemes)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    results: list[dict] = []
    try:
        for block in range(args.blocks):
            results += run_block(args.mode, schemes[block % 2], args.legs, block, stamp)
    except KeyboardInterrupt:
        print("\nStopped: both robots halted.")
    if not results:
        return
    print("\n" + summary(results, schemes))
    out = Path(f"replays/field/{stamp}_{args.mode}_planner_ab.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schemes": schemes, "legs": results}, indent=1))
    print(f"\nWritten {out}")


if __name__ == "__main__":
    main()
