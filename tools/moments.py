"""Regression moments: one field state where our robots once did something obviously wrong,
replayed for a few seconds against every strategy it concerns, with a fixed check of what they
must do instead.

A moment is a JSON file in `utama_core/tests/moments/` (`test_moments.py` plays each one):

    {"name": ..., "note": what went wrong and why it must not happen again,
     "source": {"replay": ..., "t": ...} or null for a hand-made one,
     "strategy": "all" or one strategy, "opponent": the other side (default split_shape),
     "seconds": how long to play (default 3),
     "checks": names from `CHECKS` below, or referee rule names our side must not break
               (`custom_referee/rules/`, "*" for any),
     "known_failure": null, or why it still fails (the test is then a strict xfail),
     "designated_position": a ball placement's target [x, y], else null,
     "sim_noise": the match's seeded noise (`sim_noise.py`), or null for none,
     "scenario": the field state: ball, robots and referee command (`replay/scenario.py`)}

Robots start from rest and every tactic from its initial memory (`replay/scenario.py`, Known
limitations), so a moment pins what the field and the referee command alone cause, which is
most referee failures. Start a moment where the failure is about to happen, not seconds before:
from rest a robot often takes another path. A check is a function of the played ticks, stated
as the behaviour we want, so it does not wait on the referee's persistence before calling a foul.

"all" plays every strategy in the round-robin catalog, except that during a restart the
referee override moves every outfield robot whatever the strategy, so then one strategy per
distinct override (its `referee_overrides` and goalkeeper tactic) is enough.

List our side's fouls in a round-robin run, as starting points:
    pixi run python tools/moments.py fouls replays/tournament_<id> [--rule keep_out]
Capture one from a replay (its noise and designated position come with it):
    pixi run python tools/moments.py capture replays/tournament_<id>/<a>_vs_<b>.npz --t 254.65 \\
        --name free_kick_defender_drives_in --check keeps_clear --strategy <a> --opponent <b> \\
        --note "..."
Play one:
    pixi run python tools/moments.py play utama_core/tests/moments/<name>.json [--strategy NAME]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluation.match import (  # noqa: E402
    N_OUTFIELD,
    OUTFIELD_ROBOT_IDS,
    TICKS_PER_SECOND,
)
from utama_core.entities.referee.referee_command import RefereeCommand  # noqa: E402
from utama_core.replay.scenario import (  # noqa: E402
    RobotState,
    Scenario,
    scenario_from_replay,
)

MOMENTS_DIR = REPO_ROOT / "utama_core" / "tests" / "moments"
DEFAULT_OPPONENT = "build_split_shape_kernel_strategy"
DEFAULT_SECONDS = 3.0

# Rulebook distances (`custom_referee/rules/keep_out_rule.py`, `robot_stop_speed_rule.py`).
KEEP_OUT_M = 0.5
STOP_SPEED_MPS = 1.5
# A robot inside the keep-out circle may drift with noise, never drive in: closer by more than
# this over `APPROACH_WINDOW_S` is driving in.
APPROACH_TOLERANCE_M = 0.05
APPROACH_WINDOW_S = 0.25
# ... and leaves it within the rulebook's grace for placement interference and STOP speed.
LEAVE_WITHIN_S = 2.0


@dataclasses.dataclass
class Moment:
    name: str
    note: str
    scenario: Scenario
    checks: list[str]
    strategy: str = "all"
    opponent: str = DEFAULT_OPPONENT
    seconds: float = DEFAULT_SECONDS
    known_failure: Optional[str] = None
    designated_position: Optional[tuple[float, float]] = None
    source: Optional[dict] = None
    sim_noise: Optional[dict] = None


def short_name(name: str) -> str:
    return name.removeprefix("build_").removesuffix("_kernel_strategy")


def _full_name(name: str) -> str:
    if name == "all" or name.startswith("build_"):
        return name
    return f"build_{name}_kernel_strategy"


def to_dict(m: Moment) -> dict:
    s = m.scenario
    return {
        "name": m.name,
        "note": m.note,
        "source": m.source,
        "strategy": m.strategy,
        "opponent": m.opponent,
        "seconds": m.seconds,
        "checks": m.checks,
        "known_failure": m.known_failure,
        "designated_position": list(m.designated_position) if m.designated_position else None,
        "sim_noise": m.sim_noise,
        "scenario": {
            "referee_command": s.referee_command.name if s.referee_command is not None else None,
            "ball": [s.ball_x, s.ball_y, s.ball_vx, s.ball_vy],
            "friendly": [dataclasses.asdict(r) for r in s.friendly_robots],
            "enemy": [dataclasses.asdict(r) for r in s.enemy_robots],
        },
    }


def from_dict(d: dict) -> Moment:
    sc = d["scenario"]
    bx, by, bvx, bvy = sc["ball"]
    command = sc.get("referee_command")
    scenario = Scenario(
        sim_time=0.0,
        ball_x=bx,
        ball_y=by,
        ball_vx=bvx,
        ball_vy=bvy,
        friendly_robots=tuple(RobotState(**r) for r in sc["friendly"]),
        enemy_robots=tuple(RobotState(**r) for r in sc["enemy"]),
        referee_command=RefereeCommand[command] if command else None,
        source_replay=Path((d.get("source") or {}).get("replay") or "hand-made"),
    )
    dp = d.get("designated_position")
    return Moment(
        name=d["name"],
        note=d["note"],
        scenario=scenario,
        checks=list(d["checks"]),
        strategy=_full_name(d.get("strategy", "all")),
        opponent=_full_name(d.get("opponent") or DEFAULT_OPPONENT),
        seconds=float(d.get("seconds", DEFAULT_SECONDS)),
        known_failure=d.get("known_failure"),
        designated_position=tuple(dp) if dp else None,
        source=d.get("source"),
        sim_noise=d.get("sim_noise"),
    )


def load(path: Path) -> Moment:
    return from_dict(json.loads(Path(path).read_text()))


def save(m: Moment, path: Optional[Path] = None) -> Path:
    path = Path(path or MOMENTS_DIR / f"{m.name}.json")
    path.write_text(json.dumps(to_dict(m), indent=2) + "\n")
    return path


def catalog() -> list[str]:
    """The round-robin's strategies: every factory `kernel_strategy.py` re-exports, minus the
    retired ones."""
    from tools.evaluation.round_robin import _CONFIG_NAMES, RETIRED

    return [n for n in _CONFIG_NAMES if n not in RETIRED]


def _referee_behaviour(name: str) -> tuple:
    """What decides a strategy's robots during a restart: its referee overrides and its
    goalkeeper tactic. Strategies equal here move identically while the override holds."""
    from utama_core.strategy import kernel_strategy

    strategy = getattr(kernel_strategy, name)(OUTFIELD_ROBOT_IDS)(None)
    overrides = tuple(
        sorted((c.name, getattr(f, "__qualname__", repr(f))) for c, f in strategy.referee_overrides.items())
    )
    pinned = strategy._pinned_slot
    return overrides, type(pinned.tactic).__qualname__ if pinned is not None else None


def strategies_for(m: Moment) -> list[str]:
    """The strategies a moment is played against (see the module docstring)."""
    from utama_core.engine.referee_override import is_override_command

    if m.strategy != "all":
        return [m.strategy]
    names = catalog()
    if not is_override_command(m.scenario.referee_command):
        return names
    seen, out = set(), []
    for n in names:
        key = _referee_behaviour(n)
        if key not in seen:
            seen.add(key)
            out.append(n)
    return out


@dataclasses.dataclass(frozen=True)
class Tick:
    """What the checks see of one played tick, in our side's frame."""

    t: float  # seconds since the moment started
    command: Optional[RefereeCommand]
    ball: Optional[tuple[float, float]]
    ours: dict[int, tuple[float, float]]  # robot id -> position
    our_speeds: dict[int, float]
    our_foul: Optional[tuple[str, tuple[int, ...]]]  # (rule name, robot ids) charged to us this tick


def _restart_against_us(command: Optional[RefereeCommand], we_are_yellow: bool) -> bool:
    """STOP, or a restart the other team takes: when every robot of ours keeps out."""
    if command is RefereeCommand.STOP:
        return True
    theirs = "BLUE" if we_are_yellow else "YELLOW"
    return command is not None and command.name.endswith(theirs) and not command.name.startswith("TIMEOUT")


def keeps_clear(ticks: list[Tick], we_are_yellow: bool = True) -> list[str]:
    """During STOP and the other team's restarts, a robot of ours inside the keep-out circle
    (0.5 m from the ball) never moves closer to the ball, and leaves within 2 s."""
    window = round(APPROACH_WINDOW_S * TICKS_PER_SECOND)
    failures = []
    inside_since: dict[int, float] = {}
    for tick in ticks:
        if not _restart_against_us(tick.command, we_are_yellow) or tick.ball is None:
            inside_since.clear()
            continue
        for rid, (x, y) in tick.ours.items():
            if math.hypot(x - tick.ball[0], y - tick.ball[1]) >= KEEP_OUT_M:
                inside_since.pop(rid, None)
                continue
            since = inside_since.setdefault(rid, tick.t)
            if tick.t - since > LEAVE_WITHIN_S:
                failures.append(
                    f"t={tick.t:.2f}s {tick.command.name}: robot {rid} inside the keep-out circle for {LEAVE_WITHIN_S} s"
                )
    for before, now in zip(ticks, ticks[window:]):
        if not _restart_against_us(now.command, we_are_yellow) or now.ball is None or before.ball is None:
            continue
        for rid, (x, y) in now.ours.items():
            d_now = math.hypot(x - now.ball[0], y - now.ball[1])
            if rid not in before.ours or d_now >= KEEP_OUT_M:
                continue
            bx, by = before.ours[rid]
            d_before = math.hypot(bx - before.ball[0], by - before.ball[1])
            if d_before - d_now > APPROACH_TOLERANCE_M:
                failures.append(
                    f"t={now.t:.2f}s {now.command.name}: robot {rid} drove in to {d_now:.2f} m from the ball "
                    f"(from {d_before:.2f} m {APPROACH_WINDOW_S} s before)"
                )
    return _first_per_robot(failures)


def stop_speed(ticks: list[Tick], we_are_yellow: bool = True) -> list[str]:
    """During STOP no robot of ours moves faster than 1.5 m/s."""
    failures = [
        f"t={tick.t:.2f}s STOP: robot {rid} at {v:.2f} m/s"
        for tick in ticks
        if tick.command is RefereeCommand.STOP
        for rid, v in tick.our_speeds.items()
        if v > STOP_SPEED_MPS
    ]
    return _first_per_robot(failures)


def _first_per_robot(failures: list[str]) -> list[str]:
    """One failure per robot: the first is the one to look at."""
    seen, out = set(), []
    for f in failures:
        robot = f.split("robot ")[1].split(" ")[0]
        if robot not in seen:
            seen.add(robot)
            out.append(f)
    return out


# Named checks a moment can ask for. Add one only for a failure no referee rule names.
CHECKS = {"keeps_clear": keeps_clear, "stop_speed": stop_speed}


def judge(m: Moment, ticks: list[Tick], we_are_yellow: bool = True) -> list[str]:
    """Every way `ticks` broke `m.checks`."""
    failures = []
    for name in m.checks:
        if name in CHECKS:
            failures += CHECKS[name](ticks, we_are_yellow)
    rules = [c for c in m.checks if c not in CHECKS]
    for tick in ticks:
        if tick.our_foul is not None and ("*" in rules or tick.our_foul[0] in rules):
            rule, robots = tick.our_foul
            failures.append(f"t={tick.t:.2f}s referee: {rule} foul by robot(s) {list(robots)}")
    return failures


def play(m: Moment, strategy: str) -> list[Tick]:
    """Play `m` with `strategy` on our side (yellow) for `m.seconds`."""
    from utama_core.custom_referee import CustomReferee
    from utama_core.engine.abstract_strategy import AbstractStrategy
    from utama_core.replay.scenario import apply_scenario
    from utama_core.rsoccer_simulator.src.Utils.sim_noise import SimNoise
    from utama_core.run import StrategyRunner
    from utama_core.strategy import kernel_strategy

    def side(name: str) -> AbstractStrategy:
        return AbstractStrategy(build_kernel_strategy=getattr(kernel_strategy, name)(OUTFIELD_ROBOT_IDS))

    referee = CustomReferee.from_profile_name(
        "simulation", n_robots_yellow=N_OUTFIELD + 1, n_robots_blue=N_OUTFIELD + 1
    )
    runner = StrategyRunner(
        strategy=side(strategy),
        opp_strategy=side(m.opponent),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=N_OUTFIELD + 1,
        exp_enemy=N_OUTFIELD + 1,
        exp_ball=True,
        referee=referee,
        enable_vision_stream=False,
        sim_noise=SimNoise.from_dict(m.sim_noise) if m.sim_noise else None,
    )
    ticks = []
    try:
        apply_scenario(runner, m.scenario, verify=True)
        command = m.scenario.referee_command
        if m.designated_position is not None and command is not None and "PLACEMENT" in command.name:
            ts = runner.my.current_game_frame.ts
            # From STOP, or the referee puts a STOP of its own first (`state_machine.set_command`).
            # A placement leads to the free kick for the same team, as `state_machine.py` has it.
            referee.set_command(RefereeCommand.STOP, ts)
            referee.set_command(
                command,
                ts,
                designated_position=m.designated_position,
                next_command=RefereeCommand[command.name.replace("BALL_PLACEMENT", "DIRECT_FREE")],
            )
        start = runner.my.current_game_frame.ts
        for _ in range(int(m.seconds * TICKS_PER_SECOND)):
            runner.step_once()
            game = runner.my.game
            v = referee.last_violation
            foul = None
            if v is not None and runner.my_team_is_yellow in v.offending_teams:
                foul = (
                    v.rule_name,
                    tuple(rid for yellow, rid in v.offending_robots if yellow == runner.my_team_is_yellow),
                )
            ticks.append(
                Tick(
                    t=round(game.ts - start, 4),
                    command=game.referee.referee_command if game.referee is not None else None,
                    ball=(game.ball.p.x, game.ball.p.y) if game.ball is not None else None,
                    ours={rid: (r.p.x, r.p.y) for rid, r in game.friendly_robots.items()},
                    our_speeds={rid: r.v.mag() for rid, r in game.friendly_robots.items()},
                    our_foul=foul,
                )
            )
    finally:
        runner.close()
    return ticks


def capture(replay: Path, t: float, **fields) -> Moment:
    """A moment from the frame of `replay` nearest `t` (its referee command and designated
    position from the replay's match log, its noise from the run's summary.json). `fields` are
    the other `Moment` fields."""
    scenario = scenario_from_replay(replay, t)
    scenario = _on_the_pitch(scenario)
    summary = Path(replay).parent / "summary.json"
    if fields.get("sim_noise") is None and summary.exists():
        fields["sim_noise"] = json.loads(summary.read_text()).get("sim_noise")
    placement = scenario.referee_command is not None and "PLACEMENT" in scenario.referee_command.name
    if fields.get("designated_position") is None and placement:
        fields["designated_position"] = _designated_position(Path(replay), t)
    return Moment(scenario=scenario, source={"replay": str(replay), "t": t}, **fields)


def _on_the_pitch(scenario: Scenario) -> Scenario:
    """`scenario` with every robot moved onto the field lines, the farthest rsim teleports to:
    a keeper in its goal mouth is a few centimetres behind the line. A ball off the pitch is an
    error: pick a time once a placement has brought it back."""
    from utama_core.config.field_params import STANDARD_FIELD_DIMS

    half_x, half_y = STANDARD_FIELD_DIMS.full_field_half_length, STANDARD_FIELD_DIMS.full_field_half_width
    if abs(scenario.ball_x) > half_x or abs(scenario.ball_y) > half_y:
        raise SystemExit(
            f"The ball is off the pitch at ({scenario.ball_x:.2f}, {scenario.ball_y:.2f}): pick a later --t."
        )

    def clamp(robots):
        return tuple(
            dataclasses.replace(r, x=max(-half_x, min(half_x, r.x)), y=max(-half_y, min(half_y, r.y))) for r in robots
        )

    return dataclasses.replace(
        scenario, friendly_robots=clamp(scenario.friendly_robots), enemy_robots=clamp(scenario.enemy_robots)
    )


def _designated_position(replay: Path, t: float) -> Optional[tuple[float, float]]:
    """The referee's designated position at `t`, from the replay's match log, if it has one."""
    log = replay.with_name(replay.name.split(".")[0] + ".intentions.jsonl")
    if not log.exists():
        return None
    designated = None
    with log.open() as f:
        for line in f:
            event = json.loads(line)
            if event.get("event") != "referee":
                continue
            if event["sim_time"] > t:
                break
            designated = event.get("designated")
    return tuple(designated) if designated else None


def our_fouls(run_dir: Path, rule: Optional[str] = None) -> list[dict]:
    """Every foul config_a (our side in the replay) committed in a round-robin run, from its
    summary.json: the starting points for new moments."""
    summary = json.loads((Path(run_dir) / "summary.json").read_text())
    out = []
    for r in summary["results"]:
        for f in (r.get("stats") or {}).get("fouls") or []:
            if f.get("side") == "friendly" and (rule is None or f["rule"] == rule):
                out.append({"config_a": r["config_a"], "config_b": r["config_b"], **f})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("capture", help="write a moment from a replay")
    p.add_argument("replay", type=Path)
    p.add_argument("--t", type=float, required=True, help="sim time (s) of the frame to start from")
    p.add_argument("--name", required=True)
    p.add_argument("--note", required=True, help="what went wrong, and why it must not happen again")
    p.add_argument(
        "--check",
        nargs="+",
        required=True,
        help=f"checks ({', '.join(CHECKS)}) and/or referee rule names our side must not break ('*' for any)",
    )
    p.add_argument("--strategy", default="all")
    p.add_argument("--opponent", default=DEFAULT_OPPONENT)
    p.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    p.add_argument("--designated-position", type=float, nargs=2, default=None)

    p = sub.add_parser("fouls", help="list our side's fouls in a round-robin run")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--rule", default=None)

    p = sub.add_parser("play", help="play a moment and print what broke its checks")
    p.add_argument("moment", type=Path)
    p.add_argument("--strategy", default=None, help="one strategy instead of the moment's own")

    args = parser.parse_args()
    if args.cmd == "capture":
        m = capture(
            args.replay,
            args.t,
            name=args.name,
            note=args.note,
            checks=args.check,
            strategy=_full_name(args.strategy),
            opponent=_full_name(args.opponent),
            seconds=args.seconds,
            designated_position=tuple(args.designated_position) if args.designated_position else None,
        )
        print(f"Wrote {save(m)} ({m.scenario.referee_command.name if m.scenario.referee_command else 'no command'})")
    elif args.cmd == "fouls":
        for f in our_fouls(args.run_dir, args.rule):
            print(
                f"{f['rule']:<28} t={f['sim_time']:>7.2f}  robot {f.get('robot_id')}  {f.get('tactic') or '':<24} "
                f"{short_name(f['config_a'])} vs {short_name(f['config_b'])}"
            )
    else:
        m = load(args.moment)
        for name in [_full_name(args.strategy)] if args.strategy else strategies_for(m):
            failures = judge(m, play(m, name))
            print(f"{short_name(name):<28} {'FAIL' if failures else 'ok'}")
            for f in failures:
                print(f"    {f}")


if __name__ == "__main__":
    main()
