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
     "requires": names from `REQUIRES` below: conditions that must hold during the play, so a
               moment whose situation never arose fails ("did not arise: ...") instead of passing,
     "known_failure": null, or why it still fails (the test is then a strict xfail),
     "designated_position": a ball placement's target [x, y], else null,
     "sim_noise": the match's seeded noise (`sim_noise.py`), or null for none,
     "tactic": null, or {"class", "robots", "mem"} (see "Tactic moments"),
     "scenario": the field state: ball, robots and referee command (`replay/scenario.py`)}

Robots start from rest and every tactic from its initial memory (`replay/scenario.py`, Known
limitations), so a moment pins what the field and the referee command alone cause, which is
most referee failures. Start a moment where the failure is about to happen, not seconds before:
from rest a robot often takes another path. A check is a function of the played ticks, stated
as the behaviour we want, so it does not wait on the referee's persistence before calling a foul.

"all" plays every strategy in the round-robin catalog, except that during a restart the
referee override moves every outfield robot whatever the strategy, so then one strategy per
distinct override (its `referee_overrides` and goalkeeper tactic) is enough.

Requires vs checks: a check says what must not happen, or (`pass_received`, `placement_completes`)
what must happen. A requirement says the situation the moment is about arose: a restart lasted,
a robot came near the ball, the ball was kicked. Without them a moment passes when the opponent
does something else, the ball goes elsewhere, or our robot never gets there.

Tactic moments: "tactic": {"class": "GiveAndGoTactic", "robots": [1, 2], "mem": {"carrier_id": 1}}
plays our side as that one Tactic (`utama_core/tactics/`) on the listed outfield robots, the other
outfield robots holding position, starting from `mem`: the tactic's own memory dataclass, its
fields by name (a field left out keeps its initial value; null means the initial memory). A
value is None, bool, int, float, str, a tuple/list of those, or a Vector2D as [x, y]. Tactic
moments are for live play (NORMAL_START, FORCE_START): the referee override still moves everyone
during a restart. Their one entry is the tactic, so "strategy" must be left out. A tactic moment
fails when its tactic never ticks with one of its robots ("GiveAndGoTactic never ran on robots
[1, 2]"): a tactic that is not applicable never runs, and its robots would silently hold.

Hand-made moments: `new` writes a template (both teams in a fixed formation, ours on the right,
the pool's noise with seed 1, source null) to edit by hand; `save` and `load` refuse an
invalid one, naming each problem (ids, field lines, check and strategy names, tactic).

List our side's fouls in a round-robin run, as starting points, each with the referee command
active then and whether that restart is ours or theirs (`--skip-own-restart` leaves out ours):
    pixi run python tools/moments.py fouls replays/tournament_<id> [--rule keep_out] [--skip-own-restart]
Capture one from a replay (its noise and designated position come with it; `--out` writes
elsewhere than utama_core/tests/moments/):
    pixi run python tools/moments.py capture replays/tournament_<id>/<a>_vs_<b>.npz --t 254.65 \\
        --name free_kick_defender_drives_in --check keeps_clear --strategy <a> --opponent <b> \\
        --note "..." [--out PATH]
Write a hand-made one, then edit its positions and tactic in the JSON:
    pixi run python tools/moments.py new <name> --command NORMAL_START [--ball X Y] \\
        [--strategy NAME] [--seconds N] --check <checks> --note "..."
Play one (`--seed N` replays its noise with another seed, the JSON untouched):
    pixi run python tools/moments.py play utama_core/tests/moments/<name>.json [--strategy NAME] [--seed N]
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import re
import sys
import types
import typing
from pathlib import Path
from typing import Optional, Union

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluation.match import (  # noqa: E402
    N_OUTFIELD,
    OUTFIELD_ROBOT_IDS,
    TICKS_PER_SECOND,
)
from utama_core.config.referee_constants import (  # noqa: E402
    BALL_PLACEMENT_DONE_DISTANCE,
)
from utama_core.entities.referee.referee_command import RefereeCommand  # noqa: E402
from utama_core.replay.scenario import (  # noqa: E402
    RobotState,
    Scenario,
    _referee_command_at,
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
    requires: list[str] = dataclasses.field(default_factory=list)
    strategy: str = "all"
    opponent: str = DEFAULT_OPPONENT
    seconds: float = DEFAULT_SECONDS
    known_failure: Optional[str] = None
    designated_position: Optional[tuple[float, float]] = None
    source: Optional[dict] = None
    sim_noise: Optional[dict] = None
    tactic: Optional[dict] = None  # {"class", "robots", "mem"}: our side plays this Tactic, not a strategy


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
        "requires": m.requires,
        "known_failure": m.known_failure,
        "designated_position": list(m.designated_position) if m.designated_position else None,
        "sim_noise": m.sim_noise,
        "tactic": m.tactic,
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
        requires=list(d.get("requires", [])),
        strategy=_full_name(d.get("strategy", "all")),
        opponent=_full_name(d.get("opponent") or DEFAULT_OPPONENT),
        seconds=float(d.get("seconds", DEFAULT_SECONDS)),
        known_failure=d.get("known_failure"),
        designated_position=tuple(dp) if dp else None,
        source=d.get("source"),
        sim_noise=d.get("sim_noise"),
        tactic=d.get("tactic"),
    )


def load(path: Path) -> Moment:
    m = from_dict(json.loads(Path(path).read_text()))
    validate(m)
    return m


def save(m: Moment, path: Optional[Path] = None) -> Path:
    validate(m)
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
    """The strategies a moment is played against (see the module docstring). A tactic moment's
    one entry is its Tactic's class name."""
    from utama_core.engine.referee_override import is_override_command

    if m.tactic is not None:
        return [m.tactic["class"]]
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


class MomentError(ValueError):
    """A moment that cannot be played: every problem found, each named."""


def referee_rule_names() -> set[str]:
    """The `rule_name=` literals of the referee's rules, read from `custom_referee/` (rules and state machine)."""
    import ast

    names = set()
    for path in (REPO_ROOT / "utama_core" / "custom_referee").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "rule_name" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                        names.add(kw.value.value)
    return names


def validate(m: Moment) -> None:
    """Raise `MomentError` naming every way `m` cannot be played."""
    from utama_core.config.field_params import STANDARD_FIELD_DIMS

    half_x, half_y = STANDARD_FIELD_DIMS.full_field_half_length, STANDARD_FIELD_DIMS.full_field_half_width
    s, problems = m.scenario, []
    for side, robots in (("friendly", s.friendly_robots), ("enemy", s.enemy_robots)):
        got = sorted(r.id for r in robots)
        if got != list(range(N_OUTFIELD + 1)):
            problems.append(f"{side} robots must have ids 0-{N_OUTFIELD} exactly once, got {got}")
        for r in robots:
            if abs(r.x) > half_x or abs(r.y) > half_y:
                problems.append(f"{side} robot {r.id} at ({r.x:.2f}, {r.y:.2f}) is off the pitch")
    if abs(s.ball_x) > half_x or abs(s.ball_y) > half_y:
        problems.append(f"ball at ({s.ball_x:.2f}, {s.ball_y:.2f}) is off the pitch")
    rules = referee_rule_names()
    if not m.checks:
        problems.append("checks is empty: name at least one")
    for c in m.checks:
        if c not in CHECKS and c != "*" and c not in rules:
            problems.append(f"unknown check {c!r}: use one of {sorted([*CHECKS, '*', *sorted(rules)])}")
    for r in m.requires:
        if r not in REQUIRES:
            problems.append(f"unknown requires {r!r}: use one of {sorted(REQUIRES)}")
    if m.seconds <= 0:
        problems.append(f"seconds must be positive, got {m.seconds}")
    names = catalog()
    if m.strategy != "all" and m.strategy not in names:
        problems.append(f"unknown strategy {short_name(m.strategy)!r}")
    if m.opponent not in names:
        problems.append(f"unknown opponent {short_name(m.opponent)!r}")
    if m.tactic is not None:
        if m.strategy != "all":
            problems.append('a tactic moment plays its tactic, so drop "strategy"')
        problems += _tactic_problems(m.tactic)
    if problems:
        raise MomentError(f"{m.name}:\n  " + "\n  ".join(problems))


def _tactic_problems(t: dict) -> list[str]:
    problems = []
    if not isinstance(t, dict):
        return [f"tactic must be an object with class, robots and mem, got {t!r}"]
    if set(t) - {"class", "robots", "mem"}:
        problems.append(f"tactic has unknown keys {sorted(set(t) - {'class', 'robots', 'mem'})}")
    classes = tactic_classes()
    cls = classes.get(t.get("class"))
    if cls is None:
        problems.append(
            f"tactic class {t.get('class')!r} is not a public Tactic in utama_core/tactics/: {sorted(classes)}"
        )
    robots = t.get("robots")
    valid = isinstance(robots, list) and robots and all(type(r) is int for r in robots)
    if not valid or not set(robots) <= set(OUTFIELD_ROBOT_IDS) or len(set(robots)) != len(robots):
        problems.append(
            f"tactic robots must be distinct outfield ids {list(OUTFIELD_ROBOT_IDS)} (0 is the goalkeeper), got {robots!r}"
        )
    if cls is not None:
        try:
            decode_mem(cls, t.get("mem"))
        except MomentError as e:
            problems.append(str(e))
    return problems


def tactic_classes() -> dict[str, type]:
    """Every public Tactic class in `utama_core/tactics/`, by name."""
    import importlib
    import inspect
    import pkgutil

    import utama_core.tactics as pkg

    found = {}
    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{pkg.__name__}.{info.name}")
        for name, obj in vars(module).items():
            if (
                inspect.isclass(obj)
                and obj.__module__ == module.__name__
                and not name.startswith("_")
                and hasattr(obj, "tick")
                and hasattr(obj, "initial_mem")
            ):
                found[name] = obj
    return found


def decode_mem(cls: type, fields: Optional[dict]):
    """`cls`'s memory with the JSON `fields` set on its initial memory (see the module docstring).
    A field left out keeps its initial value, so a mem need not spell out ones JSON cannot hold."""
    base = cls().initial_mem()
    if fields is None:
        return base
    mem_name = type(base).__name__
    if not isinstance(fields, dict) or not dataclasses.is_dataclass(base):
        raise MomentError(f"{mem_name} mem must be an object of its fields, got {fields!r}")
    hints = _field_types(type(base))
    updates = {}
    for key, value in fields.items():
        if key not in hints:
            raise MomentError(f"{mem_name} has no mem field {key!r} (fields: {sorted(hints)})")
        updates[key] = _decode(value, hints[key], f"{mem_name}.{key}")
    return dataclasses.replace(base, **updates)


def _field_types(cls: type) -> dict[str, object]:
    """Each field's annotation, evaluated in its own module (the tactics use string annotations)."""
    import random

    ns = {"random": random, **vars(sys.modules[cls.__module__])}
    out = {}
    for f in dataclasses.fields(cls):
        hint = f.type
        if isinstance(hint, str):
            try:
                hint = eval(hint, ns)
            except Exception:
                pass  # stays a string: `_decode` reports it as unsupported
        out[f.name] = hint
    return out


def _decode(value, hint, where: str):
    """A JSON value as a field of type `hint` (see `decode_mem`)."""
    from utama_core.entities.data.vector import Vector2D

    if value is None:
        return None
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin is Union or origin is types.UnionType:
        options = [a for a in args if a is not type(None)]
        if len(options) == 1:
            return _decode(value, options[0], where)
    elif hint is Vector2D and isinstance(value, list) and len(value) == 2:
        if all(type(v) in (int, float) for v in value):
            return Vector2D(value[0], value[1])
        raise MomentError(f"{where} is a Vector2D [x, y] of numbers, got {value!r}")
    elif hint is bool and type(value) is bool:
        return value
    elif hint is int and type(value) is int:
        return value
    elif hint is float and type(value) in (int, float):
        return float(value)
    elif hint is str and type(value) is str:
        return value
    elif origin is tuple and isinstance(value, list):
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode(v, args[0], where) for v in value)
        if len(args) != len(value):
            raise MomentError(f"{where} must have {len(args)} items, got {value!r}")
        return tuple(_decode(v, a, where) for v, a in zip(value, args))
    else:
        if hint in (bool, int, float, str, Vector2D) or origin is tuple:
            raise MomentError(f"{where} must be {_hint_name(hint)}, got {value!r}")
    raise MomentError(
        f"{where} has type {_hint_name(hint)}, which a moment cannot set: use None, bool, int, float, str, "
        "a list of those (for a tuple), or a Vector2D as [x, y]"
    )


def _hint_name(hint) -> str:
    return hint if isinstance(hint, str) else getattr(hint, "__name__", None) or str(hint)


def _started(cls: type, mem, ticked: Optional[set] = None) -> type:
    """`cls` whose `initial_mem()` is `mem` (a copy each time): a tactic that starts mid-play.
    With `ticked`, each tick adds its robot ids to that set, so a moment can tell it ran."""
    attrs = {
        "initial_mem": lambda self: copy.deepcopy(mem),
        "__module__": cls.__module__,
        "__qualname__": cls.__qualname__,
    }
    if ticked is not None:

        def tick(self, game, ctx, robot_ids, memory):
            ticked.update(robot_ids)
            return cls.tick(self, game, ctx, robot_ids, memory)

        attrs["tick"] = tick
    return type(cls.__name__, (cls,), attrs)


HOLD_ID = "hold"


def build_tactic_kernel_strategy(outfield_robot_ids: tuple[int, ...], tactic: dict, ticked: Optional[set] = None):
    """`build_kernel_strategy` for a tactic moment, built like the `build_*_kernel_strategy`
    factories (`strategy/clear_danger.py`): the moment's Tactic on its robots, the other outfield
    robots holding position (no motion command), and a partition that never changes. A tactic
    that is committed (pinned) or not applicable keeps its robots out of the partition. `ticked`
    collects the robot ids the Tactic ticks with (see `_started`)."""
    from utama_core.engine.context import TickContext
    from utama_core.engine.strategy import Strategy as KernelSchedulerStrategy
    from utama_core.engine.tactic import BaseTactic, TacticTag
    from utama_core.skills.src.utils.move_utils import empty_command

    class HoldPosition(BaseTactic[None]):
        tag = TacticTag.MIXED

        def initial_mem(self):
            return None

        def tick(self, game, ctx, robot_ids, mem):
            return {rid: empty_command() for rid in robot_ids}, None

    cls = tactic_classes()[tactic["class"]]
    tactic_id = cls.__name__
    robots = frozenset(tactic["robots"])
    started = _started(cls, decode_mem(cls, tactic.get("mem")), ticked)

    def partitioner(game, free_robots, prev_partition, available_tactic_ids):
        mine = robots & free_robots
        out, hold = {}, free_robots
        if tactic_id in available_tactic_ids and mine:
            out[tactic_id] = mine
            hold = free_robots - mine
        if hold:
            out[HOLD_ID] = hold
        return out

    def _build(motion_controller):
        ctx = TickContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={tactic_id: started(), HOLD_ID: HoldPosition()},
            partitioner=partitioner,
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build


def template(
    name: str,
    note: str,
    command: str,
    checks: list[str],
    ball: tuple[float, float] = (0.0, 0.0),
    strategy: str = "all",
    seconds: float = DEFAULT_SECONDS,
) -> Moment:
    """A hand-made moment: both teams in a fixed formation (ours yellow on the right, facing left),
    the given ball and referee command, the reference pool's noise with seed 1, source null."""
    from utama_core.replay.scenario import RobotState

    ours = [(4.0, 0.0), (2.2, 1.2), (2.2, -1.2), (1.0, 2.0), (1.0, -2.0), (0.6, 0.0)]

    def robots(positions, facing):
        return tuple(
            RobotState(id=i, x=x, y=y, orientation=facing, vx=0.0, vy=0.0) for i, (x, y) in enumerate(positions)
        )

    pool = json.loads((REPO_ROOT / "tools" / "evaluation" / "pools" / "pool_v1.json").read_text())
    scenario = Scenario(
        sim_time=0.0,
        ball_x=ball[0],
        ball_y=ball[1],
        ball_vx=0.0,
        ball_vy=0.0,
        friendly_robots=robots(ours, math.pi),
        enemy_robots=robots([(-x, y) for x, y in ours], 0.0),
        referee_command=RefereeCommand[command],
        source_replay=Path("hand-made"),
    )
    return Moment(
        name=name,
        note=note,
        scenario=scenario,
        checks=list(checks),
        strategy=_full_name(strategy),
        seconds=seconds,
        sim_noise={**pool["sim_noise"], "seed": 1},
    )


def with_seed(m: Moment, seed: int) -> Moment:
    """`m` played with its sim noise's seed replaced (`play --seed`), the JSON untouched."""
    if m.sim_noise is None:
        raise MomentError(f"{m.name}: no sim_noise to seed (it plays without noise)")
    return dataclasses.replace(m, sim_noise={**m.sim_noise, "seed": seed})


@dataclasses.dataclass(frozen=True)
class Tick:
    """What the checks see of one played tick, in our side's frame."""

    t: float  # seconds since the moment started
    command: Optional[RefereeCommand]
    ball: Optional[tuple[float, float]]
    ours: dict[int, tuple[float, float]]  # robot id -> position
    our_speeds: dict[int, float]
    our_foul: Optional[tuple[str, tuple[int, ...]]]  # (rule name, robot ids) charged to us this tick
    tactic_robots: frozenset[int] = frozenset()  # robots the moment's tactic ticked with this tick
    ball_speed: float = 0.0  # m/s
    designated: Optional[tuple[float, float]] = None  # the referee's designated position (placement, free kick)


def restart_side(command: Optional[RefereeCommand], we_are_yellow: bool = True) -> Optional[str]:
    """ "ours" for a restart our team takes, "theirs" for one the other team takes, None for
    STOP, live play and timeouts."""
    if command is None or command.name.startswith("TIMEOUT"):
        return None
    if command.name.endswith("YELLOW" if we_are_yellow else "BLUE"):
        return "ours"
    if command.name.endswith("BLUE" if we_are_yellow else "YELLOW"):
        return "theirs"
    return None


def _restart_against_us(command: Optional[RefereeCommand], we_are_yellow: bool) -> bool:
    """STOP, or a restart the other team takes: when every robot of ours keeps out."""
    return command is RefereeCommand.STOP or restart_side(command, we_are_yellow) == "theirs"


def keeps_clear(ticks: list[Tick], we_are_yellow: bool = True) -> list[str]:
    """During STOP and the other team's restarts, a robot of ours inside the keep-out circle
    (0.5 m from the ball) never moves closer to the ball, and leaves within 2 s. Moving closer is
    the robot's own move, measured against where the ball is now: a ball that is placed or
    rolls toward a robot standing still is not the robot driving in."""
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
            d_before = math.hypot(bx - now.ball[0], by - now.ball[1])
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


# A kick: the ball at this speed or more (a moving ball at the start is below it).
KICK_SPEED_MPS = 2.0
# A receiver has the ball when it is this close to it: the dribbler takes it within ~0.08 m of the
# centre, and a robot's edge is ~0.11 m, so 0.15 m with the ball slowed is a robot taking it.
RECEIVE_M = 0.15
# A restart's command holds this long from the start, so the restart happened (a free kick ends
# only when it is taken; a ball placement needs the ball settled at the target for 2 s).
RESTART_LASTS_S = 1.0
# A robot of ours this close to the ball: the keep-out circle (0.5 m) was tested, not a robot far off.
NEAR_BALL_M = 1.0


def pass_received(ticks: list[Tick], we_are_yellow: bool = True) -> list[str]:
    """After the ball is kicked (`KICK_SPEED_MPS`), a robot of the moment's tactic other than the
    carrier takes it: the ball slows below `KICK_SPEED_MPS` within `RECEIVE_M` of that robot before
    the end. The carrier is our robot nearest the ball at the start; with no tactic ticking, any
    other robot of ours counts."""
    first = next((t for t in ticks if t.ball is not None and t.ours), None)
    if first is None:
        return ["pass_received: no ball or robots of ours to play"]
    bx, by = first.ball
    carrier = min(first.ours, key=lambda rid: math.hypot(first.ours[rid][0] - bx, first.ours[rid][1] - by))
    tactic = frozenset().union(*(t.tactic_robots for t in ticks))
    receivers = (tactic or frozenset(first.ours)) - {carrier}
    kick = next((i for i, t in enumerate(ticks) if t.ball_speed >= KICK_SPEED_MPS), None)
    if kick is None:
        return [f"pass_received: the ball never reached kick speed ({KICK_SPEED_MPS} m/s), so no pass"]
    for tick in ticks[kick:]:
        if tick.ball is None or tick.ball_speed >= KICK_SPEED_MPS:
            continue  # a ball still at pass speed past a robot is a miss, not a reception
        for rid in receivers:
            if rid in tick.ours and _dist(tick.ours[rid], tick.ball) <= RECEIVE_M:
                return []
    return [f"pass_received: no robot of {sorted(receivers)} got the ball from robot {carrier} by the end"]


def placement_completes(ticks: list[Tick], we_are_yellow: bool = True) -> list[str]:
    """Our ball placement completes: the referee moves on from our BALL_PLACEMENT before the end,
    and the ball is within the rule's tolerance (`BALL_PLACEMENT_DONE_DISTANCE`) of the designated
    position at the last tick of the placement. A timed-out placement leaves it farther off."""
    placing = RefereeCommand.BALL_PLACEMENT_YELLOW if we_are_yellow else RefereeCommand.BALL_PLACEMENT_BLUE
    if not ticks or ticks[0].command is not placing:
        return [f"placement_completes: the moment does not start from our {placing.name}"]
    left = next((i for i, t in enumerate(ticks) if t.command is not placing), None)
    if left is None:
        return [f"placement_completes: still {placing.name} at the end, the ball not placed"]
    last = ticks[left - 1]
    if last.ball is None or last.designated is None:
        return ["placement_completes: no ball or designated position at the last placement tick"]
    dist = _dist(last.ball, last.designated)
    if dist > BALL_PLACEMENT_DONE_DISTANCE:
        return [
            f"placement_completes: {placing.name} ended at t={last.t:.2f}s with the ball {dist:.2f} m from "
            f"the designated position (tolerance {BALL_PLACEMENT_DONE_DISTANCE} m)"
        ]
    return []


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _restart_lasts(ticks: list[Tick], we_are_yellow: bool = True) -> Optional[str]:
    """None when the restart the moment starts from (its first tick's referee command) stays
    active for `RESTART_LASTS_S`, else why not."""
    if not ticks or restart_side(ticks[0].command, we_are_yellow) is None:
        name = ticks[0].command.name if ticks and ticks[0].command else "none"
        return f"the moment does not start from a restart (its first command is {name})"
    start = ticks[0].command
    for tick in ticks:
        if tick.command is not start:
            if tick.t < RESTART_LASTS_S:
                return f"{start.name} ended at {tick.t:.2f} s, need {RESTART_LASTS_S} s"
            return None
    if ticks[-1].t < RESTART_LASTS_S:
        return f"{start.name} still on at the end of the play, {ticks[-1].t:.2f} s, need {RESTART_LASTS_S} s"
    return None


def _robot_near_ball(ticks: list[Tick], we_are_yellow: bool = True) -> Optional[str]:
    """None when some robot of ours comes within `NEAR_BALL_M` of the ball, else the closest."""
    dists = [_dist(pos, t.ball) for t in ticks if t.ball is not None for pos in t.ours.values()]
    closest = min(dists, default=math.inf)
    if closest <= NEAR_BALL_M:
        return None
    return f"the closest robot of ours to the ball was {closest:.2f} m, need {NEAR_BALL_M} m"


def _ball_kicked(ticks: list[Tick], we_are_yellow: bool = True) -> Optional[str]:
    """None when the ball reaches `KICK_SPEED_MPS`, else its top speed."""
    fastest = max((t.ball_speed for t in ticks), default=0.0)
    if fastest >= KICK_SPEED_MPS:
        return None
    return f"the ball's top speed was {fastest:.2f} m/s, need {KICK_SPEED_MPS} m/s"


# Named checks a moment can ask for. Add one only for a failure no referee rule names, or for an
# outcome that must happen (`pass_received`, `placement_completes`).
CHECKS = {
    "keeps_clear": keeps_clear,
    "stop_speed": stop_speed,
    "pass_received": pass_received,
    "placement_completes": placement_completes,
}

# Named conditions a moment can `require`: each returns None when it holds during the play, else
# why it did not arise. Judged in `judge`, reported as "did not arise: <name> (<why>)".
REQUIRES = {
    "restart_lasts": _restart_lasts,
    "robot_near_ball": _robot_near_ball,
    "ball_kicked": _ball_kicked,
}


def judge(m: Moment, ticks: list[Tick], we_are_yellow: bool = True) -> list[str]:
    """Every way `ticks` broke `m.checks`."""
    failures = []
    if m.tactic is not None:
        ran = frozenset().union(*(t.tactic_robots for t in ticks))
        never = sorted(r for r in m.tactic["robots"] if r not in ran)
        if never:
            failures.append(f"{m.tactic['class']} never ran on robots {never}")
    for name in m.requires:
        why = REQUIRES[name](ticks, we_are_yellow)
        if why:
            failures.append(f"did not arise: {name} ({why})")
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
    """Play `m` with `strategy` on our side (yellow) for `m.seconds`. A tactic moment's own
    Tactic plays instead, and `strategy` is ignored."""
    from utama_core.custom_referee import CustomReferee
    from utama_core.engine.abstract_strategy import AbstractStrategy
    from utama_core.replay.scenario import apply_scenario
    from utama_core.rsoccer_simulator.src.Utils.sim_noise import SimNoise
    from utama_core.run import StrategyRunner
    from utama_core.strategy import kernel_strategy

    def side(name: str) -> AbstractStrategy:
        return AbstractStrategy(build_kernel_strategy=getattr(kernel_strategy, name)(OUTFIELD_ROBOT_IDS))

    ticked: set[int] = set()  # robots the tactic ticked with since the last played tick

    def ours() -> AbstractStrategy:
        if m.tactic is None:
            return side(strategy)
        return AbstractStrategy(
            build_kernel_strategy=build_tactic_kernel_strategy(OUTFIELD_ROBOT_IDS, m.tactic, ticked)
        )

    referee = CustomReferee.from_profile_name(
        "simulation", n_robots_yellow=N_OUTFIELD + 1, n_robots_blue=N_OUTFIELD + 1
    )
    runner = StrategyRunner(
        strategy=ours(),
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
                    tactic_robots=frozenset(ticked),
                    ball_speed=game.ball.v.mag() if game.ball is not None else 0.0,
                    designated=(
                        tuple(game.referee.designated_position)
                        if game.referee is not None and game.referee.designated_position is not None
                        else None
                    ),
                )
            )
            ticked.clear()
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


def without_own_restarts(fouls: list[dict]) -> list[dict]:
    """`fouls` minus those during a restart our team takes: `keeps_clear` does not apply there."""
    return [f for f in fouls if f.get("restart") != "ours"]


def our_fouls(run_dir: Path, rule: Optional[str] = None) -> list[dict]:
    """Every foul config_a (our side in the replay) committed in a round-robin run, from its
    summary.json: the starting points for new moments. Each carries the referee `command` active
    at its time (from the match's `.intentions.jsonl`, None if absent) and its `restart` side
    ("ours", "theirs", or None for live play and STOP)."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    out = []
    for r in summary["results"]:
        sidecar = run_dir / f"{short_name(r['config_a'])}_vs_{short_name(r['config_b'])}.intentions.jsonl"
        for f in (r.get("stats") or {}).get("fouls") or []:
            if f.get("side") == "friendly" and (rule is None or f["rule"] == rule):
                command = _referee_command_at(sidecar, f["sim_time"])
                out.append(
                    {
                        "config_a": r["config_a"],
                        "config_b": r["config_b"],
                        **f,
                        "command": command.name if command else None,
                        "restart": restart_side(command),
                    }
                )
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
    p.add_argument("--out", type=Path, default=None, help=f"where to write it (default {MOMENTS_DIR}/<name>.json)")

    p = sub.add_parser("fouls", help="list our side's fouls in a round-robin run")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--rule", default=None)
    p.add_argument(
        "--skip-own-restart",
        action="store_true",
        help="leave out fouls during our own restarts (keeps_clear does not apply there)",
    )

    p = sub.add_parser("new", help="write a hand-made moment template to edit by hand")
    p.add_argument("name", help="lowercase letters, digits and underscores")
    p.add_argument("--command", required=True, choices=[c.name for c in RefereeCommand])
    p.add_argument("--ball", type=float, nargs=2, default=(0.0, 0.0), metavar=("X", "Y"))
    p.add_argument("--strategy", default="all")
    p.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    p.add_argument("--check", nargs="+", required=True, help=f"checks ({', '.join(CHECKS)}), rule names or '*'")
    p.add_argument("--note", required=True, help="what went wrong, and why it must not happen again")

    p = sub.add_parser("play", help="play a moment and print what broke its checks")
    p.add_argument("moment", type=Path)
    p.add_argument("--strategy", default=None, help="one strategy instead of the moment's own")
    p.add_argument("--seed", type=int, default=None, help="the sim noise seed for this run (the JSON is not changed)")

    args = parser.parse_args()
    if args.cmd == "new":
        if not re.fullmatch(r"[a-z0-9_]+", args.name):
            parser.error("the name must be lowercase letters, digits and underscores")
        path = MOMENTS_DIR / f"{args.name}.json"
        if path.exists():
            parser.error(f"{path} exists")
        try:
            m = template(
                args.name,
                args.note,
                args.command,
                args.check,
                ball=tuple(args.ball),
                strategy=args.strategy,
                seconds=args.seconds,
            )
            print(f'Wrote {save(m)}: edit its positions (and "tactic") by hand, then `play` it')
        except MomentError as e:
            parser.error(str(e))
    elif args.cmd == "capture":
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
        print(
            f"Wrote {save(m, args.out)} ({m.scenario.referee_command.name if m.scenario.referee_command else 'no command'})"
        )
    elif args.cmd == "fouls":
        fouls = our_fouls(args.run_dir, args.rule)
        if args.skip_own_restart:
            fouls = without_own_restarts(fouls)
        for f in fouls:
            print(
                f"{f['rule']:<28} t={f['sim_time']:>7.2f}  robot {f.get('robot_id')}  {f.get('tactic') or '':<24} "
                f"{(f['command'] or '?'):<22} {f['restart'] or 'live':<7} "
                f"{short_name(f['config_a'])} vs {short_name(f['config_b'])}"
            )
    else:
        m = load(args.moment)
        if m.tactic is not None and args.strategy:
            parser.error("a tactic moment plays its tactic: --strategy does not apply")
        if args.seed is not None:
            try:
                m = with_seed(m, args.seed)
            except MomentError as e:
                parser.error(str(e))
        for name in [_full_name(args.strategy)] if args.strategy else strategies_for(m):
            failures = judge(m, play(m, name))
            print(f"{short_name(name):<28} {'FAIL' if failures else 'ok'}")
            for f in failures:
                print(f"    {f}")


if __name__ == "__main__":
    main()
