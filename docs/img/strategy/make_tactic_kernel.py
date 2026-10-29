"""Draws tactic_kernel.png for docs/STRATEGY_DEVELOPMENT.md ("The tactic-kernel model"): the classes that make up
one team's Strategy, who holds or calls whom, and the path a tick takes from StrategyRunner down to the skills.

pixi run python docs/img/strategy/make_tactic_kernel.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagram_style import LOOP, Diagram  # noqa: E402

# Every name a box shows; utama_core/tests/tools/test_diagram_names.py checks they still exist.
NAMES = [
    "StrategyRunner",
    "AbstractStrategy",
    "Strategy",
    "RefereeOverride",
    "_TacticSlot",
    "Partitioner",
    "TickContext",
    "MatchLog",
    "Tactic",
    "BaseTactic",
    "AbstractRobotController",
    "DEFAULT_COMMITMENT_DEADLINE_S",
    "utama_core/run/strategy_runner.py",
    "utama_core/engine/abstract_strategy.py",
    "utama_core/engine/strategy.py",
    "utama_core/engine/referee_override.py",
    "utama_core/engine/referee_reset.py",
    "utama_core/engine/tactic.py",
    "utama_core/engine/context.py",
    "utama_core/engine/match_log.py",
    "utama_core/strategy/split_shape.py",
    "utama_core/tactics",
    "utama_core/skills/src",
]

d = Diagram(
    150,
    112,
    "The tactic-kernel model: one team's Strategy",
    "Each tick the Strategy re-partitions the outfield robots across its tactics. The blue line is the path "
    "one tick takes.",
    figsize=(14, 10.5),
)

L, M, R = 22, 80, 130

# --- top: the driver and the wrapper -------------------------------------------------------------
d.box(L, 98, 38, 10, "StrategyRunner", "run/strategy_runner.py,\nonce per side per tick", kind="source")
d.box(
    M,
    98,
    46,
    13,
    "AbstractStrategy",
    "engine/abstract_strategy.py: wraps the Strategy\n"
    "a factory built; robots missing from\nthe command map get execute_default_action",
    kind="decision",
)
d.main((41, 98), (57, 98))
d.note(49, 100.8, "step()", colour=LOOP, ha="center", style="italic")

# --- the scheduler -------------------------------------------------------------------------------
d.box(
    M,
    76,
    46,
    17,
    "Strategy.tick(game)",
    "engine/strategy.py. Barrier reset at a restart,\npins committed slots, filters applicable(),\n"
    "validates the partition, ticks each slot",
    kind="decision",
)
d.main((M, 91.5), (M, 84.5))
d.box(
    L,
    76,
    38,
    15,
    "_TacticSlot, one per tactic id",
    "tactic, mem, assigned_robots,\ncommitted_ticks. Cleared at each\nrestart barrier. Deadline:\n"
    "DEFAULT_COMMITMENT_DEADLINE_S (15 s)",
    kind="state",
)
d.arrow((M - 23, 76), (L + 19, 76), colour="#5c5c5c", ls="--")
d.box(
    R,
    76,
    36,
    13,
    "RefereeOverride",
    "engine/referee_override.py:\nrestart or STOP, robots\nplaced before tactics tick",
    kind="decision",
)
d.arrow((M + 23, 76), (R - 18, 76))

# --- the partition and the context ---------------------------------------------------------------
d.box(
    M,
    54,
    46,
    12,
    "Partitioner",
    "strategy/split_shape.py and similar: splits\nthe free robots over the applicable tactics",
    kind="decision",
)
d.main((M, 67.5), (M, 60))
d.note(M + 2, 63.8, "free robots", colour=LOOP, ha="left", va="center", style="italic")
d.box(
    R,
    54,
    36,
    12,
    "TickContext",
    "motion_controller, match_log:\npassed to every tactic tick",
    kind="buffer",
)

# --- tactics -------------------------------------------------------------------------------------
d.box(
    M,
    32,
    46,
    14,
    "Tactic (tactics/)",
    "tick(game, ctx, robot_ids, mem)\n-> commands, new mem. BaseTactic\ngives the defaults. tag: colour only",
    kind="decision",
)
d.main((M, 47.5), (M, 39))
d.arrow((R - 18, 48), (M + 23, 36), colour="#5c5c5c", ls="--")
d.box(R, 30, 36, 12, "MatchLog", "intention, trace, referee events;\nto_jsonl after the match", kind="state")
d.arrow((R, 48), (R, 36), "match_log", lx=1.4, ly=0, ha="left", colour="#5c5c5c", ls="--")

# --- skills and output ---------------------------------------------------------------------------
d.box(
    M,
    10,
    46,
    12,
    "Skills (skills/src/)",
    "go_to_ball, go_to_point, ... each\nreturns a RobotCommand",
    kind="motion",
)
d.main((M, 25), (M, 16))
d.box(
    L,
    10,
    38,
    12,
    "AbstractRobotController",
    "team_controller/: add_robot_commands,\nthen send_robot_commands",
    kind="output",
)
d.arrow((M - 23, 10), (L + 19, 10), "RobotCommand", lx=0, ly=2.6)

d.note(
    3,
    60,
    "One writer per robot: the partition is set\nonce per tick, before any tactic runs. A restart\n"
    "barrier clears every slot's mem and commitment.",
    ha="left",
    va="top",
)

d.key(
    108,
    22,
    [
        ("source", "driver (run/)"),
        ("decision", "strategy layer"),
        ("buffer", "per-tick context"),
        ("state", "slot and log state"),
        ("motion", "skills"),
        ("output", "robot controller"),
    ],
)
print(d.save(__file__, "tactic_kernel.png"))
