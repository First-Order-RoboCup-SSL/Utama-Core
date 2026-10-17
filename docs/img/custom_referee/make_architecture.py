"""Draws architecture.png for docs/custom_referee.md ("## Architecture"): one tick of CustomReferee, from the
GameFrame through the rule checks and the game state machine to the strategy layer and the simulator.

pixi run python docs/img/custom_referee/make_architecture.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagram_style import INK_2, LOOP, Diagram  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

# Every name a box shows; utama_core/tests/tools/test_diagram_names.py checks they still exist.
NAMES = [
    "CustomReferee",
    "GameFrame",
    "GameStateMachine",
    "RefereeData",
    "RefereeRefiner",
    "RefereeOverride",
    "RefereeMessageReceiver",
    "StrategyRunner",
    "Strategy",
    "MatchStats",
    "_disallowed_goal",
    "_TRANSITION_COOLDOWN",
    "utama_core/custom_referee/custom_referee.py",
    "utama_core/custom_referee/state_machine.py",
    "utama_core/custom_referee/rules",
    "utama_core/entities/game/game_frame.py",
    "utama_core/engine/strategy.py",
    "utama_core/tactics",
]

d = Diagram(
    150,
    128,
    "CustomReferee, one tick",
    "custom_referee/custom_referee.py steps on each GameFrame. The result reaches the strategy through the "
    "same buffer as a real referee.",
    figsize=(14, 12),
)

# --- rule checks (a dashed group) ----------------------------------------------------------------
d.ax.add_patch(
    FancyBboxPatch(
        (2, 55),
        68,
        44,
        boxstyle="round,pad=0,rounding_size=1.2",
        facecolor="none",
        edgecolor=INK_2,
        linewidth=1.0,
        linestyle="--",
        zorder=1,
    )
)
d.note(4, 97.3, "CustomReferee.step: rule checks", fontsize=9.5, weight="bold", va="center")

d.box(
    36,
    109,
    50,
    9,
    "GameFrame, one tick",
    "entities/game/game_frame.py",
    kind="state",
)
d.box(
    36,
    89,
    62,
    13,
    "Stopping rules: the first violation wins",
    "custom_referee/rules/, in priority order: goal, out of bounds, double touch,\n"
    "defense area, pushing, keeper held ball, excessive dribbling, no progress,\n"
    "penalty time limit, defense area stoppage (2nd foul: HALT)",
    kind="decision",
    size=10,
)
d.box(
    36,
    71,
    62,
    15,
    "Non-stopping rules: the first one is kept",
    "they only feed the foul counter; a stopping rule later still wins\n"
    "ball speed, keep out, crashing, robot stop speed,\n"
    "ball placement interference, defense area (attacker touch),\n"
    "defense area stoppage (1st foul)",
    kind="decision",
    size=10,
)
d.note(
    4,
    59.2,
    "Scorer fouled in the last 2 s: the goal is invalid_goal (_disallowed_goal)",
    va="center",
)

# --- state machine and the referee's data --------------------------------------------------------
d.box(
    36,
    41,
    62,
    13,
    "GameStateMachine.step",
    "custom_referee/state_machine.py\n"
    "0.3 s cooldown (_TRANSITION_COOLDOWN), then the violation\n"
    "(a stopping foul or goal changes the command), then auto-advance",
    kind="decision",
)
d.box(
    36,
    24,
    50,
    10,
    "Referee buffer",
    "deque(maxlen=1): only the latest\nStrategyRunner (real: RefereeMessageReceiver)",
    "buffer",
)
d.box(
    36,
    8,
    50,
    10,
    "RefereeRefiner",
    "sets GameFrame.referee: command, stage,\nscores, designated position",
    "refiner",
)

d.main((36, 104.5), (36, 99))
d.main((36, 55), (36, 47.5))
d.main((36, 34.5), (36, 29))
d.main((36, 19), (36, 13))
d.note(37.5, 51.2, "one violation, or none", colour=INK_2, style="italic", va="center")
d.note(37.5, 31.8, "RefereeData", colour=INK_2, style="italic", va="center")

# --- strategy and simulator ----------------------------------------------------------------------
d.box(
    116,
    8,
    60,
    13,
    "Strategy.step",
    "restart, STOP or TIMEOUT: RefereeOverride places the robots\n"
    "otherwise tactics and the partitioner run as usual\n"
    "engine/strategy.py, tactics/",
    kind="decision",
)
d.box(
    116,
    24,
    60,
    12,
    "Simulator",
    "teleports the ball to designated_position;\nMatchStats records last_violation",
    kind="source",
)
d.main((61, 8), (86, 8))
d.note(73.5, 11.4, "game.referee", colour=INK_2, style="italic", ha="center", va="center")
d.arrow((61, 24), (86, 24), colour=INK_2, ls="--")

d.key(
    86,
    62,
    [
        ("source", "outside the process"),
        ("state", "game state"),
        ("decision", "referee or strategy decision"),
        ("buffer", "holds data between stages"),
        ("refiner", "turns data into game state"),
    ],
)
print(d.save(__file__, "architecture.png"))
