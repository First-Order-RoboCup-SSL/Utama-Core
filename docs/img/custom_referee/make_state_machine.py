"""Draws state_machine.png for docs/custom_referee.md ("## State machine and auto-advance"): the referee
commands GameStateMachine moves between, with the trigger or timer on each transition. The blue line is the
live-play cycle: STOP, a free kick, NORMAL_START, and a stopping foul or goal back to STOP.

pixi run python docs/img/custom_referee/make_state_machine.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagram_style import INK_2, SURFACE, Diagram  # noqa: E402

LABEL_BOX = {"facecolor": SURFACE, "edgecolor": "none", "pad": 1.0}

# Every name a box or note shows; utama_core/tests/tools/test_diagram_names.py checks they still exist.
NAMES = [
    "GameStateMachine",
    "RefereeCommand",
    "utama_core/custom_referee/state_machine.py",
]

d = Diagram(
    200,
    138,
    "Referee commands and their transitions",
    "custom_referee/state_machine.py, GameStateMachine. A box is a RefereeCommand (* = the team); each arrow is "
    "its trigger or timer.",
    figsize=(16, 11.5),
)

# --- states --------------------------------------------------------------------------------------
d.box(14, 100, 24, 10, "HALT", "starts here", kind="output")
d.box(50, 76, 30, 12, "STOP", "robots clear the ball", kind="decision")
d.box(102, 114, 34, 10, "BALL_PLACEMENT_*", kind="decision", size=10)
d.box(102, 94, 34, 10, "PREPARE_KICKOFF_*", kind="decision", size=10)
d.box(102, 74, 34, 10, "PREPARE_PENALTY_*", kind="decision", size=10)
d.box(102, 54, 34, 10, "DIRECT_FREE_*", kind="decision", size=10)
d.box(102, 34, 34, 10, "FORCE_START", kind="decision", size=10)
d.box(176, 74, 30, 12, "NORMAL_START", "the ball is in play", kind="decision", size=10)
d.box(50, 24, 30, 10, "POST_GAME", "stage, with STOP", kind="output")

# --- the live-play cycle (blue): STOP -> free kick -> NORMAL_START -> STOP --------------------------
d.main((63, 72), (85, 57))
d.main((119, 56), (161, 70))
d.loop([(176, 80), (176, 124), (50, 124), (50, 82)])
d.note(112, 126.5, "a stopping foul or goal, or the half's time up", colour=INK_2, style="italic", ha="center")

# --- queued restarts out of STOP -----------------------------------------------------------------
d.arrow((60, 82), (85, 112), "placement", lx=-1.5, ly=2.0, ha="right")
d.arrow((64, 80), (85, 94), "kick-off", lx=-1.0, ly=-3.6, ha="right")
d.arrow((65, 76), (85, 74), "penalty", lx=1.0, ly=2.4)
d.arrow((60, 70), (85, 36), "FORCE_START", lx=-2.5, ly=0.0, ha="right")
d.arrow((50, 70), (50, 29), "second half's time used up", lx=-2.0, ly=-6.0, ha="right")
d.arrow((24, 95), (40, 82), "resumed after\n5 s (sim)", lx=-1.5, ly=-2.5, ha="right")

# --- back into the live-play path and between restarts -------------------------------------------
d.arrow((102, 109), (102, 99), "placed, then the queued restart", lx=1.5, ly=0.0, ha="left")
d.arrow((102, 49), (102, 39), "no kick in 10 s", lx=1.5, ly=0.0, ha="left")
d.arrow((119, 92), (161, 78), "3 s prepared, held 2 s", lx=0.0, ly=4.0, ha="left")
d.arrow((119, 74), (161, 74), "penalty ready, held 2 s", lx=-19.0, ly=3.0, ha="left")
d.note(142.0, 51.5, "ball unmoved 10 s", colour=INK_2, style="italic", ha="center", va="center", bbox=LABEL_BOX)
d.note(
    134.0, 68.0, "ball moved, or kicker ready", colour=INK_2, style="italic", ha="center", va="center", bbox=LABEL_BOX
)
d.arrow((165, 68), (119, 35))

# --- notes ---------------------------------------------------------------------------------------
d.note(
    4,
    13,
    "A queued restart leaves STOP once every robot is 0.5 m from the ball, or after 15 s.\n"
    "Any stopping foul or goal sends play to STOP; a second defense area stoppage foul sends any state to HALT.\n"
    "Non-stopping fouls change no state. A placement that is not made in 10 s counts as a failure.",
    colour=INK_2,
    va="top",
    linespacing=1.5,
)

d.key(
    160,
    30,
    [
        ("output", "halted or ended"),
        ("decision", "referee command"),
    ],
)
d.note(160, 19, "blue: the live-play cycle", colour=INK_2, style="italic")

print(d.save(__file__, "state_machine.png"))
