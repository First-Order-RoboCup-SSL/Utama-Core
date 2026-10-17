"""Draws one_tick.png for docs/pipeline_method.md: one tick of StrategyRunner, from the data sources to the
robots, with the class or module behind each box.

pixi run python docs/img/pipeline/make_one_tick.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

INK, INK_2, SURFACE = "#1f1f1f", "#5c5c5c", "#fcfcfb"
LOOP = "#2a78d6"  # the GameFrame on its way round the loop
KINDS = {  # fill, edge
    "source": ("#ececec", "#8a8a8a"),
    "receiver": ("#e6dcf3", "#7c5aa6"),
    "buffer": ("#ffffff", "#8a8a8a"),
    "refiner": ("#fde6cc", "#dd8a25"),
    "state": ("#dcebfa", "#2a78d6"),
    "decision": ("#d8f0e5", "#1b9b6c"),
    "motion": ("#fff1c9", "#c99400"),
    "output": ("#f4f4f4", "#5c5c5c"),
}

fig, ax = plt.subplots(figsize=(14, 11.5))
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.set_xlim(0, 140)
ax.set_ylim(0, 115)
ax.axis("off")


def box(x, y, w, h, title, sub="", kind="buffer", size=10.5):
    """A rounded box centred on (x, y): bold title, smaller detail lines under it."""
    fill, edge = KINDS[kind]
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.25,rounding_size=1.2",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.4,
            zorder=3,
        )
    )
    if sub:
        ax.text(x, y + h / 2 - 1.6, title, ha="center", va="top", fontsize=size, weight="bold", color=INK, zorder=4)
        ax.text(x, y + h / 2 - 4.6, sub, ha="center", va="top", fontsize=8.6, color=INK_2, zorder=4, linespacing=1.35)
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=size, weight="bold", color=INK, zorder=4)


def receiver(x, y, w, h, title, sub):
    """A parallelogram, as in the old drawio picture: a thread or call that pulls data in."""
    fill, edge = KINDS["receiver"]
    s = 1.6
    ax.add_patch(
        Polygon(
            [
                (x - w / 2 + s, y + h / 2),
                (x + w / 2 + s, y + h / 2),
                (x + w / 2 - s, y - h / 2),
                (x - w / 2 - s, y - h / 2),
            ],
            closed=True,
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.4,
            zorder=3,
        )
    )
    ax.text(x, y + 1.3, title, ha="center", va="center", fontsize=10.5, weight="bold", color=INK, zorder=4)
    ax.text(x, y - 1.9, sub, ha="center", va="center", fontsize=8.6, color=INK_2, zorder=4)


def arrow(a, b, label="", colour=INK_2, lw=1.3, style="-|>", ls="-", rad=0.0, lx=0.0, ly=0.0, ha="left"):
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle=style,
            mutation_scale=13,
            color=colour,
            linewidth=lw,
            linestyle=ls,
            connectionstyle=f"arc3,rad={rad}",
            zorder=2,
            shrinkA=0,
            shrinkB=0,
        )
    )
    if label:
        mx, my = (a[0] + b[0]) / 2 + lx, (a[1] + b[1]) / 2 + ly
        ax.text(mx, my, label, ha=ha, va="center", fontsize=8.6, color=INK_2, style="italic", zorder=4)


def loop(points, label_at=None, label=""):
    """The blue main-loop line through `points`, arrowhead at the end."""
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], color=LOOP, linewidth=3.2, solid_capstyle="round", zorder=1)
    arrow(points[-2], points[-1], colour=LOOP, lw=3.2)
    if label_at:
        ax.text(*label_at, label, ha="center", va="bottom", fontsize=9.5, color=LOOP, weight="bold", zorder=4)


ax.text(2, 112, "One tick of StrategyRunner", fontsize=16, weight="bold", color=INK)
ax.text(
    2,
    108.6,
    "utama_core/run/strategy_runner.py, 60 times a second (CONTROL_FREQUENCY). In rsim both teams run in the "
    "same runner: each side refines and steps in turn.",
    fontsize=9.5,
    color=INK_2,
)

# --- sources -------------------------------------------------------------------------------------
box(24, 99, 36, 10, "Vision", "SSL Vision over UDP (real, grSim)\nor the rsim observation", kind="source")
box(70, 99, 36, 10, "Robots", "radio feedback (real)\nor the simulator's responses", kind="source")
box(116, 99, 36, 10, "Referee", "GameController over UDP, or\nCustomReferee stepped in process", kind="source")

# --- receivers -----------------------------------------------------------------------------------
receiver(24, 84, 34, 7, "VisionReceiver", "data_processing/receivers/, own thread")
receiver(70, 84, 34, 7, "get_robots_responses()", "team_controller/ RobotController")
receiver(116, 84, 34, 7, "RefereeReceiver", "or CustomReferee.step (rsim)")
for x in (24, 70, 116):
    arrow((x, 93.6), (x, 88))

# --- buffers -------------------------------------------------------------------------------------
box(24, 71, 34, 6.5, "Vision buffers", "deque(maxlen=1) per camera", kind="buffer", size=10)
box(116, 71, 34, 6.5, "Referee buffer", "deque(maxlen=1): only the latest", kind="buffer", size=10)
arrow((24, 80.4), (24, 74.6), "RawVisionData", lx=1.2)
arrow((116, 80.4), (116, 74.6), "RefereeData", lx=1.2)

# --- refiners on the main loop -------------------------------------------------------------------
Y = 50
box(
    24,
    Y,
    30,
    15,
    "PositionRefiner",
    "CameraCombiner merges the\ncameras, then a Kalman\nfilter per robot and ball",
    kind="refiner",
)
box(58, Y, 22, 15, "VelocityRefiner", "velocities and\naccelerations,\nfrom GameHistory", kind="refiner")
box(
    87,
    Y,
    24,
    15,
    "RobotInfoRefiner",
    "has-ball from our IR,\nor inferred from positions;\nthe enemy's in rsim",
    kind="refiner",
)
box(116, Y, 22, 15, "RefereeRefiner", "command, stage,\nscores, designated\nposition", kind="refiner")
arrow((24, 67.6), (24, 57.6), "latest frame of each camera", lx=1.2)
arrow((70, 80.4), (87, 57.6), "RobotResponse\n(IR has-ball)", rad=0.0, lx=1.5)
arrow((116, 67.6), (116, 57.6), "RefereeData", lx=1.2)

for a, b in ((39.2, 46.8), (69.2, 74.8), (99.2, 104.8)):
    arrow((a, Y), (b, Y), colour=LOOP, lw=3.2)
ax.text(135.2, 40, "GameFrame", fontsize=9, color=LOOP, weight="bold", ha="left", va="center", rotation=90)

# --- state ---------------------------------------------------------------------------------------
box(
    116,
    30,
    28,
    11,
    "Game",
    "latest GameFrame + GameHistory\n(entities/game/), immutable",
    kind="state",
)
loop([(127.2, Y), (134, Y), (134, 30), (130.6, 30)], label_at=(134, Y + 1.2), label="")
box(116, 13, 26, 9, "Recorded", "replay writer, MatchStats,\ndashboard vision stream", kind="output", size=10)
arrow((116, 24.4), (116, 17.6), ls="--")

# --- decision ------------------------------------------------------------------------------------
box(
    77,
    30,
    38,
    16,
    "Strategy.step",
    "engine/strategy.py. Restart, STOP or timeout:\nRefereeOverride places the robots.\nOtherwise the partitioner "
    "splits\nthe robots between tactics (tactics/),\nwhich call skills (skills/)",
    kind="decision",
)
arrow((101.4, 30), (96.6, 30), colour=LOOP, lw=3.2)
box(
    34,
    30,
    26,
    16,
    "MotionController",
    "motion_planning/: fpp (default),\ndwa or trajsample turn each\ntarget into a "
    "velocity,\navoiding robots and the\nball's keep-out",
    kind="motion",
)
arrow((57.4, 30), (47.6, 30))
ax.text(52.5, 27.6, "targets", fontsize=8.6, color=INK_2, style="italic", ha="center", va="top")

# --- output --------------------------------------------------------------------------------------
box(34, 13, 28, 8, "RobotController", "team_controller/src/controllers/", kind="output", size=10)
arrow((34, 21.8), (34, 17.2), "RobotCommand per robot", lx=1.2)
for x, name in ((12, "rsim"), (27, "grSim"), (54, "Real robots")):
    box(x, 3.5, 13 if x != 54 else 16, 4.5, name, kind="output", size=9.5)
arrow((26, 8.8), (12, 5.9))
arrow((30, 8.8), (27, 5.9))
arrow((42, 8.8), (54, 5.9))
loop([(19.8, 13), (1.5, 13), (1.5, Y), (8.8, Y)])
ax.text(2.6, 31, "next tick:\noff rsim, sleep\nto TIMESTEP", fontsize=8.6, color=LOOP, ha="left", va="center")

# --- key -----------------------------------------------------------------------------------------
kx, ky = 62, 3
ax.text(kx + 21, ky + 13.6, "Key", fontsize=9.5, weight="bold", color=INK, ha="center")
for i, (kind, label) in enumerate(
    (
        ("receiver", "receiver"),
        ("refiner", "refiner"),
        ("state", "game state"),
        ("decision", "strategy layer"),
        ("motion", "motion"),
    )
):
    fill, edge = KINDS[kind]
    yy = ky + 10.5 - i * 2.8
    ax.add_patch(FancyBboxPatch((kx + 13, yy - 0.9), 4, 1.8, boxstyle="round,pad=0.1", facecolor=fill, edgecolor=edge))
    ax.text(kx + 18.5, yy, label, fontsize=8.6, color=INK_2, va="center")

out = Path(__file__).with_name("one_tick.png")
fig.savefig(out, dpi=110, bbox_inches="tight", facecolor=SURFACE)
print(out)
