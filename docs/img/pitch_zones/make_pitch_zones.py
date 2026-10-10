"""Draws pitch_zones.png for docs/pitch_zones.md, from STANDARD_FIELD_DIMS.

pixi run python docs/img/pitch_zones/make_pitch_zones.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrow, Rectangle  # noqa: E402

from utama_core.config.field_params import STANDARD_FIELD_DIMS as F  # noqa: E402

L, W = F.full_field_half_length, F.full_field_half_width
BOX_D, BOX_W, GOAL = 2 * F.half_defense_area_depth, F.half_defense_area_width, F.half_goal_width
THIRD = 2 * L / 3
CENTRE = BOX_W  # the centre lane is as wide as the box

THIRDS = [("Defensive third", "#c9e4c5"), ("Middle third", "#e8f0d8"), ("Attacking third", "#f6e3c6")]
LANES = [(W, CENTRE, "Left wing"), (CENTRE, -CENTRE, "Centre"), (-CENTRE, -W, "Right wing")]

_LABEL = {"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.85}

fig, ax = plt.subplots(figsize=(11, 7.6))
for i, (name, colour) in enumerate(THIRDS):
    x0 = -L + i * THIRD
    ax.add_patch(Rectangle((x0, -W), THIRD, 2 * W, facecolor=colour, edgecolor="none"))
    ax.text(x0 + THIRD / 2, W + 0.22, f"{name}\n(3 m)", ha="center", va="bottom", fontsize=11, weight="bold")
for y in (CENTRE, -CENTRE):
    ax.plot([-L, L], [y, y], ls="--", lw=1.2, color="#555")
for top, bottom, name in LANES:
    ax.text(L + 0.25, (top + bottom) / 2, f"{name}\n({top - bottom:g} m)", ha="left", va="center", fontsize=10.5)
    for i in range(3):
        x = -L + (i + 0.5) * THIRD
        cell = f"{THIRDS[i][0].split()[0].lower()}\n{name.lower()}"
        ax.text(x, (top + bottom) / 2, cell, ha="center", va="center", fontsize=8.5, color="#444", bbox=_LABEL)

# pitch lines, boxes, goals, centre circle
ax.add_patch(Rectangle((-L, -W), 2 * L, 2 * W, fill=False, lw=2))
ax.plot([0, 0], [-W, W], lw=1.5, color="k")
ax.add_patch(plt.Circle((0, 0), F.center_circle_radius, fill=False, lw=1.5))
for sign, label in ((-1, "our box"), (1, "their box")):
    x0 = sign * L - (BOX_D if sign > 0 else 0)
    ax.add_patch(Rectangle((x0, -BOX_W), BOX_D, 2 * BOX_W, fill=False, lw=2, edgecolor="#b03030"))
    ax.text(sign * (L - BOX_D / 2), -BOX_W - 0.12, label, ha="center", va="top", fontsize=9, color="#b03030")
    gx = sign * L
    ax.add_patch(Rectangle((gx if sign > 0 else gx - F.goal_depth, -GOAL), F.goal_depth, 2 * GOAL, color="k"))
ax.text(-L - 0.3, 0, "our goal", rotation=90, ha="center", va="center", fontsize=9)

# danger: the opponent holding the ball in our defensive third
ax.add_patch(Rectangle((-L, -W), THIRD, 2 * W, fill=False, hatch="//", edgecolor="#e8a0a0", lw=0))
ax.text(
    -L + THIRD / 2,
    -W + 0.3,
    "danger: the opponent has the ball here",
    ha="center",
    fontsize=9,
    color="#b03030",
    bbox=_LABEL,
)

ax.add_patch(FancyArrow(-1.6, -W - 0.55, 3.2, 0, width=0.06, head_width=0.25, head_length=0.3, color="k"))
ax.text(0, -W - 0.8, "we attack this way: left and right are as seen facing this way", ha="center", va="top")

# axis ticks in metres, from our goal line
for i in range(4):
    x = -L + i * THIRD
    ax.text(x, -W - 0.15, f"{i * THIRD:g} m", ha="center", va="top", fontsize=8, color="#444")
ax.set_xlim(-L - 0.6, L + 1.6)
ax.set_ylim(-W - 1.2, W + 0.9)
ax.set_aspect("equal")
ax.axis("off")
fig.tight_layout()
out = Path(__file__).with_name("pitch_zones.png")
fig.savefig(out, dpi=110)
print(out)
