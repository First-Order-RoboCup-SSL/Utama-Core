"""The shared style of the drawn architecture pictures (docs/img/*/make_*.py): boxes by kind, receivers,
arrows, the blue main-loop line and a key, on one canvas in made-up units.

Each picture script also lists, in a module-level `NAMES`, every class, function, constant or path its
boxes name; `utama_core/tests/tools/test_diagram_names.py` checks they still exist, so a rename fails
`pixi run test` instead of leaving a picture that is quietly wrong.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

INK, INK_2, SURFACE = "#1f1f1f", "#5c5c5c", "#fcfcfb"
LOOP = "#2a78d6"  # the main data path: thick and blue
KINDS = {  # fill, edge. A kind means the same thing in every picture.
    "source": ("#ececec", "#8a8a8a"),  # outside the process: network, simulator, robots
    "receiver": ("#e6dcf3", "#7c5aa6"),  # pulls data in (drawn as a parallelogram)
    "buffer": ("#ffffff", "#8a8a8a"),  # holds data between stages
    "refiner": ("#fde6cc", "#dd8a25"),  # turns data into game state
    "state": ("#dcebfa", "#2a78d6"),  # game state, files on disk, anything read back later
    "decision": ("#d8f0e5", "#1b9b6c"),  # the strategy layer and the referee's decisions
    "motion": ("#fff1c9", "#c99400"),  # motion planning and control
    "output": ("#f4f4f4", "#5c5c5c"),  # where commands or results leave
}


class Diagram:
    """One picture: a canvas `w` by `h` units, title and subtitle at the top left."""

    def __init__(self, w: float, h: float, title: str, subtitle: str = "", figsize=(14, 11.5)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.fig.patch.set_facecolor(SURFACE)
        self.ax.set_facecolor(SURFACE)
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(0, h)
        self.ax.axis("off")
        self.ax.text(2, h - 3, title, fontsize=16, weight="bold", color=INK)
        if subtitle:
            self.ax.text(2, h - 6.4, subtitle, fontsize=9.5, color=INK_2)

    def box(self, x, y, w, h, title, sub="", kind="buffer", size=10.5):
        """A rounded box centred on (x, y): bold title, smaller detail lines under it."""
        fill, edge = KINDS[kind]
        self.ax.add_patch(
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
        ax = self.ax
        if sub:
            ax.text(x, y + h / 2 - 1.6, title, ha="center", va="top", fontsize=size, weight="bold", color=INK, zorder=4)
            ax.text(
                x, y + h / 2 - 4.6, sub, ha="center", va="top", fontsize=8.6, color=INK_2, zorder=4, linespacing=1.35
            )
        else:
            ax.text(x, y, title, ha="center", va="center", fontsize=size, weight="bold", color=INK, zorder=4)

    def receiver(self, x, y, w, h, title, sub):
        """A parallelogram: a thread or call that pulls data in."""
        fill, edge = KINDS["receiver"]
        s = 1.6
        corners = [
            (x - w / 2 + s, y + h / 2),
            (x + w / 2 + s, y + h / 2),
            (x + w / 2 - s, y - h / 2),
            (x - w / 2 - s, y - h / 2),
        ]
        self.ax.add_patch(Polygon(corners, closed=True, facecolor=fill, edgecolor=edge, linewidth=1.4, zorder=3))
        self.ax.text(x, y + 1.3, title, ha="center", va="center", fontsize=10.5, weight="bold", color=INK, zorder=4)
        self.ax.text(x, y - 1.9, sub, ha="center", va="center", fontsize=8.6, color=INK_2, zorder=4)

    def arrow(self, a, b, label="", colour=INK_2, lw=1.3, ls="-", rad=0.0, lx=0.0, ly=0.0, ha="left"):
        """An arrow from `a` to `b`; `label` in italics beside its midpoint, moved by (lx, ly)."""
        self.ax.add_patch(
            FancyArrowPatch(
                a,
                b,
                arrowstyle="-|>",
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
            self.ax.text(mx, my, label, ha=ha, va="center", fontsize=8.6, color=INK_2, style="italic", zorder=4)

    def loop(self, points):
        """The thick blue main-path line through `points`, arrowhead at the end."""
        xs, ys = zip(*points)
        self.ax.plot(xs[:-1], ys[:-1], color=LOOP, linewidth=3.2, solid_capstyle="round", zorder=1)
        self.arrow(points[-2], points[-1], colour=LOOP, lw=3.2)

    def main(self, a, b):
        """One straight segment of the main path."""
        self.arrow(a, b, colour=LOOP, lw=3.2)

    def note(self, x, y, text, colour=INK_2, **kw):
        """Free text, 8.6 pt unless `fontsize` says otherwise."""
        self.ax.text(x, y, text, **{"fontsize": 8.6, "color": colour, **kw})

    def key(self, x, y, entries, title="Key"):
        """A key at (x, y) (its top left): one swatch per (kind, label). A second column passes title=""."""
        self.ax.text(x, y, title, fontsize=9.5, weight="bold", color=INK, va="top")
        for i, (kind, label) in enumerate(entries):
            fill, edge = KINDS[kind]
            yy = y - 3.2 - i * 2.8
            self.ax.add_patch(
                FancyBboxPatch((x, yy - 0.9), 4, 1.8, boxstyle="round,pad=0.1", facecolor=fill, edgecolor=edge)
            )
            self.ax.text(x + 5.5, yy, label, fontsize=8.6, color=INK_2, va="center")

    def save(self, script: str, name: str) -> Path:
        """Write `name` beside the calling script (pass `__file__`)."""
        out = Path(script).with_name(name)
        self.fig.savefig(out, dpi=110, bbox_inches="tight", facecolor=SURFACE)
        return out
