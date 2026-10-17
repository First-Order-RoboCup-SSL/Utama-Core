"""Figures of a round-robin's strategy signals (`docs/signals.md`) for `docs/signal_report.md`.

    pixi run python tools/signal_report.py replays/tournament_<id> [--out docs/img/signals]

Replays the run's saved matches through `turnover_breakdown.analyse_match` (which carries
`chances`), so it works on runs recorded before chances were measured; it covers the matches
whose replay is in the run directory. Every figure lists the strategies in one order, by points
per match, top first. Writes:

- `results.png`: goals, shots and goals per shot, each for and against: goals per match is shots
  times goals per shot, so it shows whether a strategy wins on volume or on quality;
- `signals.png`: one panel per main signal, every strategy a dot on the signal's own scale, the
  median dashed;
- `shot_quality.png`: goals per shot by distance and open goal mouth, over the whole run;
- `fouls.png`: fouls per match by rule (the ball going out is not a foul: its own panel), with
  the stalled matches;
- `ball_losses.png`: real ball losses per match by kind, for strategies with at least
  `MIN_LOSS_MATCHES` matches as config_a (losses are measured for config_a only).

It also prints the figures' numbers as a markdown table, for the report and for reading without
the images.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utama_core.analysis import turnover_breakdown  # noqa: E402
from utama_core.analysis.strategy_table import strategy_table  # noqa: E402

INK, INK_2, INK_3, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e4e3df"
FOR, AGAINST = "#2a78d6", "#eb6834"  # categorical slots 1 and 2
# Categorical slots in fixed order (validated adjacent-pair for stacks); never cycled.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
MIN_LOSS_MATCHES = 5

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": INK_3,
        "axes.labelcolor": INK_2,
        "axes.titlesize": 10,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_2,
        "ytick.color": INK,
        "font.size": 9,
    }
)


def short(name: str) -> str:
    return name.removeprefix("build_").removesuffix("_kernel_strategy")


def points(row: dict) -> float:
    return (3 * row["wins"] + row["draws"]) / max(1, row["matches"])


def _order(table: dict) -> list[str]:
    return sorted(table, key=lambda n: -points(table[n]))


def _per_match(key: str) -> Callable[[dict], Optional[float]]:
    return lambda r: r[key] / r["matches"] if r["matches"] else None


def _chance(key: str) -> Callable[[dict], Optional[float]]:
    return lambda r: r["chances"].get(key)


def _ratio(num: str, den: str) -> Callable[[dict], Optional[float]]:
    return lambda r: r[num] / r[den] if r[den] else None


# (label, value, which way usually helps: "higher" / "lower" / "", format)
SIGNALS = [
    ("attacking-third entries / match", _per_match("attacking_third_entries"), "higher", "{:.1f}"),
    ("regains followed by a shot", _chance("regain_to_shot"), "higher", "{:.0%}"),
    ("free kicks followed by a shot", _chance("free_kick_to_shot"), "higher", "{:.0%}"),
    ("shot distance (m)", _chance("shot_distance_m"), "lower", "{:.1f}"),
    ("goal mouth open, own shots", _chance("shot_open_goal"), "higher", "{:.0%}"),
    ("completed passes / match", _per_match("completed_passes"), "", "{:.0f}"),
    ("metres gained per pass", lambda r: r["pass_progress_m"], "higher", "{:.2f}"),
    ("danger conceded (s / match)", _chance("danger_s_per_match"), "lower", "{:.0f}"),
    ("goal mouth open, shots faced", _chance("faced_open_goal"), "lower", "{:.0%}"),
    ("save rate", _chance("save_rate"), "higher", "{:.0%}"),
]


def _ylabels(ax, names: list[str], table: dict, show: bool = True) -> None:
    ax.set_yticks(range(len(names)))
    if show:  # shared y: clearing one panel's labels would clear them all, so hide them instead
        ax.set_yticklabels([f"{short(n)}  {points(table[n]):.2f}" for n in names])
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0, labelleft=show)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _dumbbell(ax, names, table, value_for, value_against, fmt) -> None:
    for i, n in enumerate(names):
        a, b = value_for(table[n]), value_against(table[n])
        if a is None or b is None:
            continue
        ax.plot([a, b], [i, i], color=INK_3, lw=2, solid_capstyle="round", zorder=1)
        ax.scatter([b], [i], s=64, color=AGAINST, edgecolors=SURFACE, linewidths=2, zorder=2)
        ax.scatter([a], [i], s=64, color=FOR, edgecolors=SURFACE, linewidths=2, zorder=3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fmt.format(v)))


def results(table: dict, path: Path) -> None:
    names = _order(table)
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.36 * len(names) + 1.9), sharey=True)
    panels = [
        ("goals per match", _per_match("goals_for"), _per_match("goals_against"), "{:.0f}"),
        ("shots per match", _per_match("shots"), _per_match("shots_faced"), "{:.0f}"),
        ("goals per shot", _ratio("goals_for", "shots"), _ratio("goals_against", "shots_faced"), "{:.0%}"),
    ]
    for k, (ax, (title, f, a, fmt)) in enumerate(zip(axes, panels)):
        _dumbbell(ax, names, table, f, a, fmt)
        _ylabels(ax, names, table, show=k == 0)
        ax.set_title(title, loc="left")
    axes[0].set_xlim(left=0)
    axes[1].set_xlim(left=0)
    axes[2].set_xlim(left=0)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=FOR, markersize=8, label="for (scored, shots made)"),
        plt.Line2D([], [], marker="o", ls="", color=AGAINST, markersize=8, label="against (conceded, shots faced)"),
    ]
    fig.legend(handles=handles, loc="upper left", ncol=2, frameon=False, bbox_to_anchor=(0.0, 1.0))
    fig.suptitle(
        "Why each strategy gets its results: goals per match = shots x goals per shot "
        "(strategies by points per match, the number after the name)",
        x=0.0,
        ha="left",
        y=1.07,
        fontsize=10,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def signals(table: dict, path: Path, cols: int = 5) -> None:
    names = _order(table)
    rows = -(-len(SIGNALS) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols + 2.2, (0.3 * len(names) + 1.1) * rows), sharey=True)
    for k, (ax, (label, value, helps, fmt)) in enumerate(zip(axes.flat, SIGNALS)):
        vals = [value(table[n]) for n in names]
        present = [(i, v) for i, v in enumerate(vals) if v is not None]
        ax.scatter(
            [v for _, v in present], [i for i, _ in present], s=56, color=FOR, edgecolors=SURFACE, lw=2, zorder=3
        )
        if present:
            ax.axvline(float(np.median([v for _, v in present])), color=INK_3, lw=1, ls="--", zorder=1)
            lo = min(present, key=lambda p: p[1])
            hi = max(present, key=lambda p: p[1])
            for i, v in (lo, hi):  # label only the extremes; the table has every value
                ax.annotate(
                    fmt.format(v),
                    (v, i),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=INK_2,
                )
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _, f=fmt: f.format(v)))
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        _ylabels(ax, names, table, show=k % cols == 0)
        ax.set_title(label + (f"  ({helps} helps)" if helps else ""), loc="left", fontsize=9)
    for ax in list(axes.flat)[len(SIGNALS) :]:
        ax.set_visible(False)
    fig.suptitle(
        "Signals per strategy, each on its own scale (dashed: median of the 12; labels: lowest and highest)",
        x=0.0,
        ha="left",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


DIST_BINS = [0.0, 1.5, 2.0, 2.5, 3.0, 3.5, 10.0]
OPEN_BINS = [0.0, 0.25, 0.5, 0.75, 1.01]
MIN_CELL_SHOTS = 15


def shot_quality(shots: list[dict], path: Path) -> None:
    goals = np.zeros((len(OPEN_BINS) - 1, len(DIST_BINS) - 1))
    count = np.zeros_like(goals)
    for s in shots:
        i = np.searchsorted(OPEN_BINS, s["open_goal"], side="right") - 1
        j = np.searchsorted(DIST_BINS, s["distance_m"], side="right") - 1
        count[i, j] += 1
        goals[i, j] += s["scored"]
    rate = np.divide(goals, count, out=np.full_like(goals, np.nan), where=count >= MIN_CELL_SHOTS)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
    cmap.set_bad(GRID)
    im = ax.imshow(np.ma.masked_invalid(rate), cmap=cmap, vmin=0, vmax=1, origin="lower", aspect="auto")
    for i in range(rate.shape[0]):
        for j in range(rate.shape[1]):
            n = int(count[i, j])
            if n >= MIN_CELL_SHOTS:
                ax.text(
                    j,
                    i,
                    f"{rate[i, j]:.0%}\n{n} shots",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=SURFACE if rate[i, j] > 0.45 else INK,
                )
            elif n:
                ax.text(j, i, f"{n} shots", ha="center", va="center", fontsize=7, color=INK_2)
    ax.set_xticks(range(len(DIST_BINS) - 1))
    ax.set_xticklabels([f"{a:g}-{b:g}" if b < 10 else f">{a:g}" for a, b in zip(DIST_BINS, DIST_BINS[1:])])
    ax.set_yticks(range(len(OPEN_BINS) - 1))
    ax.set_yticklabels([f"{a:.0%}-{min(b, 1):.0%}" for a, b in zip(OPEN_BINS, OPEN_BINS[1:])])
    ax.set_xlabel("shot distance to goal centre (m)")
    ax.set_ylabel("goal mouth open at the shot")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(
        f"Goals per shot, all {len(shots)} shots of the run (grey: under {MIN_CELL_SHOTS} shots, no rate)", loc="left"
    )
    fig.colorbar(im, ax=ax, label="goals per shot", format=plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _stacked(ax, names, parts: list[tuple[str, list[float]]]) -> None:
    left = np.zeros(len(names))
    for k, (label, vals) in enumerate(parts):
        v = np.array(vals, dtype=float)
        ax.barh(
            range(len(names)), v, left=left, height=0.62, color=SLOTS[k], label=label, edgecolor=SURFACE, linewidth=2
        )
        left += v


LOSS_KINDS = [
    "tackled",
    "ball_out_after_kick",
    "shot_saved_or_blocked",
    "pass_intercepted",
    "loose_ball_lost",
    "foul",
    "ball_out_other",
]


def ball_losses(table: dict, path: Path) -> None:
    names = [n for n in _order(table) if table[n]["matches_as_a"] >= MIN_LOSS_MATCHES]
    fig, ax = plt.subplots(figsize=(10, 0.36 * len(names) + 1.8))
    parts = [
        (
            kind.replace("_", " "),
            [table[n]["real_loss_kinds_as_a"].get(kind, 0) / table[n]["matches_as_a"] for n in names],
        )
        for kind in LOSS_KINDS
    ]
    _stacked(ax, names, parts)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{short(n)}  ({table[n]['matches_as_a']} matches)" for n in names])
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("real ball losses per match (not won back within 1 s)")
    ax.set_title(
        f"How each strategy loses the ball (as config_a only; strategies with {MIN_LOSS_MATCHES}+ such matches)",
        loc="left",
    )
    ax.legend(ncol=4, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.12 - 0.3 / max(1, len(names))))
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


BALL_OUT = "out_of_bounds"


def _fouls_per_match(summary: dict, table: dict) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {n: {} for n in table}
    for rule, info in summary["fouls"].items():
        for key, count in info["by_tactic"].items():
            strategy = f"build_{key.split('/')[0]}_kernel_strategy"
            if strategy in table:
                out[strategy][rule] = out[strategy].get(rule, 0.0) + count / table[strategy]["matches"]
    return out


def fouls(summary: dict, table: dict, path: Path, top: int = 5) -> None:
    names = _order(table)
    per = _fouls_per_match(summary, table)
    rules = [r for r in summary["fouls"] if r != BALL_OUT]
    shown = sorted(rules, key=lambda r: -summary["fouls"][r]["total"])[:top]
    parts = [(r.replace("_", " "), [per[n].get(r, 0.0) for n in names]) for r in shown]
    parts.append(("other", [sum(v for r, v in per[n].items() if r not in shown and r != BALL_OUT) for n in names]))
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(13, 0.36 * len(names) + 2.0), sharey=True, gridspec_kw={"width_ratios": [3, 1]}
    )
    _stacked(a, names, parts)
    for i, n in enumerate(names):
        total = sum(v for _, vals in parts for v in [vals[i]])
        stalled = table[n]["stalled"]
        if stalled:
            a.text(total + 0.1, i, f"{stalled} stalled", va="center", fontsize=8, color=INK_2)
    _ylabels(a, names, table)
    a.set_title("fouls per match, by rule", loc="left")
    a.legend(ncol=6, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.08 - 0.3 / len(names)))
    b.barh(range(len(names)), [per[n].get(BALL_OUT, 0.0) for n in names], height=0.62, color=INK_3)
    _ylabels(b, names, table, show=False)
    b.set_title("ball out of play per match (not a foul)", loc="left")
    fig.suptitle(
        "Fouls by the offending strategy, and the matches it stalled in (each stall counts for both teams)",
        x=0.0,
        ha="left",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def markdown(table: dict) -> str:
    """The figures' numbers, one row per strategy in the figures' order."""
    cols = [
        ("points / match", points, "{:.2f}"),
        ("GF", _per_match("goals_for"), "{:.1f}"),
        ("GA", _per_match("goals_against"), "{:.1f}"),
        ("shots", _per_match("shots"), "{:.1f}"),
        ("shots faced", _per_match("shots_faced"), "{:.1f}"),
        ("goals / shot", _ratio("goals_for", "shots"), "{:.0%}"),
        ("conceded / shot faced", _ratio("goals_against", "shots_faced"), "{:.0%}"),
    ] + [(label, value, fmt) for label, value, _, fmt in SIGNALS]
    lines = ["| strategy | " + " | ".join(c[0] for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for n in _order(table):
        cells = []
        for _, value, fmt in cols:
            v = value(table[n])
            cells.append("-" if v is None else fmt.format(v))
        lines.append(f"| `{short(n)}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build(summary: dict, losses: list[dict], out: Path) -> dict:
    """Write every figure for `summary` (a round-robin's summary.json) and its replays'
    `turnover_breakdown.analyse_match` records. Returns the strategy table plotted, each row
    with `shots_faced` added."""
    out.mkdir(parents=True, exist_ok=True)
    real = {r["match"]: turnover_breakdown.real_loss_kinds(r) for r in losses}
    chances_by_match = {r["match"]: r["chances"] for r in losses if "chances" in r}
    table = strategy_table(summary["results"], real, chances_by_match)
    for row in table.values():
        row["shots_faced"] = 0
    for r in summary["results"]:
        shots = (r.get("stats") or {}).get("shots") or {}
        for name, theirs in ((r["config_a"], "enemy"), (r["config_b"], "friendly")):
            if name in table:
                table[name]["shots_faced"] += shots.get(theirs, 0)
    shots = [s for c in chances_by_match.values() for s in c["shots"]]
    results(table, out / "results.png")
    signals(table, out / "signals.png")
    shot_quality(shots, out / "shot_quality.png")
    ball_losses(table, out / "ball_losses.png")
    fouls(summary, table, out / "fouls.png")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "img" / "signals")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    summary = json.loads((args.run_dir / "summary.json").read_text())
    losses = turnover_breakdown.analyse_run(args.run_dir, args.workers)
    table = build(summary, losses, args.out)
    print(f"{len(losses)} of {len(summary['results'])} matches had a replay; figures in {args.out}\n")
    print(markdown(table))


if __name__ == "__main__":
    main()
