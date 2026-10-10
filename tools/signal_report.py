"""Figures of a round-robin's strategy signals (`docs/signals.md`) for `docs/signal_report.md`.

    pixi run python tools/signal_report.py replays/tournament_<id> [--out docs/img/signals]

Replays the run's saved matches through `turnover_breakdown.analyse_match` (which carries
`chances`), so it works on runs recorded before chances were measured; it covers the matches
whose replay is in the run directory. Writes:

- `signal_heatmap.png`: strategies (by points per match) x the main signals, each column coloured
  by rank among the strategies, green where the value usually helps, with the value printed;
- `attack_defense.png`: shots created against danger conceded, and the openness of the shots a
  strategy faces against its save rate, marker size by points per match;
- `shot_quality.png`: goals per shot by distance and open goal mouth, over the whole run;
- `ball_losses.png`: real ball losses per match by kind (config_a matches only);
- `fouls.png`: fouls per match by rule, and stalled matches.
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

from tools.evaluation.round_robin import strategy_table  # noqa: E402
from utama_core.analysis import turnover_breakdown  # noqa: E402


def short(name: str) -> str:
    return name.removeprefix("build_").removesuffix("_kernel_strategy")


def points(row: dict) -> float:
    return (3 * row["wins"] + row["draws"]) / max(1, row["matches"])


def _per_match(key: str) -> Callable[[dict], Optional[float]]:
    return lambda r: r[key] / r["matches"] if r["matches"] else None


def _chance(key: str) -> Callable[[dict], Optional[float]]:
    return lambda r: r["chances"].get(key)


def _per_chance_match(key: str) -> Callable[[dict], Optional[float]]:
    return lambda r: r["chances"][key] / r["chances"]["matches"] if r["chances"]["matches"] else None


# (group, label, value, +1 higher usually helps / -1 lower does / 0 neither, format)
COLUMNS = [
    ("Attack", "shots", _per_match("shots"), 1, "{:.1f}"),
    ("Attack", "entries", _per_match("attacking_third_entries"), 1, "{:.1f}"),
    ("Attack", "regain>shot", _chance("regain_to_shot"), 1, "{:.0%}"),
    ("Attack", "regain>shot s", _chance("regain_to_shot_s"), -1, "{:.1f}"),
    ("Attack", "fk>shot", _chance("free_kick_to_shot"), 1, "{:.0%}"),
    ("Attack", "shot dist m", _chance("shot_distance_m"), -1, "{:.1f}"),
    ("Attack", "open goal", _chance("shot_open_goal"), 1, "{:.0%}"),
    ("Ball", "passes", _per_match("completed_passes"), 0, "{:.0f}"),
    ("Ball", "progress m", lambda r: r["pass_progress_m"], 1, "{:.2f}"),
    ("Ball", "forward", lambda r: r["forward_pass_share"], 1, "{:.0%}"),
    (
        "Ball",
        "real losses",
        lambda r: r["real_losses_as_a"] / r["matches_as_a"] if r["matches_as_a"] else None,
        -1,
        "{:.0f}",
    ),
    ("Defense", "GA", _per_match("goals_against"), -1, "{:.1f}"),
    ("Defense", "danger s", _chance("danger_s_per_match"), -1, "{:.0f}"),
    ("Defense", "open faced", _chance("faced_open_goal"), -1, "{:.0%}"),
    ("Defense", "save", _chance("save_rate"), 1, "{:.0%}"),
    ("Defense", "regains", _per_chance_match("regains"), 1, "{:.0f}"),
    ("Health", "fouls", _per_match("fouls"), -1, "{:.1f}"),
    ("Health", "stalled", lambda r: r["stalled"], -1, "{:.0f}"),
]


def _ranked(values: list[Optional[float]], direction: int) -> list[float]:
    """0..1 by rank among the non-missing values, 1 where it usually helps; 0.5 if missing."""
    present = sorted(v for v in values if v is not None)
    out = []
    for v in values:
        if v is None or len(present) < 2:
            out.append(0.5)
            continue
        below = sum(p < v for p in present) + 0.5 * (sum(p == v for p in present) - 1)
        r = below / (len(present) - 1)
        out.append(r if direction >= 0 else 1 - r)
    return out


def heatmap(table: dict, path: Path) -> None:
    names = sorted(table, key=lambda n: -points(table[n]))
    rows = [table[n] for n in names]
    cells = np.array([_ranked([c[2](r) for r in rows], c[3]) for c in COLUMNS]).T
    fig, ax = plt.subplots(figsize=(1.0 * len(COLUMNS) + 3.5, 0.42 * len(rows) + 2.2))
    for j, col in enumerate(COLUMNS):
        cmap = plt.get_cmap("Blues" if col[3] == 0 else "RdYlGn")
        for i, r in enumerate(rows):
            v = col[2](r)
            shade = cells[i, j] * 0.6 + 0.2 if col[3] == 0 else cells[i, j]
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=cmap(shade)))
            ax.text(j + 0.5, i + 0.5, "-" if v is None else col[4].format(v), ha="center", va="center", fontsize=8)
    ax.set_xlim(0, len(COLUMNS))
    ax.set_ylim(len(rows), 0)
    ax.set_xticks(np.arange(len(COLUMNS)) + 0.5)
    ax.set_xticklabels([c[1] for c in COLUMNS], rotation=40, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels([f"{short(n)}  {points(table[n]):.2f}" for n in names], fontsize=9)
    groups: dict[str, list[int]] = {}
    for j, c in enumerate(COLUMNS):
        groups.setdefault(c[0], []).append(j)
    for g, js in groups.items():
        ax.text((js[0] + js[-1] + 1) / 2, -0.4, g, ha="center", va="bottom", fontsize=10, weight="bold")
        ax.axvline(js[0], color="white", lw=3)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(
        "Signals per strategy (rows by points per match; colour = rank among strategies, green where it usually "
        "helps, blue = neither)",
        fontsize=10,
        pad=40,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _scatter(ax, table, x, y, xlabel, ylabel, title):
    names = [n for n in table if x(table[n]) is not None and y(table[n]) is not None]
    xs, ys = [x(table[n]) for n in names], [y(table[n]) for n in names]
    pts = [points(table[n]) for n in names]
    sc = ax.scatter(xs, ys, s=[40 + 90 * p for p in pts], c=pts, cmap="viridis", alpha=0.8, edgecolors="k")
    for n, a, b in zip(names, xs, ys):
        ax.annotate(short(n), (a, b), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.axvline(float(np.median(xs)), color="grey", lw=0.8, ls="--")
    ax.axhline(float(np.median(ys)), color="grey", lw=0.8, ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    return sc


def attack_defense(table: dict, path: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 6))
    _scatter(
        a,
        table,
        _chance("danger_s_per_match"),
        _per_match("shots"),
        "danger conceded (s per match the enemy holds the ball in our defensive third)",
        "shots per match",
        "Attack vs defense (top left: creates a lot, concedes little)",
    )
    sc = _scatter(
        b,
        table,
        _chance("faced_open_goal"),
        _chance("save_rate"),
        "open goal mouth on the shots faced",
        "save rate",
        "Shots allowed (left: the defense blocks the lane)",
    )
    fig.colorbar(sc, ax=[a, b], label="points per match", fraction=0.025)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


DIST_BINS = [0.0, 1.5, 2.0, 2.5, 3.0, 3.5, 10.0]
OPEN_BINS = [0.0, 0.25, 0.5, 0.75, 1.01]


def shot_quality(shots: list[dict], path: Path) -> None:
    goals = np.zeros((len(OPEN_BINS) - 1, len(DIST_BINS) - 1))
    count = np.zeros_like(goals)
    for s in shots:
        i = np.searchsorted(OPEN_BINS, s["open_goal"], side="right") - 1
        j = np.searchsorted(DIST_BINS, s["distance_m"], side="right") - 1
        count[i, j] += 1
        goals[i, j] += s["scored"]
    rate = np.divide(goals, count, out=np.full_like(goals, np.nan), where=count > 0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(rate, cmap="RdYlGn", vmin=0, vmax=1, origin="lower", aspect="auto")
    for i in range(rate.shape[0]):
        for j in range(rate.shape[1]):
            if count[i, j]:
                ax.text(j, i, f"{rate[i, j]:.0%}\nn={int(count[i, j])}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(DIST_BINS) - 1))
    ax.set_xticklabels([f"{a:g}-{b:g}" if b < 10 else f">{a:g}" for a, b in zip(DIST_BINS, DIST_BINS[1:])])
    ax.set_yticks(range(len(OPEN_BINS) - 1))
    ax.set_yticklabels([f"{a:.0%}-{min(b, 1):.0%}" for a, b in zip(OPEN_BINS, OPEN_BINS[1:])])
    ax.set_xlabel("shot distance to goal centre (m)")
    ax.set_ylabel("goal mouth open at the shot")
    ax.set_title(f"Goals per shot, all {len(shots)} shots of the run", fontsize=10)
    fig.colorbar(im, ax=ax, label="goals per shot")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


LOSS_KINDS = [
    "tackled",
    "shot_saved_or_blocked",
    "ball_out_after_kick",
    "pass_intercepted",
    "loose_ball_lost",
    "foul",
    "ball_out_other",
]


def ball_losses(table: dict, path: Path) -> None:
    names = sorted((n for n in table if table[n]["matches_as_a"]), key=lambda n: -points(table[n]))[::-1]
    fig, ax = plt.subplots(figsize=(10, 0.35 * len(names) + 1.5))
    left = np.zeros(len(names))
    colors = plt.get_cmap("tab10")
    for k, kind in enumerate(LOSS_KINDS):
        vals = np.array(
            [table[n]["real_loss_kinds_as_a"].get(kind, 0) / table[n]["matches_as_a"] for n in names], dtype=float
        )
        ax.barh(range(len(names)), vals, left=left, color=colors(k), label=kind)
        left += vals
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{short(n)} (n={table[n]['matches_as_a']})" for n in names], fontsize=8)
    ax.set_xlabel("real ball losses per match, as config_a (n = matches as config_a)")
    ax.set_title("How each strategy loses the ball (top = most points)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fouls(summary: dict, table: dict, path: Path, top: int = 7) -> None:
    rules = list(summary["fouls"])[:top]
    names = sorted(table, key=lambda n: -points(table[n]))
    m = np.zeros((len(names), len(rules) + 1))
    for j, rule in enumerate(rules):
        for key, n in summary["fouls"][rule]["by_tactic"].items():
            strategy = f"build_{key.split('/')[0]}_kernel_strategy"
            if strategy in table:
                m[names.index(strategy), j] += n / table[strategy]["matches"]
    for i, n in enumerate(names):
        m[i, -1] = table[n]["stalled"]
    fig, ax = plt.subplots(figsize=(1.0 * len(rules) + 4, 0.38 * len(names) + 1.8))
    norm = m / np.maximum(m.max(axis=0), 1e-9)
    ax.imshow(norm, cmap="Reds", vmin=0, vmax=1.2, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m[i, j]
            text = f"{int(v)}" if j == len(rules) else f"{v:.1f}" if v >= 1 or v == 0 else f"{v:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(rules) + 1))
    ax.set_xticklabels(rules + ["stalled matches"], rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([short(n) for n in names], fontsize=9)
    ax.set_title("Fouls per match by rule (both sides, the offending strategy), and stalled matches", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def build(summary: dict, losses: list[dict], out: Path) -> dict:
    """Write every figure for `summary` (a round-robin's summary.json) and its replays'
    `turnover_breakdown.analyse_match` records. Returns the strategy table plotted."""
    out.mkdir(parents=True, exist_ok=True)
    real = {r["match"]: turnover_breakdown.real_loss_kinds(r) for r in losses}
    chances_by_match = {r["match"]: r["chances"] for r in losses if "chances" in r}
    table = strategy_table(summary["results"], real, chances_by_match)
    shots = [s for c in chances_by_match.values() for s in c["shots"]]
    heatmap(table, out / "signal_heatmap.png")
    attack_defense(table, out / "attack_defense.png")
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
    build(summary, losses, args.out)
    print(f"{len(losses)} of {len(summary['results'])} matches had a replay; figures in {args.out}")


if __name__ == "__main__":
    main()
