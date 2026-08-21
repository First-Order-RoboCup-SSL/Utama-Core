"""plot_elo.py — Visualize elo.py's output: rating history, W/D/L matrix, goal diff.

Run:
    pixi run python plot_elo.py replays/<run>/elo_history.json replays/<run>/summary.json [out_dir]

What this does
---------------
Three plots from the same pair of files elo.py/tournament.py already
produce — no new data collection, purely a visualization layer:

1. Elo rating vs. match index, one line per config.
2. Head-to-head win/draw/loss matrix as a heatmap.
3. Goal differential (goals for - goals against) per config, bar chart.

Saved as PNGs under `out_dir` (default: alongside elo_history.json).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _short_name(config_name: str) -> str:
    return config_name.removeprefix("build_").removesuffix("_kernel_strategy")


def plot_elo_history(history: list[dict], configs: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ratings_over_time: dict[str, list[tuple[int, float]]] = {c: [] for c in configs}

    for row in history:
        ratings_over_time[row["config_a"]].append((row["match_index"], row["rating_a_after"]))
        ratings_over_time[row["config_b"]].append((row["match_index"], row["rating_b_after"]))

    for config in configs:
        points = sorted(ratings_over_time[config])
        if not points:
            continue
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker="o", label=config, linewidth=2)

    ax.set_xlabel("Match index")
    ax.set_ylabel("Elo rating")
    ax.set_title("Elo rating over matches played")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_wdl_matrix(results: list[dict], configs: list[str], out_path: Path) -> None:
    n = len(configs)
    idx = {c: i for i, c in enumerate(configs)}
    # matrix[i][j] = row config's win count against column config, from row's perspective
    win_matrix = np.zeros((n, n))
    game_matrix = np.zeros((n, n))

    for r in results:
        a, b = _short_name(r["config_a"]), _short_name(r["config_b"])
        if a not in idx or b not in idx:
            continue
        ia, ib = idx[a], idx[b]
        game_matrix[ia][ib] += 1
        game_matrix[ib][ia] += 1
        if r["score_a"] > r["score_b"]:
            win_matrix[ia][ib] += 1
        elif r["score_b"] > r["score_a"]:
            win_matrix[ib][ia] += 1
        else:
            win_matrix[ia][ib] += 0.5
            win_matrix[ib][ia] += 0.5

    with np.errstate(invalid="ignore", divide="ignore"):
        win_rate = np.divide(win_matrix, game_matrix, out=np.full((n, n), np.nan), where=game_matrix > 0)

    fig, ax = plt.subplots(figsize=(6 + 0.4 * n, 5 + 0.4 * n))
    im = ax.imshow(win_rate, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(configs, rotation=45, ha="right")
    ax.set_yticklabels(configs)
    ax.set_xlabel("Opponent")
    ax.set_ylabel("Config (row's win rate vs. column)")
    ax.set_title("Head-to-head win rate")

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val = win_rate[i][j]
            text = "-" if np.isnan(val) else f"{val:.0%}\n({int(game_matrix[i][j])}g)"
            ax.text(j, i, text, ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, label="Win rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_goal_diff(results: list[dict], configs: list[str], out_path: Path) -> None:
    goals_for = {c: 0 for c in configs}
    goals_against = {c: 0 for c in configs}

    for r in results:
        a, b = _short_name(r["config_a"]), _short_name(r["config_b"])
        if a in goals_for:
            goals_for[a] += r["score_a"]
            goals_against[a] += r["score_b"]
        if b in goals_for:
            goals_for[b] += r["score_b"]
            goals_against[b] += r["score_a"]

    ordered = sorted(configs, key=lambda c: -(goals_for[c] - goals_against[c]))
    diffs = [goals_for[c] - goals_against[c] for c in ordered]
    colors = ["#2a9d8f" if d >= 0 else "#e76f51" for d in diffs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(ordered, diffs, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Goal differential (GF - GA)")
    ax.set_title("Goal differential across all recorded matches")
    for i, c in enumerate(ordered):
        ax.text(diffs[i], i, f"  GF{goals_for[c]}-GA{goals_against[c]}", va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = sys.argv[1:]
    if len(args) < 2:
        raise SystemExit("Usage: pixi run python plot_elo.py <elo_history.json> <summary.json> [out_dir]")

    elo_path = Path(args[0])
    summary_path = Path(args[1])
    out_dir = Path(args[2]) if len(args) > 2 else elo_path.parent

    with open(elo_path) as f:
        elo_data = json.load(f)
    with open(summary_path) as f:
        summary = json.load(f)

    configs = sorted(_short_name(c) for c in summary["config_names"])
    history = elo_data["history"]
    results = summary["results"]

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_elo_history(history, configs, out_dir / "elo_history.png")
    plot_wdl_matrix(results, configs, out_dir / "wdl_matrix.png")
    plot_goal_diff(results, configs, out_dir / "goal_diff.png")

    print(f"Wrote plots to {out_dir}/: elo_history.png, wdl_matrix.png, goal_diff.png")


if __name__ == "__main__":
    main()
