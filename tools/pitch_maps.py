"""On-pitch pictures of how each strategy plays, from one round-robin run.

    pixi run python tools/pitch_maps.py replays/tournament_<id> [--out docs/img/signals]

Every strategy gets one panel per figure, in `signal_report`'s order (points per match). All
panels use one frame: the strategy attacks toward +x, its own goal is on the left (-x), and
the pitch is rotated 180 degrees for a team that attacks toward -x, so its left wing is +y in
every panel. Writes:

- `positions_attack.png`: where the strategy's outfield robots stand while it has the ball
  (share of live samples per cell, one scale for all panels); five dots mark each robot's mean;
- `positions_defence.png`: the same while the opponent has the ball;
- `shots.png`: every shot the strategy took (filled if it scored, hollow if not) at the spot it
  was struck from, and the shots it conceded (AGAINST colour, faded) at the other end;
- `fouls_map.png`: where the strategy committed its fouls, coloured by rule: the four most
  common rules in the run, then all other rules.

Possession is `MatchStatsAccumulator`'s notion without its hysteresis: the nearest robot to the
ball, within `_POSSESSION_RADIUS_M`, while the ball is slower than `_PASS_RELEASE_MPS`. Positions
are sampled every `STEP` ticks in live play (NORMAL_START, FORCE_START); the goalkeeper is the
robot that spends the most time in its own box, and is left out. Shots come from
`turnover_breakdown.analyse_match` (the `chances` record), so each match is replayed once.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import signal_report as sr  # noqa: E402
from utama_core.analysis import turnover_breakdown  # noqa: E402
from utama_core.config.field_params import STANDARD_FIELD_DIMS as F  # noqa: E402
from utama_core.engine.match_stats import (  # noqa: E402
    _PASS_RELEASE_MPS,
    _POSSESSION_RADIUS_M,
)
from utama_core.entities.referee.referee_command import RefereeCommand  # noqa: E402
from utama_core.replay.columnar_reader import (  # noqa: E402
    ColumnarReplay,
    load_columnar_replay,
)

L, W = F.full_field_half_length, F.full_field_half_width
BOX_D, BOX_W = 2 * F.half_defense_area_depth, F.half_defense_area_width  # the 2 m x 1 m box, from the goal line
GOAL_D = F.goal_depth
BIN_M = 0.25  # occupancy cell size
STEP = 10  # sample every 10th tick
LIVE_IDS = np.array([RefereeCommand.NORMAL_START.value, RefereeCommand.FORCE_START.value])
BALL_OUT = "out_of_bounds"
TOP_RULES = 4
OTHER_SLOT = 4  # sr.SLOTS[4] for "other": the top four rules take slots 0-3

SHARE_CMAP = LinearSegmentedColormap.from_list("share", [sr.SURFACE, "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])


@dataclass
class MatchView:
    """One strategy's view of one match, with every coordinate in the common frame."""

    strategy: str
    pos_x: np.ndarray  # outfield robot samples in live play
    pos_y: np.ndarray
    pos_in: np.ndarray  # bool: the strategy had the ball at the sample
    pos_robot: np.ndarray  # robot id of each sample
    shots: list[dict] = field(default_factory=list)  # {"x", "y", "scored", "own"}
    fouls: list[dict] = field(default_factory=list)  # {"x", "y", "rule"}


# --- frame and possession -------------------------------------------------------------------


def attack_sign(defends_right: bool) -> float:
    """+1 for a team that attacks toward +x (it defends the left goal), -1 for one that attacks
    toward -x. Rotating the pitch by 180 degrees for the -1 case keeps the team's left on the
    same side of every panel."""
    return -1.0 if defends_right else 1.0


def to_attacking_frame(x: float, y: float, defends_right: bool) -> tuple[float, float]:
    s = attack_sign(defends_right)
    return s * x, s * y


def _nearest(players: np.ndarray, ball_xy: np.ndarray) -> np.ndarray:
    """Distance from the ball to the nearest present robot at each tick (inf if none)."""
    if players.shape[1] == 0:
        return np.full(len(ball_xy), np.inf)
    d = np.linalg.norm(players - ball_xy[:, None, :], axis=2)
    return np.nan_to_num(d, nan=np.inf).min(axis=1)


def possession_codes(
    friendly_p: np.ndarray, enemy_p: np.ndarray, ball_p: np.ndarray, ball_v: np.ndarray, live: np.ndarray
) -> np.ndarray:
    """Per tick: 1 if the friendly team has the ball, -1 the enemy, 0 nobody (or not live)."""
    ball_xy = ball_p[:, :2]
    d_f = _nearest(friendly_p, ball_xy)
    d_e = _nearest(enemy_p, ball_xy)
    speed = np.linalg.norm(ball_v[:, :2], axis=1)
    controlled = live & (np.minimum(d_f, d_e) <= _POSSESSION_RADIUS_M) & (speed < _PASS_RELEASE_MPS)
    return np.where(controlled, np.where(d_f < d_e, 1, -1), 0).astype(np.int8)


def keeper_slot(norm_p: np.ndarray) -> int:
    """Index of the robot that spends the most ticks in its own box. `norm_p` is (ticks, robots, 2)
    in the common frame, where the team's own goal is at -x."""
    if norm_p.shape[1] == 0:
        return -1
    in_box = (norm_p[:, :, 0] < -L + BOX_D) & (np.abs(norm_p[:, :, 1]) < BOX_W)
    return int(in_box.sum(axis=0).argmax())


def _tick(ts: np.ndarray, t: float) -> int:
    return min(int(np.searchsorted(ts, t)), len(ts) - 1)


def build_views(replay: ColumnarReplay, names: dict[str, str], shots: list[dict], fouls: list[dict]) -> list[MatchView]:
    """The two strategy views of one replay. `names` maps "friendly"/"enemy" to the strategy
    (config_a / config_b); `shots` are `chances` shot records and `fouls` are the `stats.fouls`
    records of the match, both in its own (friendly/enemy) terms."""
    n = replay.n_ticks
    live = np.isin(replay.referee_command, LIVE_IDS)
    defends_right = (
        replay.side_is_right.astype(bool) if replay.side_is_right is not None else np.full(n, replay.my_team_is_right)
    )
    sign = {"friendly": np.where(defends_right, -1.0, 1.0), "enemy": np.where(defends_right, 1.0, -1.0)}
    poss = possession_codes(replay.friendly_p, replay.enemy_p, replay.ball_p, replay.ball_v, live)
    idx = np.arange(0, n, STEP)
    views = []
    for side, opp in (("friendly", "enemy"), ("enemy", "friendly")):
        players, ids = (
            (replay.friendly_p, replay.friendly_ids) if side == "friendly" else (replay.enemy_p, replay.enemy_ids)
        )
        norm = players[idx] * sign[side][idx][:, None, None]
        keeper = keeper_slot(norm)
        code = 1 if side == "friendly" else -1
        xs, ys, ins, rids = [], [], [], []
        for j, rid in enumerate(ids):
            if j == keeper:
                continue
            ok = live[idx] & ~np.isnan(norm[:, j, 0])
            xs.append(norm[ok, j, 0])
            ys.append(norm[ok, j, 1])
            ins.append(poss[idx][ok] == code)
            rids.append(np.full(int(ok.sum()), int(rid)))
        view = MatchView(
            strategy=names[side],
            pos_x=np.concatenate(xs) if xs else np.empty(0),
            pos_y=np.concatenate(ys) if ys else np.empty(0),
            pos_in=np.concatenate(ins) if ins else np.empty(0, dtype=bool),
            pos_robot=np.concatenate(rids) if rids else np.empty(0, dtype=int),
        )
        for s in shots:
            if s["side"] not in (side, opp):
                continue
            tick = _tick(replay.ts, s["t"])
            x, y = s["xy"][0] * sign[side][tick], s["xy"][1] * sign[side][tick]
            view.shots.append({"x": float(x), "y": float(y), "scored": bool(s["scored"]), "own": s["side"] == side})
        for f in fouls:
            if f["side"] != side or f["rule"] == BALL_OUT:
                continue
            tick = _tick(replay.ts, f["sim_time"])
            x, y = f["ball_xy"][0] * sign[side][tick], f["ball_xy"][1] * sign[side][tick]
            view.fouls.append({"x": float(x), "y": float(y), "rule": f["rule"]})
        views.append(view)
    return views


# --- aggregation -----------------------------------------------------------------------------


def _cat(views: list[MatchView], strategy: str, attr: str) -> np.ndarray:
    parts = [getattr(v, attr) for v in views if v.strategy == strategy]
    return np.concatenate(parts) if parts else np.empty(0)


def position_grid(views: list[MatchView], strategy: str, in_possession: bool) -> dict:
    """Share of the strategy's outfield samples per cell (rows: y, columns: x), and each robot's
    mean position, for the samples taken while it had the ball (or not)."""
    x, y = _cat(views, strategy, "pos_x"), _cat(views, strategy, "pos_y")
    inp = _cat(views, strategy, "pos_in").astype(bool)
    rid = _cat(views, strategy, "pos_robot")
    keep = inp if in_possession else ~inp
    x, y, rid = x[keep], y[keep], rid[keep]
    xe = np.linspace(-L, L, round(2 * L / BIN_M) + 1)
    ye = np.linspace(-W, W, round(2 * W / BIN_M) + 1)
    hist, _, _ = np.histogram2d(x, y, bins=[xe, ye])
    share = hist.T / max(1.0, hist.sum())
    means = [(float(x[rid == r].mean()), float(y[rid == r].mean())) for r in np.unique(rid)]
    return {"share": share, "means": means}


def shot_marks(views: list[MatchView], strategy: str) -> dict:
    """The strategy's own shots and the shots it conceded, as lists of (x, y, scored)."""
    mine = [s for v in views if v.strategy == strategy for s in v.shots]
    return {
        "own": [(s["x"], s["y"], s["scored"]) for s in mine if s["own"]],
        "faced": [(s["x"], s["y"], s["scored"]) for s in mine if not s["own"]],
    }


def top_rules(views: list[MatchView], k: int = TOP_RULES) -> list[str]:
    counts = Counter(f["rule"] for v in views for f in v.fouls)
    return [rule for rule, _ in counts.most_common(k)]


def rule_colours(rules: list[str]) -> dict[str, str]:
    """Fixed slots: the i-th top rule takes slot i, every other rule takes the 'other' slot."""
    return {rule: sr.SLOTS[i] for i, rule in enumerate(rules)}


# --- drawing ---------------------------------------------------------------------------------


def _pitch(ax) -> None:
    ax.set_facecolor(sr.SURFACE)
    ax.add_patch(Rectangle((-L, -W), 2 * L, 2 * W, fill=False, edgecolor=sr.INK_3, lw=0.8))
    ax.plot([0, 0], [-W, W], color=sr.INK_3, lw=0.6)
    ax.add_patch(Circle((0, 0), F.center_circle_radius, fill=False, edgecolor=sr.INK_3, lw=0.6))
    for x0 in (-L, L - BOX_D):
        ax.add_patch(Rectangle((x0, -BOX_W), BOX_D, 2 * BOX_W, fill=False, edgecolor=sr.INK_3, lw=0.6))
    ax.add_patch(Rectangle((-L - GOAL_D, -F.half_goal_width), GOAL_D, 2 * F.half_goal_width, color=sr.INK_3))
    ax.add_patch(Rectangle((L, -F.half_goal_width), GOAL_D, 2 * F.half_goal_width, color=sr.INK_3))
    ax.set_xlim(-L - GOAL_D - 0.2, L + GOAL_D + 0.2)
    ax.set_ylim(-W - 0.2, W + 0.2)
    ax.set_aspect("equal")
    ax.axis("off")


def _panel_grid(n: int, cols: int = 3):
    """A grid of pitch panels for `n` strategies; unused cells are hidden at the end of drawing."""
    rows = -(-n // cols)
    return plt.subplots(rows, cols, figsize=(3.6 * cols, 2.55 * rows + 0.8), squeeze=False)


def _hide_unused(axes, n: int) -> None:
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)


def plot_positions(views, order, labels, path: Path, in_possession: bool) -> None:
    grids = {n: position_grid(views, n, in_possession) for n in order}
    nonzero = np.concatenate([g["share"][g["share"] > 0] for g in grids.values() if (g["share"] > 0).any()])
    vmax = float(np.percentile(nonzero, 99)) if nonzero.size else 1.0
    fig, axes = _panel_grid(len(order))
    for k, name in enumerate(order):
        ax = axes.flat[k]
        _pitch(ax)
        ax.imshow(
            grids[name]["share"],
            extent=[-L, L, -W, W],
            origin="lower",
            cmap=SHARE_CMAP,
            vmin=0,
            vmax=vmax,
            interpolation="bilinear",
            zorder=1,
        )
        if grids[name]["means"]:
            mx, my = zip(*grids[name]["means"])
            ax.scatter(mx, my, s=26, color=sr.INK, edgecolors=sr.SURFACE, linewidths=0.8, zorder=3)
        ax.set_title(labels[name], loc="left", fontsize=9, color=sr.INK)
    _hide_unused(axes, len(order))
    state = "with the ball" if in_possession else "without the ball"
    fig.suptitle(
        f"Where the outfield robots stand {state}: share of live samples per cell, one scale. "
        "Every strategy attacks right; dots are each robot's mean position",
        x=0.0,
        ha="left",
        fontsize=10,
        color=sr.INK,
    )
    sm = plt.cm.ScalarMappable(cmap=SHARE_CMAP, norm=plt.Normalize(0, vmax))
    fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02, label="share of the samples in a cell")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_shots(views, order, labels, path: Path) -> None:
    fig, axes = _panel_grid(len(order))
    for k, name in enumerate(order):
        ax = axes.flat[k]
        _pitch(ax)
        marks = shot_marks(views, name)
        for pts, colour, alpha in ((marks["faced"], sr.AGAINST, 0.35), (marks["own"], sr.FOR, 1.0)):
            for x, y, scored in pts:
                ax.scatter(
                    [x],
                    [y],
                    s=16,
                    color=colour if scored else sr.SURFACE,
                    edgecolors=colour,
                    linewidths=0.9,
                    alpha=alpha,
                    zorder=3 if colour == sr.FOR else 2,
                )
        goals = sum(s for _, _, s in marks["own"])
        ax.set_title(
            f"{labels[name]}\n{len(marks['own'])} shots, {goals} goals", loc="left", fontsize=8.5, color=sr.INK
        )
    _hide_unused(axes, len(order))
    handles = [
        plt.Line2D([], [], marker="o", ls="", mfc=sr.FOR, mec=sr.FOR, ms=5, label="own shot, goal"),
        plt.Line2D([], [], marker="o", ls="", mfc=sr.SURFACE, mec=sr.FOR, ms=5, label="own shot, no goal"),
        plt.Line2D([], [], marker="o", ls="", mfc=sr.AGAINST, mec=sr.AGAINST, ms=5, alpha=0.35, label="conceded, goal"),
        plt.Line2D(
            [], [], marker="o", ls="", mfc=sr.SURFACE, mec=sr.AGAINST, ms=5, alpha=0.35, label="conceded, no goal"
        ),
    ]
    fig.legend(handles=handles, loc="upper left", ncol=4, frameon=False, bbox_to_anchor=(0.0, 1.0))
    fig.suptitle(
        "Where each strategy shoots from, and where it concedes: shots at the spot struck from, "
        "attacking right; conceded shots faded",
        x=0.0,
        ha="left",
        y=1.04,
        fontsize=10,
        color=sr.INK,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_fouls(views, order, labels, path: Path) -> None:
    rules = top_rules(views)
    colours = rule_colours(rules)
    other = sr.SLOTS[OTHER_SLOT]
    fig, axes = _panel_grid(len(order))
    for k, name in enumerate(order):
        ax = axes.flat[k]
        _pitch(ax)
        fouls = [f for v in views if v.strategy == name for f in v.fouls]
        for rule in rules + [None]:
            pts = [f for f in fouls if (f["rule"] == rule if rule else f["rule"] not in colours)]
            if pts:
                ax.scatter(
                    [f["x"] for f in pts],
                    [f["y"] for f in pts],
                    s=18,
                    color=colours[rule] if rule else other,
                    edgecolors=sr.SURFACE,
                    linewidths=0.4,
                    zorder=3,
                )
        ax.set_title(f"{labels[name]}\n{len(fouls)} fouls", loc="left", fontsize=8.5, color=sr.INK)
    _hide_unused(axes, len(order))
    handles = [
        plt.Line2D([], [], marker="o", ls="", mfc=colours[r], mec=sr.SURFACE, ms=6, label=r.replace("_", " "))
        for r in rules
    ]
    handles.append(plt.Line2D([], [], marker="o", ls="", mfc=other, mec=sr.SURFACE, ms=6, label="other rules"))
    fig.legend(handles=handles, loc="upper left", ncol=len(handles), frameon=False, bbox_to_anchor=(0.0, 1.0))
    fig.suptitle(
        "Where each strategy commits its fouls, by rule (attacking right; the ball's spot at the foul)",
        x=0.0,
        ha="left",
        y=1.04,
        fontsize=10,
        color=sr.INK,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def summary_rows(views: list[MatchView], order: list[str]) -> list[dict]:
    """The numbers the figures show, one row per strategy in `order`: the share of its in-possession
    samples in the attacking half, the share of its out-of-possession samples in its own half, the
    share of its shots struck from the attacking third and of its fouls in its defensive third."""
    third = -L + 2 * L / 3  # the defensive third's far edge, in the common frame
    rows = []
    for name in order:
        x, y = _cat(views, name, "pos_x"), _cat(views, name, "pos_y")
        inp = _cat(views, name, "pos_in").astype(bool)
        shots = shot_marks(views, name)["own"]
        fouls = [f for v in views if v.strategy == name for f in v.fouls]
        rows.append(
            {
                "strategy": sr.short(name),
                "in_possession_attacking_half": float(np.mean(x[inp] > 0)) if inp.any() else None,
                "in_possession_mean_abs_y": float(np.mean(np.abs(y[inp]))) if inp.any() else None,
                "out_possession_own_half": float(np.mean(x[~inp] < 0)) if (~inp).any() else None,
                "shots_from_attacking_third": float(np.mean([sx > -third for sx, _, _ in shots])) if shots else None,
                "fouls_in_defensive_third": float(np.mean([f["x"] < third for f in fouls])) if fouls else None,
            }
        )
    return rows


# --- run ------------------------------------------------------------------------------------


def _match(job: tuple[str, str, str, list[dict]]) -> list[MatchView]:
    npz, config_a, config_b, fouls = job
    names = {"friendly": config_a, "enemy": config_b}
    shots = turnover_breakdown.analyse_match(npz)["chances"]["shots"]
    return build_views(load_columnar_replay(npz), names, shots, fouls)


def run_views(run_dir: Path, workers: int) -> list[MatchView]:
    summary = json.loads((run_dir / "summary.json").read_text())
    jobs = []
    for r in summary["results"]:
        stem = f"{sr.short(r['config_a'])}_vs_{sr.short(r['config_b'])}"
        npz = run_dir / f"{stem}.npz"
        if npz.exists():
            jobs.append((str(npz), r["config_a"], r["config_b"], (r.get("stats") or {}).get("fouls") or []))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return [v for views in pool.map(_match, jobs) for v in views]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "img" / "signals")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    summary = json.loads((args.run_dir / "summary.json").read_text())
    order = sr._order(summary["strategies"])
    labels = {n: f"{sr.short(n)}  {sr.points(summary['strategies'][n]):.2f}" for n in order}
    views = run_views(args.run_dir, args.workers)
    args.out.mkdir(parents=True, exist_ok=True)
    plot_positions(views, order, labels, args.out / "positions_attack.png", in_possession=True)
    plot_positions(views, order, labels, args.out / "positions_defence.png", in_possession=False)
    plot_shots(views, order, labels, args.out / "shots.png")
    plot_fouls(views, order, labels, args.out / "fouls_map.png")
    for row in summary_rows(views, order):
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}))
    print(f"{len(views)} strategy-match views; figures in {args.out}")


if __name__ == "__main__":
    main()
