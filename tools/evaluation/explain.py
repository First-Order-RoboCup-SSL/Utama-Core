"""On-pitch pictures of a candidate against the champion, from the replays of one run dir.

    pixi run python tools/evaluation/explain.py RUN_DIR CANDIDATE CHAMPION --out DIR

Explains a verdict; it never makes one. Goals (`evaluate.py`'s verdict) are the only ranking;
these pictures and numbers show how the two strategies played, so a search agent can see WHY a
candidate won or lost. Each strategy's view is every match it played in the run dir, against any
opponent (`find_matches`): a match file's stem is `<config_a>_vs_<config_b>` (round-robin) or
`<config_a>_vs_<config_b>_s<seed>` (an evaluate.py run), and config_a is the friendly side.

Writes to DIR:

- `positions.png`: 2x2, candidate and champion, with and without the ball; one colour scale;
- `shots.png`: each strategy's shots for and against (`pitch_maps.plot_shots`);
- `fouls.png`: each strategy's fouls by rule, one fixed colour per rule (`pitch_maps.plot_fouls`);
- `summary.json`: the numbers behind the pictures per side, and the paired goal difference per
  opponent (same opponent, seed and colour, as `evaluate.py` pairs them), where the run has them.

Everything is in the attacking frame of `pitch_maps.py` (attacking right). Goals come from the
replay's final referee state (the score), fouls from the match's `.stats.json`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import pitch_maps as pm  # noqa: E402
from tools import signal_report as sr  # noqa: E402
from tools.evaluation.match import _short_name  # noqa: E402
from utama_core.replay.columnar_reader import load_columnar_replay  # noqa: E402

STEM = re.compile(r"^(?P<a>.+?)_vs_(?P<b>.+?)(?:_s(?P<seed>\d+))?$")
ROLES = ("candidate", "champion")


@dataclass
class Match:
    """One replay in the run dir. `a` is config_a (the friendly side), `b` config_b (enemy)."""

    npz: Path
    a: str  # short names
    b: str
    seed: Optional[int]
    score_a: Optional[int]  # None when the replay has no referee state to read a score from
    score_b: Optional[int]
    fouls: list[dict]

    def job(self) -> tuple[str, str, str, list[dict]]:
        return (str(self.npz), self.a, self.b, self.fouls)

    def goal_difference(self, name: str) -> Optional[int]:
        if self.score_a is None:
            return None
        return self.score_a - self.score_b if name == self.a else self.score_b - self.score_a


def _final_score(npz: Path) -> tuple[Optional[int], Optional[int]]:
    """The score at the end of a replay: the last referee state the writer stored (scores only
    change through stored referee messages). Yellow is config_a (friendly) in every run."""
    replay = load_columnar_replay(npz)
    if not replay.sparse_referee:
        return None, None
    ref = replay.sparse_referee[max(replay.sparse_referee)]
    return int(ref.yellow_team.score), int(ref.blue_team.score)


def find_matches(run_dir: Path, names: set[str]) -> list[Match]:
    """Every replay in `run_dir` with `names` (short names) on either side, in file-name order."""
    matches = []
    for npz in sorted(run_dir.glob("*.npz")):
        m = STEM.match(npz.name.removesuffix(".npz"))
        if m is None or not ({m["a"], m["b"]} & names):
            continue
        stats = npz.with_name(npz.name.removesuffix(".npz") + ".stats.json")
        fouls = []
        if stats.exists():
            fouls = json.loads(stats.read_text()).get("fouls") or []
        score_a, score_b = _final_score(npz)
        seed = int(m["seed"]) if m["seed"] is not None else None
        matches.append(Match(npz, m["a"], m["b"], seed, score_a, score_b, fouls))
    return matches


def _mean(values: list[Optional[int]]) -> Optional[float]:
    known = [v for v in values if v is not None]
    return float(np.mean(known)) if known else None


def _gd_by_pair(matches: list[Match], name: str, opp: str) -> dict[tuple, int]:
    """{(seed, name is config_a): goal difference of `name`} over the matches of `name` vs `opp`."""
    out = {}
    for m in matches:
        if {m.a, m.b} != {name, opp} or m.a == m.b:
            continue
        gd = m.goal_difference(name)
        if gd is not None:
            out[(m.seed, m.a == name)] = gd
    return out


def paired_rows(matches: list[Match], cand: str, champ: str) -> dict[str, dict]:
    """Per opponent: the candidate's and the champion's goal difference over the matches both
    played with the same seed and colour (evaluate.py's pairing), and how many pairs there are.
    The last row is the candidate against the champion itself."""
    opponents = sorted({n for m in matches for n in (m.a, m.b)} - {cand, champ})
    rows = {}
    for o in opponents:
        c, ch = _gd_by_pair(matches, cand, o), _gd_by_pair(matches, champ, o)
        common = [k for k in c if k in ch]
        rows[o] = {
            "candidate": sum(c[k] for k in common),
            "champion": sum(ch[k] for k in common),
            "matches": len(common),
        }
    h2h = _gd_by_pair(matches, cand, champ)
    rows["head-to-head"] = {"candidate": sum(h2h.values()), "champion": -sum(h2h.values()), "matches": len(h2h)}
    return rows


def side_numbers(name: str, matches: list[Match], views: list[pm.MatchView], rows: dict[str, dict]) -> dict:
    mine = [m for m in matches if name in (m.a, m.b)]
    n = len(mine)
    marks = pm.shot_marks(views, name)
    fouls = Counter(f["rule"] for v in views if v.strategy == name for f in v.fouls)
    goals_for = [None if m.score_a is None else (m.score_a if name == m.a else m.score_b) for m in mine]
    goals_against = [None if m.score_a is None else (m.score_b if name == m.a else m.score_a) for m in mine]
    row = rows[name]
    return {
        "matches": n,
        "goals_for_per_match": _mean(goals_for),
        "goals_against_per_match": _mean(goals_against),
        "shots_for_per_match": len(marks["own"]) / n if n else None,
        "shots_against_per_match": len(marks["faced"]) / n if n else None,
        "in_possession_attacking_half": row.get("in_possession_attacking_half"),
        "in_possession_mean_abs_y": row.get("in_possession_mean_abs_y"),
        "out_possession_own_half": row.get("out_possession_own_half"),
        "shots_from_attacking_third": row.get("shots_from_attacking_third"),
        "fouls_in_defensive_third": row.get("fouls_in_defensive_third"),
        "fouls_per_match": {rule: round(c / n, 3) for rule, c in sorted(fouls.items())} if n else {},
    }


# --- pictures -----------------------------------------------------------------------------


def plot_positions(views: list[pm.MatchView], cand: str, champ: str, path: Path) -> None:
    """2x2: columns candidate and champion, rows with and without the ball, one colour scale."""
    cols = (cand, champ)
    states = (True, False)
    grids = {(n, s): pm.position_grid(views, n, s) for n in cols for s in states}
    nonzero = np.concatenate([g["share"][g["share"] > 0] for g in grids.values() if (g["share"] > 0).any()])
    vmax = float(np.percentile(nonzero, 99)) if nonzero.size else 1.0
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.4), squeeze=False)
    for i, state in enumerate(states):
        for j, name in enumerate(cols):
            ax = axes[i][j]
            pm._pitch(ax)
            grid = grids[(name, state)]
            ax.imshow(
                grid["share"],
                extent=[-pm.L, pm.L, -pm.W, pm.W],
                origin="lower",
                cmap=pm.SHARE_CMAP,
                vmin=0,
                vmax=vmax,
                interpolation="bilinear",
                zorder=1,
            )
            if grid["means"]:
                mx, my = zip(*grid["means"])
                ax.scatter(mx, my, s=22, color=sr.INK, edgecolors=sr.SURFACE, linewidths=0.8, zorder=3)
            ax.set_title(
                f"{ROLES[j]}: {name}, {'with' if state else 'without'} the ball", loc="left", fontsize=8.5, color=sr.INK
            )
    fig.suptitle(
        "Where the outfield robots stand: share of live samples per cell, one scale for all four panels; "
        "attacking right, dots are each robot's mean",
        x=0.0,
        ha="left",
        fontsize=9,
        color=sr.INK,
    )
    sm = plt.cm.ScalarMappable(cmap=pm.SHARE_CMAP, norm=plt.Normalize(0, vmax))
    fig.colorbar(
        sm,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        shrink=0.6,
        pad=0.08,
        label="share of the samples in a cell",
    )
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# --- entry point -----------------------------------------------------------------------------


def explain(run_dir, candidate: str, champion: str, out_dir, workers: int = 2) -> dict:
    """Write positions.png, shots.png, fouls.png and summary.json for `candidate` against
    `champion` into `out_dir`, from the replays in `run_dir`. Returns the summary."""
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    cand, champ = _short_name(candidate), _short_name(champion)
    if cand == champ:
        raise ValueError(f"candidate and champion are the same strategy: {cand}")
    matches = find_matches(run_dir, {cand, champ})
    if not any(cand in (m.a, m.b) for m in matches) or not any(champ in (m.a, m.b) for m in matches):
        raise FileNotFoundError(f"no replays of both {cand} and {champ} in {run_dir}")
    views = [v for v in pm.views_for_matches([m.job() for m in matches], workers) if v.strategy in (cand, champ)]
    order = [cand, champ]
    labels = {cand: f"candidate: {cand}", champ: f"champion: {champ}"}

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_positions(views, cand, champ, out_dir / "positions.png")
    pm.plot_shots(views, order, labels, out_dir / "shots.png")
    pm.plot_fouls(views, order, labels, out_dir / "fouls.png")

    rows = {r["strategy"]: r for r in pm.summary_rows(views, order)}
    summary = {
        "run_dir": str(run_dir),
        "candidate": cand,
        "champion": champ,
        "replays_read": len(matches),
        "sides": {name: side_numbers(name, matches, views, rows) for name in order},
        "paired": paired_rows(matches, cand, champ),
        "fouls_rules_shown": pm.top_rules(views),
        "notes": (
            "Matches are those in the run dir; fixtures reused from the ledger have no replay here. "
            "Positions and fouls exclude the goalkeeper and out-of-bounds (as pitch_maps.py)."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("candidate")
    parser.add_argument("champion")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    summary = explain(args.run_dir, args.candidate, args.champion, args.out, args.workers)
    for name, side in summary["sides"].items():
        print(
            f"{name:<28} {side['matches']} matches, goals for/against per match "
            f"{side['goals_for_per_match']} / {side['goals_against_per_match']}"
        )
    print(f"wrote positions.png, shots.png, fouls.png and summary.json to {args.out}")


if __name__ == "__main__":
    main()
