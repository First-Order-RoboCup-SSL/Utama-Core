"""striker_grsim.py — the 1v1 duel (`striker_duel.py`) in an external grSim, in real time: one attacker
(yellow, attacking +x) against one keeper (blue), a fresh random start every episode, the attacker's
choices from the trained network (`--weights`) or the rule (no --weights).

grSim must already be running, with its field matching --field and the standard ports from
`docs/setup_external.md`. Each episode holds its start for half a second, then plays up to 8 s; it prints the outcome
and the running goal rate per keeper, and writes them to --out at the end.

  pixi run python -m tools.striker_grsim --weights striker_rl2_w150.npz --keepers default --episodes 50
  pixi run python -m tools.striker_grsim --keepers default --episodes 50   # the rule, for comparison
  pixi run python -m tools.striker_grsim --weights ... --mode rsim          # the same, without grSim
  pixi run python -m tools.striker_grsim --weights ... --field gerf         # 3 m x 2.25 m

The same --seed gives the same starts whichever attacker plays, so the two are compared start by start.
Real robots use examples/real_striker.py, with operator-controlled starts and the field check.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from tools.striker_duel import Duel, StrikerTactic, goal_rates, load_params, net_chooser
from utama_core.config.field_params import (
    GREAT_EXHIBITION_FIELD_DIMS,
    STANDARD_FIELD_DIMS,
)

FIELDS = {"standard": STANDARD_FIELD_DIMS, "gerf": GREAT_EXHIBITION_FIELD_DIMS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, help="the trained network (.npz); none: the rule")
    parser.add_argument("--keepers", default="default,flat", help="default (GoalkeeperTactic), flat (flat_rule's)")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=9999, help="9999: the starts the network was evaluated on")
    parser.add_argument("--mode", choices=("grsim", "rsim"), default="grsim")
    parser.add_argument("--field", choices=FIELDS, default="standard", help="standard: 9 x 6 m; gerf: 3 x 2.25 m")
    parser.add_argument("--out", type=Path, help="the outcomes and goal rates (.json)")
    a = parser.parse_args()
    keepers = a.keepers.split(",")
    params = load_params(a.weights) if a.weights is not None else None
    duel = Duel(mode=a.mode, field_dims=FIELDS[a.field])
    results = []
    try:
        for i in range(a.episodes):
            seed, keeper = a.seed * 1_000_000 + i, keepers[i % len(keepers)]
            choose = net_chooser(params, True, np.random.default_rng(seed)) if params is not None else None
            striker = StrikerTactic(choose)
            outcome = duel.episode(random.Random(seed), striker, keeper)
            results.append({"seed": seed, "keeper": keeper, "outcome": outcome, "trajectory": striker.trajectory})
            rates = goal_rates(results)
            running = ", ".join(f"{k} {v['goals']}/{v['episodes']}" for k, v in rates.items())
            print(f"episode {i + 1}/{a.episodes} keeper={keeper}: {outcome}  (goals: {running})", flush=True)
    finally:
        duel.runner.close()
    summary = {
        "policy": str(a.weights) if a.weights is not None else "rule",
        "mode": a.mode,
        "field": a.field,
        "seed": a.seed,
    }
    summary.update(goal_rates(results))
    print(json.dumps(summary, indent=1))
    if a.out is not None:
        per_episode = [{k: r[k] for k in ("seed", "keeper", "outcome")} for r in results]
        a.out.write_text(json.dumps({**summary, "episodes": per_episode}, indent=1))


if __name__ == "__main__":
    main()
