"""striker_grsim.py — the 1v1 duel (`striker_duel.py`) in an external grSim, in real time: one attacker
(yellow, attacking +x) against one keeper (blue), a fresh random start every episode, the attacker's
choices from a trained network (`--weights`) or the rule (no --weights). The network is either
`striker_rl.py`'s, choosing shots and carries, or `striker_e2e_rl.py`'s, driving the robot itself
(speeds, kick, dribbler); which one is read from the file.

grSim must already be running, set up as `docs/setup_external.md` says (Division B, the standard
ports). Each episode holds its start for half a second, then plays up to 8 s; it prints the outcome
and the running goal rate per keeper, and writes them to --out at the end.

  pixi run python -m tools.striker_grsim --weights striker_rl2_w150.npz --keepers default --episodes 50
  pixi run python -m tools.striker_grsim --weights striker_e2e2.npz --keepers default --episodes 50
  pixi run python -m tools.striker_grsim --keepers default --episodes 50   # the rule, for comparison
  pixi run python -m tools.striker_grsim --weights ... --mode rsim          # the same, without grSim

The same --seed gives the same starts whichever attacker plays, so the two are compared start by start.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from tools.striker_duel import Duel, StrikerTactic, goal_rates, load_params, net_chooser
from tools.striker_e2e import E2ETactic, actor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, help="the trained network (.npz); none: the rule")
    parser.add_argument("--keepers", default="default,flat", help="default (GoalkeeperTactic), flat (flat_rule's)")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=9999, help="9999: the starts the network was evaluated on")
    parser.add_argument("--mode", choices=("grsim", "rsim"), default="grsim")
    parser.add_argument("--out", type=Path, help="the outcomes and goal rates (.json)")
    a = parser.parse_args()
    keepers = a.keepers.split(",")
    params = load_params(a.weights) if a.weights is not None else None
    duel = Duel(mode=a.mode)
    results = []
    try:
        for i in range(a.episodes):
            seed, keeper = a.seed * 1_000_000 + i, keepers[i % len(keepers)]
            rng = np.random.default_rng(seed)
            if params is not None and "log_std" in params:  # end to end
                striker = E2ETactic(act=actor(params, True, rng))
            else:
                striker = StrikerTactic(net_chooser(params, True, rng) if params is not None else None)
            outcome = duel.episode(random.Random(seed), striker, keeper)
            results.append({"seed": seed, "keeper": keeper, "outcome": outcome, "trajectory": striker.trajectory})
            rates = goal_rates(results)
            running = ", ".join(f"{k} {v['goals']}/{v['episodes']}" for k, v in rates.items())
            print(f"episode {i + 1}/{a.episodes} keeper={keeper}: {outcome}  (goals: {running})", flush=True)
    finally:
        duel.runner.close()
    summary = {"policy": str(a.weights) if a.weights is not None else "rule", "mode": a.mode, "seed": a.seed}
    summary.update(goal_rates(results))
    print(json.dumps(summary, indent=1))
    if a.out is not None:
        per_episode = [{k: r[k] for k in ("seed", "keeper", "outcome")} for r in results]
        a.out.write_text(json.dumps({**summary, "episodes": per_episode}, indent=1))


if __name__ == "__main__":
    main()
