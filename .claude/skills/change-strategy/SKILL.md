---
name: change-strategy
description: Write a new strategy or change an existing one, then test it, evaluate it against the champion and promote it if it wins - from branch to pull request. Use when asked to try a strategy idea, improve how a strategy plays, add a tactic for a strategy, or promote a strategy to the core. Not for explaining an existing result (explain-result) or fixing one incident (debug-incident).
---

# Change a strategy

The model behind every step (tactics, partitioners, referee restarts, past tactic bugs) is
`docs/STRATEGY_DEVELOPMENT.md`; read its "Writing and evaluating a strategy" and "Strategy
branches" sections first. This skill is the order of the steps and the rules that are easy
to forget.

## 1. Branch

`strategy/<idea>` from `main`. In a git worktree run `tools/setup_worktree.sh`, and symlink
`replays` from the main checkout so the match results store is shared.

## 2. Change one thing

- **New strategy:** a new file `utama_core/strategy/<name>.py` with
  `build(outfield_robot_ids, motion_controller)` returning a `Strategy`. The catalog finds it; it
  plays as a reserve until it is added to the core. Reuse `strategy/pickers.py`.
- **Changed strategy:** edit exactly one existing strategy module. Never new modules and an
  edited one in the same branch.
- **New tactic:** a new module in `tactics/`. Never edit an existing tactic, `pickers.py` or
  `catalog.py` on a strategy branch: the opponents run them. To improve a tactic, copy it and
  change the copy.
- No module-level state (keep it inside `build`), no import-time effects, no imports of another
  strategy. CI checks a `strategy/*` branch touches only `strategy/`, new `tactics/` modules,
  `tests/strategy/` and `docs/strategies.md` (`tools/check_strategy_branch.py`).
- Every change needs a reason in game terms (what the robots will do differently, and why
  that wins goals), not a threshold nudged until results move.

## 3. Test

1. Pure partitioner tests in `utama_core/tests/strategy/test_<name>.py` (a hand-built `Game`, no rsim).
2. `pixi run pytest utama_core/tests/engine/test_all_strategy_configs.py --headless`.
3. `git add` new files, then `pixi run test` and `pixi run lint`.

## 4. Evaluate (ask first)

Never start matches without the user's yes. Then:

```
pixi run python tools/evaluation/evaluate.py NAME [NAME ...]
```

- Read the first line of each round: "N matches, X from the match results store, M to play".
  After a change to one strategy, M is about its own 5 matches per round. Much more means
  shared code changed since the store was filled; say so before waiting on it.
- Round 1 is a round robin: a stall reported there is a bug. Fix it (the `debug-incident`
  skill) before reading the verdict.
- The verdict is goals only: better (P ≥ 0.99), worse (P ≤ 0.2) or undecided after 6 rounds.
  Every other signal explains it (the `explain-result` skill), never overrules it.

## 5. Record and promote

- Add or update the strategy's row in `docs/strategies.md` with the verdict and margin.
- Better than the champion: promoting is one line in `tools/evaluation/reference/players.json`
  (`champion`, and the core if it should play every round), in the pull request. It becomes the
  reference at the next release.
- Open a draft pull request into `main` with the result. An agent never merges.

## Rules

- **Shared code is not a strategy change.** A change to `tactics/` (existing modules), `skills/`,
  `pickers.py`, the planner or the referee reruns most matches: put it on an ordinary branch,
  batch such changes, and evaluate once at the end.
- **Strategy and evaluation in separate commits**, so results stay comparable.
- **`world.json` / `robots.json` changes** replay every stored match: batch them and ask first.
- Keep evaluation at the sim settings in `tools/evaluation/reference/`; don't tune a strategy
  to real-robot limits in sim (those are compensated in the control layer).
