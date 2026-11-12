# Strategy tools: one question, one tool

Design note, agreed after two review rounds; status updated 2026-11-11. It says which tool answers which question, what
is planned and in what order, and what is still undecided. Items marked **planned** do not
exist yet. The current tools are in [tools.md](tools.md), and the "Which tool?" table there
links back here.

## The problem

- **Too many overlapping tools.** Several scripts answer the same question in slightly
  different ways. A reader cannot tell which one to run first, or which result to believe.
- **The docs list scripts, not jobs.** `tools.md` says what each script is. It does not say
  which question the script answers, so a reader has to work that out.
- **Runner set-up was written five times** (step 5 fixed them all). Each of these built its own strategy runner,
  referee and dashboard: `tools/evaluation/match.py`, `tools/debug_match.py`,
  `tools/repro_from_replay.py`, `tools/moments.py` and `main.py`. Now they all build their referee and
  runner in `tools/sim_setup.py`.
  Before step 5 only `match.py` loaded `.strategy` files (`utama_core/bundle/`), so the other four
  could not run a frozen baseline.

## The question, the tool

"Today" is a tool that exists. "Planned" is in the steps at the end.

| Question | Tool | Status |
|---|---|---|
| Does it break? | `pixi run test`, which includes the regression moments in `utama_core/tests/moments/`. Stalls: recorded in the match results for every match and reported by `evaluate.py` with coverage; the round-robin's STALLS section and `--strict` | Today |
| Is it better? | `tools/evaluation/evaluate.py`: paired against the champion, through the reference opponents. `--held-out` before promoting. The only verdict | Today |
| Does everything still run, after a shared-code change? | `round_robin.py --vs-champion`: each strategy plays the champion once (12 matches), under reference noise seed 2, so each (strategy, champion) match is the fixture `evaluate.py` plays in round 2 and is not wasted. A full round-robin (66 matches) only occasionally, for example before a release. A round-robin is not an evaluation: it gives one unpaired match per pair | Today |
| What went wrong in these matches (bugs: thrashing, stalls, restart geometry, fouls)? | `tools/triage.py` (`--fouls` for our side's fouls), then the picture tools. Triage finds bugs, not why a strategy loses in football terms. For that, see `ball_losses.md` in the run folder, `docs/signals.md`, and the goal ledger (`utama_core/analysis/goal_ledger.py`) | Today |
| Keep a fixed bug fixed? | A unit test at the cause. For restart and placement geometry, a regression moment, which is already part of `pixi run test`. `tools/moments.py` writes new ones | Today |
| Look closely at one thing? | `tools/repro_from_replay.py`, the picture tools, `tools/debug_match.py` | Today |
| Is the simulator realistic? | Simulator tools: `realism_scorecard.py`, `physics_ranges.py`, `field_physics.py`, `robot_calibration.py`, `convert_ssl_log.py`. Not part of strategy development | Today |

Notes on the table:

- A round-robin answers "is the whole catalog healthy?". `evaluate.py` answers "is this candidate
  stronger than the champion?". Keep the two apart in the docs and in the output.
- "Look closely" tools explain a result. They never judge one.

## The screen (dropped)

The screen was built and dropped on 2026-11-12: any code change made the paired starts diverge within 0.1 s, and goals in 20 s starts were too rare to say "skip", so it could not save evaluations. `evaluate.py` stays the only gate.

## Moments

Regression moments are fixed-state checks of shared-code bugs (restart and placement geometry),
played by `pixi run test`. Changes to `tools/moments.py`, done 2026-11-11:

- **`moments.py fouls` moved into `tools/triage.py --fouls`** (the code in
  `utama_core/analysis/fouls.py`), so there is one place to find what went wrong in a run.
- **`new` (hand-made templates) removed.** Moments come from `capture`, from a replay.
- **`--seeds` kept.** `play --seeds 0-4` is how a moment is checked (it must fail on 4 of 5
  seeds before the fix).
- **`--against` fixed.** Its worktrees sit beside the repo in `<repo>-moments-against/`, set once in
  `AGAINST_DIR`, no longer in a dated folder from one session.

## Planned steps, in order

1. **Fix `main.py`'s import.** Done: it loads strategies through `strategy/catalog.py`. Its runner set-up is still one of the five hand-built ones above.
2. **This note and the docs.** `docs/strategy_tools_plan.md`, the "Which tool?" table in
   `tools.md`, the debugging entry point in `STRATEGY_DEVELOPMENT.md`, the round-robin's "taken from" counts.
3. **Stalls in the match results.** Done. Stalls are recorded in `replays/match_results/`, and `evaluate.py`
   reports them with coverage: how many of the matches it used were checked for stalls. The
   stall watchdog covers config_a only, so coverage is partial. A match whose stats were not
   collected counts as **unknown, not clean**.
4. **`round_robin.py --vs-champion`.** Done. One match per strategy against the champion, plus
   the champion against the first reference opponent, under reference noise seed 2 by default.
   `evaluate.py` puts the side under test second in even rounds, so each (strategy, champion)
   match is its round-2 fixture (not with `--frozen`, whose opponents are packed files).
5. **One shared match and runner construction** for every tool. `tools/sim_setup.py` has `load_side` (a
   catalog name, a `.strategy` file or `PATH.py:FUNCTION`), `sim_referee` (the referee, its kickoff team and
   seeded restart fuzzing) and `sim_runner` (the rsim `StrategyRunner`). `repro_from_replay.py`, `moments.py`,
   `debug_match.py`, `main.py` and `tools/evaluation/match.py` all build through them, so every tool plays
   `.strategy` files and the referee and runner exist in one place. `main.py` keeps its dashboard: the
   referee is built before the dashboard attaches, the runner after. `match.py`'s fingerprint changed with
   this (its set-up is now in `sim_setup.py`), so every stored match replays once. The match results store
   (`replay/match_results.py`) changed the same fingerprints, so one replay covers both.
6. **The screen.** The screen was built and dropped on 2026-11-12: any code change made the paired starts diverge within 0.1 s, and goals in 20 s starts were too rare to say "skip", so it could not save evaluations. `evaluate.py` stays the only gate.
7. **Triage takes the foul listing.** Done: `triage.py --fouls`; `moments.py` lost `fouls` and `new`,
   and `--against` no longer uses a dated folder.

Every step that changes shared code follows the rule in `docs/STRATEGY_DEVELOPMENT.md`: batch
such changes, and run one round-robin at the end.

## Names considered (undecided)

- **Triage**: `scan_replays.py` or `find_bugs.py`. Undecided. `triage.py` stays until then.
- **Evaluate**: `challenge` or `compare` for the `evaluate.py` family. Undecided.

Nothing above is decided by a name. Pick the name when the tool is merged, not before.
