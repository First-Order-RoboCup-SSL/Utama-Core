# Strategy tools: one question, one tool

Design note, agreed after two review rounds. It says which tool answers which question, what
is planned and in what order, and what is still undecided. Items marked **planned** do not
exist yet. The current tools are in [tools.md](tools.md), and the "Which tool?" table there
links back here.

## The problem

- **Too many overlapping tools.** Several scripts answer the same question in slightly
  different ways. A reader cannot tell which one to run first, or which result to believe.
- **The docs list scripts, not jobs.** `tools.md` says what each script is. It does not say
  which question the script answers, so a reader has to work that out.
- **Runner set-up was written five times** (step 5 fixed four of them). Each of these builds its own strategy runner,
  referee and dashboard:
  `tools/evaluation/match.py`, `tools/debug_match.py`, `tools/repro_from_replay.py`,
  `tools/moments.py` and `main.py`.
  Only `match.py` loads `.strategy` files (`utama_core/bundle/`). The other four cannot run a
  frozen baseline, so they cannot compare against one.

## The question, the tool

"Today" is a tool that exists. "Planned" is in the steps at the end.

| Question | Tool | Status |
|---|---|---|
| Does it break? | `pixi run test`, which includes the regression moments in `utama_core/tests/moments/`. Stalls: recorded in the ledger for every match and reported by `evaluate.py` with coverage; the round-robin's STALLS section and `--strict` | Today |
| Is it better? | `tools/evaluation/evaluate.py`: paired against the champion, through the reference opponents. `--held-out` before promoting. The only verdict | Today |
| Does everything still run, after a shared-code change? | `round_robin.py --vs-champion`: each strategy plays the champion once (12 matches), under reference noise seed 2, so each (strategy, champion) match is the fixture `evaluate.py` plays in round 2 and is not wasted. A full round-robin (66 matches) only occasionally, for example before a release. A round-robin is not an evaluation: it gives one unpaired match per pair | Today |
| What went wrong in these matches (bugs: thrashing, stalls, restart geometry)? | `tools/triage.py`, then the picture tools. Triage finds bugs, not why a strategy loses in football terms. For that, see `ball_losses.md` in the run folder, `docs/signals.md`, and the goal ledger (`utama_core/analysis/goal_ledger.py`) | Today |
| Is this change worth evaluating? | A screen: short paired starts of one strategy against itself, at a git ref. Never a verdict. Never two different strategies. Never written to the ledger. It must report how often the change fired. See "The screen" below | Planned (step 6) |
| Keep a fixed bug fixed? | A unit test at the cause. For restart and placement geometry, a regression moment, which is already part of `pixi run test`. `tools/moments.py` writes new ones | Today |
| Look closely at one thing? | `tools/repro_from_replay.py`, the picture tools, `tools/debug_match.py` | Today |
| Is the simulator realistic? | Simulator tools: `realism_scorecard.py`, `physics_ranges.py`, `field_physics.py`, `robot_calibration.py`, `convert_ssl_log.py`. Not part of strategy development | Today |

Notes on the table:

- A round-robin answers "is the whole catalog healthy?". `evaluate.py` answers "is this candidate
  stronger than the champion?". Keep the two apart in the docs and in the output.
- "Look closely" tools explain a result. They never judge one.

## The screen (planned)

The screen asks a narrower question than `evaluate.py`: "did this change do anything in
play?" It is for deciding whether a change is worth the cost of an evaluation.

- It plays short paired starts of **one** strategy against **itself**, at two git refs: the
  change and its base. The same starts and the same noise are used on both sides.
- It is never a verdict. Its output says where the change had an effect, not whether it helps.
- It never plays two different strategies, and its matches never go into the ledger.
- It must report how often the change fired (how many of the starts reached the code the change
  touched). A screen that never fired says nothing.

**Pass mark (proposed 2026-11-11, not yet agreed).** The screen is a filter in front of
`evaluate.py`, so its two errors cost different amounts. Screening out a change that would have won
loses an improvement for good. Letting a useless change through only costs one evaluation. The
pass mark guards the first error hard and asks only enough of the second to make the screen worth
running.

Past verdicts are too few to judge it on: three `evaluate.py` runs on a change to one strategy
(`split_shape` against the frozen v2.5.0, 2026-11-10, all "better"); the other nine compare
different strategies, which the screen never does. So the validation set is built with known answers:

- **Known worse:** 6 sabotages of the champion, each a small, plausible slip of the size a real change
  makes (a pass gate a little too strict, a receiver who turns late, a marker 0.3 m off, a slower
  restart, a shot threshold off, a missing keeper step-out). Each sabotage is confirmed worse by
  `evaluate.py` first; one that `evaluate.py` cannot tell apart is not a known answer and is dropped.
- **Known better:** each confirmed sabotage reversed (base = sabotaged, change = the champion), plus
  the three real "better" verdicts above.
- **Known nothing:** 3 changes that cannot alter play (a renamed variable, a reordered pure function,
  a log line).

The screen passes validation when all of these hold, on a design (number of starts, seconds per start,
the measure, the threshold) fixed before any of the cases are screened:

1. **Never screens out a known better change.** Every one of them gets "worth evaluating".
2. **Screens out at least half of the known worse ones.** Below that it saves too little to be worth
   running.
3. **Calls every known-nothing change "did not fire"**, or reports no difference on it. A difference
   there means the screen measures noise.
4. **Repeatable:** screened again with other seeds, at least 9 cases in 10 get the same call.
5. **Cheap:** at most a fifth of the playing time of a 6-round `evaluate.py` run.

If it fails, the design is changed once and the whole set is screened again. If it fails twice, the
screen is dropped and `evaluate.py` stays the only gate. Nothing is tuned after the results are in.

**Why not the retired scenario bench.** The scenario bench (retired 2026-10-17) also started
matches from fixed states. It ranked *different* strategies. Its estimate was biased per
strategy, so the ranking failed: Spearman -0.09 against round-robin results (see
[roadmap.md](roadmap.md), "Strategy evaluation"; the retirement is in
[sim_realism.md](sim_realism.md), "Evaluation infrastructure"). The screen differs on both
points: it compares one strategy with itself, so per-strategy bias cancels, and it does not
rank anything. The screen is still a new mechanism, so it is built only after the scratch
validation.

## Moments

Regression moments are fixed-state checks of shared-code bugs (restart and placement geometry).
Changes planned for `tools/moments.py`:

- **Drop `moments.py fouls`.** Its job moves into `tools/triage.py`, so there is one place to
  find what went wrong in a run.
- **Drop `new` (hand-made templates).** Moments come from `capture`, from a replay.
- **Keep `--seeds`.** `play --seeds 0-4` is how a moment is checked (it must fail on 4 of 5
  seeds before the fix).
- **Fix `--against`.** Its worktree folder is a hard-coded dated name next to the repo
  (`AGAINST_DIR = REPO_ROOT.parent / "moments-work-20261017" / "against"` in `tools/moments.py`).
  Planned: a folder that is set in one place and does not carry a date.

## Planned steps, in order

1. **Fix `main.py`'s import.** Done: it loads strategies through `strategy/catalog.py`. Its runner set-up is still one of the five hand-built ones above.
2. **This note and the docs.** `docs/strategy_tools_plan.md`, the "Which tool?" table in
   `tools.md`, the debugging entry point in `STRATEGY_DEVELOPMENT.md`, the `--reuse` counts.
3. **Stalls in the ledger.** Done. Stalls are recorded in `replays/eval/ledger.jsonl`, and `evaluate.py`
   reports them with coverage: how many of the matches it used were checked for stalls. The
   stall watchdog covers config_a only, so coverage is partial. Old ledger entries count as
   **unknown, not clean**.
4. **`round_robin.py --vs-champion`.** Done. One match per strategy against the champion, plus
   the champion against the first reference opponent, under reference noise seed 2 by default.
   `evaluate.py` puts the side under test second in even rounds, so each (strategy, champion)
   match is its round-2 fixture (not with `--frozen`, whose opponents are packed files).
5. **One shared match and runner construction** for the other tools. Done as `tools/sim_setup.py`
   (`load_side`: a catalog name, a `.strategy` file or `PATH.py:FUNCTION`; `sim_runner`: the referee and
   rsim runner), used by `repro_from_replay.py`, `moments.py` and `debug_match.py`; `main.py` loads its
   strategies through it but keeps its own runner for the dashboard. So all four play `.strategy` files.
   `match.py` is unchanged, and moves onto `sim_setup` with the next change that replays every match
   anyway: editing `match.py`, or anything it imports, changes the
   fingerprint of every cached match, so every match replays and the ledger is invalidated
   (`utama_core/replay/fingerprint.py`). If it must change, batch it with other changes to shared
   code and say so in the commit.
6. **The screen.** First validated in scratch code, against past `evaluate.py` verdicts, with the
   pass mark agreed beforehand. Only then built in. Then a `compare.py` tool with two
   subcommands, `verdict` (wraps `evaluate.py`) and `screen`, with no options shared between
   them. Renames happen together with the merge they belong to, not before.

Every step that changes shared code follows the rule in `docs/STRATEGY_DEVELOPMENT.md`: batch
such changes, and run one round-robin at the end.

## Names considered (undecided)

- **`drill`** for the screen: rejected as a name, because `examples/shooting_drill.py` already
  uses "drill".
- **Triage**: `scan_replays.py` or `find_bugs.py`. Undecided. `triage.py` stays until then.
- **Evaluate**: `challenge` or `compare` for the `evaluate.py` family. Step 6 uses `compare.py`
  as a placeholder. Undecided.

Nothing above is decided by a name. Pick the name when the tool is merged, not before.
