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
- **Runner set-up was written five times** (step 5 fixed four of them). Each of these builds its own strategy runner,
  referee and dashboard:
  `tools/evaluation/match.py`, `tools/debug_match.py`, `tools/repro_from_replay.py`,
  `tools/moments.py` and `main.py`.
  Before step 5 only `match.py` loaded `.strategy` files (`utama_core/bundle/`), so the other four
  could not run a frozen baseline.

## The question, the tool

"Today" is a tool that exists. "Planned" is in the steps at the end.

| Question | Tool | Status |
|---|---|---|
| Does it break? | `pixi run test`, which includes the regression moments in `utama_core/tests/moments/`. Stalls: recorded in the ledger for every match and reported by `evaluate.py` with coverage; the round-robin's STALLS section and `--strict` | Today |
| Is it better? | `tools/evaluation/evaluate.py`: paired against the champion, through the reference opponents. `--held-out` before promoting. The only verdict | Today |
| Does everything still run, after a shared-code change? | `round_robin.py --vs-champion`: each strategy plays the champion once (12 matches), under reference noise seed 2, so each (strategy, champion) match is the fixture `evaluate.py` plays in round 2 and is not wasted. A full round-robin (66 matches) only occasionally, for example before a release. A round-robin is not an evaluation: it gives one unpaired match per pair | Today |
| What went wrong in these matches (bugs: thrashing, stalls, restart geometry, fouls)? | `tools/triage.py` (`--fouls` for our side's fouls), then the picture tools. Triage finds bugs, not why a strategy loses in football terms. For that, see `ball_losses.md` in the run folder, `docs/signals.md`, and the goal ledger (`utama_core/analysis/goal_ledger.py`) | Today |
| Is this change worth evaluating? | `tools/screen.py`: short paired starts of one strategy against itself, at a git ref. Never a verdict. Never two different strategies. Never written to the ledger. Reports how often the change fired. See "The screen" below | Built 2026-11-11, on trial |
| Keep a fixed bug fixed? | A unit test at the cause. For restart and placement geometry, a regression moment, which is already part of `pixi run test`. `tools/moments.py` writes new ones | Today |
| Look closely at one thing? | `tools/repro_from_replay.py`, the picture tools, `tools/debug_match.py` | Today |
| Is the simulator realistic? | Simulator tools: `realism_scorecard.py`, `physics_ranges.py`, `field_physics.py`, `robot_calibration.py`, `convert_ssl_log.py`. Not part of strategy development | Today |

Notes on the table:

- A round-robin answers "is the whole catalog healthy?". `evaluate.py` answers "is this candidate
  stronger than the champion?". Keep the two apart in the docs and in the output.
- "Look closely" tools explain a result. They never judge one.

## The screen (built, on trial)

The screen asks a narrower question than `evaluate.py`: "did this change do anything in
play?" It is for deciding whether a change is worth the cost of an evaluation.

- It plays short paired starts of **one** strategy against **itself**, at two git refs: the
  change and its base. The same starts and the same noise are used on both sides.
- It is never a verdict. Its output says where the change had an effect, not whether it helps.
- It never plays two different strategies, and its matches never go into the ledger.
- It must report how often the change fired (how many of the starts reached the code the change
  touched). A screen that never fired says nothing.

**As built** (`tools/screen.py`, 2026-11-11). The design was fixed before any change was screened:

- **Starts:** 60 moments of live play (NORMAL_START or FORCE_START), drawn uniformly at random from
  the replays of a run in which the strategy played, so situations come as often as in matches.
  20 s each. Robots start from rest with fresh tactic memory.
- **Sides:** the change (the working tree) and the base (a `.strategy` file at REF, as `evaluate.py
  --frozen` builds) each play our side against the base, under the reference world and robots with
  one seed per start.
- **Fired:** the two starts diverged (a robot or the ball more than 1 µm apart). Checked: the same
  code, file against code, never diverged on 3 starts; a strategy whose code is the same at both refs
  is not played at all.
- **Call:** `SKIP: did not fire` if under 10% of starts fired; `SKIP: fewer goals in the starts` if,
  over the fired starts, the goal difference (change minus base) is lower by more than 2 standard
  errors; otherwise `WORTH EVALUATING`.
- **Cost:** about 30 minutes on 8 workers, against several hours for a 6-round `evaluate.py` run.
- Every run appends to `replays/eval/screen_log.jsonl` (commit, fingerprints of both sides, every
  pair, the call), so its calls can be checked against verdicts later.

**Pass mark: a track record on real changes (agreed 2026-11-11).** The screen is a filter in front of
`evaluate.py`, so its two errors cost different amounts. Screening out a change that would have won
loses an improvement for good. Letting a useless change through only costs one evaluation. Past verdicts
are too few to judge it on (three `evaluate.py` runs on a change to one strategy, all "better"), so it
is judged going forward:

- On every real change to one strategy, run the screen first and keep its call, then run
  `evaluate.py` as usual. **`evaluate.py` stays the only gate meanwhile:** the screen decides nothing.
- It earns the right to skip evaluations after **at least 8 real changes with a clear verdict**
  (better or worse) where it **never** called "skip" on one `evaluate.py` called better, and would have
  skipped **at least half** of those called worse.
- The rule above is not tuned during the trial. If it fails, it is changed once and the record starts
  again; if it fails twice, the screen is dropped and `evaluate.py` stays the only gate.
- If the record grows too slowly, deliberate small mistakes in the champion (each confirmed worse by
  `evaluate.py` first), their reversals and changes that cannot alter play can be added as known answers.

**Why not the retired scenario bench.** The scenario bench (retired 2026-10-17) also started
matches from fixed states. It ranked *different* strategies. Its estimate was biased per
strategy, so the ranking failed: Spearman -0.09 against round-robin results (see
[roadmap.md](roadmap.md), "Strategy evaluation"; the retirement is in
[sim_realism.md](sim_realism.md), "Evaluation infrastructure"). The screen differs on both
points: it compares one strategy with itself, so per-strategy bias cancels, and it does not
rank anything. It is still a new mechanism, so it decides nothing until its track record is in.

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
6. **The screen.** Built as `tools/screen.py` (2026-11-11), on trial: its calls are kept beside
   `evaluate.py`'s verdicts on real changes until the track record above is in. Only if it passes,
   a `compare.py` tool with two subcommands, `verdict` (wraps `evaluate.py`) and `screen`, with no
   options shared between them. Renames happen together with the merge they belong to, not before.
7. **Triage takes the foul listing.** Done: `triage.py --fouls`; `moments.py` lost `fouls` and `new`,
   and `--against` no longer uses a dated folder.

Every step that changes shared code follows the rule in `docs/STRATEGY_DEVELOPMENT.md`: batch
such changes, and run one round-robin at the end.

## Names considered (undecided)

- **`drill`** for the screen: rejected as a name, because `examples/shooting_drill.py` already
  uses "drill". The trial tool is `screen.py`; its final name comes with the `compare.py` merge.
- **Triage**: `scan_replays.py` or `find_bugs.py`. Undecided. `triage.py` stays until then.
- **Evaluate**: `challenge` or `compare` for the `evaluate.py` family. Step 6 uses `compare.py`
  as a placeholder. Undecided.

Nothing above is decided by a name. Pick the name when the tool is merged, not before.
