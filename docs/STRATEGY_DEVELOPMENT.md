# Strategy development

Context for work under `utama_core/engine/`, `utama_core/tactics/`, `utama_core/skills/`,
`utama_core/strategy/`, or `smoke_tournament.py`/`docs/strategies.md`. Assumes you've read the
root `AGENTS.md`. Design rationale and rejected alternatives: `docs/tactic_model_design_decisions.md`
— read it before proposing a change to the kernel's shape.

`engine/` is the scheduler/protocol infra; `strategy/kernel_strategy.py` holds the
`build_*_kernel_strategy` factories, where day-to-day strategy edits land. (`engine/` was
renamed from `kernel/` so it can't be confused with "kernel strategy", the model's own term.)

## The tactic-kernel model

A team's play is a `Strategy` (`engine/strategy.py`): a scheduler that re-partitions the
outfield robots across concurrently-running `Tactic`s every tick. The goalkeeper (robot 0 by
default, `Strategy.set_goalkeeper()`) is pinned outside the scheduler.

- **`Tactic`** (`engine/tactic.py`) — a plain object: `tick(game, ctx, robot_ids, mem) ->
  (commands, mem)`, a required `tag` (ATTACK/DEFENSE/MIXED — used only for dashboard
  colouring), and optional `applicable()`/`is_committed()`/`suggest_next()`. `mem` is a plain
  dataclass; no blackboard, no state-machine base class.
- **`Partitioner`** — a plain function that splits the *free* robots (those no committed slot
  holds) across tactic slots. It receives `available_tactic_ids` (applicable and not pinned by
  a commitment) and may only assign robots to those. No bid/fitness scoring.
- **`AbstractStrategy`** (`engine/abstract_strategy.py`) — what `StrategyRunner` drives; wraps
  a `Strategy` built by a factory.

**Invariants:**
- *Single writer:* the partition is decided once per tick, before any `Tactic` runs, so two
  tactics can never contend for a robot. Don't add a path that lets a `Tactic` claim or release
  robots outside the `Partitioner`.
- *Commitment:* while `is_committed()` is True the slot's robots can't be reassigned. Backstop:
  `commitment_deadline_s` (default 15s) releases a slot committed that long with the ball moved
  <5cm, and logs it. A deadline release means your tactic lacks a release path — fix the
  tactic, don't raise the deadline.
- *Resets* (`engine/referee_reset.py`): entering a restart (kickoff, penalty, free kick, ball
  placement, goal), resuming from one, or `FORCE_START` straight out of `STOP`/`HALT` is a
  barrier — every slot's `mem` and commitment clears. `STOP`/`HALT` alone only pause; a pause
  resumed by `NORMAL_START` keeps state.

## Referee handling

`CustomReferee` (`utama_core/custom_referee/`) runs in-process and identically across
rsim/grsim/real. During a restart, `RefereeOverride` (`engine/referee_override.py`) takes over
every outfield robot *before* any `Tactic` ticks, reusing the `*Step` classes in
`custom_referee/actions.py`. Strategies can override a restart formation via
`Strategy(referee_overrides=...)`. See `docs/custom_referee.md`.

## Writing a Tactic

Lessons from bugs that recurred (mostly `SwitchOfPlayTactic`, `tactics/switch_of_play.py`):

- **`go_to_point()` always faces the ball.** If orientation matters while stationary (facing a
  pass target), use `move()` with `target_oren`. Check every `go_to_point(` call site — fixing
  one doesn't fix the pattern.
- **`intercept_point()` (`shared/pass_and_score_geometry.py`) projects along the passer's
  current orientation,** not toward the receiver. A mis-facing passer sends the receiver to a
  nonsense point; it fails quietly and looks like a vague stall.
- **Every path that sets `is_committed()` True needs a path back to False** — success, failure
  and timeout. Check a timeout reset actually sticks for a full tick rather than being
  re-advanced past in the same tick.
- **Verify by tracing a real match** (`ctx.match_log.trace`, below), not by reading the phase
  logic.
- **A tactic can trip referee rules unrelated to its logic** (e.g. a relay walking an attacker
  into its own defense area). On a foul stall, check which robot and which rule first.

## Writing and evaluating a strategy

A strategy is a combination of existing tactics plus a partitioner that decides how many robots
each gets. Most new strategies need no new tactic; if one does, propose the tactic first
(see `AGENTS.md`, minimalism).

**Writing one.** Add `build_<name>_kernel_strategy(outfield_robot_ids)` to
`strategy/kernel_strategy.py`, returning a `Strategy(tactics={...}, partitioner=...)`. Every
`build_*_kernel_strategy` is discovered by name (`tournament_lib`), so it joins round-robins and
the bench as `<name>` with no registry to edit. Reuse the shared partitioner pieces rather than
re-deriving them:
- `_friendly_closer_to_ball(game)` — the possession edge. True/False is a clear edge; None is a
  near-tie or unreadable state, where a sticky picker keeps its previous split.
- `_carrier_first(game, free)` / `_clearer_first(game, ordered)` — robot order for a slot that
  must take the ball: the carrier (or the kicker at a still ball) first, else the nearest.
- `_fixed_ratio_picker`, `_possession_split_picker` — the two common split shapes.

Then add it to `_CONFIGS` in `tests/engine/test_all_strategy_configs.py` (builds it and runs it
through the kernel invariants), give its partitioner pure-function tests in
`tests/strategy/test_<name>.py` (a `Game` built by hand, no rsim), and add a catalog row to
`docs/strategies.md` with status `experimental`.

**Evaluating one,** cheapest first; stop as soon as a step fails:
1. **Tests:** its own, `test_all_strategy_configs.py`, then the full suite `--headless`.
2. **One saved match** against a few opponents: `smoke_tournament.py --pair <name> <opp>`.
   Saved, so stalls are recorded (`--no-save` cannot see them). A stall is a bug to fix
   before anything else.
3. **Bench A/B** against its nearest existing strategy, the one it differs from in a single
   idea, so the A/B tests that idea: `tools/scenario_bench.py --load-bank <newest bank>
   --candidate <name> --baseline <nearest> --opponent <opp> --stop-at-t 4` (see below).
   Use an opponent outside the pair.
4. **Matches** to confirm: a strict round-robin with it in (`smoke_tournament.py --strict`),
   0 stalls. Record the result in `docs/strategies.md`.

**What counts as better.** Results: goals and W-D-L in matches, and the bench's outcome delta.
Everything in [Reading a tournament run](#reading-a-tournament-run) explains a result; none of
it is a target. Don't tune a threshold until the bench moves: a change needs a reason in game
terms, and a bench gain that matches don't confirm means distrust the bench, not that the
strategy got better. The bench agrees with round-robin standings at Spearman +0.66 (two
round-robins agree at +0.91), so it screens changes; it doesn't rank strategies.

## Observability — use these before adding a debug print

- **`MatchLog`** (`engine/match_log.py`) — one JSONL per match. `Strategy` records an
  intention row per slot-assignment *change*; call `ctx.match_log.trace(tick, sim_time, key,
  value)` yourself from a `Tactic` or skill (guard with `if ctx.match_log is not None:`; it's
  `None` on tournament/CI runs, so traces can stay in). Examples: `skills/src/go_to_ball.py`,
  `tactics/give_and_go.py`. Enable with `match_log_path=` or `tournament_lib.run_match(...,
  run_dir=...)`; read back with `load_jsonl(path)`.
- **`render_window()` / `render_around_event()`** (`replay/render_window.py`) — PNG of
  robot/ball trails over a window, optionally anchored on a `MatchLog` event. **Default to this
  over `load_frames_in_range()`**: a coordinate dump is expensive in context and easy to misread
  spatially. Use raw frames only for an exact number once the picture has localized the issue.
- **`render_clip()`** (`replay/render_clip.py`) — MP4 of a window for humans, ball-following
  or full-pitch camera; sim ball teleports are never shown. Needs `ffmpeg`. Commands behind
  committed clips: `demo_clips/README.md`.
- **`docs/strategies.md`** — every factory's status and the latest results. `baseline`
  strategies aren't meant to win; don't tune them to.
- **`smoke_tournament.py`** — round-robin runner (`--max-workers N`, `--both-sides`,
  `--strict`, `--stop-at-first-stall`, `--fuzz-restarts SEED`, `--fuzz-interval LO HI`,
  `--no-save`, `--pair A B` for one fixture with A as config_a — reruns a stalled match from a
  round-robin; see the determinism caveat below).
- **Ball losses** (`replay/turnover_breakdown.py`) — runs after every saved tournament and
  writes `ball_losses.md`; see [Reading a tournament run](#reading-a-tournament-run).
  `python -m utama_core.replay.turnover_breakdown <run_dir>` re-runs it on an older run.
- **Stall watchdog** (`engine/match_stats.py`) — records `StallEvent`s, never affects play:
  `RESTART_STALL` (a restart/stoppage command held >15s) and `COMMITTED_FROZEN` (ball moved
  <5cm for >10s in live play while a slot is committed). Each `RESTART_STALL` carries a one-line
  `diagnosis` (ball in goal / past a line, taker not closing on the ball, taker at the ball but
  not kicking). The tournament prints a STALLS section, writes it to `summary.json`, and `--strict` exits non-zero on any stall; a
  possession-pinned backstop flags what the watchdog misses.
- **Restart fuzzing** (`custom_referee/restart_fuzzer.py`) — injects legal restarts at
  seeded-random times to exercise auto-advance paths; same seed and interval reproduce the
  schedule. Details: `docs/custom_referee.md`.
- **`tools/replay_trace.py REPLAY.npz T0 T1`** — text trace of a window: referee command and
  why it changed, ball, nearest robot per side. The first look at a stall or a voided restart.
- **`repro_from_replay.py`** — reload a replay's field state at time *t* into a fresh headless
  match with tracing on, instead of re-running the whole match (`--help` for flags).

**Frame convention trap:** rsim's own frame stores y negated relative to ours. Anything that
writes into the sim (e.g. `teleport_robot`) must negate y *and* heading — `543741e` fixed a
teleported robot facing the mirrored heading, which had silently corrupted scenario benches.

**Determinism:** an rsim match is reproducible: `--pair A B` replays a round-robin match exactly (13 of
13 sampled from `tournament_20260928_221404` at `1de1e18d`, every replay array and sidecar).
One cause of "differs run to run" was state that outlived a match: `has_ball`'s and shielding's
hysteresis were module dicts keyed by robot id, shared by both teams and carried from match to
match in a round-robin or bench worker process (`c27bfad7`). Module-level state in a tactic or
skill must be keyed by team and cleared at match start (`StrategyRunner.__init__`). If a
round-robin match and its `--pair` rerun still differ, suspect more of the same; compare the two
replays' `ball_p` to find the first differing tick.

**Fast and exact numerics:** the motion planner, refiners and Kalman filter use faster float code
by default (about 19% less wall time per match under full load). `UTAMA_EXACT_MATH=1` restores the
original numpy paths, byte-identical to before (`8c191eea`); the round-robin, bench pools and rsim
all inherit it. The two modes agree to 1e-9 per call but matches diverge after the first differing
tick, so a baseline recorded in one mode is only comparable with a candidate run in the same mode.

## Reading a tournament run

Every saved `smoke_tournament.py` run prints these sections and writes the same data to
`replays/<run>/summary.json`. Look here before adding a new metric: it probably exists.

| Printed section | `summary.json` key | What it tells you | Caveat |
|---|---|---|---|
| Standings, STRATEGIES | `strategies` | per strategy: W-D-L, goals, shots, passes, entries, fouls, real losses, `stalled` | a stalled match stays in W-D-L, flagged |
| LOSS KINDS | `strategies[*].real_loss_kinds_as_a` | where each strategy gives the ball away: tackled, kicked out, shot saved, intercepted, loose ball lost, foul (`turnover_breakdown.TURNOVER_KINDS` and `RESTART_KINDS`) | config_a matches only (the side with an intentions log) |
| BALL LOSSES | `ball_losses` | the same kinds over the run, fouls by rule, losses by the tactic holding the ball; full tables in `ball_losses.md` | raw `MatchStats.turnovers` is ~40% two robots on one ball flipping "nearest": use real losses |
| `passes:` line | `ball_losses.receptions` | every pass to `received` / `missed_reception` (reached a teammate, no contact) / `intercepted` / `off_target`; catch rate by receiver facing | config_a only |
| FOULS | `fouls` | every foul, both sides, by rule, then strategy/tactic of the offending robot | `*` rules name only a team: attributed to its robot nearest the ball |
| STALLS | `stalled_match_count`, `stall_incidents`, per-match `stats.stall_events` | `RESTART_STALL` with a one-line diagnosis, `COMMITTED_FROZEN` with the committed tactics; `stall_incidents` merges one freeze seen against two opponents (same kind and ticks, a shared strategy) | a stall may be the strategy, the planner, the referee or the sim |
| RESTARTS | `restarts` | every kickoff, free kick and penalty: how many reached NORMAL_START and were `taken`, or were `voided` / `stopped_before_kick` / `timeout` / `match_ended` (`replay/restart_outcomes.py`) | both sides' restarts |
| — | `run` | git commit, dirty flag, argv | compare runs only at clean commits |

These are diagnostics, not objectives. Fewer losses is not better on its own: a strategy that
never passes or shoots loses the ball least. Rank strategies by results (goals, W-D-L), and use
the rest to explain why one wins or loses, and which shared primitive (reception, carrying, the
planner, the referee) is failing every strategy at once.

## A/B on the scenario bank

For a targeted A/B of one change (a tactic, the planner) without an hour-long round-robin, use
`tools/scenario_bench.py` on the committed bank (`utama_core/replay/banks/`, newest version):
every start harvested from one round-robin (kickoffs, free kicks, penalties, and open play: a
pass about to be made, a ball just lost), near-duplicates dropped, each played 20 s once, the
candidate against a fixed opponent. rsim is deterministic, so the same code gives the same
outcomes, and a baseline recorded once serves every later candidate.

    # once, at the baseline commit
    pixi run python tools/scenario_bench.py --load-bank utama_core/replay/banks/bank_vN.json \
        --candidate press_and_pass --opponent low_block --workers 15 --output-dir bench_base
    # per candidate commit
    pixi run python tools/scenario_bench.py --load-bank utama_core/replay/banks/bank_vN.json \
        --candidate press_and_pass --opponent low_block --workers 15 --output-dir bench_new \
        --against-results bench_base/scenario_bench_<timestamp>.json

(or `--baseline <config>` to compare two strategies at one commit). The last line printed is
the paired result: mean outcome delta per scenario, its standard error, and t = mean / stderr.
|t| under about 2 is within chance. The sign says which side did better, and the per-family
table says where. Treat it as a screen, then confirm a real improvement with matches.

A bench worker keeps its rsim subprocess between starts (`enable_sim_reuse` in
`robosim_wrapper.py`), asking it for a new native world each time instead of paying about 0.4 s wall
and 0.5 s CPU to start one: outcomes and every sim state are bit-identical to a fresh process per
start, in exact and fast numerics (checked on 106 starts in two shuffled orders). Only the sim is
kept: a `StrategyRunner` is still built per start, which costs about 0.2 s.

To screen many candidates, add `--stop-at-t 4` to the candidate run: it scores a shuffled sample 100
starts at a time and stops once |t| reaches 4 (2 would give false alarms, since t is looked at
repeatedly). Passes aimed 10° off stop after 100 of 846 starts, about 1.5 min. It also stops as
futile once the mean delta is confidently under 0.15, that is once |mean| + 2.5 x stderr < 0.15
(`FUTILE_BELOW`, `FUTILE_Z`). The printed summary and the JSON's `stopped` say which
("detected" or "futile"; null when every start was scored). 0.15 is about the smallest mean delta a
full bank_v5 pass detects at |t| >= 4 (per-start deltas have spread 1.02, and 4 x 1.02 / sqrt(846)
= 0.14), and 2.5 is a one-sided 5% bound split over the 8 checks. Calibration on the recorded
10°-off run (mean -0.27 over 845 starts), with 2000 simulated passes each:

| Simulated candidate | Stopped futile | Detected | Mean starts scored |
|---|---:|---:|---:|
| 10° off, resampled | 0% | 100% | ~295 |
| half that effect (-0.13) | 4.5%, none a full pass would have detected | 39% | ~730 |
| null, deltas as spread as 10° off (38% of starts changed) | 89% | 0% | ~565 |
| null, 20% of starts changed | ~100% | 0% | ~320 |
| null, 10% of starts changed | 100% | 0% | ~185 |
| identical code (all deltas 0) | 100% | 0% | 100 |

On the recorded run in the bench's own order the futility bound never falls below 0.36. So a
candidate no different from the baseline now costs a fifth to two thirds of a pass, less the
fewer starts it changes. "Futile" means no difference of 0.15 or more, not no difference.

The current bank is `bank_v7` (from `tournament_20261001_094103`, the first stall-free
round-robin): press_and_pass vs low_block against itself is identical on all 860 scored starts,
and passes aimed 10° off are detected after 400 (mean -0.247, t -4.98). The calibration figures
above were measured on bank_v5.

How far to trust the bench: every strategy scored on 150 bank_v5 starts against counter_press
ranks them with Spearman +0.66 against round-robin points, where two round-robins agree at +0.91
(`tools/bench_vs_standings.py`). It gets the top and bottom right and shuffles the middle, so use
it to screen a change against a baseline and confirm with matches, not to rank strategies.

A new bank from a new round-robin (keep its replays until this is done) takes a few minutes and
no simulation:

    pixi run python tools/scenario_bench.py --harvest-from replays/tournament_<id> --open-play 2 \
        --save-bank utama_core/replay/banks/bank_vN+1.json --list-scenarios

Starts where a robot is past the field lines are dropped (the sim can't place it there), and a
scenario whose run errored on either side has no delta. There is no screen that plays starts
forward to pick "informative" ones: the one tried varied the opponent, not the candidate, and
threw away as many useful starts as it kept. A new bank needs a new baseline run.
