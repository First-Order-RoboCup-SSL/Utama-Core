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
- **`repro_from_replay.py`** — reload a replay's field state at time *t* into a fresh headless
  match with tracing on, instead of re-running the whole match (`--help` for flags).

**Frame convention trap:** rsim's own frame stores y negated relative to ours. Anything that
writes into the sim (e.g. `teleport_robot`) must negate y *and* heading — `543741e` fixed a
teleported robot facing the mirrored heading, which had silently corrupted scenario benches.

**Determinism caveat:** rsim matches are *mostly* reproducible, but some (seen with
`press_and_pass`) differ run to run on identical code; cause not yet known. Before attributing a
result difference to a code change, re-run the baseline.

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

For a targeted A/B of one change (a tactic, the planner) without an hour-long round-robin, use
`tools/scenario_bench.py`: restart moments harvested from a run, played 20s from jittered
starts, candidate vs baseline (another config, or `--against-results` from another commit),
reported per family with a standard error.
