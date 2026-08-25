# Roadmap / TODO

Running list of larger, not-yet-scheduled workstreams. Unlike
`tactic_model_design_decisions.md` (a decision log for the tactic-kernel specifically),
this is just a place to park bigger ideas so they don't live only in someone's head or
in chat history. Entries get promoted out of here into an actual plan/PR when someone
picks them up — this file isn't itself a design doc.

Resolved work is kept here only as a one-line pointer (what + commit hash) —
the full investigation narrative for anything already fixed lives in git log
(`git log --all --grep=<topic>`) and commit messages, not in this file.

## Done (one-line pointers — see git log for detail)

- **More tactics (first pass)** — `press_and_contain`, `give_and_go`, 5 example
  `Strategy` configs (`dd73bbc`).
- **BT/py_trees removal** — `AbstractStrategy` rewritten kernel-native, 13 BT
  example strategies + blackboard/tree scaffolding deleted, all riding tests
  ported to kernel-based strategies first (`087ee4b`, `960662c`). `strategy/
  referee/actions.py` + `abstract_behaviour.py` deliberately kept (load-bearing
  for `kernel.RefereeOverride`).
- **CI** — already existed (`.github/workflows/ci.yml`/`lint.yml`); found and
  fixed 2 pre-existing test bugs blocking a green run on `spike/tactic-kernel`
  once it was actually pushed.
- **Tournament scoreless-draw debugging** — 86% → 72% scoreless-draw rate via:
  rSim kick-direction physics fix (`docs/patches/rSim-kick-direction.diff`),
  two `FastPathPlanner` bugs (NaN divide-by-zero in `_find_subgoal`,
  ball-adjacent-obstacle target exemption), `SwitchOfPlayTactic` (new),
  `DefenseTactic`/`defend_parameter` foul-loop fix, `TwoDPID` braking-distance
  cap (`v <= sqrt(2*max_acceleration*error)`). `two_robot_attack` renamed to
  `pass_and_shoot`.
- **Motion-controller discontinuity handling** — `AbstractPID.calculate()` now
  auto-resets PID state on a target jump, orientation-only (`0.5 rad`
  threshold; translation deliberately left alone — see below). Removed 9
  manual `ctx.motion_controller.reset()` call sites this replaced (`91100ff`).
- **CustomReferee gaps** (double-touch rule, ball-speed rule, full-episode
  `reset()` for RL reuse, `set_bt_data` → `set_debug_status` rename) — all
  done; see `docs/custom_referee.md`'s "Known gaps" section.
- **Repo root cleanup** — stray committed session transcripts removed; 7 files
  importing deleted `strategy.examples` triaged file-by-file (2 ported:
  `demo_dribbler_test.py`, `main.py`; 5 deleted as having no kernel-tactic
  equivalent worth building).
- **Strategy-computation perf pass** — `FastPathPlanner._find_subgoal`
  bounding-box prune, `VelocityRefiner`/`KalmanFilter` scalar rewrites,
  `robosim` pipe I/O de-dup. ~3.0x end-to-end speedup measured (30s 6v6 match:
  ~47.5s → ~16.0s wall time), compounding across all perf commits this
  session (`3086337`, `0cf1e17`, `679e8cd`, `b79b863`, `b98cd29`, `4493002`).
- **`AGENTS.md`** — agent-agnostic contributor doc: kernel/`Tactic`/`Strategy`/
  `Partitioner` model, single-writer-partition invariant, minimalism
  discipline, headless/CI conventions, "Writing a Tactic" guide.
- **Goalkeeper overshoot (2026-08-24)** — two independent root causes, both
  fixed. (1) `predict_ball_pos_at_x` returning `None` right as a shot crosses
  the goal line snapped the keeper's target to the goal center for one tick;
  `goalkeep.py` now holds the ball's own position instead when within 0.5m of
  the line. (2) `FastPathPlanner`'s lookahead "carrot" (up to 1m ahead of the
  robot) was being used for `TwoDPID`'s braking-distance cap instead of the
  true target, so the cap never engaged until the last ~1m — confirmed
  against a real tournament replay (`counter_flow_vs_tiki_taka_plus_Lk.pkl`,
  t=13-17.5s: keeper velocity reversed sign 6+ times approaching a target
  that had already settled). Fixed via `TwoDPID.set_final_target()`; measured
  ~40% avg / ~60-65% worst-case overshoot reduction in the reproduction
  match. Residual overshoot (~0.17-0.26m) is now consistent with the robot's
  physical acceleration limit, not a logic bug.
- **Defense-area retrieval stall** — a robot legally retrieving a ball resting
  in the opponent's defense area during a stoppage (`DIRECT_FREE_*`,
  `BALL_PLACEMENT_*`) previously stalled ~0.25m short — `FastPathPlanner`
  treated the defense area as an unconditional obstacle with no exemption for
  a legal dead-ball retrieval. Fixed via
  `_enemy_defense_area_retrieval_exempt()`.
- **Dashboard rebuild** — `custom_referee/gui.py` replaced with a unified
  Live/Replay/Tournament dashboard (`utama_core/dashboard/`); sparse
  (change-only) tactic/referee event logging instead of dense per-tick
  duplication; ~97% redundant per-tick `trace()` calls deduped via
  `MatchLog.trace_if_changed()`. `dashboard_server.py` is the standing way to
  browse replays/tournaments without a live match.
- **Touchline avoidance + ball-placement-into-defense-area stall** — same
  routing gap as the ball-contest deadlock below, triggered by a static
  obstacle (field wall / enemy defense-area rect) instead of a robot:
  `ball_adjacent_obstacles` was only applied to target sanitization, never to
  `check_segment`/`smooth_path` routing, so a ball near a touchline or a
  `BALL_PLACEMENT_OURS` carry into the opponent's box both converged just
  short and stalled. Fixed via a routing exemption scoped to *static*
  obstacles only (never robots — that's the reverted case below) plus
  extending `_enemy_defense_area_retrieval_exempt` to cover carry-to-place,
  not just retrieval.
- **SSL rulebook §8.3/8.4 audit + 7 new referee rules** — Pushing, Crashing
  (both position/velocity-only, no `has_ball` — see `robot_contact.py`),
  Keeper Held Ball, Excessive Dribbling, Robot Stop Speed, Ball Placement
  Interference, stoppage-time Robot-Too-Close-To-Opponent-Defense-Area, plus
  a Multiple Defenders sanction fix (penalty kick + ball-touch gating, not
  occupancy + free kick). Foul-counter/yellow-card mechanism
  (`TeamInfo.increment_foul_counter()`) wired up for the first time — no
  prior rule incremented it. Built via 3 parallel agents; see
  `docs/testing_gaps.md` for what the merge process caught and what it
  didn't.
- **Referee restart-formation fixes + strategy override hook** (2026-08-26).
  `PrepareKickoffOursStep`/`PrepareKickoffTheirsStep` (`custom_referee/
  actions.py`) hardcoded "goalkeeper = robot id 0" and folded the keeper into
  the outfield kickoff formation — a real bug once `AbstractStrategy
  .goalkeeper_id` is ever non-zero (item 3 below), and wrong even at id 0
  (unlike `PreparePenalty{Ours,Theirs}Step`, which already read the real
  keeper ID off the referee packet). Both kickoff steps now do the same:
  read `ref.{yellow,blue}_team.goalkeeper`, exempt that robot from formation
  entirely (absent from `cmd_map`, not just excluded from kicker choice), and
  pass `clear_own_defense_area=True`/`clear_opp_defense_area=True` to
  `_clear_to_legal_positions` as a live legality net (previously only
  `StopStep` did this for kickoff-adjacent formations — no live check meant a
  bad ratio/non-standard field could silently place a robot in a defense area
  with nothing to catch it before `NORMAL_START` fires and immediately
  refouls). Companion fix in `AbstractStrategy.step()`: `GoalkeeperTactic` now
  ticks whenever the goalkeeper's robot ID is absent from `cmd_map`, not only
  when no override command is active — needed because the keeper is now
  legitimately absent from the kickoff steps' output and must fall through to
  real goalkeeper logic instead of freezing.
  Also added: strategy implementers can now override any restart formation
  (`RefereeOverride`/`Strategy`/`AbstractStrategy` all gained a
  `referee_overrides: dict[RefereeCommand, Callable[[Game, MotionController],
  dict[RobotId, RobotCommand]]]` — pass it to `AbstractStrategy(...)` and a
  registered command bypasses the built-in `*Step` entirely; unregistered
  commands are unaffected). Previously there was no extension point at all —
  the only documented customization path was editing `actions.py` directly.
  12 new tests (`test_kickoff_goalkeeper_exemption.py`,
  `test_referee_overrides_customization.py`) plus 4 pre-existing
  `test_referee_unit.py` kickoff tests fixed (they asserted the old buggy
  behavior). Full suite: 781 passed, 0 failed.

## Open

1. **A shared `Sticky`/hysteresis helper** — 5-6 independently-invented
   instances of "keep the previous choice unless a new candidate beats it by
   a margin" exist across skill/tactic/scheduler layers (`go_to_ball.py`'s
   `_COMMIT_RANGE`, `switch_of_play.py`'s `_WEAK_SIDE_MARGIN`,
   `pass_and_shoot.py`'s `_REASSIGN_MARGIN_M`, etc.). Each is well-reasoned in
   isolation but shares no common primitive. **Explicitly not recommended to
   build yet** per the project's minimalism preference — the instances aren't
   quite the same shape (scalar-distance vs. gap-membership vs. boolean edge
   trigger). Revisit only if/when a clearly-6th instance of the exact same
   shape shows up.

2. **More tactics — football-inspired plays/formations.** Only a handful of
   tactics exist today (goalkeeper, defense, pass_and_shoot, lead_and_support,
   switch_of_play, ...). Real football/SSL tactical vocabulary (formations,
   set plays, pressing schemes, overlap patterns) is worth mining for
   genuinely new tactics rather than variations on what exists — also the
   natural forcing function for exercising `TacticTag`/`applicable()` at more
   than toy scale.

3. **AbstractStrategy follow-ups**, deferred from the BT-removal rewrite, not
   urgent:
   - `KernelContext` — reconsider whether it's still needed as a
     `motion_controller`-threading wrapper once `AbstractStrategy` itself is
     simpler.
   - (Resolved 2026-08-26: `goalkeeper_id` now has a real, load-bearing
     override path — the kickoff-formation goalkeeper-exemption fix reads
     the actual keeper ID off the referee packet rather than assuming 0, and
     the new `referee_overrides` hook gives strategy implementers a way to
     customize restart formations per-command. See "Done" above.)

4. **Idea: geometric intention data for Replay-tab overlays** (user's idea,
   2026-08-24). The intention log currently surfaces *what* changed (which
   tactic a robot holds) but not the *geometry* of a decision — which enemy a
   marker covers, a pass-and-shoot's intended receive point, etc. Looks
   tractable, not speculative: `ShadowAndMarkTactic._assign_marks()`
   (`tactics/shadow_and_mark.py:94-108`) already computes exactly this kind of
   `{marker_id: opponent_id}` mapping every tick and discards it. Natural fit
   for `MatchLog.trace_if_changed()` (tactic-side half is small/mechanical).
   The canvas-overlay half (`dashboard/static/field_canvas.js` has no line/
   arrow primitive yet) is the real design work — not started.

5. **Developer documentation** for a new contributor: how kernel/`Strategy`/
   `Tactic`/`Partitioner` fit together, how to author a new `Tactic` end to
   end, testing conventions (headless rsim for most things, grsim for
   anything dribble-related — see [[project_rsim_dribble_issues]]).

6. **CI: dedicated marker for kernel-specific tests?** `tests/engine/` and
   `tests/strategy_runner/` are the real tactic-kernel surface (no test files
   elsewhere are kernel-specific) — an `@pytest.mark.engine` marker could run
   them fast/prioritized on every push instead of only via the full-suite
   sweep. No pytest markers of any kind exist yet. Flagged for whenever CI's
   actual bottleneck becomes clear from real run times, not done speculatively.

7. **grsim as a CI/tournament environment, alongside rsim** (raised by user).
   Three separable sub-problems:
   - **rsim ball-stickiness bug (fix ready, NOT shipped — blocked).** Root
     cause found and fixed in `rc-robosim`/`vendor/rSim` (upstream C++, ODE
     physics): `SSLWorld::setActions()` never called `setDribbler(false)` (a
     one-way latch), and `unholdBall()` didn't actually move the ball clear of
     the kicker collision envelope. Both fixed in
     `docs/patches/rSim-dribbler-release.diff` (source: `vendor/rSim/`, see
     `FORK_NOTES.md`). **Blocker**: installing the patched build regresses
     `test_referee_override.py::test_their_kickoff_clears_our_robots_outside_center_circle`
     (a robot's path planner stalls at `dist_to_center≈0.24m`, well short of
     the required 0.75m; `has_ball` is `False` throughout, so not
     dribbler-related). Leading hypothesis, not confirmed: the ball's
     slightly different post-release resting position pushes
     `FastPathPlanning`'s geometry into a degenerate case (an `invalid value
     encountered in divide` warning from `planner.py`'s `perp_dir /
     np.linalg.norm(perp_dir)` appears in the same run). **Do not install the
     patched build over stock `rc-robosim` in `.pixi/envs/robosim` until this
     is root-caused** — currently reverted to stock 1.2. Whoever picks this up:
     reproduce standalone, confirm/rule out the degenerate-perpendicular-
     vector hypothesis, only then re-attempt installing over stock. Not
     confirmed whether this is the same issue as
     [[project_rsim_dribble_issues]], but at minimum consistent with it.
     Scope note: `DribbleTactic` (the tactic that surfaced this) is not wired
     into any of the tournament configs, so this did not cause the
     scoreless-draw pattern above.
   - **grsim headless/dependency/speed investigation** — not yet verified:
     what grsim's actual runtime dependencies are and whether they're
     installable in a GitHub Actions runner at all (grsim is an external
     process — nothing in this codebase starts/manages one today); whether
     grsim genuinely cannot exceed real-time even headless, or whether
     that's a fixable bottleneck the way rsim's was.
   - **CI integration**, blocked on both items above. `.github/workflows/
     ci.yml` currently hardcodes `--ignore-glob "**/*grsim*"`.

8. **Agentic coding infra** (exploratory, no design decided):
   - CI/testing infra shaped for agent iteration loops — fast feedback on
     whether a newly authored `Tactic` is well-formed before a full rsim/grsim
     run.
   - grsim/rsim feedback surfaced back to an LLM in a usable form — today
     results are numbers/logs/plots meant for a human; agents authoring
     tactics need some translation layer (match summaries, failure
     characterizations, maybe rendered trajectory snapshots).

9. **Repo root cleanup — still pending.** 9 `demo_*.py` scripts plus `main.py`
   sit loose in the repo root, no `demos/`/`scripts/` directory. Worth
   deciding whether the survivors (post broken-demo triage) move into a
   proper subdirectory.

10. **Known open bug: FastPathPlanning convergence stall
    (`test_mirror_swap`).** `tests/motion_planning/multiple_robots_test.py::
    test_mirror_swap` is `xfail(strict=False)`. In one specific 6v6 mirrored
    geometry, the two outer "wing" robots (starting at `(-3.5, ±0.75)`)
    consistently stall 0.53-0.54m from their target — inside the episode
    timeout but outside `endpoint_tolerance=0.3`. Reproduced identically via
    plain `move()` commands independent of strategy class, so this is a
    genuine `FastPathPlanning` convergence/local-minimum behavior for this
    geometry, not test flakiness. Not root-caused — a planner-level gap, not
    a one-line patch. Worth a dedicated investigation since it's a real
    behavior that could show up in an actual match with similar spacing.

11. **Gameplay bugs observed via dashboard Live view** (flagged 2026-08-24).
    User-observed watching a live `tiki_taka_plus` 6v6 match. Three of the
    original four are resolved (goalkeeper overshoot — see "Done" above;
    direct-free-kick retrieval — see "Defense-area retrieval stall" above,
    though worth re-verifying that fix fully covers the originally-reported
    symptom; touchline/defense-area placement stalls — see "Done" above).
    One remains open:
    - **Ball-contest deadlock** (traced 2026-08-25, not fixed). Root-caused
      via a real replay (`counter_press_vs_tiki_taka_plus_Rk.pkl`,
      t=40.23-46.17s: a `GiveAndGoTactic` carrier held 0.11-0.34m from a
      stationary ball for 5.9s of live play, orbiting rather than closing).
      Mechanism: `FastPathPlanner._path_to`'s `ball_adjacent_obstacles`
      exemption (added for an earlier, similar stall) only exempts a
      ball-adjacent opponent from *target* sanitization, not from
      `check_segment`'s path *routing* — so the last approach segment to a
      contested ball is never collision-free and the planner detours
      forever, sweeping the carrot around the opponent's `OBSTACLE_CLEARANCE`
      ring instead of closing the gap.
      Tried and reverted: exempting the same obstacle from routing too
      (mirroring the existing defense-area-retrieval exemption) does let the
      robot reach the ball, but then exposes a worse failure — both robots'
      dribblers register `has_ball` simultaneously and grind in place
      (bodies pinned at exactly `ROBOT_DIAMETER` apart, ball crawling
      ~0.3m/5s instead of frozen). That's a genuine 50/50-contest physics
      case with no possession-arbitration logic to resolve it, a different
      and likely larger problem than the routing gap. Left unfixed pending a
      decision on whether/how simultaneous-touch possession should be
      arbitrated — not a `go_to_ball`/planner fix. (Note: the touchline/
      defense-area routing fix above deliberately does NOT apply here — it's
      scoped to static, non-robot obstacles only, for exactly this reason.)

    All open items need an actual match trace (via the dashboard's Replay
    tab, or `debug_match.py` + a temporary `trace()`/print hook) before
    attempting a fix — root-cause from real per-tick state, not from the
    symptom description alone.

12. **Testing-gap follow-ups from the §8.4 referee-rules audit** — see
    `docs/testing_gaps.md` for full detail. Short version: (1) add at least
    one integration-shaped test per new referee rule that goes through
    `CustomReferee.step()` itself, not just the rule class directly; (2) a
    test driving 3 real fouls through `GameStateMachine` and asserting a
    yellow card lands; (3) a test for the stopping/non-stopping scan-order
    interaction in `CustomReferee.step()`; (4) investigate whether adopting
    mypy/pyright in CI (Ruff-only today) is worth it — would have caught the
    signature-drift bug behind (1) for free; (5) `pushing_rule.py`/
    `crashing_rule.py`/`robot_stop_speed_rule.py` share
    `ball_placement_interference_rule.py`'s latent "assumes `game_frame` is
    never `None`" bug, just never exercised; (6) Pushing/Keeper Held Ball/
    Ball Placement Interference never fired in a 3-match live tournament
    sanity check — least field-validated of the 7 new rules.
