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
   - `goalkeeper_id`/`exp_ball` as `AbstractStrategy.__init__` params —
     `goalkeeper_id` has zero real overrides today; worth reconsidering
     whether it belongs as a constructor param at all.
   - `KernelContext` — reconsider whether it's still needed as a
     `motion_controller`-threading wrapper once `AbstractStrategy` itself is
     simpler.

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
    User-observed watching a live `tiki_taka_plus` 6v6 match. Two of the
    original four are resolved (goalkeeper overshoot — see "Done" above;
    direct-free-kick retrieval — see "Defense-area retrieval stall" above,
    though worth re-verifying that fix fully covers the originally-reported
    symptom). Two remain open, not yet traced:
    - **Ball-contest deadlock.** Two robots (one per team) converging on a
      loose/contested ball don't resolve possession — push against each
      other and slowly drift together, neither gaining clear control nor
      backing off. Possibly `PressAndContainTactic`/`go_to_ball` obstacle-
      avoidance interaction when both sides' targets coincide on the ball, or
      an rSim contact-physics artifact — not confirmed. Worth checking
      whether either side's tactic has any "I'm losing this contest, back
      off" branch at all.
    - **Ball placement into/near the defense area stalls the restart.** When
      the ball ends up inside/near a defense area and the opposing team needs
      to retrieve/place it to restart play, placement itself appears
      unreliable (distinct from a failed kick/pass attempt after a
      successful placement). Suspect areas: `DefenseAreaRule`'s keep-out
      enforcement fighting the placing robot's approach path,
      `BALL_PLACEMENT_*` handling for a designated position landing inside/
      near a defense area boundary, or a tactic with no explicit "placement
      point is inside a keep-out zone" case. Not traced. User's suspicion,
      not yet confirmed: this may be the single largest contributor to
      matches "going stale after a certain time."

    All open items need an actual match trace (via the dashboard's Replay
    tab, or `debug_match.py` + a temporary `trace()`/print hook) before
    attempting a fix — root-cause from real per-tick state, not from the
    symptom description alone.
