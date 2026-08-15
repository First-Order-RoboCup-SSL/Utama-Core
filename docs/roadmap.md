# Roadmap / TODO

Running list of larger, not-yet-scheduled workstreams. Unlike
`tactic_model_design_decisions.md` (a decision log for the tactic-kernel specifically),
this is just a place to park bigger ideas so they don't live only in someone's head or
in chat history. Entries get promoted out of here into an actual plan/PR when someone
picks them up — this file isn't itself a design doc.

## Sequencing (current thinking, 2026-08-15)

1. **BT/py_trees cleanup** and **more tactics** first — in either order, no hard
   dependency between them.
2. **CI** deliberately comes *after* those two, not before. Reason: CI gates
   (lint scope, which test dirs are stable enough to require green) should be
   written against the codebase we intend to keep, not against a tree that's
   about to have a bunch of BT code deleted and a bunch of new tactics/tests
   added. Writing CI first just means rewriting it once cleanup and the new
   tactics land and change what "passing" even means.
3. Tournament/multi-strategy infra waits on there being enough of a tactic
   catalog for comparisons to mean anything.
4. Agentic coding infra (`AGENTS.md` at least) can start any time — cheap,
   and best written while the tactic-kernel reasoning is still fresh.

## Multi-strategy / tournament evaluation infra

Today's evaluation is strategy-vs-strategy via `StrategyRunner` (one strategy per
side). We'll eventually want something tournament-shaped: many strategies (or many
tactic-kernel configs) round-robined or bracketed against each other, with aggregate
results, not just a single head-to-head match. Much later priority — revisit once
there's enough of a tactic catalog to make comparisons meaningful.

Note: an earlier plan (`snug-hugging-sutton.md`, now deleted) explored a
multi-strategy `Runner` built directly on `AbstractStrategy`/py_trees, to let several
functional strategies each dynamically own a subset of robots. That specific
mechanism problem — many things concurrently owning dynamic robot subsets — is what
the tactic-kernel (`Strategy` + `Tactic` + `Partitioner`) already solves, on a
different (non-BT) substrate. Any future tournament/multi-strategy infra should build
on the kernel, not resurrect the BT-based Runner design. A few ideas from that plan
are still worth keeping in mind when this gets built:
- Reassignment should reset a tactic's `mem` exactly when its robot set changes, not
  otherwise (already the pattern inside `two_robot_attack.py` and generalized into the
  kernel's `Strategy` tick loop).
- Conflict detection: never allow the same robot to be double-assigned in one tick
  silently — assert loudly instead of last-write-wins.
- An allocator/partitioner is cleanest as a pure function `(Game, ...) -> assignment`,
  re-run every tick; "static" allocation is just the trivial case of a function that
  ignores `Game`.

## More tactics — football-inspired plays/formations

Only a handful of tactics exist today (goalkeeper, defense, two_robot_attack,
lead_and_support), each tagged via the closed `TacticTag` vocabulary
(see `tactic_model_design_decisions.md` §15). There's a lot of real football/SSL
tactical vocabulary worth mining for genuinely new tactics — formations, set plays,
pressing schemes, overlap/give-and-go patterns, etc. — rather than growing the
catalog by variations on what's already there. Also the natural forcing function for
actually exercising `applicable()`/tags at more than toy scale.

## Codebase cleanup — remove remaining BT/py_trees junk

`AbstractStrategy`, the py_trees blackboard plumbing, and BT-only strategies/tests are
still in the tree even though the tactic-kernel is the live path going forward. Needs
an isolated cleanup pass (previously attempted inline alongside unrelated rsim-env
cleanup and reverted because the diffs got entangled — see kernel-cleanup commit
history around `97839a6`). Should be its own PR, not mixed with other work.

## AbstractStrategy follow-ups (from the BT-removal rewrite)

Deferred during the `AbstractStrategy` rewrite (merging `KernelStrategy` into it,
dropping py_trees) — not urgent, revisit once there's a concrete forcing case:

- `goalkeeper_id`/`exp_ball` as `AbstractStrategy.__init__` params: `goalkeeper_id`
  has zero real overrides today (every `build_*_kernel_strategy` factory uses the
  default `0`) — worth reconsidering whether it belongs as a constructor param at
  all, or should just be hardcoded until a config actually needs a different
  keeper id. `exp_ball` is genuinely read by `StrategyRunner`'s validation before
  any tactic runs, so it likely does need to live somewhere the runner can see it
  — but worth a closer look at whether the constructor is the right place once
  more of `AbstractStrategy`'s shape has settled.
- `KernelContext` — reconsider whether it's still needed as a wrapper once the
  BT-removal pass is fully done. It exists to thread `motion_controller` through
  every `Tactic.tick()` call; worth checking whether that indirection earns its
  keep once `AbstractStrategy` itself is simpler.

## Developer documentation

Beyond `tactic_model_design_decisions.md` (internal decision log, not onboarding
material), need real docs aimed at a new contributor: how kernel/`Strategy`/`Tactic`/
`Partitioner` fit together, how to author a new `Tactic` end to end, testing
conventions (headless rsim for most things, grsim for anything dribble-related since
rsim has known dribble simulation bugs).

## CI

No CI exists yet. At minimum: lint/format check (black, matching local pre-commit
hooks), and a headless test run (`pixi run pytest --headless`, matching local
convention — never run simulator/integration tests without `--headless`). Needs a
decision on scope — which test directories are stable enough to gate merges on,
given known rsim dribble flakiness — and probably GitHub Actions given the org
(`First-Order-RoboCup-SSL`) already lives on GitHub.

## Agentic coding infra

As the tactic catalog and contributor base potentially includes coding agents (not
just humans), worth deliberately investing in:

- **`AGENTS.md`** (agent-agnostic, not Claude-specific) — durable context a coding
  agent needs before touching this repo: the kernel/Tactic/Partitioner model, the
  single-writer-partition invariant, `--headless` requirement, minimalism discipline
  (add concepts only after a concrete forcing case), where design rationale lives
  (`docs/tactic_model_design_decisions.md`).
- **CI/testing infra shaped for agent iteration loops**, not just human PR gating —
  e.g. fast feedback on whether a newly authored `Tactic` is well-formed
  (`tag` declared, `applicable()`/`is_committed()` behave sanely) before a full
  rsim/grsim run.
- **grsim/rsim feedback surfaced back to an LLM in a usable form** — today simulator
  results are numbers/logs/plots meant for a human to read; if agents are going to
  author and iterate on tactics, they need some translation layer (match summaries,
  failure characterizations, maybe rendered trajectory snapshots) that's actually
  legible to an LLM, not just a human staring at a viewer.

This is explicitly exploratory — no design decisions made yet, just the shape of the
problem worth thinking about before committing to a mechanism.
