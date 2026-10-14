# A mathematical model of tactic scheduling

Companion to [`tactic_model_design_decisions.md`](tactic_model_design_decisions.md) §15-§16. It
places our `Partitioner`s and Sumatra's role assignment in one design space, and records — without
building — how the *allocation* stage could be opened up later. Nothing here reopens §15's boolean
eligibility filter (`applicable()`).

## 1. The general object

State $s \in S$ is a `Game` snapshot. A **partition** $\pi : F \to \mathcal{T}$ assigns each free
outfield robot to at most one applicable tactic slot; $\Pi(s)$ is the feasible set (arity,
`applicable()`, slots pinned by `is_committed()`). A scheduling policy is a map $s \mapsto \pi \in
\Pi(s)$. Since reassigning a robot mid-action has a cost, the real objective is over a trajectory:

$$\max_{\pi_1,\dots,\pi_T} \sum_t V(s_t, \pi_t) \;-\; \lambda \cdot \text{switch}(\pi_{t-1}, \pi_t)$$

## 2. Our design

A hand-written `Partitioner` emits $\pi = p(s)$ directly: $V$ collapses to an indicator on one
candidate, and $p$ is piecewise-constant over hand-drawn regions of $S$ (e.g. "ball closer to us").
Switching cost is bang-bang: `is_committed()` gives $\lambda = \infty$ for pinned robots, $0$ for
free ones. Deliberate: fully auditable, zero hyperparameters, and no self-reported score to
distrust.

## 3. Sumatra (TIGERs, `Athena`/`Metis`)

Traced from source (`RoleAssigner`, `ADesiredBotCalc`, `Metis.register()` order): per-role
calculators claim robots from a shared `desiredBotMap` in a fixed order — **lexicographic
optimization** (defense fully first, then offense on the remainder, ...). Needs no switching
margin because a deterministic claim doesn't thrash. Inherent weakness: a 9/10 defender who is an
8/10 attacker always goes to defense, however close the call.

## 4. Opening allocation later: filter, then fitness vector

Both optional, both behind the unchanged `Partitioner` interface (`Strategy` can't tell
implementations apart, and `_validate_partition` checks all of them the same way):

- **Layer A — partition-level heuristic filter** before allocation (e.g. "never leave the keeper's
  third empty"). Same kind of asserted domain knowledge a `Partitioner` already encodes, reusable.
- **Layer B — optional `Tactic.fitness(game, candidate_robots) -> dict[str, float]`** (default
  `{}`), per named criterion, with a swappable collapse $\Phi$: (a) weighted sum — asserts an
  exchange rate between unlike criteria; (b) pure lexicographic — Sumatra's mechanism,
  generalized to criteria; (c) **lexicographic with tolerance** (recommended) — criterion $i$
  dominates only outside a band $\epsilon_i$, otherwise falls through. $\epsilon\to0$ recovers
  (b); wide $\epsilon$ approaches (a). "Close enough not to matter" is easier to pick than an
  exchange rate.
- A `Partitioner` may be hybrid: hand-code where auditability matters (defense), delegate the
  remaining free robots to a fitness partitioner.

Limits: still an approximation of $V$; a named per-criterion score is more falsifiable than a
scalar but a tactic can still misreport one criterion, so Layer B stays optional and
config-scoped; no learning loop is implied. Build neither layer until a concrete `Partitioner`
strains to express a real trade-off.

## 5. Keep switching cost bang-bang

A per-robot `interrupt_cost()` was considered and rejected. A wrong switch cost fails silently in
both directions — too low gives flapping (visible only in aggregate traces), too high gives stuck
assignments (no visible anomaly, just worse play) — so it is a worse hand-design ask than $V$. $V$
(memoryless) and switch (depends on the previous partition) also stay separate terms rather than
one vector. If flapping is ever observed, reach first for a single global **minimum dwell time**
per slot ($k=0$ reproduces today).

## 6. The far end: learned $Q_\theta(s_t, \pi_{t-1}, \pi_t)$

Once both terms are fit end-to-end rather than hand-set, merging them into one action-value
function is natural; Layer B's fitness vector is a hand-designed approximation to it. Adopting it
would need: shaped reward (reward design relocates, it doesn't disappear); far more rollouts, with
simulator fidelity gating feasibility (a policy can learn to exploit rsim dribble bugs);
enumeration over the small feasible $\Pi(s_t)$ (the one easy part); and `is_committed()` kept as a
hard mask, never delegated to the network. Conceptual only — start only if §4/§5's auditable moves
plateau.
