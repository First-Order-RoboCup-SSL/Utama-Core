# A mathematical model of tactic scheduling

Companion to [`tactic_model_design_decisions.md`](tactic_model_design_decisions.md), specifically
§15–§16 (eligibility filtering vs. allocation, and why allocation stays hand-written and
config-local). This document gives the general mathematical object our `Partitioner`s are one
point in, shows where our design and Sumatra's sit in that space, and proposes — without building
yet — how the allocation stage's design space could be opened up later. Nothing here proposes
reopening §15's boolean-not-scored decision for *eligibility filtering* (`applicable()`); the
subject here is *allocation*, which §16 already scoped as disposable, config-local logic free to
vary per `Strategy`.

## 1. The general object

At each tick, state $s \in S$ (a `Game` snapshot) is given. A **partition** $\pi$ assigns each free
robot to at most one tactic slot: $\pi : F \to \mathcal{T}$, where $F \subseteq$ outfield robots is
the free pool and $\mathcal{T}$ is the set of currently-applicable tactics. Write $\Pi(s)$ for the
set of partitions that are *feasible* at $s$ — respecting each tactic's arity, `applicable()`, and
which slots are pinned by `is_committed()`.

A **scheduling policy** is a map $s \mapsto \pi \in \Pi(s)$. Every mechanism this document
discusses — ours, Sumatra's, and the proposed extension — is a different way of constructing this
map. They differ only in *how* $\pi$ is chosen from $\Pi(s)$, not in the underlying object.

Trajectory quality is not just the sum of per-tick choices; reassigning a robot away from an
in-progress tactic action has a cost. So the real objective, for any of these mechanisms, is over a
whole trajectory:

$$\max_{\pi_1,\dots,\pi_T} \sum_t V(s_t, \pi_t) \;-\; \lambda \cdot \text{switch}(\pi_{t-1}, \pi_t)$$

where $V : S \times \Pi \to \mathbb{R}$ scores a candidate partition and $\text{switch}$ penalizes
reassigning robots away from their previous tactic. $\lambda$ is a switching-cost weight — it need
not be a single global constant; see §5.

## 2. Our current design as a point in this space

A hand-written `Partitioner` is a function $p : S \to \Pi(s)$ — it does not score candidates, it
*emits* one directly. Expressed in the $V$-framing above, this is the degenerate case where $V$ has
collapsed to an indicator over a singleton feasible set per state:

$$V(s, \pi) = \begin{cases} 1 & \pi = p(s) \\ -\infty & \text{otherwise} \end{cases}$$

More precisely, $p$ is **piecewise-constant**: the state space $S$ is partitioned by hand
(`_possession_split_picker`'s "ball closer to us" branch, etc.) into regions $R_1, \dots, R_k$, and
$p(s) = \pi_i$ for $s \in R_i$ — one hand-picked output per hand-drawn region. Both the *region
boundaries* and the *output per region* are asserted by a developer reading game replays, not fit
from data or compared against alternatives at runtime.

Switching cost is likewise degenerate, not absent: $\text{is\_committed()}$ makes $\lambda = \infty$
for a pinned slot's robots and $\lambda = 0$ for every free robot — bang-bang, per-slot, no
continuum. This is a genuine, deliberate design point (§3, §16), not an oversight: it is fully
auditable (read the if/else, trace one `s` through it), has zero hyperparameters to mistune before a
match, and — per §15's rejection of scored eligibility — avoids a self-reported-score's structural
unfalsifiability problem entirely, since a hand-written $p$ makes no claims about itself at all.

## 3. Sumatra (TIGERs Mannheim, `Athena`/`Metis`) as a point in this space

Traced directly from source (`RoleAssigner`, `ADesiredBotCalc`, `Metis.register()` order — not
secondhand): a fixed sequence of per-role calculators (`DesiredDefendersCalc`,
`DesiredOffendersCalc`, `DesiredSupportBotsCalc`, keeper, ball-placement, ...) each claim from a
shared, mutable `desiredBotMap` in a hardcoded order. Each calculator's internal claim logic is a
per-role $V_i(s, G)$; there is no comparison *across* $V_i$'s. This is **lexicographic
optimization**: maximize defense's satisfaction fully first, then offense on whatever free robots
remain, then support, etc. — not a weighted sum over a shared scalar.

$$\pi^{\text{Sumatra}}_t = \text{greedy-sequential-claim under a fixed total order } \tau_1 \succ \tau_2 \succ \dots$$

This is a different, non-degenerate point from ours: $V$ is real per-role, but the collapse across
roles is a hard priority order rather than either an indicator (ours) or a scalar blend (§4). Its
switching-cost analog — `RoleAssigner` only reassigns bots whose desired-set membership actually
changed — needs no tunable margin, because a deterministic lexicographic claim doesn't thrash the
way a close scalar comparison can.

**Known weakness, inherent to the mechanism, not an implementation gap:** a robot that would be a
9/10 defender and an 8/10 attacker always goes to defense, however close the call, because
criterion 1 strictly dominates criterion 2 with no tolerance for "close enough to be a toss-up."

## 4. Opening the design space: filter, then fitness vector

Both points above are legitimate, currently-shipping corners of the same space — this section is
not a claim that either is wrong. It's the mechanism for moving *within* the space later, if a
concrete `Partitioner` is ever found straining to express a trade-off a hand-written if/else can't
express cleanly. Two additive layers, both optional, neither touching `engine/Strategy`'s existing
contract (`Partitioner` stays `Callable[[Game, frozenset, Optional[dict], frozenset], dict]` —
`Strategy` cannot tell a hand-written, filtered, or fitness-driven `Partitioner` apart):

**Layer A — a feasibility/heuristic filter, ahead of allocation.** Distinct from §15's
`applicable()` (a *tactic's* self-declared precondition, already shipping) — this is a
*partition-level* filter: given a candidate $\pi$, cheaply reject it by hand-written heuristic
before it's ever scored (e.g. "never leave the keeper's third empty," "never assign a group smaller
than a tactic's minimum viable size even if arity technically allows it"). This shrinks $\Pi(s)$
using exactly the same kind of asserted, auditable domain knowledge a hand-written `Partitioner`
already encodes — it just runs as a reusable filter rather than being re-derived inside every
`Partitioner`'s own branches.

**Layer B — vector-valued fitness, with an explicit, swappable collapse.** A new optional `Tactic`
method, defaulted to indifferent so no existing tactic needs to change:

```python
FitnessVector = dict[str, float]  # criterion name -> score, e.g. {"ball_proximity": 0.8}

def fitness(self, game: Game, candidate_robots: frozenset[RobotId]) -> FitnessVector:
    """Self-reported, per-criterion scores for this candidate group, right now.
    Default: {} (indifferent on every criterion)."""
    return {}
```

Per-criterion, not per-tactic — this is the generalization of §3's observation: Sumatra's
lexicographic order is over *roles*, but the same collapse machinery applies equally well to any
named set of *criteria* a tactic can self-report against (ball proximity, formation risk, shot-line
coverage, ...). A vector is strictly more expressive than either existing point in this space: it
subsumes a scalar $V_\theta$ (weighted-sum collapse, §4a below) and it subsumes Sumatra's
lexicographic order (§4b below) as two different, swappable choices of collapse function $\Phi$
applied to the same underlying vector, rather than committing to one at the type level.

Three honest choices for $\Phi : \mathbb{R}^k \to \mathbb{R}$ (or to a total order):

- **(a) Weighted sum** $\Phi(v) = w \cdot v$ — simple, differentiable, but asserts an exchange rate
  between criteria that a hand-written if/else never had to state explicitly.
- **(b) Pure lexicographic** — exactly Sumatra's mechanism, generalized from roles to named
  criteria. No exchange rate assumed; inherits Sumatra's known weakness (§3) at criterion
  boundaries.
- **(c) Lexicographic-with-tolerance** (the recommended middle ground) — criterion $i$ dominates
  criterion $i{+}1$ only outside a tolerance band $\epsilon_i$; scores within $\epsilon_i$ of each
  other are treated as tied and the comparison falls through. $\epsilon \to 0$ recovers (b);
  widening every $\epsilon$ approaches (a). Fixes §3's dominance weakness without inventing a
  cross-criterion exchange rate — "these are close enough not to matter" is an easier, more
  defensible number for a developer to pick than an exchange rate between unlike quantities.

```python
@dataclass
class LexicographicPolicy:
    criteria_order: tuple[str, ...]
    tolerances: dict[str, float]

    def compare(self, a: FitnessVector, b: FitnessVector) -> int:
        for c in self.criteria_order:
            va, vb = a.get(c, 0.0), b.get(c, 0.0)
            if abs(va - vb) > self.tolerances.get(c, 0.0):
                return -1 if va < vb else 1
        return 0
```

`fitness_partitioner(policy)` is one more `Partitioner` implementation living in `strategy/`,
selected by whichever `build_*_kernel_strategy` factory opts in — the existing 14 factories are
unaffected. A `Partitioner` may also be a **hybrid**: hand-code the slice of the roster where
auditability matters most (e.g. defense), and delegate only the remaining free robots to
`fitness_partitioner` where more tactics genuinely compete for the same marginal robot. This is
composition of plain functions, not a global mode switch — `_choose_partition`'s existing
validation (partition must exactly cover `free_robots`, must respect `applicable_tactic_ids`)
already catches bugs in any of hand-written, filtered, fitness-driven, or hybrid `Partitioner`s
identically, with no new validation path needed.

## 5. What this does not solve, and what it deliberately leaves out

- **Still an approximation.** Vector fitness with an explicit collapse does not "capture the true
  V" — no finite representation does. What it buys is a *cheaper axis of revision*: changing
  `criteria_order`/`tolerances` (or which `Partitioner` a factory uses) needs no tactic code
  changes, versus rewriting if/else branches for a hand-written picker.
- **§15's unfalsifiability objection still applies, in reduced form.** A self-reported *scalar*
  score from an agent-authored tactic has every incentive to report high confidence, with no local,
  falsifiable claim to check it against. A *named, per-criterion* vector is more falsifiable than a
  scalar (each criterion is individually unit-testable against a concrete game state, the way
  `applicable()` already is) but is not immune — a tactic can still misreport a single named
  criterion. This is a real, open cost of Layer B that Layer A (a heuristic filter with no
  self-reporting at all) does not share, and is why Layer B should stay optional and
  config-scoped, not become the default allocation path.
- **No learning loop is proposed here.** Fitting `tolerances`, `criteria_order`, or $w$ from
  `match_log` traces is a legitimate future direction but a separate, larger project (reward
  design, offline data, safety guarantees under a fitted policy) — not a consequence of adding the
  `fitness()` protocol method itself.
- **Per the standing minimalism rule ([[feedback_minimal_architecture]]) and §16's precedent**,
  neither layer should be built until a concrete `Partitioner` is found straining to express a
  real trade-off — this document fixes the design so that moment doesn't require re-deriving it,
  not a signal to start building now.

## 6. Opening `switch`, and why it should stay bang-bang for now

§2 noted switch cost is also degenerate today: `is_committed()` gives $\lambda \in \{0, \infty\}$,
per-slot, no continuum. The natural-looking generalization — a tactic-declared, per-robot
`interrupt_cost(game, mem, robot_id) -> float` replacing the boolean, combined with fitness in the
same objective ($V - \text{switch}$, never merged into one vector; see below for why not) — was
considered and is **deliberately not recommended**, for a reason specific to switch cost rather
than a generic "not yet" deferral.

**Why switch cost is a harder hand-design problem than $V$, not an equally hard one.** A wrong
fitness score has a short, legible feedback loop: watch a match, see a robot go somewhere dumb, fix
the heuristic. A wrong switch cost does not: too low produces *flapping* (a robot oscillating
between tactics tick to tick — individually-reasonable-looking assignments that are only visible as
a bug in aggregate, e.g. in a `match_log` trace, not from watching live play) and too high produces
*stuck* assignments (a tactic holding robots long after it should have released them) — a failure
that produces no visible anomaly at all, just quietly worse play. The two failure directions need
opposite fixes, and neither is easy to notice, which is why `_choose_partition`'s existing
`committed_ticks % 100` warning exists — "stuck" is exactly the kind of bug that doesn't announce
itself. Asking every tactic author to hand-pick a per-robot cost on a scale with no natural units,
against failure modes that are this hard to observe, is a worse ask than anything in §4.

**Why $V$ and $\text{switch}$ combine at the objective level, not by merging into one vector.**
`fitness(τ, G, s)` is memoryless — a property of the current candidate only. `interrupt_cost`
necessarily depends on `prev_partition`/`mem` — a property of the transition. Folding both into one
`FitnessVector` would force a single collapse policy (§4's `Φ`) to reason about "is this good" and
"is this sticky" through the same tolerance machinery, conflating two quantities with different
natural units. They stay two terms, $V(s,\pi) - \lambda\cdot\text{switch}(\pi_{\text{prev}},\pi)$,
exactly as §1 already had it — only the grading of each term is up for revision, not their
separation.

**What to reach for instead, if flapping is ever concretely observed:** a single **global
minimum-dwell-time**, enforced by `Strategy` around any `Partitioner` — once a slot's tactic
changes, no further change to that slot for $k$ ticks. One number, one place, same unit
`committed_ticks` already uses, `k=0` reproduces today's behaviour exactly. It doesn't capture
"some interruptions are worse than others," but it directly targets the one concrete failure mode
without asking any tactic author to invent a currency for interruption cost. Reach for a richer,
per-tactic mechanism only if a dwell-time floor concretely fails to be enough — same
forcing-case discipline as §4.

## 7. The far end of the space: learned, end-to-end $Q_\theta(s_{t-1}, \pi_{t-1}, \pi_t)$

§6 argued $V$ and $\text{switch}$ should stay separate, hand-authored terms — but that argument is
about *human calibration cost specifically*, not about the math. It stops applying the moment a
human is no longer the one picking the numbers. Once both terms are being fit end-to-end by
gradient descent against one differentiable objective, there is no calibration-incommensurability
problem to avoid — the optimizer can freely learn its own coupling between "this partition is good"
and "this transition is costly," including interaction terms (a switch can be cheap *because* the
resulting formation is good) that two separately-authored terms can't represent at all. At that
point, merging into a single network is the more natural design, not merely a permissible one: a
learned $Q_\theta(s_t, \pi_{t-1}, \pi_t)$ — a standard action-value function over (state,
previous-partition, candidate-partition) triples — trained via TD-learning/actor-critic against a
return signal, played by enumerating feasible $\pi_t \in \Pi(s_t)$ (small and structured — arity
constraints already bound it, exactly as in §4's Layer A) and taking $\arg\max_\pi Q_\theta$.

**§4's vector `fitness()` is a hand-designed, interpretable approximation to this $Q_\theta$, not a
competing proposal** — one is a special case of the other, differing only in whether the scoring
function is authored or fit. This section is the far end of the same axis §4 opened, not a fork in
it.

**What actually adopting this would require, stated plainly so the gap is visible:**

1. **Reward design doesn't disappear, it relocates.** Goals are sparse and slow; TD-learning from
   raw goal reward alone is a poor training signal. Shaped reward (possession, shot quality,
   danger denied) would almost certainly be needed — which is hand-design again, just moved from
   "switch cost" into "reward function." This is the same point raised earlier against "less
   inductive bias, trust search": removing structure from the policy does not remove it from the
   system, it moves it somewhere less visible.
2. **Sample source becomes the bottleneck, not architecture.** RL needs far more rollouts than a
   hand-tuned `Partitioner` needs test cases. grsim/rsim throughput, not model design, would gate
   feasibility — and [[project_rsim_dribble_issues]] matters directly here: training against a
   simulator with known dribble-physics bugs risks the policy learning to exploit the bug rather
   than the game.
3. **The action space stays enumerable, which is the one piece of good news.** $\pi_t$ ranges over
   feasible partitions of a handful of robots across a bounded tactic set — small enough to
   enumerate rather than requiring a structured-output policy network. Only $\Phi$ (§4) is replaced
   by $Q_\theta$; the surrounding machinery (feasibility filtering, enumeration, the `Partitioner`
   interface itself) is unchanged.
4. **Safety cannot be delegated to the network.** `is_committed()`'s hard veto is a
   physical-correctness constraint, not a preference a reward function should be trusted to
   rediscover. It must remain a hard mask applied to $\Pi(s_t)$ before $\arg\max_\pi Q_\theta$ ever
   runs — an RL policy that hasn't sampled a rare unsafe transition during training has no
   guarantee of avoiding it at inference time.

**Scope of this section:** conceptual, not a proposal to build. It is a materially larger
undertaking than §4/§6 — different data requirements, no equivalent of "read the if/else" for
validating a trained $Q_\theta$ before trusting it in a real match, and real ownership cost for a
training pipeline. Per the same forcing-case discipline as the rest of this document: worth having
mapped now, precisely so it doesn't need re-deriving later, but only worth starting if §4/§6's
cheaper, auditable moves are actually tried and found to plateau.
