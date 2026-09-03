"""Two small, pure primitives factoring out a hand-rolled pattern that had
reached six independent instances across the tactic/skill layer (see
`docs/roadmap.md`'s "Open" item 1 for the inventory that triggered this).

Both bugs below are boundary-condition bugs on otherwise-correct logic, not
missing features — every instance already *knew* it needed a tolerance or a
margin, and hand-rolled one. The point of this module is not "add slop
nobody asked for"; it's "stop re-deriving the same two primitives with
slightly different bugs each time."

- `approx_eq` — exact `==`/`!=` on a float or 2D point is wrong whenever
  either side comes from live sensor/sim state, because simulator physics
  (and real hardware) never reproduces a value bit-for-bit tick to tick.
  Concretely: `trajsampling/planner.py`'s `_try_reuse` compared the ball's
  raw target position by exact tuple equality, so sub-millimetre rsim
  jitter on a DIRECT_FREE restart's stationary ball was read as "the target
  changed," forcing a full replan every tick forever (fixed ad hoc via
  `_TRAJECTORY_TARGET_TOLERANCE`, commit `5183ed1`).
- `Sticky` — "keep the previous choice unless a new candidate beats it by a
  margin" is the fix for a *different* bug: a bare `min(candidates,
  key=score)`/`<` comparison recomputed fresh every tick lets simulator
  noise alone flip which candidate currently "wins" a near-tie, producing a
  visible tick-to-tick flicker in whatever the choice drives (a target
  point, a role assignment). `clear_ball.py`'s `_best_clear_target` and
  `pass_and_shoot.py`'s `assign_passer_receiver` are the two clearest real
  instances of this exact shape (score a fixed candidate set against the
  current game state, stick with the incumbent unless a challenger clears a
  margin) and are what `Sticky.update()`'s signature is designed to fit,
  not a hypothetical generalization of them.

Deliberately NOT covered here (see roadmap item 1's classification):
- Tick-count timeouts (`_PHASE_TIMEOUT_TICKS` and friends) — a different
  shape (a counter vs. a threshold), and a kernel-level commitment deadline
  is arriving separately; per-tactic timeouts are fine to leave hand-rolled.
- Boolean edge-trigger hysteresis with separate commit/release bands
  (`shielding.py`'s `COMMIT_RANGE`/`_RELEASE_RANGE`,
  `press_and_contain.py`'s `_LOOSE_BALL_HYSTERESIS_TICKS`) — these gate a
  binary mode switch on crossing one of two different thresholds depending
  on current mode, not "does a challenger beat the incumbent by a margin
  measured the same way for both." Forcing them onto `Sticky` would need a
  second scoring function per mode, which is the "config framework nobody
  asked for" this module is explicitly trying to avoid.
- `find_best_shot`'s gap-membership hysteresis (`_pass_and_score.py`'s
  `_SHOT_SWITCH_MARGIN`) — the "candidate" there is a goal-line interval,
  not a point, and "does the previous choice still exist" is a containment
  check, not a score comparison; genuinely a different shape.
- Bare `min(items, key=distance)` picks with no margin at all today
  (`block_shape.py`'s presser, `press_and_contain.py`'s presser,
  `give_and_go.py`'s initial carrier, `shadow_and_mark.py`'s/
  `press_and_contain.py`'s greedy mark assignment, `decoy_and_overload.py`'s
  nearest marker, `defense.py`'s/`shadow_and_mark.py`'s loose-ball
  retriever). Wrapping these in `Sticky` with an invented margin would
  silently change strategy behaviour (and therefore tournament results,
  `docs/strategies.md`) rather than just fix a boundary bug — out of scope
  here, listed as follow-up candidates in the roadmap entry instead.
"""

from __future__ import annotations

from typing import Callable, Generic, Optional, Sequence, Tuple, TypeVar, Union

Point = Union["_HasXY", Tuple[float, float]]


class _HasXY:
    """Structural stand-in for `Vector2D` — anything with `.x`/`.y` floats."""

    x: float
    y: float


def _as_xy(p: Point) -> Tuple[float, float]:
    if hasattr(p, "x") and hasattr(p, "y"):
        return float(p.x), float(p.y)
    x, y = p
    return float(x), float(y)


def approx_eq(a: Union[float, Point], b: Union[float, Point], tol: float) -> bool:
    """True if `a` and `b` are within `tol` of each other.

    Accepts either two scalars (compared by absolute difference) or two 2D
    points (`Vector2D` or a plain `(x, y)` tuple, compared by Euclidean
    distance) — dispatch is by shape, not a separate function per type,
    since every real call site already knows which kind of value it has and
    a `points=True` flag would just be ceremony around that.

    The bug this prevents: comparing a live sensor/sim-derived float or
    point with `==` treats sub-millimetre/sub-epsilon jitter as a genuine
    change. `trajsampling/planner.py`'s `_try_reuse` did exactly this on a
    stationary restart ball's target position, forcing a full replan every
    single tick forever even though the ball never actually moved
    (`5183ed1`). `tol` should be picked the same way that fix's
    `_TRAJECTORY_TARGET_TOLERANCE` was: comfortably above the known noise
    floor, comfortably below the smallest change that should actually count
    as "different" (e.g. well under `ROBOT_RADIUS` for a position).

    `tol` must be given explicitly — there is no default, because the right
    tolerance is a property of the noise floor and the smallest meaningful
    change at each call site, not a codebase-wide constant.
    """
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol
    ax, ay = _as_xy(a)  # type: ignore[arg-type]
    bx, by = _as_xy(b)  # type: ignore[arg-type]
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 <= tol


T = TypeVar("T")


class Sticky(Generic[T]):
    """Holds a current choice; `update()` only switches it when a fresh
    candidate beats it by more than `margin`, on a score computed by the
    same `score_fn` for both.

    The bug this prevents: re-deriving the "best" candidate from scratch
    every tick with a bare `max(candidates, key=score_fn)`/`<` comparison
    means ordinary simulator noise on the *inputs* to `score_fn` (enemy
    positions, distances) can flip which candidate currently scores highest
    between two near-tied options — even though nothing about the actual
    tactical picture changed. The visible symptom is whatever the choice
    drives (a target point, a role assignment) flickering tick to tick
    instead of holding steady. `clear_ball.py`'s `_best_clear_target` and
    `pass_and_shoot.py`'s `assign_passer_receiver` are both exactly this
    shape today (hand-rolled, not using this class) — a fixed candidate set
    scored fresh each tick, sticking with the incumbent unless a challenger
    clears a margin.

    Not a general "state machine" or "commitment" primitive — it holds
    exactly one thing (the current choice) and does exactly one thing
    (compare a score). It has no notion of phases, timeouts, or committing
    a choice permanently; callers that need those compose them on top (a
    tactic's own `mem`/`is_committed()`), same as today.

    Usage: keep a `Sticky[...]` instance (or just the raw choice) in a
    tactic's `mem` dataclass across ticks, and call `.update(candidates,
    score_fn)` each tick with this tick's freshly-computed candidate list.
    """

    def __init__(self, margin: float, current: Optional[T] = None):
        """`margin`: how much a challenger's score must exceed the current
        choice's score by before it wins. `current`: an optional pre-seeded
        choice (e.g. restored from `mem`); `None` means no choice yet."""
        self.margin = margin
        self.current = current

    def update(
        self,
        candidates: Sequence[T],
        score_fn: Callable[[T], float],
        key_fn: Optional[Callable[[T], object]] = None,
    ) -> Optional[T]:
        """Re-score `candidates` and return the (possibly unchanged)
        winner, updating `self.current` to match.

        Algorithm (deliberately a sequential fold, not "compute the global
        max, then compare it to the incumbent" — the two are *not*
        equivalent when more than one candidate can each individually clear
        the margin over the *previous* running best): start from the
        current choice's own freshly-computed score (if any candidate
        matches it — see below), then walk `candidates` in order, replacing
        the running best each time a candidate clears the running best's
        score by more than `margin`. This is exactly `clear_ball.py`'s
        `_best_clear_target` inlined, generalized past its specific
        `segment_clearance` scorer — confirmed to agree with it on a few
        hundred seeded-random candidate sets, see
        `test_sticky_matches_hand_rolled_margin_logic_seeded_random`.

        `key_fn`, if given, identifies "the same candidate" across ticks
        (e.g. a robot id or an assignment tuple) when `T` itself isn't
        directly comparable/hashable or when object identity wouldn't
        survive being recomputed fresh each tick (a new `Vector2D` with the
        same coordinates, a new tuple with the same ids). Without it,
        candidates are matched by `==`.

        If `candidates` is empty, `self.current` is returned unchanged (and
        may be `None`) — there is nothing to switch to, and clearing a
        held choice just because this tick had nothing to offer would
        itself be a source of flicker, not a fix for one.

        If there is no current choice, or it is no longer present among
        `candidates` (its underlying option disappeared — e.g. a
        previously-chosen enemy no longer exists), there is no stale score
        left to gate a challenger against, so the plain highest-scoring
        candidate wins outright — no margin applied. This matches
        `_best_clear_target`'s own `prev_target is None` branch
        (`max(candidates, key=_score)`, unconditionally).
        """
        if not candidates:
            return self.current

        def _k(item: T) -> object:
            return key_fn(item) if key_fn is not None else item

        still_present = None
        if self.current is not None:
            current_key = _k(self.current)
            still_present = next((c for c in candidates if _k(c) == current_key), None)

        if still_present is None:
            self.current = max(candidates, key=score_fn)
            return self.current

        best_point, best_score = still_present, score_fn(still_present)
        for candidate in candidates:
            if _k(candidate) == _k(best_point):
                continue
            score = score_fn(candidate)
            if score > best_score + self.margin:
                best_point, best_score = candidate, score

        self.current = best_point
        return self.current
