"""Boundary-condition tests for `utama_core.shared.tolerance`.

Every test here pins an exact boundary (at tolerance, just under/over it, a
tie, an empty list, a vanished previous choice) rather than exercising happy
paths only — see AGENTS.md's testing guidance on what a boundary-condition
fix needs from its regression test.
"""

import random

import pytest

from utama_core.entities.data.vector import Vector2D
from utama_core.shared.tolerance import Sticky, approx_eq

# ---------------------------------------------------------------------------
# approx_eq — scalars
# ---------------------------------------------------------------------------


def test_approx_eq_scalar_exactly_at_tolerance_is_equal():
    # 1.0 -> 1.25 is exactly 0.25 in float64 (no representation error at
    # this boundary), unlike e.g. 1.0/1.01 which is 0.010000000000000009.
    assert approx_eq(1.0, 1.25, tol=0.25)


def test_approx_eq_scalar_just_over_tolerance_is_not_equal():
    assert not approx_eq(1.0, 1.2500001, tol=0.25)


def test_approx_eq_scalar_jitter_well_below_tolerance_is_equal():
    assert approx_eq(1.0, 1.0 + 1e-9, tol=0.01)


def test_approx_eq_scalar_identical_values():
    assert approx_eq(3.5, 3.5, tol=0.0)


def test_approx_eq_scalar_negative_difference():
    assert approx_eq(-1.0, -1.005, tol=0.01)
    assert not approx_eq(-1.0, -1.02, tol=0.01)


# ---------------------------------------------------------------------------
# approx_eq — 2D points (Vector2D and tuple)
# ---------------------------------------------------------------------------


def test_approx_eq_point_vector2d_exactly_at_tolerance():
    a = Vector2D(0.0, 0.0)
    b = Vector2D(0.01, 0.0)
    assert approx_eq(a, b, tol=0.01)


def test_approx_eq_point_vector2d_just_over_tolerance():
    a = Vector2D(0.0, 0.0)
    b = Vector2D(0.0100001, 0.0)
    assert not approx_eq(a, b, tol=0.01)


def test_approx_eq_point_submillimetre_jitter_within_tolerance():
    """Mirrors the trajsample DIRECT_FREE restart bug (5183ed1): a
    stationary ball's raw position jitters by sub-millimetres tick to tick;
    a real tolerance must absorb that."""
    a = Vector2D(1.234567, -0.987654)
    b = Vector2D(1.234567 + 0.0000003, -0.987654 - 0.0000002)
    assert approx_eq(a, b, tol=0.001)


def test_approx_eq_point_tuple_input():
    assert approx_eq((0.0, 0.0), (0.005, 0.0), tol=0.01)
    assert not approx_eq((0.0, 0.0), (0.05, 0.0), tol=0.01)


def test_approx_eq_point_mixed_vector2d_and_tuple():
    assert approx_eq(Vector2D(1.0, 1.0), (1.005, 1.0), tol=0.01)


def test_approx_eq_point_diagonal_distance():
    # 3-4-5 triangle scaled down: distance is exactly 0.05 at these deltas.
    a = Vector2D(0.0, 0.0)
    b = Vector2D(0.03, 0.04)
    assert approx_eq(a, b, tol=0.05)
    assert not approx_eq(a, b, tol=0.0499)


# ---------------------------------------------------------------------------
# Sticky — basic switching behaviour
# ---------------------------------------------------------------------------


def test_sticky_no_current_choice_picks_best_candidate():
    sticky = Sticky[str](margin=0.1)
    result = sticky.update(["a", "b", "c"], score_fn={"a": 1.0, "b": 3.0, "c": 2.0}.get)
    assert result == "b"
    assert sticky.current == "b"


def test_sticky_empty_candidates_returns_current_unchanged():
    sticky = Sticky[str](margin=0.1, current="a")
    result = sticky.update([], score_fn=lambda x: 0.0)
    assert result == "a"
    assert sticky.current == "a"


def test_sticky_empty_candidates_with_no_current_returns_none():
    sticky = Sticky[str](margin=0.1)
    result = sticky.update([], score_fn=lambda x: 0.0)
    assert result is None


def test_sticky_challenger_beating_by_exactly_the_margin_does_not_switch():
    """A challenger must *clearly* beat the incumbent -- beating it by
    exactly `margin` is not enough (matches clear_ball._best_clear_target's
    `>` -- not `>=` -- comparison)."""
    scores = {"incumbent": 1.0, "challenger": 1.5}
    sticky = Sticky[str](margin=0.5, current="incumbent")
    result = sticky.update(["incumbent", "challenger"], score_fn=scores.get)
    assert result == "incumbent"


def test_sticky_challenger_beating_by_more_than_margin_switches():
    scores = {"incumbent": 1.0, "challenger": 1.50001}
    sticky = Sticky[str](margin=0.5, current="incumbent")
    result = sticky.update(["incumbent", "challenger"], score_fn=scores.get)
    assert result == "challenger"


def test_sticky_challenger_beating_by_less_than_margin_does_not_switch():
    scores = {"incumbent": 1.0, "challenger": 1.2}
    sticky = Sticky[str](margin=0.5, current="incumbent")
    result = sticky.update(["incumbent", "challenger"], score_fn=scores.get)
    assert result == "incumbent"


def test_sticky_exact_tie_does_not_switch():
    scores = {"incumbent": 1.0, "challenger": 1.0}
    sticky = Sticky[str](margin=0.1, current="incumbent")
    result = sticky.update(["incumbent", "challenger"], score_fn=scores.get)
    assert result == "incumbent"


def test_sticky_incumbent_still_scores_itself_fresh_each_tick():
    """The incumbent's own score is re-read from this tick's candidate list
    (not cached from a previous tick), matching `_best_clear_target`'s
    "score everything fresh, then compare" behaviour."""
    sticky = Sticky[str](margin=0.5, current="a")
    scores_tick1 = {"a": 1.0, "b": 1.0}
    sticky.update(["a", "b"], score_fn=scores_tick1.get)
    assert sticky.current == "a"

    # Next tick: "a"'s own underlying score dropped (e.g. a lane closed up),
    # but "b" only ties the new value -- still shouldn't switch without
    # clearing the margin.
    scores_tick2 = {"a": 0.2, "b": 0.2}
    result = sticky.update(["a", "b"], score_fn=scores_tick2.get)
    assert result == "a"


# ---------------------------------------------------------------------------
# Sticky — previous choice no longer in the candidate list
# ---------------------------------------------------------------------------


def test_sticky_previous_choice_vanished_picks_best_fresh_candidate():
    """Mirrors `_best_clear_target`'s "the gap closed entirely -- a real
    change, not jitter" case: if the incumbent isn't even offered this
    tick, the margin has nothing to compare against, so the best available
    candidate wins outright."""
    scores = {"b": 1.0, "c": 1.05}
    sticky = Sticky[str](margin=0.5, current="a")
    result = sticky.update(["b", "c"], score_fn=scores.get)
    assert result == "c"
    assert sticky.current == "c"


def test_sticky_key_fn_identifies_recomputed_equal_candidate():
    """Vector2D/tuple candidates are usually freshly constructed each tick
    (new object, same coordinates) -- `key_fn` must be able to match those
    as "the same candidate" so equality isn't defeated by object identity
    or floating point reconstruction."""
    incumbent = Vector2D(1.0, 2.0)
    sticky = Sticky[Vector2D](margin=0.5, current=incumbent)

    fresh_same_point = Vector2D(1.0, 2.0)  # different object, same coords
    other_point = Vector2D(5.0, 5.0)
    scores = {id(fresh_same_point): 1.0, id(other_point): 1.2}

    result = sticky.update(
        [fresh_same_point, other_point],
        score_fn=lambda p: scores[id(p)],
        key_fn=lambda p: (round(p.x, 6), round(p.y, 6)),
    )
    # other_point's 1.2 doesn't clear fresh_same_point's 1.0 by the 0.5
    # margin, so the incumbent (matched via key_fn) should hold.
    assert result.x == pytest.approx(1.0)
    assert result.y == pytest.approx(2.0)


def test_sticky_default_key_uses_equality_when_no_key_fn_given():
    sticky = Sticky[tuple](margin=1.0, current=(1, 2))
    scores = {(1, 2): 1.0, (3, 4): 1.4}
    result = sticky.update([(1, 2), (3, 4)], score_fn=scores.get)
    assert result == (1, 2)


# ---------------------------------------------------------------------------
# Equivalence test: Sticky vs. clear_ball._best_clear_target's hand-rolled
# hysteresis on seeded random inputs, per the task's requirement that a
# migrated (C)-class site prove behavioural equivalence at the same margin.
# ---------------------------------------------------------------------------


def _hand_rolled_best(candidates, prev, score_fn, margin):
    """Re-implementation of clear_ball._best_clear_target's selection logic
    in scoring-agnostic form, for a direct behavioural comparison against
    Sticky.update() -- see test_sticky_matches_hand_rolled_margin_logic."""
    if prev is not None:
        prev_score = score_fn(prev)
        best_point, best_score = prev, prev_score
        for c in candidates:
            if score_fn(c) > best_score + margin:
                best_point, best_score = c, score_fn(c)
        return best_point
    return max(candidates, key=score_fn)


def test_sticky_matches_hand_rolled_margin_logic_seeded_random():
    """`clear_ball._best_clear_target` always re-scores its previous target
    even when it's not literally re-offered as a fresh candidate this tick
    (the ball moved, so the old landing point is scored against the new
    ball position). Sticky's `update()` needs the previous choice present
    in `candidates` to reproduce that -- this test drives both
    implementations with the previous choice re-included, at a few hundred
    seeded random candidate sets, and asserts identical output.
    """
    rng = random.Random(12345)
    margin = 0.15
    mismatches = 0
    for _ in range(300):
        num_candidates = rng.randint(1, 5)
        candidates = [Vector2D(rng.uniform(-5, 5), rng.uniform(-3, 3)) for _ in range(num_candidates)]
        prev = Vector2D(rng.uniform(-5, 5), rng.uniform(-3, 3)) if rng.random() < 0.8 else None

        # A score function keyed by object identity (as the real game-state
        # dependent score would be, recomputed per call).
        score_lookup = {id(c): rng.uniform(0, 1) for c in candidates}
        if prev is not None:
            score_lookup[id(prev)] = rng.uniform(0, 1)

        def score_fn(p, _lookup=score_lookup):
            return _lookup[id(p)]

        hand_rolled = _hand_rolled_best(candidates, prev, score_fn, margin)

        sticky = Sticky[Vector2D](margin=margin, current=prev)
        offered = list(candidates)
        if prev is not None and prev not in offered:
            offered = offered + [prev]
        sticky_result = sticky.update(offered, score_fn=score_fn)

        hr_x, hr_y = (hand_rolled.x, hand_rolled.y) if hand_rolled is not None else (None, None)
        sr_x, sr_y = (sticky_result.x, sticky_result.y) if sticky_result is not None else (None, None)
        if (hr_x, hr_y) != (sr_x, sr_y):
            mismatches += 1

    assert mismatches == 0


def _hand_rolled_passer(distances, robot_ids, prev_assignment, margin):
    """Re-implementation of pass_and_shoot.assign_passer_receiver's
    selection logic (distances/margin only, no Game/fallback plumbing), for
    a direct behavioural comparison against Sticky.update() -- see
    test_sticky_matches_pass_and_shoot_margin_logic."""
    if prev_assignment is not None and prev_assignment[0] in distances and prev_assignment[1] in distances:
        prev_passer, prev_receiver = prev_assignment
        if distances[prev_receiver] + margin >= distances[prev_passer]:
            return prev_passer
    return min(distances, key=distances.get)


def test_sticky_matches_pass_and_shoot_margin_logic_seeded_random():
    """`pass_and_shoot.assign_passer_receiver` is the other real (C)-class
    hysteresis site migrated onto `Sticky` (score = negative distance, so
    "closer" maximizes the score). Drives both the original min-distance-
    with-margin logic and `Sticky.update()` at 500 seeded random distance
    pairs and assignments, asserting identical passer choice.
    """
    rng = random.Random(2718)
    margin = 0.3
    ids = (7, 12)
    mismatches = 0
    for _ in range(500):
        distances = {7: rng.uniform(0, 5), 12: rng.uniform(0, 5)}
        prev_assignment = rng.choice([(7, 12), (12, 7), None])

        hand_rolled = _hand_rolled_passer(distances, ids, prev_assignment, margin)

        prev_passer = prev_assignment[0] if prev_assignment is not None else None
        sticky = Sticky[int](margin=margin, current=prev_passer)
        sticky_result = sticky.update(list(ids), score_fn=lambda rid: -distances[rid])

        if hand_rolled != sticky_result:
            mismatches += 1

    assert mismatches == 0
