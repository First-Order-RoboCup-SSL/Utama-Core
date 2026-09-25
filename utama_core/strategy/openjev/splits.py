"""OpenJev's action space: named splits of the outfield pool across a fixed tactic registry.

The model picks one split key per decision; `allocate()` turns it into a
`Partitioner`-shaped `{tactic_id: robots}` dict for whichever robots are free
this tick. Robots are assigned in ascending-id order, the same convention as
every other picker in `kernel_strategy.py` (`_allocate_ordered`), so a match
result reflects the *choice of split*, not a different robot-to-slot policy.
"""

from __future__ import annotations

from typing import Optional

from utama_core.engine.tactic import RobotId, Tactic
from utama_core.tactics.block_shape import BlockShapeTactic
from utama_core.tactics.clear_ball import ClearBallTactic
from utama_core.tactics.decoy_and_overload import DecoyOverloadTactic
from utama_core.tactics.give_and_go import GiveAndGoTactic
from utama_core.tactics.press_and_contain import PressAndContainTactic
from utama_core.tactics.shadow_and_mark import ShadowAndMarkTactic

# Tactic slot id -> one-line description (goes into the cached static state block).
TACTIC_GLOSSARY: dict[str, str] = {
    "attack": "give-and-go: 2+ attackers pass and move in repeated one-twos until a shot opens",
    "overload": "decoy-overload: one attacker drags a marker wide, another attacks the space it leaves",
    "press": "press: one robot closes down the ball carrier, the rest mark passing outlets "
    "(only possible when an opponent is near the ball)",
    "mark": "shadow-mark: two defenders block the ball-to-goal shot line, the rest man-mark",
    "block": "block-shape: a zone screen in front of our own area, no man-marking",
    "clear": "clear-ball: win the ball deep in our third and kick it out of danger "
    "(only possible when the ball is in danger near our goal)",
}


def make_tactics() -> dict[str, Tactic]:
    """Fresh tactic instances for one Strategy (tactics keep per-match state)."""
    return {
        "attack": GiveAndGoTactic(),
        "overload": DecoyOverloadTactic(),
        "press": PressAndContainTactic(),
        "mark": ShadowAndMarkTactic(),
        "block": BlockShapeTactic(),
        "clear": ClearBallTactic(),
    }


# When a slot's tactic is not applicable this tick, its robots go here instead.
_FALLBACK: dict[str, str] = {"press": "mark", "clear": "block"}

# (key, description, [(tactic_id, count or None for "the rest")]). Order is fixed on purpose:
# OpenJev's answer moves with option order in ~2% of cases, so never reorder between runs.
SPLITS: list[tuple[str, str, list[tuple[str, Optional[int]]]]] = [
    ("build_up", "3 attack + 2 mark", [("attack", 3), ("mark", None)]),
    ("build_up_heavy", "4 attack + 1 mark", [("attack", 4), ("mark", None)]),
    ("counter", "3 attack + 2 block", [("attack", 3), ("block", None)]),
    ("overload", "2 overload + 3 mark", [("overload", 2), ("mark", None)]),
    (
        "overload_support",
        "2 overload + 2 attack + 1 mark",
        [("overload", 2), ("attack", 2), ("mark", None)],
    ),
    (
        "balanced",
        "2 attack + 1 press + 2 mark",
        [("attack", 2), ("press", 1), ("mark", None)],
    ),
    ("press", "3 press + 2 mark", [("press", 3), ("mark", None)]),
    ("press_block", "3 press + 2 block", [("press", 3), ("block", None)]),
    ("all_press", "all 5 press", [("press", None)]),
    ("mark_all", "all 5 mark", [("mark", None)]),
    ("low_block", "all 5 block", [("block", None)]),
    ("clear_danger", "2 clear + 3 block", [("clear", 2), ("block", None)]),
]
SPLIT_KEYS: list[str] = [k for k, _, _ in SPLITS]
_SPLIT_SLOTS = {k: slots for k, _, slots in SPLITS}


def split_glossary() -> dict[str, str]:
    return {k: d for k, d, _ in SPLITS}


def allocate(
    split_key: str,
    free_robots: frozenset[RobotId],
    applicable_tactic_ids: frozenset[str],
) -> dict[str, frozenset[RobotId]]:
    """Assign `free_robots` (ascending id) to the split's slots, folding inapplicable slots into
    their fallback and giving every leftover robot to the split's final "rest" slot."""
    ordered = sorted(free_robots)
    groups: dict[str, list[RobotId]] = {}
    i = 0
    slots = _SPLIT_SLOTS[split_key]
    for n, (tactic_id, count) in enumerate(slots):
        take = (
            len(ordered) - i
            if count is None or n == len(slots) - 1
            else min(count, len(ordered) - i)
        )
        target = tactic_id
        while target not in applicable_tactic_ids and target in _FALLBACK:
            target = _FALLBACK[target]
        if target not in applicable_tactic_ids:
            target = next(
                (t for t in ("mark", "block", "attack") if t in applicable_tactic_ids),
                None,
            )
            if target is None:
                break
        groups.setdefault(target, []).extend(ordered[i : i + take])
        i += take
    return {t: frozenset(r) for t, r in groups.items() if r}
