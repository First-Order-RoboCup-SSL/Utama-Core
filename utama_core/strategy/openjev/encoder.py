"""Game -> OpenJev state text.

Layout follows FOR-Engine's prefix cache and OpenJev's own guidance: a static
block (identical for every tick of every match) first, then the live block,
joined by a newline — a comma at the join breaks prefix reuse (measured with
`tools/openjev_bench.py`). The state is sent as a pre-rendered string so the
static block is a literal text prefix of every request.

Everything in the live block is egocentric (zones/lanes from our attacking
point of view, so side of the pitch never matters) and bucketed or rounded:
OpenJev is weak at raw coordinates and number comparison, and identical
consecutive states let the client reuse the previous answer.
"""

from __future__ import annotations

import json
import math
from typing import Any, Optional

from utama_core.engine.tactic import RobotId
from utama_core.entities.data.object import TeamType
from utama_core.entities.game import Game
from utama_core.strategy.kernel_strategy import _CLOSER_TO_BALL_MARGIN
from utama_core.strategy.openjev.splits import TACTIC_GLOSSARY, split_glossary
from utama_core.tactics.clear_ball import in_danger

STATIC_STATE: dict[str, Any] = {
    "role": "You are the coach of our team in a 6v6 RoboCup small-size-league robot soccer match. Each "
    "moment you choose how our 5 outfield robots are split across team tactics. Our goalkeeper is "
    "handled separately and is never part of the split.",
    "pitch": "Everything is described from our point of view. Thirds: own (nearest our goal), mid, final "
    "(nearest the goal we attack). Lanes: left, centre, right when facing the goal we attack.",
    "tactics": TACTIC_GLOSSARY,
    "splits": split_glossary(),
    "reading_the_state": "possession says who is clearly closer to the ball; distances are metres rounded "
    "to 0.5; counts are precomputed; `recent` lists the latest events, newest first.",
}

STATIC_TEXT = (
    json.dumps({"static": STATIC_STATE}, ensure_ascii=False, separators=(",", ":"))
    + "\n"
)

QUESTION_KEY = "split"
INSTRUCTIONS = "Which split of our 5 outfield robots should we use right now?"


def _attack_dir(game: Game) -> float:
    return -1.0 if game.my_team_is_right else 1.0


def _zone(game: Game, x: float) -> str:
    half = game.field.half_length
    progress = (x - (-_attack_dir(game) * half)) * _attack_dir(game)
    third = 2.0 * half / 3.0
    return "own" if progress < third else "mid" if progress < 2.0 * third else "final"


def _lane(game: Game, y: float) -> str:
    # Facing +x, left is +y; facing -x, left is -y.
    lateral = y * _attack_dir(game)
    edge = game.field.half_width / 3.0
    return "left" if lateral > edge else "right" if lateral < -edge else "centre"


def _half_metre(d: Optional[float]) -> Optional[float]:
    return None if d is None else round(d * 2) / 2


def possession(game: Game) -> tuple[str, Optional[float], Optional[float]]:
    """('us' | 'them' | 'contested', our closest distance, their closest distance)."""
    _, ours = game.proximity_lookup.closest_to_ball(team_type_filter=TeamType.FRIENDLY)
    _, theirs = game.proximity_lookup.closest_to_ball(team_type_filter=TeamType.ENEMY)
    if ours is None or theirs is None:
        return "contested", ours, theirs
    if ours < theirs - _CLOSER_TO_BALL_MARGIN:
        return "us", float(ours), float(theirs)
    if theirs < ours - _CLOSER_TO_BALL_MARGIN:
        return "them", float(ours), float(theirs)
    return "contested", float(ours), float(theirs)


def live_state(
    game: Game,
    *,
    current_split: Optional[str],
    split_held_s: float,
    robot_tactic: dict[RobotId, str],
    applicable: frozenset[str],
    recent: list[str],
) -> dict[str, Any]:
    ref = game.referee
    score: dict[str, Any] = {}
    referee: dict[str, Any] = {}
    if ref is not None:
        us, them = (
            (ref.yellow_team, ref.blue_team)
            if game.my_team_is_yellow
            else (ref.blue_team, ref.yellow_team)
        )
        score = {"us": us.score, "them": them.score}
        referee = {
            "stage": ref.stage.name.lower(),
            "command": ref.referee_command.name.lower(),
            "time_left_s": int(ref.stage_time_left // 10 * 10),
        }

    ball: dict[str, Any] = {}
    if game.ball is not None:
        b = game.ball
        speed = math.hypot(b.v.x, b.v.y)
        forward = b.v.x * _attack_dir(game)
        ball = {
            "zone": _zone(game, b.p.x),
            "lane": _lane(game, b.p.y),
            "speed": "still" if speed < 0.2 else "slow" if speed < 1.5 else "fast",
            "moving": (
                "none"
                if speed < 0.2
                else (
                    "towards_their_goal"
                    if forward > 0.5 * speed
                    else "towards_our_goal" if forward < -0.5 * speed else "sideways"
                )
            ),
        }

    side, ours_d, theirs_d = possession(game)
    enemy_zones = [_zone(game, r.p.x) for r in game.enemy_robots.values()]
    our_robots = [
        {
            "id": rid,
            "zone": _zone(game, r.p.x),
            "lane": _lane(game, r.p.y),
            "tactic": robot_tactic.get(rid, "goalkeeper" if rid == 0 else "none"),
        }
        for rid, r in sorted(game.friendly_robots.items())
        if rid != 0
    ]
    return {
        "score": score,
        "referee": referee,
        "ball": ball,
        "possession": {
            "side": side,
            "our_closest_m": _half_metre(ours_d),
            "their_closest_m": _half_metre(theirs_d),
        },
        "threat": {
            "opponents_in_our_third": enemy_zones.count("own"),
            "opponents_in_mid": enemy_zones.count("mid"),
            "opponents_in_final_third": enemy_zones.count("final"),
            "ball_in_danger": in_danger(game),
            "press_possible": "press" in applicable,
        },
        "our_robots": our_robots,
        "current_split": current_split,
        "split_held_s": round(split_held_s * 2) / 2,
        "recent": recent[:3],
    }


def render(live: dict[str, Any]) -> str:
    return STATIC_TEXT + json.dumps(
        {"live": live}, ensure_ascii=False, separators=(",", ":")
    )


def question() -> dict[str, Any]:
    return {
        QUESTION_KEY: {
            "type": "choice",
            "instructions": INSTRUCTIONS,
            # keys only: descriptions live in the cached static block (shorter uncacheable tail)
            "criteria": {k: None for k in split_glossary()},
        }
    }
