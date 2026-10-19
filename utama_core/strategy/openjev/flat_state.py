"""Flat OpenJev: the 9 cells, each robot's action menu, and the state text the model reads.

Flat means no tactics in between: at each decision the model picks one action per outfield robot,
one robot at a time, and `flat.py` carries it out with existing skills. Questions in one decision are
chained: each robot's question lists what the robots asked before it chose, so the robots' actions
are decided jointly (p(a1..a5) = p(a1) p(a2|a1) ...), not independently.

Everything is egocentric (thirds and strips from our attacking point of view, names from
`docs/pitch_zones.md`) and given in words, not coordinates: OpenJev is weak at raw numbers. The
geometry it can't see in words (is a pass lane open, is a shot open) is computed here and stated as
a fact about each option; what to do about it is left to the model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Optional

from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.shared.pass_and_score_geometry import (
    enemy_goal_line,
    enemy_positions,
    find_best_shot,
    has_ball,
    segment_blocked,
)
from utama_core.strategy.openjev.encoder import possession

THIRDS = ("defensive", "middle", "attacking")
STRIPS = ("left", "centre", "right")
CELLS: tuple[str, ...] = tuple(f"{t}_{s}" for t in THIRDS for s in STRIPS)

ACTION_GLOSSARY: dict[str, str] = {
    "shoot": "(ball carrier) shoot at the widest open part of their goal; if no shot is open yet, edge sideways "
    "with the ball until one opens",
    "pass:<id>": "(ball carrier) pass to that teammate; the teammate moves to receive it",
    "dribble:<cell>": "(ball carrier) carry the ball towards that cell (at most 1 m per touch)",
    "clear": "(ball carrier) kick the ball long and wide, upfield towards the nearer touch line, away from our goal",
    "get_ball": "go for the ball: collect a loose ball or challenge the opponent who has it",
    "move:<cell>": "run to open space in that cell (the best spot there for receiving a pass)",
    "mark:<id>": "stand between that opponent and our goal, so it can't receive or run at goal freely",
    "block_shot": "stand on the line between the ball and our goal, just outside our defense area",
    "hold": "stay where you are, facing the ball",
}

STATIC_STATE: dict[str, Any] = {
    "role": "You are the coach of our team in a 6v6 RoboCup small-size-league robot soccer match, and you "
    "control each of our 5 outfield robots directly. At each moment you choose one action for one robot at a "
    "time; the question lists what you already chose for the other robots at this moment, so make the robots "
    "work together as a team. Our goalkeeper (robot 0) plays by itself. Referee restarts (kick-offs, free "
    "kicks, penalties) are taken automatically: you only decide during open play.",
    "pitch": "Everything is from our point of view: we attack towards their goal. Thirds: defensive (nearest "
    "our goal), middle, attacking (nearest their goal). Strips: left, centre, right as we face their goal. "
    "A cell is a third and a strip, e.g. attacking_left.",
    "rules": "Only our goalkeeper may enter our own defense area, and none of our robots may enter theirs. A "
    "robot may carry the ball at most 1 m before it must pass, shoot or clear. A foul gives them a free kick. "
    "Aimless kick: a ball we kick from our own half that crosses their goal line without touching anyone gives "
    "them a free kick from where we kicked it.",
    "actions": ACTION_GLOSSARY,
    "coaching": [
        "Shots win matches here: the ball carrier shoots whenever its shot is open or narrow, from any range. "
        "It passes only when its shot is blocked, to a free teammate with an open lane, preferring one whose own "
        "shot is open; otherwise it dribbles into space, or clears when pressed in our own half.",
        "When they have the ball, two robots go for it (get_ball: the nearer blocks its shot, the other tackles), "
        "two block_shot, and one moves into an attacking cell, ready for the counter-attack. In our defensive "
        "third three go for it and two block_shot: nobody stays forward.",
        "When we have the ball, three robots move into open attacking cells on different strips and one stays "
        "back on block_shot.",
        "A loose ball near our goal is always someone's to get.",
    ],
    "reading_the_state": "Distances are in words: touching (<0.3 m), very close (<0.7 m), close (<1.5 m), "
    "medium (<3 m), far. 'free' means no opponent within 0.7 m. Shots: open, narrow or blocked, and close "
    "(<2 m), medium (<4 m) or long range. `recent` lists the latest events, newest first.",
}

STATIC_TEXT = json.dumps({"static": STATIC_STATE}, ensure_ascii=False, separators=(",", ":")) + "\n"

QUESTION_KEY = "action"

FREE_RADIUS_M = 0.7
MAX_BLOCKERS = 2
_NARROW_SHOT_M = 0.25  # open gap along their goal line below which a shot is "narrow"


def attack_dir(game: Game) -> float:
    return -1.0 if game.my_team_is_right else 1.0


def cell_coords(game: Game, p: Vector2D) -> tuple[int, int]:
    """(third, strip) indices of world point `p`, from our point of view."""
    a = attack_dir(game)
    half = game.field.half_length
    progress = p.x * a + half  # metres from our own goal line towards theirs
    third = min(2, max(0, int(progress / (2.0 * half / 3.0))))
    lateral = p.y * a  # facing their goal, left is +lateral
    edge = game.field.half_width / 3.0
    strip = 0 if lateral > edge else 2 if lateral < -edge else 1
    return third, strip


def cell_of(game: Game, p: Vector2D) -> str:
    third, strip = cell_coords(game, p)
    return CELLS[third * 3 + strip]


def cell_bounds(game: Game, cell: str) -> tuple[float, float, float, float]:
    """World (x_lo, x_hi, y_lo, y_hi) of `cell`."""
    k = CELLS.index(cell)
    third, strip = divmod(k, 3)
    a = attack_dir(game)
    half, width = game.field.half_length, game.field.half_width
    edge = width / 3.0
    p_lo, p_hi = third * 2.0 * half / 3.0, (third + 1) * 2.0 * half / 3.0
    l_lo, l_hi = [(edge, width), (-edge, edge), (-width, -edge)][strip]
    xs = sorted(((p_lo - half) * a, (p_hi - half) * a))
    ys = sorted((l_lo * a, l_hi * a))
    return xs[0], xs[1], ys[0], ys[1]


def adjacent_cells(cell: str) -> list[str]:
    """`cell` and its up-to-8 neighbours, in `CELLS` order."""
    third, strip = divmod(CELLS.index(cell), 3)
    return [c for i, c in enumerate(CELLS) if abs(i // 3 - third) <= 1 and abs(i % 3 - strip) <= 1]


def dist_word(d: Optional[float]) -> str:
    if d is None or not math.isfinite(d):
        return "far"
    return (
        "touching" if d < 0.3 else "very close" if d < 0.7 else "close" if d < 1.5 else "medium" if d < 3.0 else "far"
    )


def nearest_enemy_dist(game: Game, p: Vector2D) -> float:
    return min((r.p.distance_to(p) for r in game.enemy_robots.values()), default=math.inf)


def shot_word(game: Game, p: Vector2D) -> str:
    """'open, medium range' / 'narrow, close' / 'blocked, long range' for a shot from `p`."""
    goal_x, y1, y2 = enemy_goal_line(game)
    best_y, gap = find_best_shot(p, list(game.enemy_robots.values()), goal_x, y1, y2)
    width = (gap[1] - gap[0]) if (best_y is not None and gap is not None) else 0.0
    quality = "blocked" if best_y is None else "narrow" if width < _NARROW_SHOT_M else "open"
    d = p.distance_to(Vector2D(goal_x, (y1 + y2) / 2.0))
    rng = "close" if d < 2.0 else "medium range" if d < 4.0 else "long range"
    return f"{quality}, {rng}"


def lane_open(game: Game, start: Vector2D, end: Vector2D) -> bool:
    return not segment_blocked(start, end, enemy_positions(game))


def carrier(game: Game, robot_ids) -> Optional[int]:
    """Our robot with the ball on (or visibly at) its dribbler, nearest the ball if several."""
    holders = [rid for rid in robot_ids if rid in game.friendly_robots and has_ball(game, rid, visual=True)]
    if not holders:
        return None
    ball = game.ball.p.to_2d()
    return min(holders, key=lambda rid: game.friendly_robots[rid].p.distance_to(ball))


def side_of(game: Game, robot_ids) -> str:
    """Who has the ball, as the state and the questions say it: 'us' only with a carrier of ours (`carrier`),
    as `flat.scripted_choice` counts it; else `possession`'s 'them' or 'contested'. `possession` says 'us'
    whenever one of ours is nearest, our keeper included: told "we have the ball" about a loose ball in our
    third, the model sent the robot called to it forward instead, a quarter of the time."""
    if carrier(game, robot_ids) is not None:
        return "us"
    side = possession(game)[0]
    return "contested" if side == "us" else side


def enemy_keeper_id(game: Game) -> Optional[int]:
    ref = game.referee
    if ref is None:
        return None
    return (ref.blue_team if game.my_team_is_yellow else ref.yellow_team).goalkeeper


# The coaching line for each kind of moment, repeated in that moment's questions: the model weighs an
# instruction next to the question far more than the same line in the long static block.
PHASE_COACHING = {
    "carrier": "shoot whenever your shot is open or narrow, from any range; pass only when it is blocked, to a "
    "free teammate with an open lane; otherwise dribble into space",
    "them": "two robots go for the ball (the two nearest it), two block_shot, and one moves into an attacking "
    "cell, ready for the counter-attack",
    "contested": "two robots go for the ball (the two nearest it), two block_shot, and one moves into an attacking "
    "cell, ready for the counter-attack",
    "deep": "the ball is loose or theirs in our defensive third: three robots go for it (the three nearest it) "
    "and two block_shot; nobody stays forward",
    "us": "three robots move into open attacking cells on different strips and one stays back on block_shot",
}


# The roles each moment's coaching asks for, so a question can say which are still unfilled: a single
# readout does not count reliably, and asked "two robots go for the ball" it left the second one out
# (half the time, in the first full match against split_shape).
PHASE_ROLES = {
    "them": (("get_ball", 2), ("block_shot", 2), ("attacking move", 1)),
    "contested": (("get_ball", 2), ("block_shot", 2), ("attacking move", 1)),
    "deep": (("get_ball", 3), ("block_shot", 2)),
    "us": (("attacking move", 3), ("block_shot", 1)),
}


def still_to_fill(phase: str, decided: dict[int, str]) -> str:
    """'get_ball 1, block_shot 2' for the roles of `phase` not yet taken this moment."""
    if phase not in PHASE_ROLES:
        return ""
    taken = {
        "get_ball": sum(a == "get_ball" for a in decided.values()),
        "block_shot": sum(a == "block_shot" for a in decided.values()),
        "attacking move": sum(a.startswith("move:attacking") for a in decided.values()),
    }
    left = [(role, n - taken[role]) for role, n in PHASE_ROLES[phase] if n - taken[role] > 0]
    if not left:
        return "Every role the coaching asks for is taken."
    return (
        "Still to fill for this moment: "
        + ", ".join(
            f"{role} ({n} more)" if role != "attacking move" else f"move into an attacking cell ({n} more)"
            for role, n in left
        )
        + "."
    )


@dataclass
class RobotView:
    """What one robot's question needs: its own situation and its options with their facts."""

    robot_id: int
    has_ball: bool
    options: dict[str, Optional[str]]
    situation: str
    phase: str = "contested"  # PHASE_COACHING key; "deep": no carrier of ours and the ball in our defensive third


def live_state(
    game: Game,
    robot_ids: tuple[int, ...],
    actions: dict[int, str],
    marked_by: dict[int, list[int]],
    recent: list[str],
) -> dict[str, Any]:
    ref = game.referee
    score: dict[str, Any] = {}
    clock: dict[str, Any] = {}
    if ref is not None:
        us, them = (ref.yellow_team, ref.blue_team) if game.my_team_is_yellow else (ref.blue_team, ref.yellow_team)
        score = {"us": us.score, "them": them.score}
        clock = {"stage": ref.stage.name.lower(), "time_left_s": int(ref.stage_time_left // 10 * 10)}

    ball = game.ball.p.to_2d()
    speed = math.hypot(game.ball.v.x, game.ball.v.y)
    forward = game.ball.v.x * attack_dir(game)
    moving = (
        "still"
        if speed < 0.2
        else (
            "towards their goal"
            if forward > 0.5 * speed
            else "towards our goal" if forward < -0.5 * speed else "sideways"
        )
        + (", fast" if speed >= 1.5 else ", slow")
    )
    side = side_of(game, robot_ids)
    holder = carrier(game, robot_ids)
    ours = []
    for rid in sorted(robot_ids):
        if rid not in game.friendly_robots:
            continue
        p = game.friendly_robots[rid].p
        ours.append(
            {
                "id": rid,
                "cell": cell_of(game, p),
                "to_ball": dist_word(p.distance_to(ball)),
                "free": nearest_enemy_dist(game, p) >= FREE_RADIUS_M,
                "shot": shot_word(game, p),
                "doing": actions.get(rid, "nothing yet"),
            }
        )
    keeper = enemy_keeper_id(game)
    theirs = []
    for eid, r in sorted(game.enemy_robots.items()):
        theirs.append(
            {
                "id": eid,
                "cell": cell_of(game, r.p),
                "to_ball": dist_word(r.p.distance_to(ball)),
                **({"role": "goalkeeper"} if eid == keeper else {"marked_by": marked_by.get(eid, [])}),
            }
        )
    return {
        "score": score,
        "clock": clock,
        "ball": {
            "cell": cell_of(game, ball),
            "moving": moving,
            "held_by": f"our robot {holder}" if holder is not None else "nobody of ours",
        },
        "possession": side,
        "our_robots": ours,
        "their_robots": theirs,
        "recent": recent[:4],
    }


def render(live: dict[str, Any]) -> str:
    return STATIC_TEXT + json.dumps({"live": live}, ensure_ascii=False, separators=(",", ":"))


def robot_view(
    game: Game,
    robot_id: int,
    robot_ids: tuple[int, ...],
    *,
    carry_left_m: Optional[float],
    marked_by: dict[int, list[int]],
    blockers: int,
    chasers: tuple[int, ...] = (),
) -> RobotView:
    """`robot_id`'s options, each stated with the facts that bear on it, and a one-line situation.

    The carrier gets ball actions (no `dribble` once its carry is used up); everyone else gets the
    off-ball actions. Teammates that can't take part (the goalkeeper, robots not in `robot_ids`) are
    not offered as pass targets.
    """
    me = game.friendly_robots[robot_id].p
    ball = game.ball.p.to_2d()
    here = cell_of(game, me)
    holding = has_ball(game, robot_id, visual=True)
    mates = [rid for rid in sorted(robot_ids) if rid != robot_id and rid in game.friendly_robots]
    options: dict[str, Optional[str]] = {}
    if holding:
        options["shoot"] = f"shot {shot_word(game, me)}"
        for rid in mates:
            p = game.friendly_robots[rid].p
            options[f"pass:{rid}"] = (
                f"robot {rid} in {cell_of(game, p)} at {dist_word(p.distance_to(me))} distance, lane "
                f"{'open' if lane_open(game, ball, p) else 'blocked'}, "
                f"{'free' if nearest_enemy_dist(game, p) >= FREE_RADIUS_M else 'marked'}, its shot {shot_word(game, p)}"
            )
        if carry_left_m is None or carry_left_m > 0.2:
            for cell in adjacent_cells(here):
                options[f"dribble:{cell}"] = "stay in this cell, beat the nearest opponent" if cell == here else None
        options["clear"] = (
            "kick it long and wide, out towards the touch line: gives the ball away, so only when it is about to "
            "be lost near our goal"
        )
        left = "" if carry_left_m is None else f", can carry about {round(max(0.0, carry_left_m) * 4) / 4:.2f} m more"
        situation = f"Robot {robot_id} has the ball in {here}{left}; its shot is {shot_word(game, me)}."
        phase = "carrier"
    else:
        order = sorted(mates + [robot_id], key=lambda rid: game.friendly_robots[rid].p.distance_to(ball))
        rank = order.index(robot_id) + 1
        near = f"the ball is {dist_word(me.distance_to(ball))}, you are number {rank} of ours to it"
        if not chasers:
            options["get_ball"] = f"nobody of ours is going for the ball yet; {near}"
        elif len(chasers) == 1:
            # A second robot on get_ball is executed differently (the nearer blocks, the other tackles),
            # so say what it does here instead of repeating the first chaser's description.
            first = f"robot {chasers[0]}"
            options["get_ball"] = f"help {first} win the ball: tackle while {first} blocks its shot; {near}"
        else:
            mates = " and ".join(f"robot {c}" for c in chasers)
            options["get_ball"] = f"join {mates} on the ball: tackle with them; {near}"
        for cell in CELLS:
            options[f"move:{cell}"] = "you are here" if cell == here else None
        keeper = enemy_keeper_id(game)
        for eid, r in sorted(game.enemy_robots.items()):
            if eid == keeper:
                continue
            by = marked_by.get(eid, [])
            options[f"mark:{eid}"] = (
                f"their robot {eid} in {cell_of(game, r.p)}, {dist_word(r.p.distance_to(ball))} from the ball, "
                f"{'marked by ' + ', '.join(map(str, by)) if by else 'unmarked'}"
            )
        if blockers < MAX_BLOCKERS:  # a third blocker crowds our defense area (`defend_parameter` places two)
            options["block_shot"] = f"{blockers} teammate{'s' if blockers != 1 else ''} already blocking"
        options["hold"] = None
        side = side_of(game, robot_ids)
        who = {"us": "we have the ball", "them": "they have the ball", "contested": "the ball is loose or contested"}[
            side
        ]
        situation = f"Robot {robot_id} is in {here}, {dist_word(me.distance_to(ball))} to the ball; {who}."
        # Nobody of ours on the ball in our defensive third: `flat.scripted_choice`'s three-chaser case.
        deep = cell_of(game, ball).startswith("defensive") and carrier(game, robot_ids) is None
        phase = "deep" if deep else side
    return RobotView(robot_id=robot_id, has_ball=holding, options=options, situation=situation, phase=phase)


def question(view: RobotView, decided: list[str], roles_left: str = "", call: Optional[str] = None) -> dict[str, Any]:
    """`call`: the coaching's pick for this robot (`flat.scripted_choice`), stated as advice; the answer is
    still the model's. Without it the model chose as the coaching would on 62% of questions, with it on
    93% (`docs/openjev.md`)."""
    already = "; ".join(decided) if decided else "nothing yet"
    left = f" {roles_left}" if roles_left else ""
    called = f" The coaching's call for robot {view.robot_id}: {call}." if call else ""
    return {
        QUESTION_KEY: {
            "type": "choice",
            "instructions": f"{view.situation} Already decided for this moment: {already}. "
            f"Coaching for this moment: {PHASE_COACHING[view.phase]}.{left}{called} "
            f"What should robot {view.robot_id} do now?",
            "criteria": view.options,
        }
    }
