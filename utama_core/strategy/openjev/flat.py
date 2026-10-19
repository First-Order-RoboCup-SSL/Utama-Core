"""Flat OpenJev engine: the model controls every outfield robot directly, no tactics in between.

One `FlatJevTactic` owns all 5 outfield robots (the goalkeeper stays on its pinned slot, referee
restarts on `RefereeOverride`). At each decision it asks OpenJev one question per robot, chained in
the same moment: the ball carrier first, then the others nearest the ball first, each question
listing what was already chosen (see `flat_state.py`). A robot keeps its action until the next
decision, which comes at once when the situation changes (our carrier, possession, the ball's cell,
or a pass, shot or clearance ending), and at least every `1 / decision_hz` s of sim time (1 Hz:
the scripted rule plays as well at 1 Hz with these triggers as at 5 Hz).

Passes, shots and clearances are locked once chosen: the robots involved aren't asked again until
the kick happens or the action fails or times out, so a new decision can't abandon a half-aimed
pass.

`policy="scripted"` replaces the model with `scripted_choice`, a split_shape-like rule over the
same menu: it is the fallback when the server errors, and a model-free control that tells the
action layer's strength apart from the model's choices.

The sim is stepped synchronously, so inference costs wall time, never game time.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from typing import IO, Any, Optional

from utama_core.engine.tactic import BaseTactic, RobotId, TacticTag
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.shared.pass_and_score_geometry import (
    clamp_outside_enemy_defense_area,
    clamp_outside_own_defense_area,
    clamp_to_field,
    enemy_goal_line,
    enemy_positions,
    find_best_shot,
    has_ball,
    segment_blocked,
)
from utama_core.strategy.openjev import flat_state as fs
from utama_core.strategy.openjev.client import OpenJevClient
from utama_core.strategy.openjev.encoder import possession

LOCKED_ACTIONS = ("pass:", "receive", "shoot", "clear")
PASS_AIM_TIMEOUT_S = 3.0
PASS_ROLL_TIMEOUT_S = 2.5
SHOT_TIMEOUT_S = 2.0
BALL_LOST_GRACE_S = 0.3


@dataclass
class Plan:
    action: str
    since: float
    target: Optional[Vector2D] = None  # move/dribble/hold: fixed when chosen, so the planner keeps one goal
    partner: Optional[int] = None  # pass: the receiver; receive: the passer
    kicked_at: Optional[float] = None  # receive: when the pass was kicked
    lost_since: Optional[float] = None


@dataclass
class FlatMem:
    plans: dict[int, Plan] = field(default_factory=dict)
    carry_origins: dict[int, Optional[Vector2D]] = field(default_factory=dict)
    shot_y: dict[int, Optional[float]] = field(default_factory=dict)
    last_decision_t: Optional[float] = None
    replan: bool = True
    last_carrier: Optional[int] = None
    last_key: Optional[tuple] = None  # (possession, ball cell) at the last tick


def is_locked(plan: Optional[Plan]) -> bool:
    return plan is not None and plan.action.startswith(LOCKED_ACTIONS)


def open_point(game: Game, cell: str, robot_id: int, taken: list[Vector2D]) -> Vector2D:
    """The best spot in `cell` to be available for a pass: an open lane from the ball, room from the
    nearest opponent, a shot from there in the attacking third, away from teammates' spots."""
    x_lo, x_hi, y_lo, y_hi = fs.cell_bounds(game, cell)
    inset = 0.35
    ball = game.ball.p.to_2d()
    enemies = enemy_positions(game)
    me = game.friendly_robots[robot_id].p
    attacking = cell.startswith("attacking")
    goal_x, y1, y2 = enemy_goal_line(game)
    best, best_score = None, -math.inf
    for i in range(3):
        for j in range(3):
            c = Vector2D(
                x_lo + inset + (x_hi - x_lo - 2 * inset) * i / 2.0,
                y_lo + inset + (y_hi - y_lo - 2 * inset) * j / 2.0,
            )
            c = clamp_to_field(clamp_outside_enemy_defense_area(game, clamp_outside_own_defense_area(game, c)), game)
            score = 1.5 * (not segment_blocked(ball, c, enemies)) + min(fs.nearest_enemy_dist(game, c), 1.5)
            if attacking:
                best_y, gap = find_best_shot(c, list(game.enemy_robots.values()), goal_x, y1, y2)
                score += 2.0 * ((gap[1] - gap[0]) if best_y is not None and gap is not None else 0.0)
            score -= 0.15 * c.distance_to(me)
            if any(c.distance_to(o) < 0.8 for o in taken):
                score -= 10.0
            if score > best_score:
                best, best_score = c, score
    return best


def scripted_choice(game: Game, view: fs.RobotView, decided: dict[int, str], ours_has_ball: bool) -> str:
    """The best fixed rule found over this menu, model-free (see `docs/openjev.md`): the carrier shoots
    whenever its shot isn't blocked, from any range, else passes to a free teammate with an open lane,
    else carries towards their goal. Off the ball, when they have it: two robots on the ball, two
    blocking the shot, one forward as the counter-attack outlet; when we have it: two forward into
    attacking cells, two back blocking."""
    opts = view.options
    if view.has_ball:
        if not opts["shoot"].startswith("shot blocked"):
            return "shoot"
        passes = [k for k, d in opts.items() if k.startswith("pass:") and "lane open" in d and ", free," in d]
        if passes:
            return max(passes, key=lambda k: "its shot open" in opts[k])
        dribbles = [k for k in opts if k.startswith("dribble:")]
        if dribbles:
            return max(dribbles, key=lambda k: (fs.THIRDS.index(k.split(":")[1].split("_")[0]), "centre" in k))
        return "shoot"
    chosen = list(decided.values())
    if not ours_has_ball:
        if chosen.count("get_ball") < 2:
            return "get_ball"
        if "block_shot" in opts:
            return "block_shot"
        cells = ("attacking_centre", "attacking_left", "attacking_right")
    else:
        forward = [a for a in chosen if a.startswith("move:") and not a.startswith("move:defensive")]
        if len(forward) >= 2 and "block_shot" in opts:
            return "block_shot"
        cells = ("attacking_left", "attacking_right", "attacking_centre", "middle_left", "middle_right")
    return next((f"move:{c}" for c in cells if f"move:{c}" not in chosen), "hold")


class FlatJevTactic(BaseTactic[FlatMem]):
    tag = TacticTag.MIXED
    keeps_mem_on_robot_change = True

    def __init__(
        self,
        client: Optional[OpenJevClient],
        decision_hz: float = 1.0,
        log_path: Optional[str] = None,
        policy: str = "openjev",
    ):
        if policy not in ("openjev", "scripted"):
            raise ValueError(f"unknown policy {policy!r}")
        self.client = client
        self.policy = policy
        self.period = 1.0 / decision_hz
        self._log: Optional[IO[str]] = open(log_path, "w", buffering=1) if log_path else None
        self._recent: deque[tuple[float, str]] = deque(maxlen=4)
        self._last_side: Optional[str] = None
        self._last_score: Optional[tuple[int, int]] = None
        self.n_decisions = self.n_questions = self.n_cached = self.n_fallback = 0
        self.latency_ms_total = 0.0

    def initial_mem(self) -> FlatMem:
        return FlatMem()

    def highlights(self, mem: FlatMem) -> dict[RobotId, str]:
        return {rid: p.action for rid, p in mem.plans.items()}

    # --- plan progress ------------------------------------------------------------------------

    def _finish(self, mem: FlatMem, rid: int) -> None:
        mem.plans.pop(rid, None)
        mem.replan = True

    def _advance(self, game: Game, t: float, mem: FlatMem) -> None:
        """Drop plans that succeeded, failed or timed out."""
        for rid, plan in list(mem.plans.items()):
            if rid not in mem.plans:
                continue
            holding = has_ball(game, rid, visual=True)
            if plan.action.startswith("pass:"):
                lost = self._lost(plan, holding, t)
                if lost or t - plan.since > PASS_AIM_TIMEOUT_S:
                    self._note(t, f"robot {rid}'s pass to robot {plan.partner} failed")
                    self._finish(mem, rid)
                    if plan.partner in mem.plans and mem.plans[plan.partner].action == "receive":
                        self._finish(mem, plan.partner)
            elif plan.action == "receive":
                if plan.kicked_at is None:
                    if plan.partner not in mem.plans:  # the pass was called off before the kick
                        self._finish(mem, rid)
                elif holding:
                    self._note(t, f"robot {rid} received the pass")
                    self._finish(mem, rid)
                else:
                    ball_speed = math.hypot(game.ball.v.x, game.ball.v.y)
                    rolling = t - plan.kicked_at
                    taken = possession(game)[0] == "them" and rolling > BALL_LOST_GRACE_S
                    if rolling > PASS_ROLL_TIMEOUT_S or (ball_speed < 0.3 and rolling > 0.5) or taken:
                        self._note(t, f"pass to robot {rid} did not arrive")
                        self._finish(mem, rid)
            elif plan.action in ("shoot", "clear"):
                if self._lost(plan, holding, t) or t - plan.since > SHOT_TIMEOUT_S:
                    self._finish(mem, rid)

    @staticmethod
    def _lost(plan: Plan, holding: bool, t: float) -> bool:
        if holding:
            plan.lost_since = None
            return False
        if plan.lost_since is None:
            plan.lost_since = t
        return t - plan.lost_since > BALL_LOST_GRACE_S

    # --- events for the state's `recent` list -------------------------------------------------

    def _note(self, t: float, event: str) -> None:
        self._recent.appendleft((t, event))

    def _track_events(self, game: Game, t: float, holder: Optional[int], mem: FlatMem) -> None:
        side = possession(game)[0]
        if self._last_side is not None and side != self._last_side and side != "contested":
            self._note(t, "we won the ball" if side == "us" else "they won the ball")
        if side != "contested":
            self._last_side = side
        ref = game.referee
        if ref is not None:
            us, them = (ref.yellow_team, ref.blue_team) if game.my_team_is_yellow else (ref.blue_team, ref.yellow_team)
            score = (us.score, them.score)
            if self._last_score is not None and score != self._last_score:
                self._note(t, "we scored" if score[0] > self._last_score[0] else "they scored")
            self._last_score = score

    def _recent_text(self, t: float) -> list[str]:
        """Events of the last 10 s with coarse ages, so the state text repeats while nothing changes
        and the client's cache answers repeated questions (exact ages made every state unique)."""
        out = []
        for et, e in self._recent:
            age = t - et
            if age < 10.0:
                out.append(f"{'just now' if age < 1.0 else 'a moment ago' if age < 3.0 else 'a few seconds ago'}: {e}")
        return out

    # --- logging ------------------------------------------------------------------------------

    def _write(self, row: dict[str, Any]) -> None:
        if self._log is not None:
            self._log.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def summary(self) -> dict[str, Any]:
        live = self.n_questions - self.n_cached - self.n_fallback
        return {
            "policy": self.policy,
            "decisions": self.n_decisions,
            "questions": self.n_questions,
            "cached": self.n_cached,
            "fallback": self.n_fallback,
            "mean_latency_ms": round(self.latency_ms_total / live, 1) if live else None,
        }

    def close(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None
