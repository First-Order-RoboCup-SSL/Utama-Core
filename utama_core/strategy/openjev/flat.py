"""Flat OpenJev engine: the model controls every outfield robot directly, no tactics in between.

One `FlatJevTactic` owns all 5 outfield robots (the goalkeeper stays on its pinned slot, referee
restarts on `RefereeOverride`). At each decision it asks OpenJev one question per robot, chained in
the same moment: the ball carrier first, then the others nearest the ball first, each question
listing what was already chosen (see `flat_state.py`) and the coaching's call for that robot, the
scripted rule's pick: advice, the answer is still the model's. A robot keeps its action until the next
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
from pathlib import Path
from types import SimpleNamespace
from typing import IO, Any, Callable, Optional

from utama_core.custom_referee.actions import (
    DirectFreeOursStep,
    _clear_to_legal_positions,
)
from utama_core.engine.context import TickContext
from utama_core.engine.strategy import Strategy as KernelSchedulerStrategy
from utama_core.engine.tactic import BaseTactic, RobotId, TacticTag
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.shared.pass_and_score_geometry import (
    CARRY_LIMIT_M,
    ball_in_own_defense_area,
    ball_line_receive_point,
    carry_exhausted,
    carry_origin,
    clamp_outside_enemy_defense_area,
    clamp_outside_own_defense_area,
    clamp_to_field,
    enemy_goal_line,
    enemy_positions,
    find_best_shot,
    has_ball,
    oriented_towards,
    own_defense_area_exit_point,
    segment_blocked,
)
from utama_core.skills.src.block import block_attacker
from utama_core.skills.src.defend_parameter import defend_parameter
from utama_core.skills.src.go_to_ball import go_to_ball
from utama_core.skills.src.go_to_point import go_to_point
from utama_core.skills.src.man_mark import mark_target
from utama_core.skills.src.utils.move_utils import (
    empty_command,
    kick,
    move,
    turn_on_spot,
)
from utama_core.strategy.openjev import flat_state as fs
from utama_core.strategy.openjev.client import OpenJevClient
from utama_core.strategy.openjev.encoder import possession
from utama_core.tactics._pass_and_score import _pass_exec, _score_goal

LOCKED_ACTIONS = ("pass:", "receive", "shoot", "clear")
EVENT_MIN_GAP_S = 0.1  # an event-triggered decision never comes sooner than this after the last one
PASS_KICK_NOW_S = 1.2  # a pass not kicked by then is kicked at the receiver as it stands
PASS_AIM_TIMEOUT_S = 3.0
PASS_ROLL_TIMEOUT_S = 2.5
SHOT_TIMEOUT_S = 2.0
BALL_LOST_GRACE_S = 0.3
ENEMY_HOLD_M = 0.3
BLOCK_HOLD_M = 0.3  # a blocker with a tackling teammate holds the shot line this far in front of the ball


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


def shot_line_point(game: Game, dist_m: float) -> Vector2D:
    """The point `dist_m` from the ball towards our goal's centre, kept out of our defense area."""
    ball = game.ball.p.to_2d()
    goal = Vector2D(game.field.my_goal_line[0][0], 0.0)
    d = goal - ball
    k = min(1.0, dist_m / max(d.mag(), 1e-6))
    return clamp_outside_own_defense_area(game, Vector2D(ball.x + d.x * k, ball.y + d.y * k))


def enemy_holder(game: Game) -> Optional[int]:
    """The opponent with the ball: nearest it, within `ENEMY_HOLD_M`, and clearly nearer than us."""
    if possession(game)[0] != "them":
        return None
    ball = game.ball.p.to_2d()
    eid, r = min(game.enemy_robots.items(), key=lambda kv: kv[1].p.distance_to(ball), default=(None, None))
    return eid if r is not None and r.p.distance_to(ball) <= ENEMY_HOLD_M else None


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
    blocking the shot, one forward as the counter-attack outlet (in our defensive third, three on the
    ball and no outlet); when we have it: two forward into attacking cells, two back blocking."""
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
        # Three on the ball in our defensive third, no outlet: against split_shape 8-0-4 (78-53) to two's
        # 6-0-6 (62-61), 12 matches each (4 cells x 3 decision rates).
        if chosen.count("get_ball") < (3 if view.phase == "deep" else 2):
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
        call: bool = True,
    ):
        if policy not in ("openjev", "scripted"):
            raise ValueError(f"unknown policy {policy!r}")
        self.client = client
        self.policy = policy
        self.call = call
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

    # --- tick ---------------------------------------------------------------------------------

    def tick(
        self, game: Game, ctx: TickContext, robot_ids: tuple[RobotId, ...], mem: FlatMem
    ) -> tuple[dict[RobotId, RobotCommand], FlatMem]:
        robot_ids = tuple(rid for rid in robot_ids if rid in game.friendly_robots)
        if game.ball is None:
            return {rid: empty_command() for rid in robot_ids}, mem
        t = game.ts
        for rid in list(mem.plans):
            if rid not in robot_ids:
                del mem.plans[rid]
        for rid in robot_ids:
            mem.carry_origins[rid] = carry_origin(game, rid, mem.carry_origins.get(rid))
        self._advance(game, t, mem)

        holder = fs.carrier(game, robot_ids)
        self._track_events(game, t, holder, mem)
        key = (possession(game)[0], fs.cell_of(game, game.ball.p.to_2d()))
        if holder != mem.last_carrier or key != mem.last_key:
            mem.replan = True
        mem.last_carrier, mem.last_key = holder, key

        since = None if mem.last_decision_t is None else t - mem.last_decision_t
        if since is None or since >= self.period - 1e-9 or (mem.replan and since >= EVENT_MIN_GAP_S - 1e-9):
            self._decide(game, t, robot_ids, holder, mem)
            mem.last_decision_t = t
            mem.replan = False
        return self._execute(game, ctx, t, robot_ids, mem), mem

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

    # --- decisions ----------------------------------------------------------------------------

    def _decide(self, game: Game, t: float, robot_ids: tuple[int, ...], holder: Optional[int], mem: FlatMem) -> None:
        ball = game.ball.p.to_2d()
        free = [rid for rid in robot_ids if not is_locked(mem.plans.get(rid))]
        order = sorted(free, key=lambda rid: (rid != holder, game.friendly_robots[rid].p.distance_to(ball)))
        if not order:
            return
        self.n_decisions += 1
        doing = {rid: self._describe(rid, p) for rid, p in mem.plans.items()}
        state = fs.render(fs.live_state(game, robot_ids, doing, self._marked_by(mem.plans, {}), self._recent_text(t)))
        if self.policy == "openjev" and self.client is not None and len(order) > 1:
            # One prefill of the shared state, so each question below computes only its own tail.
            # Needs SGLang's no_buffer mamba cache (FOR-Engine's sglang-server task): it keeps the
            # hybrid model's state at the end of this request, exactly where the questions diverge.
            try:
                self.client.prewarm(state)
            except Exception:  # a failed warm-up only costs speed; the questions still go out
                pass
        decided: dict[int, str] = {}
        lines = [f"robot {rid} -> {doing[rid]}" for rid in robot_ids if is_locked(mem.plans.get(rid))]
        row: dict[str, Any] = {"t": round(t, 3), "chain": []}
        taken = [game.friendly_robots[rid].p for rid in robot_ids if rid not in order]
        ours_has_ball = holder is not None
        while order:
            rid = order.pop(0)
            view = fs.robot_view(
                game,
                rid,
                robot_ids,
                carry_left_m=self._carry_left(game, mem, rid),
                marked_by=self._marked_by({}, decided),
                blockers=sum(a == "block_shot" for a in decided.values()),
                chasers=tuple(r for r, a in decided.items() if a == "get_ball"),
            )
            choice, info = self._choose(game, state, view, lines, decided, ours_has_ball)
            row["chain"].append({"robot": rid, **info})
            decided[rid] = choice
            plan = Plan(choice, t)
            if choice.startswith(("move:", "dribble:")):
                plan.target = open_point(game, choice.split(":", 1)[1], rid, taken)
                taken.append(plan.target)
            elif choice == "hold":
                plan.target = game.friendly_robots[rid].p
            if choice.startswith("pass:"):
                receiver = int(choice.split(":")[1])
                plan.partner = receiver
                mem.plans[receiver] = Plan("receive", t, partner=rid)
                decided[receiver] = "receive"
                lines.append(f"robot {rid} -> pass to robot {receiver}")
                lines.append(f"robot {receiver} -> receive the pass from robot {rid}")
                if receiver in order:
                    order.remove(receiver)
                self._note(t, f"robot {rid} passes to robot {receiver}")
            else:
                lines.append(f"robot {rid} -> {choice}")
                if choice == "shoot":
                    self._note(t, f"robot {rid} shoots")
            mem.plans[rid] = plan
        self._write(row)

    def _choose(
        self,
        game: Game,
        state: str,
        view: fs.RobotView,
        lines: list[str],
        decided: dict[int, str],
        ours_has_ball: bool,
    ) -> tuple[str, dict[str, Any]]:
        rule = scripted_choice(game, view, decided, ours_has_ball)
        if self.policy == "scripted" or self.client is None:
            return (rule if rule in view.options else "hold"), {"choice": None, "scripted": True}
        self.n_questions += 1
        try:
            q = fs.question(view, lines, fs.still_to_fill(view.phase, decided), call=rule if self.call else None)
            d = self.client.decide(state, q)
            ans = d.answers[fs.QUESTION_KEY]
            choice = ans["choice"]
            if choice not in view.options:
                raise ValueError(f"answer {choice!r} is not one of the options")
        except Exception as e:  # server down or a bad answer: this robot plays the scripted rule
            self.n_fallback += 1
            return rule, {"choice": rule, "fallback": f"{type(e).__name__}: {e}"[:200]}
        self.n_cached += int(d.cached)
        self.latency_ms_total += d.latency_ms
        probs = ans.get("probabilities") or {}
        top = dict(sorted(probs.items(), key=lambda kv: -kv[1])[:3]) if isinstance(probs, dict) else probs
        return choice, {
            "choice": choice,
            "why": view.options.get(choice),
            "p": top,
            "ms": round(d.latency_ms, 1),
            "cached": d.cached,
            "rule": rule,  # what the scripted rule would do (the call, when the question carries it)
        }

    @staticmethod
    def _describe(rid: int, plan: Plan) -> str:
        if plan.action.startswith("pass:"):
            return f"passing to robot {plan.partner}"
        if plan.action == "receive":
            return f"receiving a pass from robot {plan.partner}"
        return plan.action

    @staticmethod
    def _marked_by(plans: dict[int, Plan], decided: dict[int, str]) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        actions = {rid: p.action for rid, p in plans.items()}
        actions.update(decided)
        for rid, action in sorted(actions.items()):
            if action.startswith("mark:"):
                out.setdefault(int(action.split(":")[1]), []).append(rid)
        return out

    @staticmethod
    def _carry_left(game: Game, mem: FlatMem, rid: int) -> Optional[float]:
        origin = mem.carry_origins.get(rid)
        if origin is None:
            return None
        return CARRY_LIMIT_M - game.ball.p.to_2d().distance_to(origin)

    # --- execution ----------------------------------------------------------------------------

    def _execute(
        self, game: Game, ctx: TickContext, t: float, robot_ids: tuple[int, ...], mem: FlatMem
    ) -> dict[RobotId, RobotCommand]:
        mc = ctx.motion_controller
        commands: dict[RobotId, RobotCommand] = {}
        ball = game.ball.p.to_2d()
        blockers = sorted(rid for rid, p in mem.plans.items() if p.action == "block_shot")
        holder = enemy_holder(game)
        # With several robots on get_ball against their carrier, the one nearest it blocks the shot
        # line and the rest tackle the ball directly: two blockers on one line would collide.
        chasers = [rid for rid, p in mem.plans.items() if p.action == "get_ball" and rid in robot_ids]
        blocker_id = (
            min(chasers, key=lambda rid: game.friendly_robots[rid].p.distance_to(game.enemy_robots[holder].p))
            if holder is not None and chasers
            else None
        )

        for rid, plan in list(mem.plans.items()):
            if not plan.action.startswith("pass:") or rid in commands:
                continue
            receiver = plan.partner
            if receiver not in robot_ids:
                self._finish(mem, rid)
                continue
            cmds, _, _ = _pass_exec(game, ctx, rid, receiver, kick_now=t - plan.since > PASS_KICK_NOW_S)
            commands.update(cmds)
            if cmds[rid].kick:
                self._finish(mem, rid)  # the passer is free again; the receiver's plan tracks the ball
                if receiver in mem.plans:
                    mem.plans[receiver].kicked_at = t

        for rid in robot_ids:
            if rid in commands:
                continue
            plan = mem.plans.get(rid)
            action = plan.action if plan is not None else ""
            # `at_ball` (the visual box) reaches a few cm past dribbler contact: from there a robot
            # must still close in before it can carry or kick (`run_setup_phase`'s comment), so
            # ball actions check `on_ball`, the contact sensor.
            at_ball = has_ball(game, rid, visual=True)
            on_ball = has_ball(game, rid)
            if action == "receive":
                if plan.kicked_at is None:
                    commands[rid] = go_to_point(game, mc, rid, game.friendly_robots[rid].p)
                    continue
                spot = ball_line_receive_point(game, rid)
                if spot is None:
                    commands[rid] = go_to_ball(game=game, motion_controller=mc, robot_id=rid, ctx=ctx)
                else:
                    spot = clamp_outside_enemy_defense_area(game, clamp_outside_own_defense_area(game, spot))
                    commands[rid] = move(
                        game, mc, rid, spot, game.friendly_robots[rid].p.angle_to(ball), dribbling=True
                    )
            elif action == "shoot":
                commands[rid] = self._shoot(game, ctx, rid, mem)
                if commands[rid].kick and not on_ball:
                    # Aimed but only at the visual edge: a kick now would hit nothing. Close
                    # the last few cm keeping the aim (`_pass_exec` does the same).
                    commands[rid] = move(game, mc, rid, ball, game.friendly_robots[rid].orientation, dribbling=True)
                elif commands[rid].kick:
                    self._finish(mem, rid)
            elif action == "clear":
                if on_ball:
                    commands[rid] = self._clear(game, mc, rid)
                    if commands[rid].kick:
                        self._finish(mem, rid)
                else:
                    commands[rid] = go_to_ball(game=game, motion_controller=mc, robot_id=rid, ctx=ctx)
            elif action.startswith("dribble:"):
                if not on_ball:
                    commands[rid] = go_to_ball(game=game, motion_controller=mc, robot_id=rid, ctx=ctx)
                elif carry_exhausted(game, mem.carry_origins.get(rid), stops_after=True):
                    commands[rid] = empty_command(dribbler_on=True)
                    mem.replan = True  # out of carry: the carrier must pass, shoot or clear
                else:
                    commands[rid] = move(
                        game, mc, rid, plan.target, game.friendly_robots[rid].p.angle_to(ball), dribbling=True
                    )
            elif at_ball:
                # Off-ball action but the ball came to this robot: take it and keep it until the
                # next decision, which the carrier change has already brought forward.
                if on_ball:
                    commands[rid] = empty_command(dribbler_on=True)
                else:
                    commands[rid] = go_to_ball(game=game, motion_controller=mc, robot_id=rid, ctx=ctx)
            elif action == "get_ball":
                if ball_in_own_defense_area(game):
                    commands[rid] = go_to_point(game, mc, rid, own_defense_area_exit_point(game, ball.y))
                elif rid == blocker_id and len(chasers) > 1:
                    # A teammate tackles: hold the ball's line to our goal just in front of it.
                    # `block_attacker` lunges at the ball inside its steal range and opens the line.
                    commands[rid] = go_to_point(game, mc, rid, shot_line_point(game, BLOCK_HOLD_M))
                elif rid == blocker_id:
                    # Challenge from the goal side: on the carrier's shot line, lunging only once
                    # in steal range. Chasing the ball from behind leaves the shot open.
                    commands[rid] = block_attacker(game, mc, rid, holder, attacker_has_ball=True)
                else:
                    commands[rid] = go_to_ball(game=game, motion_controller=mc, robot_id=rid, ctx=ctx)
            elif action.startswith("move:") or action == "hold":
                commands[rid] = go_to_point(game, mc, rid, plan.target)
            elif action.startswith("mark:") and int(action.split(":")[1]) in game.enemy_robots:
                target = clamp_to_field(
                    clamp_outside_own_defense_area(game, mark_target(game, int(action.split(":")[1]))), game
                )
                commands[rid] = go_to_point(game, mc, rid, target)
            elif action == "block_shot":
                commands[rid] = defend_parameter(game, mc, rid, defender_group=blockers) or empty_command()
            else:
                commands[rid] = empty_command()
        return commands

    @staticmethod
    def _clear(game: Game, mc: MotionController, rid: int) -> RobotCommand:
        """Kick long and wide: upfield, towards the nearer touch line. Straight upfield
        (`kick_upfield`) from our own half rolls over their goal line too often: an aimless kick,
        their free kick from where we kicked."""
        me = game.friendly_robots[rid].p
        side = 1.0 if game.ball.p.y >= 0 else -1.0
        target = Vector2D(me.x + fs.attack_dir(game) * 3.0, side * (game.field.half_width + 1.0))
        oren = me.angle_to(target)
        if oriented_towards(game, rid, oren):
            return kick()
        return turn_on_spot(game=game, motion_controller=mc, robot_id=rid, target_oren=oren, dribbling=True)

    def _shoot(self, game: Game, ctx: TickContext, rid: int, mem: FlatMem) -> RobotCommand:
        cmd, _, best_y = _score_goal(game, ctx, rid, mem.shot_y.get(rid))
        mem.shot_y[rid] = best_y
        if best_y is None and carry_exhausted(game, mem.carry_origins.get(rid), stops_after=True):
            # No lane and no carry left: edging further would be an excessive-dribbling foul, so
            # shoot at the middle of the goal now.
            goal_x, y1, y2 = enemy_goal_line(game)
            me = game.friendly_robots[rid].p
            oren = me.angle_to(Vector2D(goal_x, (y1 + y2) / 2.0))
            if oriented_towards(game, rid, oren) and has_ball(game, rid):
                return kick()
            return turn_on_spot(
                game=game, motion_controller=ctx.motion_controller, robot_id=rid, target_oren=oren, dribbling=True
            )
        return cmd

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


# --- free kicks ----------------------------------------------------------------------------------
#
# The built-in steps leave a free kick's defenders frozen wherever play stopped (`DirectFreeTheirsStep`
# only clears the keep-out circle) and line our kicker up at one of their robots. These replace them for
# this engine: the model plays open play, set pieces are formations.

WALL_DIST_M = 0.65  # wall centre from the ball: outside the 0.5 m keep-out (KeepOutRule, _free_kick_ready)


@dataclass
class _Shim:
    """The three attributes `custom_referee/actions.py`'s steps and helpers read and write."""

    game: Game
    motion_controller: MotionController
    cmd_map: dict[RobotId, RobotCommand] = field(default_factory=dict)


class _AimedFreeKick(DirectFreeOursStep):
    """The built-in kicker approach, lined up at the widest gap in their goal instead of at a robot,
    so the kick can go the moment play restarts."""

    def _kick_target_enemy(self, game: Game, ball_pos: Vector2D):
        goal_x, y1, y2 = enemy_goal_line(game)
        best_y, _ = find_best_shot(ball_pos, list(game.enemy_robots.values()), goal_x, y1, y2)
        return None, SimpleNamespace(p=Vector2D(goal_x, best_y if best_y is not None else (y1 + y2) / 2.0))


def their_free_kick_targets(game: Game, outfield: list[int]) -> dict[int, Vector2D]:
    """Two robots in a wall on the ball's line to our goal, the rest marking the opponents nearest our
    goal (not their kicker or goalkeeper)."""
    ball = game.ball.p.to_2d()
    goal = Vector2D(game.field.my_goal_line[0][0], 0.0)
    d = goal - ball
    length = max(d.mag(), 1e-6)
    u = Vector2D(d.x / length, d.y / length)
    n = Vector2D(-u.y, u.x)
    centre = Vector2D(ball.x + u.x * WALL_DIST_M, ball.y + u.y * WALL_DIST_M)
    spots = [Vector2D(centre.x + n.x * 0.1, centre.y + n.y * 0.1), Vector2D(centre.x - n.x * 0.1, centre.y - n.y * 0.1)]
    spots = [clamp_outside_own_defense_area(game, s) for s in spots]
    targets: dict[int, Vector2D] = {}
    free = list(outfield)
    for spot in spots:
        if not free:
            break
        rid = min(free, key=lambda r: game.friendly_robots[r].p.distance_to(spot))
        targets[rid] = spot
        free.remove(rid)
    keeper = fs.enemy_keeper_id(game)
    kicker = min(game.enemy_robots, key=lambda e: game.enemy_robots[e].p.distance_to(ball), default=None)
    threats = sorted(
        (e for e in game.enemy_robots if e not in (keeper, kicker)),
        key=lambda e: game.enemy_robots[e].p.distance_to(goal),
    )
    for eid in threats:
        if not free:
            break
        rid = min(free, key=lambda r: game.friendly_robots[r].p.distance_to(game.enemy_robots[eid].p))
        targets[rid] = clamp_to_field(clamp_outside_own_defense_area(game, mark_target(game, eid)), game)
        free.remove(rid)
    for rid in free:
        targets[rid] = clamp_outside_own_defense_area(game, Vector2D((ball.x + goal.x) / 2.0, (ball.y + goal.y) / 2.0))
    return targets


def free_kick_overrides(outfield_robot_ids: tuple[int, ...]) -> dict[RefereeCommand, Callable]:
    """`Strategy(referee_overrides=...)` for both teams' free kicks. Our goalkeeper is left out of
    every returned command map, so its pinned tactic keeps covering the goal."""
    kick = _AimedFreeKick()

    def theirs(game: Game, mc: MotionController) -> dict[RobotId, RobotCommand]:
        outfield = [rid for rid in sorted(game.friendly_robots) if rid in outfield_robot_ids]
        shim = _Shim(game, mc)
        others = {rid for rid in game.friendly_robots if rid not in outfield}
        _clear_to_legal_positions(
            shim,
            ball_keep_dist=WALL_DIST_M - 0.05,
            intended_targets=their_free_kick_targets(game, outfield),
            exempt_robot_ids=others,
        )
        return shim.cmd_map

    def ours(game: Game, mc: MotionController) -> dict[RobotId, RobotCommand]:
        shim = _Shim(game, mc)
        kick.blackboard = shim
        kick.update()
        kicker = kick._kicker_sticky.current
        outfield = [rid for rid in sorted(game.friendly_robots) if rid in outfield_robot_ids and rid != kicker]
        ball = game.ball.p.to_2d()
        by_dist = sorted(outfield, key=lambda rid: game.friendly_robots[rid].p.distance_to(ball))
        taken = [ball]
        cmds = {kicker: shim.cmd_map[kicker]} if kicker in shim.cmd_map else {}
        for rid, cell in zip(by_dist[:2], ("attacking_left", "attacking_right")):
            target = open_point(game, cell, rid, taken)
            taken.append(target)
            cmds[rid] = go_to_point(game, mc, rid, target)
        backs = by_dist[2:]
        for rid in backs:
            cmds[rid] = defend_parameter(game, mc, rid, defender_group=backs) or empty_command()
        return cmds

    out = {}
    for yellow_cmds, blue_cmds in (
        (
            (RefereeCommand.DIRECT_FREE_YELLOW, RefereeCommand.INDIRECT_FREE_YELLOW),
            (RefereeCommand.DIRECT_FREE_BLUE, RefereeCommand.INDIRECT_FREE_BLUE),
        ),
    ):
        for c in yellow_cmds:
            out[c] = lambda g, mc: ours(g, mc) if g.my_team_is_yellow else theirs(g, mc)
        for c in blue_cmds:
            out[c] = lambda g, mc: theirs(g, mc) if g.my_team_is_yellow else ours(g, mc)
    return out


def build_openjev_flat_kernel_strategy(
    outfield_robot_ids: tuple[int, ...],
    *,
    url: str = "http://127.0.0.1:3000",
    decision_hz: float = 1.0,
    log_path: Optional[str | Path] = None,
    client: Optional[OpenJevClient] = None,
    policy: str = "openjev",
    call: bool = True,
    on_build: Optional[Callable[[FlatJevTactic], None]] = None,
):
    """OpenJev controls every outfield robot (see the module docstring). Same shape as the
    `build_*_kernel_strategy` factories; `policy="scripted"` needs no server, `call=False` leaves the
    coaching's call out of the questions."""

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        jev_client = client or (OpenJevClient(url) if policy == "openjev" else None)
        tactic = FlatJevTactic(
            jev_client,
            decision_hz=decision_hz,
            log_path=str(log_path) if log_path else None,
            policy=policy,
            call=call,
        )
        if on_build is not None:
            on_build(tactic)
        return KernelSchedulerStrategy(
            tactics={"flat": tactic},
            partitioner=KernelSchedulerStrategy.single_tactic_picker(lambda game, prev: "flat"),
            outfield_robot_ids=outfield_robot_ids,
            ctx=TickContext(motion_controller=motion_controller),
            referee_overrides=free_kick_overrides(outfield_robot_ids),
        )

    return _build
