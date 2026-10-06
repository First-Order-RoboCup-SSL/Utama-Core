"""Chances each side creates and concedes, over a tournament run's replays.

`turnover_breakdown.analyse_match` feeds every replay frame to a `ChanceTracker`, so this
costs no extra pass over the replay. Its record is under `chances` in each match's ball-loss
record; `round_robin.py` totals it per strategy (`side_totals`, `rates`) into
`summary.json`'s `strategies[*].chances`. Unlike the ball-loss breakdown, everything here is
measured for both sides: it needs positions and possession, not the intentions log.

- **Shots** are `MatchStats`' own (a hard ball from the attacking third heading into the
  goal mouth), each with where it was struck from: the ball where the shooting side last
  held it, its distance and angle to the goal centre, and the share of the goal mouth no
  opponent blocked from there (`open_goal`). A shot `scored` if its side scored within
  `SHOT_GOAL_WINDOW_S`; goals no shot preceded (deflections, slow rolls) are `unshot_goals`.
- **Save rate** = 1 - goals from shots / shots faced. Every strategy fields the same
  `GoalkeeperTactic`, so it measures the shots a defense allows more than the keeper.
- **Regains**: the ball won from the opponent in open play (not within
  `JUST_RESTARTED_S` of a restart) and kept `REGAIN_MIN_HOLD_S` — shorter is two robots on
  one ball flipping "nearest". `shot_after_s`: how soon the side shot, within `CHANCE_WINDOW_S`.
- **Danger**: seconds the opponent held the ball in a side's defensive third (the third
  the shot detector calls attacking) in live play, and how many separate spells.
- **Free kicks**: every `DIRECT_FREE_*` (corner and goal kicks included) that reached
  NORMAL_START, whether it was in the kicking side's attacking third, and how soon that side
  shot, within `CHANCE_WINDOW_S`.
"""

from __future__ import annotations

import math
from typing import Optional

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.config.physical_constants import BALL_RADIUS, ROBOT_RADIUS
from utama_core.engine.match_stats import _SHOT_ATTACKING_THIRD_M
from utama_core.entities.referee.referee_command import RefereeCommand

SIDES = ("friendly", "enemy")
LIVE = {RefereeCommand.NORMAL_START, RefereeCommand.FORCE_START}
SHOT_GOAL_WINDOW_S = 4.0
CHANCE_WINDOW_S = 10.0
REGAIN_MIN_HOLD_S = 1.0  # turnover_breakdown.FLICKER_S
JUST_RESTARTED_S = 3.0  # turnover_breakdown._JUST_RESTARTED_S
# How far from its goal line a side's defensive third reaches: where the shot detector's
# attacking third starts, seen from the other end.
THIRD_DEPTH_M = STANDARD_FIELD_DIMS.full_field_half_length - _SHOT_ATTACKING_THIRD_M
# Shot origins this stale fall back to where the ball was when the shot was detected.
_ORIGIN_MAX_AGE_S = 1.0
_GOAL_SAMPLES = 21
_FREE_KICKS = {RefereeCommand.DIRECT_FREE_YELLOW: True, RefereeCommand.DIRECT_FREE_BLUE: False}  # -> yellow


def _other(side: str) -> str:
    return "enemy" if side == "friendly" else "friendly"


def open_goal(origin: tuple[float, float], goal_x: float, blockers: list[tuple[float, float]]) -> float:
    """Share of the goal mouth at `goal_x` a ball from `origin` reaches without passing
    within a robot radius plus a ball radius of any of `blockers`."""
    half = STANDARD_FIELD_DIMS.half_goal_width
    reach = ROBOT_RADIUS + BALL_RADIUS
    ox, oy = origin
    clear = 0
    for i in range(_GOAL_SAMPLES):
        gx, gy = goal_x, -half + 2 * half * i / (_GOAL_SAMPLES - 1)
        dx, dy = gx - ox, gy - oy
        length_sq = dx * dx + dy * dy
        blocked = False
        for bx, by in blockers:
            s = 0.0 if length_sq == 0 else max(0.0, min(1.0, ((bx - ox) * dx + (by - oy) * dy) / length_sq))
            if math.hypot(ox + s * dx - bx, oy + s * dy - by) <= reach:
                blocked = True
                break
        clear += not blocked
    return clear / _GOAL_SAMPLES


class ChanceTracker:
    """Fed one replay frame at a time, after `MatchStatsAccumulator.record_tick` saw it."""

    def __init__(self):
        self.shots: list[dict] = []
        self.unshot_goals = {s: 0 for s in SIDES}
        self.regains: list[dict] = []
        self.danger = {s: {"s": 0.0, "spells": 0} for s in SIDES}
        self.free_kicks: list[dict] = []
        self._last_hold: dict[str, dict] = {}  # side -> where and when it last held the ball
        self._spell: Optional[dict] = None  # the current possession spell
        self._in_danger = {s: False for s in SIDES}
        self._pending_free_kick: Optional[str] = None
        self._score: Optional[dict[str, int]] = None
        self._prev_ts: Optional[float] = None

    def step(self, frame, acc, cmd, live_since: float, shots_before: dict[str, int]) -> None:
        ts = frame.ts
        dt = ts - self._prev_ts if self._prev_ts is not None and 0.0 < ts - self._prev_ts < 1.0 else 0.0
        self._prev_ts = ts
        live = cmd in LIVE
        own_goal_sign = 1.0 if frame.my_team_is_right else -1.0
        goal_x = {  # the goal each side attacks
            "friendly": -own_goal_sign * STANDARD_FIELD_DIMS.full_field_half_length,
            "enemy": own_goal_sign * STANDARD_FIELD_DIMS.full_field_half_length,
        }
        ball = frame.ball
        side = acc._poss_side

        if side is not None and acc._poss_robot_id is not None and ball is not None:
            blockers = frame.enemy_robots if side == "friendly" else frame.friendly_robots
            self._last_hold[side] = {
                "t": ts,
                "xy": (ball.p.x, ball.p.y),
                "blockers": [(r.p.x, r.p.y) for r in blockers.values()],
            }

        self._track_spell(side, ts, live, live_since)
        self._track_free_kick(frame, cmd, ts, goal_x)

        for s in SIDES:
            if acc._shots[s] > shots_before[s] and ball is not None:
                self._shot(s, ts, ball, frame, goal_x[s])

        self._track_score(frame, ts)

        if ball is not None:
            for defender in SIDES:
                attacker = _other(defender)
                in_third = live and side == attacker and abs(ball.p.x - goal_x[attacker]) < THIRD_DEPTH_M
                if in_third:
                    self.danger[defender]["s"] += dt
                    self.danger[defender]["spells"] += not self._in_danger[defender]
                self._in_danger[defender] = in_third

    def _track_spell(self, side: Optional[str], ts: float, live: bool, live_since: float) -> None:
        # A stoppage ends the spell even if the same side has the ball after it: a shot from
        # the restart is the restart's, not the open-play regain's.
        if self._spell is not None and side == self._spell["side"] and live:
            return
        if self._spell is not None and self._spell["regain"]:
            spell = self._spell
            if ts - spell["t"] >= REGAIN_MIN_HOLD_S or spell["shot_after_s"] is not None:
                self.regains.append(
                    {"side": spell["side"], "t": round(spell["t"], 2), "shot_after_s": spell["shot_after_s"]}
                )
        regain = self._spell is not None and side is not None and live and ts - live_since > JUST_RESTARTED_S
        self._spell = (
            None if side is None or not live else {"side": side, "t": ts, "regain": regain, "shot_after_s": None}
        )

    def _track_free_kick(self, frame, cmd, ts: float, goal_x: dict[str, float]) -> None:
        if cmd in _FREE_KICKS:
            yellow = _FREE_KICKS[cmd]
            self._pending_free_kick = "friendly" if yellow == frame.my_team_is_yellow else "enemy"
        elif cmd == RefereeCommand.NORMAL_START and self._pending_free_kick is not None:
            side, self._pending_free_kick = self._pending_free_kick, None
            ball = frame.ball
            attacking = ball is not None and abs(ball.p.x - goal_x[side]) < THIRD_DEPTH_M
            self.free_kicks.append(
                {"side": side, "t": round(ts, 2), "attacking_third": attacking, "shot_after_s": None}
            )
        elif cmd not in (RefereeCommand.STOP, RefereeCommand.HALT, None):
            self._pending_free_kick = None

    def _shot(self, side: str, ts: float, ball, frame, goal_x: float) -> None:
        hold = self._last_hold.get(side)
        if hold is not None and ts - hold["t"] <= _ORIGIN_MAX_AGE_S:
            xy, blockers = hold["xy"], hold["blockers"]
        else:
            opponents = frame.enemy_robots if side == "friendly" else frame.friendly_robots
            xy, blockers = (ball.p.x, ball.p.y), [(r.p.x, r.p.y) for r in opponents.values()]
        along = abs(goal_x - xy[0])
        self.shots.append(
            {
                "side": side,
                "t": round(ts, 2),
                "distance_m": round(math.hypot(along, xy[1]), 2),
                "angle_deg": round(math.degrees(math.atan2(abs(xy[1]), along)), 1),
                "open_goal": round(open_goal(xy, goal_x, blockers), 2),
                "scored": False,
            }
        )
        spell = self._spell
        if spell is not None and spell["side"] == side and spell["regain"] and spell["shot_after_s"] is None:
            if ts - spell["t"] <= CHANCE_WINDOW_S:
                spell["shot_after_s"] = round(ts - spell["t"], 2)
        for fk in reversed(self.free_kicks):
            if fk["side"] == side:
                if fk["shot_after_s"] is None and ts - fk["t"] <= CHANCE_WINDOW_S:
                    fk["shot_after_s"] = round(ts - fk["t"], 2)
                break

    def _track_score(self, frame, ts: float) -> None:
        ref = frame.referee
        if ref is None:
            return
        yellow, blue = ref.yellow_team.score, ref.blue_team.score
        score = {"friendly": yellow, "enemy": blue} if frame.my_team_is_yellow else {"friendly": blue, "enemy": yellow}
        if self._score is not None:
            for side in SIDES:
                for _ in range(score[side] - self._score[side]):
                    shot = next(
                        (
                            s
                            for s in reversed(self.shots)
                            if s["side"] == side and not s["scored"] and ts - s["t"] <= SHOT_GOAL_WINDOW_S
                        ),
                        None,
                    )
                    if shot is not None:
                        shot["scored"] = True
                    else:
                        self.unshot_goals[side] += 1
        self._score = score

    def result(self) -> dict:
        self._track_spell(None, self._prev_ts or 0.0, False, 0.0)  # close the last spell
        return {
            "shots": self.shots,
            "unshot_goals": self.unshot_goals,
            "regains": self.regains,
            "danger": {s: {"s": round(d["s"], 2), "spells": d["spells"]} for s, d in self.danger.items()},
            "free_kicks": self.free_kicks,
        }


def side_totals(record: dict, side: str) -> dict[str, float]:
    """Additive counts for `side` in one match's `ChanceTracker.result()`; sum them over
    matches and pass the sum to `rates`."""
    other = _other(side)
    shots = [s for s in record["shots"] if s["side"] == side]
    faced = [s for s in record["shots"] if s["side"] == other]
    regains = [r for r in record["regains"] if r["side"] == side]
    kicks = [k for k in record["free_kicks"] if k["side"] == side]
    attacking = [k for k in kicks if k["attacking_third"]]
    converted = [r for r in regains if r["shot_after_s"] is not None]
    return {
        "matches": 1,
        "shots": len(shots),
        "shots_scored": sum(s["scored"] for s in shots),
        "unshot_goals": record["unshot_goals"][side],
        "shot_distance_m": sum(s["distance_m"] for s in shots),
        "shot_open_goal": sum(s["open_goal"] for s in shots),
        "shots_faced": len(faced),
        "shots_faced_scored": sum(s["scored"] for s in faced),
        "faced_open_goal": sum(s["open_goal"] for s in faced),
        "regains": len(regains),
        "regains_to_shot": len(converted),
        "regain_to_shot_s": sum(r["shot_after_s"] for r in converted),
        "danger_s": record["danger"][side]["s"],
        "danger_spells": record["danger"][side]["spells"],
        "free_kicks": len(kicks),
        "free_kicks_to_shot": sum(k["shot_after_s"] is not None for k in kicks),
        "attacking_free_kicks": len(attacking),
        "attacking_free_kicks_to_shot": sum(k["shot_after_s"] is not None for k in attacking),
    }


def add(total: dict[str, float], part: dict[str, float]) -> None:
    for k, v in part.items():
        total[k] = total.get(k, 0) + v


def rates(t: dict[str, float]) -> dict[str, Optional[float]]:
    """Readable rates from summed `side_totals`; None where nothing was measured."""

    def ratio(a: str, b: str, digits: int = 2) -> Optional[float]:
        return round(t[a] / t[b], digits) if t.get(b) else None

    return {
        "matches": t.get("matches", 0),
        "shots": t.get("shots", 0),
        "conversion": ratio("shots_scored", "shots"),
        "unshot_goals": t.get("unshot_goals", 0),
        "shot_distance_m": ratio("shot_distance_m", "shots"),
        "shot_open_goal": ratio("shot_open_goal", "shots"),
        "shots_faced": t.get("shots_faced", 0),
        "save_rate": round(1 - t["shots_faced_scored"] / t["shots_faced"], 2) if t.get("shots_faced") else None,
        "faced_open_goal": ratio("faced_open_goal", "shots_faced"),
        "regains": t.get("regains", 0),
        "regain_to_shot": ratio("regains_to_shot", "regains"),
        "regain_to_shot_s": ratio("regain_to_shot_s", "regains_to_shot", 1),
        "danger_s_per_match": ratio("danger_s", "matches", 1),
        "danger_spells_per_match": ratio("danger_spells", "matches", 1),
        "free_kicks": t.get("free_kicks", 0),
        "free_kick_to_shot": ratio("free_kicks_to_shot", "free_kicks"),
        "attacking_free_kick_to_shot": ratio("attacking_free_kicks_to_shot", "attacking_free_kicks"),
    }
