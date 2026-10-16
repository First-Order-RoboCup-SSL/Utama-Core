from types import SimpleNamespace as NS

import numpy as np
import pytest

from utama_core.analysis.realism import (
    passes,
    reaction,
    scorecard,
    shot_outcomes,
    shots,
    xg,
)
from utama_core.entities.referee.referee_command import RefereeCommand as C
from utama_core.entities.referee.stage import Stage

_DT = 1 / 60
_KEEPER = 0  # both sides' keeper id


def _robot(x, y):
    return NS(p=NS(x=x, y=y))


def _frame(ts, bx, by, vx=0.0, vy=0.0, friendly=None, enemy=None, cmd=C.NORMAL_START, score=(0, 0)):
    """Friendly is yellow and attacks the +x goal."""
    referee = NS(
        referee_command=cmd,
        stage=Stage.NORMAL_FIRST_HALF,
        yellow_team=NS(score=score[0], goalkeeper=_KEEPER),
        blue_team=NS(score=score[1], goalkeeper=_KEEPER),
        game_events=[],
    )
    return NS(
        ts=ts,
        ball=NS(p=NS(x=bx, y=by), v=NS(x=vx, y=vy)),
        friendly_robots=friendly or {},
        enemy_robots=enemy or {},
        my_team_is_right=False,
        my_team_is_yellow=True,
        referee=referee,
    )


def _kick(x0, y0, vx, vy, rest_s=0.3, roll_s=1.0, goal_at_s=None, **kw):
    """A ball at rest at (x0, y0), then kicked at (vx, vy) and rolling without slowing; the
    friendly score rises `goal_at_s` after the kick if given."""
    frames, t = [], 0.0
    while t < rest_s:
        frames.append(_frame(t, x0, y0, **kw))
        t += _DT
    t_kick = t
    while t < rest_s + roll_s:
        dt = t - t_kick
        score = (1, 0) if goal_at_s is not None and dt >= goal_at_s else (0, 0)
        frames.append(_frame(t, x0 + vx * dt, y0 + vy * dt, vx, vy, score=score, **kw))
        t += _DT
    return frames, t_kick


def test_a_hard_kick_at_the_goal_mouth_is_one_shot_from_where_the_ball_rested():
    frames, t_kick = _kick(2.0, 0.0, 4.0, 0.0)
    found = shots.scan(frames)["shots"]
    assert len(found) == 1
    s = found[0]
    assert s["side"] == "friendly" and s["context"] == "open_play"
    assert s["t"] == pytest.approx(t_kick, abs=1e-3)
    assert s["distance_m"] == pytest.approx(2.5)
    assert s["peak"] == pytest.approx(4.0)


def test_a_kick_just_outside_the_on_target_margin_or_too_slow_is_not_a_shot():
    # from x = 2.0 the line reaches the goal line after 2.5 m: aim at y = 0.66 vs 0.5 + 0.15
    wide, _ = _kick(2.0, 0.0, 4.0, 4.0 * 0.66 / 2.5)
    inside, _ = _kick(2.0, 0.0, 4.0, 4.0 * 0.64 / 2.5)
    slow, _ = _kick(2.0, 0.0, shots.KICK_MPS - 0.1, 0.0)
    assert shots.scan(wide)["shots"] == []
    assert len(shots.scan(inside)["shots"]) == 1
    assert shots.scan(slow)["shots"] == []


def test_a_kick_in_a_stoppage_is_not_a_shot():
    frames, _ = _kick(2.0, 0.0, 4.0, 0.0, cmd=C.STOP)
    assert shots.scan(frames)["shots"] == []


def test_a_sim_goal_goes_to_the_shot_before_the_ball_entered_the_goal():
    # 2.5 m at 4 m/s: in the goal 0.625 s after the kick; our referee awards 0.3 s later
    frames, _ = _kick(2.0, 0.0, 4.0, 0.0, goal_at_s=0.68)
    scanned = shots.scan(frames)
    links = shots.link_goals(scanned)
    assert links["link_source"] == "ball_in_goal"
    assert links["unshot_goals"] == {"friendly": 0, "enemy": 0}
    assert scanned["shots"][0]["scored"]
    shot_outcomes.follow(frames, scanned["shots"])
    assert scanned["shots"][0]["outcome"] == "goal"


def _shot(t, x, y, side="friendly"):
    return {"side": side, "t": t, "raw_x": x, "raw_y": y, "scored": False}


def _event(kind, t, kick, origin=None):
    return {"type": kind, "t": t, "goal_sign": 1.0, "kick": kick, "message": None, "origin": origin}


def _real(shot_list, entry_t, possible_t, goal_t, kick):
    return {
        "shots": shot_list,
        "in_goal_events": [(entry_t, "friendly")],
        "game_events": [_event("POSSIBLE_GOAL", possible_t, kick), _event("GOAL", goal_t, kick)],
        "score_events": [(goal_t, "friendly", False)],
    }


def test_a_late_operator_confirmed_goal_goes_to_the_shot_at_the_autorefs_kick_location():
    near, later = _shot(100.0, 3.0, 0.2), _shot(104.0, 1.0, -1.0)
    # the operator accepts 44.5 s after POSSIBLE_GOAL; the later shot elsewhere is not it
    links = shots.link_goals(_real([near, later], 105.0, 105.5, 150.0, (3.0, 0.25)))
    assert near["scored"] and not later["scored"] and "kick_mismatch" not in near
    assert links["link_source"] == "game_events"
    assert links["unshot_goals"]["friendly"] == 0
    assert links["confirm_lags"] == [44.5]


def test_a_hand_entered_possible_goal_still_finds_the_ball_entry_up_to_the_manual_window():
    shot = _shot(100.0, 3.0, 0.2)
    entry = 101.0
    in_window = _real([dict(shot)], entry, entry + shots.MANUAL_GOAL_S, entry + shots.MANUAL_GOAL_S + 5, (3.0, 0.2))
    too_late = _real([dict(shot)], entry, entry + shots.MANUAL_GOAL_S + 1, entry + shots.MANUAL_GOAL_S + 6, (3.0, 0.2))
    shots.link_goals(in_window)
    links = shots.link_goals(too_late)
    assert in_window["shots"][0]["scored"]
    # without the entry the crossing time is the event's, far past the shot: missed
    assert not too_late["shots"][0]["scored"]
    assert links["missed"][0]["entry_found"] is False


def test_with_no_shot_at_the_kick_location_the_sides_latest_shot_scores_flagged_as_a_mismatch():
    first, last = _shot(100.0, 3.0, 0.2), _shot(103.0, 2.0, 1.0)
    links = shots.link_goals(_real([first, last], 104.0, 104.5, 110.0, (4.2, -0.3)))
    assert last["scored"] and last["kick_mismatch"] and not first["scored"]
    assert links["missed"] == []


def test_a_score_rise_with_no_goal_event_on_that_goal_is_unlinked():
    s = _shot(100.0, 3.0, 0.2)
    scanned = _real([s], 101.0, 101.5, 120.0, (3.0, 0.2))
    scanned["score_events"] = [(130.0, "friendly", False)]  # more than 1 s from the GOAL event
    links = shots.link_goals(scanned)
    assert not s["scored"] and links["unshot_goals"]["friendly"] == 1


def test_the_first_robot_the_shot_reaches_names_its_outcome():
    keeper = {_KEEPER: _robot(4.0, 0.0)}
    frames, _ = _kick(2.0, 0.0, 4.0, 0.0, enemy=keeper)
    scanned = shots.scan(frames)
    shot_outcomes.follow(frames, scanned["shots"])
    s = scanned["shots"][0]
    assert s["outcome"] == "keeper"
    assert s["keeper_depth_m"] == pytest.approx(0.5)


def test_a_shot_inside_the_margin_but_outside_the_post_ends_wide():
    frames, _ = _kick(2.0, 0.0, 4.0, 4.0 * 0.6 / 2.5)
    scanned = shots.scan(frames)
    shot_outcomes.follow(frames, scanned["shots"])
    assert scanned["shots"][0]["outcome"] == "wide"


def test_attempts_drop_passes_and_kicks_from_beyond_six_metres():
    assert shot_outcomes.is_attempt({"outcome": "keeper", "distance_m": shot_outcomes.MAX_ATTEMPT_M})
    assert not shot_outcomes.is_attempt({"outcome": "keeper", "distance_m": shot_outcomes.MAX_ATTEMPT_M + 0.01})
    assert not shot_outcomes.is_attempt({"outcome": "teammate", "distance_m": 2.0})


def test_the_logistic_fit_recovers_known_probabilities_and_survives_a_round_trip():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20000, len(xg.FEATURES)))
    true_w = np.array([-1.0, 0.8, -0.5] + [0.0] * (len(xg.FEATURES) - 2))
    p = 1 / (1 + np.exp(-(np.c_[np.ones(len(X)), X] @ true_w)))
    y = (rng.random(len(X)) < p).astype(float)
    model = xg.Logit().fit(X, y)
    assert np.abs(model.predict(X) - p).max() < 0.05
    again = xg.Logit.from_dict(model.to_dict())
    np.testing.assert_allclose(again.predict(X[:10]), model.predict(X[:10]))
    with pytest.raises(ValueError):
        xg.Logit.from_dict({**model.to_dict(), "features": ["log_dist"]})


def _attempt(scored, outcome="keeper", distance=3.0):
    return {
        "distance_m": distance,
        "angle_deg": 10.0,
        "open_goal": 0.5,
        "og_keeper": 0.7,
        "keeper_in_place": True,
        "peak": 4.0,
        "context": "open_play",
        "scored": scored,
        "outcome": "goal" if scored else outcome,
        "speed_1m": None,
    }


def test_scorecard_rates_pool_matches_and_drop_short_ones():
    model = xg.Logit().fit(np.array([xg.row(_attempt(False)), xg.row(_attempt(True))]), np.array([0.0, 1.0]))
    long_match = {"live_s": 120.0, "restarts": 4, "shots": [_attempt(True), _attempt(False)]}
    other = {"live_s": 240.0, "restarts": 2, "shots": [_attempt(False, "wide")]}
    short = {"live_s": xg.MIN_LIVE_S - 1, "restarts": 9, "shots": [_attempt(True)] * 5}
    long_match["shots"].append(_attempt(False, "teammate"))  # a pass: not an attempt
    rows = scorecard.rows(scorecard.usable([long_match, other, short]), model, n_boot=20)
    # 3 attempts over 6 live minutes for 2 teams
    assert rows["attempts per team per live minute"]["value"] == pytest.approx(3 / 12)
    assert rows["outcome: wide, post or crossed"]["value"] == pytest.approx(1 / 3)
    assert rows["outcome: wide"]["value"] == pytest.approx(1 / 3)
    assert rows["restarts per match"]["value"] == pytest.approx(3.0)


def _outcome(frames):
    scanned = shots.scan(frames)
    shot_outcomes.follow(frames, scanned["shots"])
    return scanned["shots"][0]


def _aimed_at(y):
    # from x = 2.0 the line reaches the goal line after 2.5 m; the post is at y = 0.5
    return _kick(2.0, 0.0, 4.0, 4.0 * y / 2.5)[0]


def test_where_an_unscored_shot_crosses_the_line_splits_wide_post_and_crossed_no_goal():
    wide = _outcome(_aimed_at(0.6))
    assert wide["outcome"] == "wide" and wide["miss_m"] == pytest.approx(0.1, abs=0.01)
    assert _outcome(_aimed_at(0.5 + shot_outcomes.POST_BAND_M - 0.005))["outcome"] == "post"
    assert _outcome(_aimed_at(0.5 + shot_outcomes.POST_BAND_M + 0.01))["outcome"] == "wide"
    crossed = _outcome(_aimed_at(0.3))
    assert crossed["outcome"] == "crossed_no_goal" and crossed["cross_y"] == pytest.approx(0.3, abs=0.02)


def test_a_ball_bouncing_back_off_the_post_is_post():
    frames, _ = _kick(2.0, 0.0, 4.0, 4.0 * 0.5 / 2.5)
    frames = [f for f in frames if f.ball.p.x <= 4.47]  # up to 3 cm short of the post
    last = frames[-1]
    x, y = last.ball.p.x, last.ball.p.y
    for k in range(1, 30):
        frames.append(_frame(last.ts + k * _DT, x - 3.0 * k * _DT, y, -3.0, 0.0))
    assert _outcome(frames)["outcome"] == "post"


def _keeper_frames(move_at_s, vy_after, x=4.4, y0=0.4):
    """Shot straight down y = 0 from x = 2.0 at 4 m/s; the keeper stands off the line at
    (x, y0) and from `move_at_s` after the kick moves across at `vy_after`."""
    frames, t_kick = _kick(2.0, 0.0, 4.0, 0.0)
    for f in frames:
        dt = max(0.0, f.ts - t_kick - move_at_s)
        f.enemy_robots = {_KEEPER: _robot(x, y0 + vy_after * dt)}
    return frames


def _react(frames):
    scanned = shots.scan(frames)
    reaction.measure(frames, scanned["shots"])
    return scanned["shots"][0]


def test_the_keeper_reaction_is_the_first_velocity_change_toward_the_shot_line():
    s = _react(_keeper_frames(0.2, -1.0))
    assert s["keeper_toward_s"] == pytest.approx(0.2, abs=0.04)
    assert s["keeper_react_s"] == pytest.approx(0.2, abs=0.04)
    assert s["keeper_window_s"] == pytest.approx(0.6, abs=0.03)  # the ball passes x = 4.4
    assert s["keeper_line_m"] == pytest.approx(0.4)


def test_a_keeper_moving_away_from_the_line_reacts_but_not_toward_it_and_a_still_one_not_at_all():
    away = _react(_keeper_frames(0.2, 1.0))
    assert away["keeper_react_s"] is not None and away["keeper_toward_s"] is None
    assert _react(_keeper_frames(0.0, 0.0))["keeper_react_s"] is None


def _passes(frames):
    scanned = shots.scan(frames)
    assert scanned["shots"] == []
    return passes.follow(frames, [])


def _pass_frames(enemy=None, speed=3.0):
    """Friendly robot 0 at (0, -0.1) kicks the ball from (0, 0) up the y axis; friendly robot 1
    waits at (0.1, 2.0)."""
    friendly = {0: _robot(0.0, -0.1), 1: _robot(0.1, 2.0)}
    return _kick(0.0, 0.0, 0.0, speed, friendly=friendly, enemy=enemy, roll_s=2.0)[0]


def test_a_kick_toward_a_teammate_is_a_pass_with_its_line_offset_and_completion():
    (p,) = _passes(_pass_frames())
    assert p["side"] == "friendly" and p["completed"]
    assert p["offset_m"] == pytest.approx(0.1, abs=1e-3)


def test_a_pass_softer_than_a_shot_kick_still_counts_but_one_below_the_pass_threshold_does_not():
    (p,) = _passes(_pass_frames(speed=passes.KICK_MPS + 0.1))
    assert p["completed"]
    assert _passes(_pass_frames(speed=passes.KICK_MPS - 0.1)) == []


def test_a_kick_that_is_a_shot_is_not_also_a_pass():
    frames = _pass_frames()
    kick_t = frames[passes.kicks(frames)[0][0]].ts
    assert passes.follow(frames, [kick_t - 0.05]) == []


def test_a_pass_an_opponent_reaches_first_is_incomplete_and_a_kick_to_nobody_is_no_pass():
    (p,) = _passes(_pass_frames(enemy={5: _robot(0.05, 1.0)}))
    assert not p["completed"]
    lonely, _ = _kick(0.0, 0.0, 0.0, 3.0, friendly={0: _robot(0.0, -0.1)})
    assert _passes(lonely) == []


def test_reaction_rows_use_only_shots_that_leave_time_and_count_no_reaction_in_time_as_none():
    model = xg.Logit().fit(np.array([xg.row(_attempt(False)), xg.row(_attempt(True))]), np.array([0.0, 1.0]))
    react = dict(keeper_window_s=0.6, keeper_line_m=0.3, keeper_react_s=0.2)
    attempts = [
        {**_attempt(False), **react, "keeper_toward_s": 0.2},
        {**_attempt(False), **react, "keeper_toward_s": None},
        {**_attempt(False), **react, "keeper_window_s": 0.2, "keeper_toward_s": 0.1},  # no time
    ]
    match = {"match": "m", "live_s": 120.0, "restarts": 0, "shots": attempts}
    rows = scorecard.rows(scorecard.usable([match]), model, n_boot=5)
    assert rows["keeper moves toward the shot line within 0.4 s, share"]["value"] == pytest.approx(0.5)
    assert rows["keeper moves toward the shot line, median (s)"]["value"] == pytest.approx(0.2)


def test_the_broken_tracker_match_is_excluded():
    (stem,) = scorecard.EXCLUDED_MATCHES
    assert scorecard.usable([{"match": stem, "live_s": 600.0, "shots": []}]) == []
