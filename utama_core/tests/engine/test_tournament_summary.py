from smoke_tournament import foul_table, stall_incidents, strategy_table


def _result(a: str, b: str, score_a: int, score_b: int) -> dict:
    return {
        "config_a": f"build_{a}_kernel_strategy",
        "config_b": f"build_{b}_kernel_strategy",
        "score_a": score_a,
        "score_b": score_b,
        "stats": {
            "shots": {"friendly": 3, "enemy": 1},
            "completed_passes": 5,
            "enemy_completed_passes": 2,
            "attacking_third_entries": 4,
            "enemy_attacking_third_entries": 1,
            "fouls_by_side": {"friendly": {"crashing": 2}, "enemy": {"double_touch": 1, "crashing": 1}},
        },
    }


def test_strategy_table_reads_each_side_from_its_own_perspective():
    """Match stats are config_a's view, so config_b's shots/passes/entries/fouls
    come from the enemy side. Real losses are only measured for config_a, so
    they total over `matches_as_a` alone."""
    results = [_result("x", "y", 2, 1), _result("y", "x", 0, 0)]

    table = strategy_table(
        results, real_losses_by_match={"x_vs_y": {"tackled": 3, "foul": 1}, "y_vs_x": {"pass_intercepted": 6}}
    )

    x = table["build_x_kernel_strategy"]
    assert (x["matches"], x["wins"], x["draws"], x["losses"]) == (2, 1, 1, 0)
    assert (x["goals_for"], x["goals_against"]) == (2, 1)
    assert x["shots"] == 3 + 1  # friendly in x_vs_y, enemy in y_vs_x
    assert x["completed_passes"] == 5 + 2
    assert x["attacking_third_entries"] == 4 + 1
    assert x["fouls"] == 2 + 2
    assert (x["matches_as_a"], x["real_losses_as_a"]) == (1, 4)
    assert x["real_loss_kinds_as_a"] == {"tackled": 3, "foul": 1}

    y = table["build_y_kernel_strategy"]
    assert (y["wins"], y["draws"], y["losses"]) == (0, 1, 1)
    assert (y["matches_as_a"], y["real_losses_as_a"]) == (1, 6)


def test_strategy_table_counts_stalled_matches_for_both_sides():
    """A stalled match stays in W-D-L and is counted as `stalled` for both strategies;
    so is one only the possession backstop caught."""
    stalled = _result("x", "y", 1, 0)
    stalled["stats"]["stall_events"] = [{"kind": "RESTART_STALL"}]
    backstop = _result("x", "z", 0, 0)
    backstop["possession_backstop"] = True

    table = strategy_table([stalled, backstop, _result("y", "z", 2, 0)])

    assert table["build_x_kernel_strategy"]["stalled"] == 2
    assert table["build_x_kernel_strategy"]["wins"] == 1
    assert table["build_y_kernel_strategy"]["stalled"] == 1
    assert table["build_z_kernel_strategy"]["stalled"] == 1


def test_strategy_table_without_replays_leaves_real_losses_empty():
    table = strategy_table([_result("x", "y", 1, 1)])
    assert table["build_x_kernel_strategy"]["matches_as_a"] == 0


def test_foul_table_attributes_each_side_to_its_own_strategy():
    """`side` is relative to config_a: an "enemy" foul belongs to config_b."""

    def foul(rule, side, tactic, inferred=False):
        return {"rule": rule, "side": side, "tactic": tactic, "inferred": inferred}

    results = [
        {
            "config_a": "build_alpha_kernel_strategy",
            "config_b": "build_beta_kernel_strategy",
            "stats": {
                "fouls": [
                    foul("excessive_dribbling", "friendly", "lure"),
                    foul("excessive_dribbling", "enemy", "mark"),
                    foul("excessive_dribbling", "enemy", "mark"),
                    foul("keeper_held_ball", "enemy", "goalkeeper", inferred=True),
                ]
            },
        },
        {"config_a": "build_beta_kernel_strategy", "config_b": "build_alpha_kernel_strategy", "stats": None},
    ]

    table = foul_table(results)

    assert table["excessive_dribbling"] == {"total": 3, "inferred": 0, "by_tactic": {"beta/mark": 2, "alpha/lure": 1}}
    assert table["keeper_held_ball"] == {"total": 1, "inferred": 1, "by_tactic": {"beta/goalkeeper": 1}}
    assert list(table) == ["excessive_dribbling", "keeper_held_ball"]


def _stalled(a: str, b: str, sim_time: float, duration_s: float = 11.3) -> dict:
    r = _result(a, b, 0, 0)
    r["stats"]["stall_events"] = [{"kind": "COMMITTED_FROZEN", "sim_time": sim_time, "duration_s": duration_s}]
    return r


def test_one_deterministic_freeze_against_two_opponents_is_one_incident():
    """RR 2026-09-28: overload_flow froze at t=20.5 for 11.3 s against both
    press_trigger_flow and score_aware_counter_flow -- the same freeze, since rsim is
    deterministic and neither opponent had diverged yet. The same times with no shared
    strategy, or a tick apart, are different incidents."""
    results = [
        _stalled("overload_flow", "press_trigger_flow", 20.5),
        _stalled("overload_flow", "score_aware_counter_flow", 20.5 + 1e-9),
        _stalled("tiki_taka", "zone_fluid", 20.5),
        _stalled("overload_flow", "low_block", 20.5 + 1 / 60),
        _result("overload_flow", "high_press", 0, 0),
    ]

    incidents = stall_incidents(results)

    assert [i["matches"] for i in incidents] == [
        ["overload_flow_vs_press_trigger_flow", "overload_flow_vs_score_aware_counter_flow"],
        ["tiki_taka_vs_zone_fluid"],
        ["overload_flow_vs_low_block"],
    ]
    assert incidents[0]["kind"] == "COMMITTED_FROZEN"
