from smoke_tournament import foul_table, strategy_table


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

    table = strategy_table(results, real_losses_by_match={"x_vs_y": 4, "y_vs_x": 6})

    x = table["build_x_kernel_strategy"]
    assert (x["matches"], x["wins"], x["draws"], x["losses"]) == (2, 1, 1, 0)
    assert (x["goals_for"], x["goals_against"]) == (2, 1)
    assert x["shots"] == 3 + 1  # friendly in x_vs_y, enemy in y_vs_x
    assert x["completed_passes"] == 5 + 2
    assert x["attacking_third_entries"] == 4 + 1
    assert x["fouls"] == 2 + 2
    assert (x["matches_as_a"], x["real_losses_as_a"]) == (1, 4)

    y = table["build_y_kernel_strategy"]
    assert (y["wins"], y["draws"], y["losses"]) == (0, 1, 1)
    assert (y["matches_as_a"], y["real_losses_as_a"]) == (1, 6)


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
