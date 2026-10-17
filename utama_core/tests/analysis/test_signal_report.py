from tools.signal_report import build, markdown


def _result(a: str, b: str, score_a: int, score_b: int) -> dict:
    return {
        "config_a": f"build_{a}_kernel_strategy",
        "config_b": f"build_{b}_kernel_strategy",
        "score_a": score_a,
        "score_b": score_b,
        "stats": {"shots": {"friendly": 2, "enemy": 1}, "pass_progress_m": [1.5], "enemy_pass_progress_m": [0.2]},
    }


def test_build_writes_every_figure(tmp_path):
    shot = {"side": "friendly", "distance_m": 2.2, "angle_deg": 10.0, "open_goal": 0.6, "scored": True}
    chances = {
        "shots": [shot, {**shot, "side": "enemy", "scored": False}],
        "unshot_goals": {"friendly": 0, "enemy": 0},
        "regains": [{"side": "friendly", "t": 1.0, "shot_after_s": 2.0}],
        "danger": {"friendly": {"s": 5.0, "spells": 1}, "enemy": {"s": 9.0, "spells": 2}},
        "free_kicks": [{"side": "enemy", "t": 3.0, "attacking_third": False, "shot_after_s": None}],
    }
    losses = [
        {
            "match": "x_vs_y",
            "turnovers": [
                {"kind": "tackled", "t": 0.0, "tactic": "T", "just_restarted": False, "regained_after_s": None}
            ],
            "restarts": [],
            "chances": chances,
        }
    ]
    summary = {
        "results": [_result("x", "y", 1, 0)],
        "fouls": {"crashing": {"total": 1, "inferred": 0, "by_tactic": {"x/T": 1}}},
    }

    table = build(summary, losses, tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "ball_losses.png",
        "fouls.png",
        "results.png",
        "shot_quality.png",
        "signals.png",
    ]
    assert table["build_x_kernel_strategy"]["chances"]["regain_to_shot"] == 1.0
    # Shots faced are the other side's shots: x (config_a, friendly) faced the enemy's 1, y faced 2.
    assert table["build_x_kernel_strategy"]["shots_faced"] == 1
    assert table["build_y_kernel_strategy"]["shots_faced"] == 2
    rows = markdown(table).splitlines()
    assert rows[2].startswith("| `x` | 3.00 | 1.0 | 0.0 | 2.0 | 1.0 | 50% | 0% |")
