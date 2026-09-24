from utama_core.replay.turnover_breakdown import breakdown


def _turnover(kind: str, regained_after_s, tactic: str = "GiveAndGoTactic") -> dict:
    return {"kind": kind, "t": 0.0, "tactic": tactic, "just_restarted": False, "regained_after_s": regained_after_s}


def test_breakdown_counts_only_real_losses():
    """A turnover won back within 1.0s is two robots on one ball (the nearest robot
    flips), and a `during_stoppage` handover is already counted as the foul/ball-out
    restart that caused it — neither is a real loss. Everything else is, including
    restarts given away."""
    results = [
        {
            "match": "a_vs_b",
            "acc_turnovers": 5,
            "turnovers": [
                _turnover("tackled", 1.0),  # flicker: excluded
                _turnover("tackled", 1.01, tactic="PressAndContainTactic"),
                _turnover("tackled", None, tactic="PressAndContainTactic"),  # never won back
                _turnover("pass_intercepted", 4.0),
                _turnover("during_stoppage", None, tactic="restart override"),  # excluded
            ],
            "restarts": [
                {"kind": "foul", "t": 0.0, "rule": "Excessive dribbling", "tactic": "GiveAndGoTactic"},
                {"kind": "ball_out_after_kick", "t": 0.0, "rule": "unknown", "tactic": "GiveAndGoTactic"},
            ],
        },
        {"match": "c_vs_d", "acc_turnovers": 0, "turnovers": [], "restarts": []},
    ]

    b = breakdown(results)

    assert b["matches"] == 2
    assert b["raw_turnovers"] == 5
    assert b["real_losses"] == 5
    assert b["real_losses_per_match"] == 2.5
    assert b["by_kind"] == {"tackled": 2, "pass_intercepted": 1, "foul": 1, "ball_out_after_kick": 1}
    assert b["fouls_by_rule"] == {"Excessive dribbling": 1}
    assert b["by_tactic"] == {"GiveAndGoTactic": 3, "PressAndContainTactic": 2}
