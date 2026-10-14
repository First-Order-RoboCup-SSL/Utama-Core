"""Per-strategy totals over a round-robin's `summary.json`-shaped results.

Lives in `utama_core` so the analysis code can rebuild a `summary.json`'s `strategies`
table without importing the `tools/` scripts (which are not part of the installed package).
`tools/evaluation/round_robin.py` re-exports it.
"""

from __future__ import annotations

from typing import Optional

from utama_core.analysis import chances


def _short_name(config_name: str) -> str:
    """A config's name in replay file names (mirrors `tools/evaluation/match._short_name`)."""
    return config_name.removeprefix("build_").removesuffix("_kernel_strategy")


def strategy_table(
    results: list[dict],
    real_losses_by_match: Optional[dict[str, dict[str, int]]] = None,
    chances_by_match: Optional[dict[str, dict]] = None,
) -> dict[str, dict]:
    """Per-strategy totals over `summary.json`-shaped `results`. Match stats are from
    config_a's side, so config_b reads the `enemy_*` counterparts. Real ball losses are only
    measured for config_a (the side with an intentions log), so they are totalled over the
    matches a strategy played as config_a (`matches_as_a`), in total and by kind
    (`turnover_breakdown.TURNOVER_KINDS` and `RESTART_KINDS`). `stalled` counts matches that
    recorded a stall event or tripped the possession backstop: their result is still in
    W-D-L, flagged rather than dropped, since a stall can be the strategy's own fault.
    `chances` (`analysis.chances.rates`) is measured for both sides, over the matches in
    `chances_by_match`; `pass_progress_m` is the mean metres gained toward goal per
    completed pass, `forward_pass_share` the share gaining at least 1 m."""
    table: dict[str, dict] = {}
    chance_totals: dict[str, dict[str, float]] = {}
    progress: dict[str, list[float]] = {}
    for r in results:
        stats = r.get("stats") or {}
        fouls = stats.get("fouls_by_side") or {}
        tag = f"{_short_name(r['config_a'])}_vs_{_short_name(r['config_b'])}"
        for name, own, is_a in ((r["config_a"], "friendly", True), (r["config_b"], "enemy", False)):
            row = table.setdefault(
                name,
                {
                    "matches": 0,
                    "wins": 0,
                    "draws": 0,
                    "losses": 0,
                    "goals_for": 0,
                    "goals_against": 0,
                    "shots": 0,
                    "completed_passes": 0,
                    "attacking_third_entries": 0,
                    "fouls": 0,
                    "stalled": 0,
                    "matches_as_a": 0,
                    "real_losses_as_a": 0,
                    "real_loss_kinds_as_a": {},
                    "robot_overlap_ticks": 0,
                    "robot_overlap_incidents": 0,
                    "teammate_overlap_ticks": 0,
                    "teammate_overlap_incidents": 0,
                    "near_miss_ticks": 0,
                    "worst_clearance_m": None,
                },
            )
            prefix = "" if is_a else "enemy_"
            gf, ga = (r["score_a"], r["score_b"]) if is_a else (r["score_b"], r["score_a"])
            row["matches"] += 1
            row["wins" if gf > ga else "draws" if gf == ga else "losses"] += 1
            row["goals_for"] += gf
            row["goals_against"] += ga
            row["shots"] += (stats.get("shots") or {}).get(own, 0)
            row["completed_passes"] += stats.get(f"{prefix}completed_passes", 0)
            row["attacking_third_entries"] += stats.get(f"{prefix}attacking_third_entries", 0)
            row["fouls"] += sum((fouls.get(own) or {}).values())
            row["stalled"] += bool(stats.get("stall_events") or r.get("possession_backstop"))
            row["robot_overlap_ticks"] += stats.get("robot_overlap_ticks", 0)
            row["robot_overlap_incidents"] += stats.get("robot_overlap_incidents", 0)
            row["teammate_overlap_ticks"] += (stats.get("teammate_overlap_ticks") or {}).get(own, 0)
            row["teammate_overlap_incidents"] += (stats.get("teammate_overlap_incidents") or {}).get(own, 0)
            row["near_miss_ticks"] += stats.get("near_miss_ticks", 0)
            wc = stats.get("worst_clearance_m")
            if wc is not None:
                if row["worst_clearance_m"] is None or wc < row["worst_clearance_m"]:
                    row["worst_clearance_m"] = round(wc, 3)
            progress.setdefault(name, []).extend(stats.get(f"{prefix}pass_progress_m") or [])
            if chances_by_match and tag in chances_by_match:
                chances.add(chance_totals.setdefault(name, {}), chances.side_totals(chances_by_match[tag], own))
            if is_a and real_losses_by_match is not None and tag in real_losses_by_match:
                row["matches_as_a"] += 1
                kinds = real_losses_by_match[tag]
                row["real_losses_as_a"] += sum(kinds.values())
                for kind, n in kinds.items():
                    row["real_loss_kinds_as_a"][kind] = row["real_loss_kinds_as_a"].get(kind, 0) + n
    for name, row in table.items():
        gains = progress.get(name, [])
        row["pass_progress_m"] = round(sum(gains) / len(gains), 2) if gains else None
        row["forward_pass_share"] = round(sum(g >= 1.0 for g in gains) / len(gains), 2) if gains else None
        row["chances"] = chances.rates(chance_totals.get(name, {}))
    return table
