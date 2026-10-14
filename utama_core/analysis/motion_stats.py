"""Offline motion planning collision and clearance analysis from replay files (.npz).

Computes pairwise physical robot overlaps, near-misses, teammate vs opponent
collision breakdowns, and worst observed clearance over a full match replay.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from utama_core.config.physical_constants import ROBOT_RADIUS


def _count_incidents(d_pairs: np.ndarray) -> int:
    """Count discrete collision incidents across pairs via rising-edge transitions (d < 0.0)."""
    if d_pairs.size == 0 or d_pairs.shape[0] == 0:
        return 0
    in_contact = d_pairs < 0.0
    starts = in_contact[0:1, :]
    trans = in_contact[1:, :] & ~in_contact[:-1, :]
    return int(np.sum(starts) + np.sum(trans))


def compute_motion_metrics_from_arrays(
    friendly_p: np.ndarray,
    enemy_p: np.ndarray,
) -> Dict[str, Union[int, float, Dict[str, int], None]]:
    """Compute motion planning collision metrics directly from position arrays.

    Parameters
    ----------
    friendly_p : np.ndarray
        Shape (T, N_f, 2), friendly robot positions over T ticks.
    enemy_p : np.ndarray
        Shape (T, N_e, 2), enemy robot positions over T ticks.

    Returns
    -------
    dict with:
        robot_overlap_ticks : int
        robot_overlap_incidents : int
        near_miss_ticks : int
        teammate_overlap_ticks : dict[str, int]
        teammate_overlap_incidents : dict[str, int]
        opponent_overlap_ticks : int
        opponent_overlap_incidents : int
        worst_clearance_m : float or None
        total_ticks : int
    """
    t_steps = friendly_p.shape[0]
    if t_steps == 0 or (friendly_p.shape[1] + enemy_p.shape[1]) < 2:
        return {
            "robot_overlap_ticks": 0,
            "robot_overlap_incidents": 0,
            "near_miss_ticks": 0,
            "teammate_overlap_ticks": {"friendly": 0, "enemy": 0},
            "teammate_overlap_incidents": {"friendly": 0, "enemy": 0},
            "opponent_overlap_ticks": 0,
            "opponent_overlap_incidents": 0,
            "worst_clearance_m": None,
            "total_ticks": t_steps,
        }

    two_r = 2.0 * ROBOT_RADIUS
    n_f = friendly_p.shape[1]
    n_e = enemy_p.shape[1]

    clearances = []

    # Friendly teammate clearances
    if n_f >= 2:
        idx_f = np.triu_indices(n_f, k=1)
        diff_ff = friendly_p[:, :, None, :] - friendly_p[:, None, :, :]
        d_ff = np.linalg.norm(diff_ff, axis=-1)[:, idx_f[0], idx_f[1]] - two_r
        clearances.append(d_ff)
        ff_overlap = int(np.sum(np.any(d_ff < 0.0, axis=1)))
        ff_incidents = _count_incidents(d_ff)
    else:
        d_ff = np.empty((t_steps, 0), dtype=np.float32)
        ff_overlap = 0
        ff_incidents = 0

    # Enemy teammate clearances
    if n_e >= 2:
        idx_e = np.triu_indices(n_e, k=1)
        diff_ee = enemy_p[:, :, None, :] - enemy_p[:, None, :, :]
        d_ee = np.linalg.norm(diff_ee, axis=-1)[:, idx_e[0], idx_e[1]] - two_r
        clearances.append(d_ee)
        ee_overlap = int(np.sum(np.any(d_ee < 0.0, axis=1)))
        ee_incidents = _count_incidents(d_ee)
    else:
        d_ee = np.empty((t_steps, 0), dtype=np.float32)
        ee_overlap = 0
        ee_incidents = 0

    # Cross (friendly vs enemy) clearances
    if n_f >= 1 and n_e >= 1:
        diff_fe = friendly_p[:, :, None, :] - enemy_p[:, None, :, :]
        d_fe = (np.linalg.norm(diff_fe, axis=-1) - two_r).reshape(t_steps, -1)
        clearances.append(d_fe)
        fe_overlap = int(np.sum(np.any(d_fe < 0.0, axis=1)))
        fe_incidents = _count_incidents(d_fe)
    else:
        d_fe = np.empty((t_steps, 0), dtype=np.float32)
        fe_overlap = 0
        fe_incidents = 0

    if not clearances:
        return {
            "robot_overlap_ticks": 0,
            "robot_overlap_incidents": 0,
            "near_miss_ticks": 0,
            "teammate_overlap_ticks": {"friendly": 0, "enemy": 0},
            "teammate_overlap_incidents": {"friendly": 0, "enemy": 0},
            "opponent_overlap_ticks": 0,
            "opponent_overlap_incidents": 0,
            "worst_clearance_m": None,
            "total_ticks": t_steps,
        }

    all_clear = np.concatenate(clearances, axis=1)
    overlap_ticks = int(np.sum(np.any(all_clear < 0.0, axis=1)))
    near_miss_ticks = int(np.sum(np.any((all_clear >= 0.0) & (all_clear < 0.05), axis=1)))
    worst_clearance_m = round(float(np.min(all_clear)), 3)
    total_overlap_incidents = ff_incidents + ee_incidents + fe_incidents

    return {
        "robot_overlap_ticks": overlap_ticks,
        "robot_overlap_incidents": total_overlap_incidents,
        "near_miss_ticks": near_miss_ticks,
        "teammate_overlap_ticks": {"friendly": ff_overlap, "enemy": ee_overlap},
        "teammate_overlap_incidents": {"friendly": ff_incidents, "enemy": ee_incidents},
        "opponent_overlap_ticks": fe_overlap,
        "opponent_overlap_incidents": fe_incidents,
        "worst_clearance_m": worst_clearance_m,
        "total_ticks": t_steps,
    }


def compute_motion_metrics_from_replay(
    replay_path: Union[str, Path],
) -> Dict[str, Union[int, float, Dict[str, int], None]]:
    """Compute motion planning collision metrics from an .npz replay file."""
    path = Path(replay_path)
    data = np.load(path)
    return compute_motion_metrics_from_arrays(data["friendly_p"], data["enemy_p"])


def enrich_tournament_directory(tournament_dir: Union[str, Path]) -> None:
    """Enrich all .stats.json files and summary.json in a tournament directory with motion metrics."""
    run_dir = Path(tournament_dir)
    npz_files = [p for p in run_dir.glob("*.npz") if not p.name.endswith(".tmp.npz")]
    if not npz_files:
        return

    stats_updates = {}
    for npz_path in npz_files:
        tag = npz_path.stem
        stats_path = run_dir / f"{tag}.stats.json"
        try:
            metrics = compute_motion_metrics_from_replay(npz_path)
        except Exception:
            continue
        stats_updates[tag] = metrics

        if stats_path.exists():
            try:
                with open(stats_path) as f:
                    s_data = json.load(f)
                s_data.update(
                    {
                        "robot_overlap_ticks": metrics["robot_overlap_ticks"],
                        "robot_overlap_incidents": metrics["robot_overlap_incidents"],
                        "near_miss_ticks": metrics["near_miss_ticks"],
                        "teammate_overlap_ticks": metrics["teammate_overlap_ticks"],
                        "teammate_overlap_incidents": metrics["teammate_overlap_incidents"],
                        "opponent_overlap_ticks": metrics["opponent_overlap_ticks"],
                        "opponent_overlap_incidents": metrics["opponent_overlap_incidents"],
                        "worst_clearance_m": metrics["worst_clearance_m"],
                    }
                )
                with open(stats_path, "w") as f:
                    json.dump(s_data, f, indent=2)
            except Exception:
                pass

    # Update summary.json if present
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                summary = json.load(f)

            for result in summary.get("results", []):
                cfg_a = result["config_a"].removeprefix("build_").removesuffix("_kernel_strategy")
                cfg_b = result["config_b"].removeprefix("build_").removesuffix("_kernel_strategy")
                tag = f"{cfg_a}_vs_{cfg_b}"
                if tag in stats_updates:
                    m = stats_updates[tag]
                    st = result.setdefault("stats", {})
                    st["robot_overlap_ticks"] = m["robot_overlap_ticks"]
                    st["robot_overlap_incidents"] = m["robot_overlap_incidents"]
                    st["near_miss_ticks"] = m["near_miss_ticks"]
                    st["teammate_overlap_ticks"] = m["teammate_overlap_ticks"]
                    st["teammate_overlap_incidents"] = m["teammate_overlap_incidents"]
                    st["opponent_overlap_ticks"] = m["opponent_overlap_ticks"]
                    st["opponent_overlap_incidents"] = m["opponent_overlap_incidents"]
                    st["worst_clearance_m"] = m["worst_clearance_m"]

            # Rebuild strategy_table if possible
            from utama_core.analysis.strategy_table import strategy_table

            summary["strategies"] = strategy_table(summary["results"])
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
        except Exception:
            pass
