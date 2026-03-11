"""Reward computation for SSL passing drills."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor


def compute_passing_reward(
    agent_name: str,
    agent_pos: Tensor,
    ball_pos: Tensor,
    scenario,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Per-agent reward using delta-based shaping.

    Delta-based: reward = prev_dist - curr_dist, so standing still = 0.

    Args:
        agent_name: Name of the agent (e.g. "attacker_0").
        agent_pos: Agent position, shape (batch, 2).
        ball_pos: Ball position, shape (batch, 2).
        scenario: PassingScenario instance (for accessing other agents/state).

    Returns:
        Tuple of (reward tensor shape (batch,), component dict for logging).
    """
    rc = scenario.cfg.rewards
    fc = scenario.cfg.field
    device = agent_pos.device
    batch_dim = agent_pos.shape[0]

    reward = torch.zeros(batch_dim, device=device)
    components: Dict[str, Tensor] = {}
    zeros = torch.zeros(batch_dim, device=device)

    # Compute dense shaping annealing factor
    dense_scale = _dense_annealing_factor(rc, scenario)

    is_attacker = any(a.name == agent_name for a in scenario.attackers)
    holder = scenario.confirmed_holder
    curr = scenario.current_metrics
    prev = scenario.previous_metrics

    if is_attacker and len(scenario.attackers) >= 2:
        # Determine passer (attacker_0) vs receiver (attacker_1)
        is_passer = agent_name == scenario.attackers[0].name

        if is_passer:
            passer_near_ball = holder == 0
            passer_touched_ball = scenario.last_attacker_holder == 0

            approach_delta = prev["passer_to_ball_dist"] - curr["passer_to_ball_dist"]
            phase1 = rc.passer_to_ball_weight * approach_delta * dense_scale

            pass_delta = prev["ball_to_receiver_dist"] - curr["ball_to_receiver_dist"]
            phase2 = rc.ball_to_receiver_weight * pass_delta * dense_scale

            in_phase2 = passer_near_ball | passer_touched_ball
            phased_reward = torch.where(in_phase2, phase2, phase1)
            reward = reward + phased_reward

            components["reward/approach_delta"] = torch.where(~in_phase2, phase1, zeros)
            components["reward/pass_delta"] = torch.where(in_phase2, phase2, zeros)

            facing_recv_delta = curr["passer_facing_receiver_cos"] - prev["passer_facing_receiver_cos"]
            face_recv_reward = rc.passer_face_receiver_weight * facing_recv_delta * dense_scale
            reward = reward + face_recv_reward
            components["reward/face_receiver"] = face_recv_reward

            has_ball_reward = (holder == 0).float() * rc.has_ball_reward * dense_scale
            reward = reward + has_ball_reward
            components["reward/has_ball"] = has_ball_reward

            kick_reward = zeros.clone()
            passer_name = scenario.attackers[0].name
            if hasattr(scenario, "kick_fired") and passer_name in scenario.kick_fired:
                kicked = scenario.kick_fired[passer_name]
                if kicked.any():
                    facing_recv_clamped = curr["passer_facing_receiver_cos"].clamp(min=0.0)
                    kick_reward = kicked.float() * rc.kick_alignment_weight * facing_recv_clamped * dense_scale
                    reward = reward + kick_reward
            components["reward/kick_align"] = kick_reward
        else:
            recv_approach_reward = zeros.clone()
            if rc.receiver_to_ball_weight > 0:
                delta = prev["receiver_to_ball_dist"] - curr["receiver_to_ball_dist"]
                recv_approach_reward = rc.receiver_to_ball_weight * delta * dense_scale
                reward = reward + recv_approach_reward
            components["reward/recv_approach"] = recv_approach_reward

            facing_delta = curr["receiver_facing_ball_cos"] - prev["receiver_facing_ball_cos"]
            recv_face_reward = rc.receiver_face_ball_weight * facing_delta * dense_scale
            reward = reward + recv_face_reward
            components["reward/recv_face_ball"] = recv_face_reward

        pass_reward = scenario.pass_completed.float() * rc.successful_pass
        reward = reward + pass_reward
        components["reward/pass_completed"] = pass_reward

    elif not is_attacker:
        defender_dense = zeros.clone()
        if rc.defender_delta_weight > 0:
            delta = prev["defender_to_ball_dist"][agent_name] - curr["defender_to_ball_dist"][agent_name]
            defender_dense = delta * rc.defender_delta_weight * dense_scale
        reward = reward + defender_dense
        components["reward/defender_approach"] = defender_dense

        intercept_reward = scenario.ball_intercepted.float() * 10.0
        reward = reward + intercept_reward
        components["reward/interception"] = intercept_reward

    # Out-of-zone penalty (soft constraint, NOT annealed)
    az_x = fc.active_zone_center_x
    az_y = fc.active_zone_center_y
    outside_x = agent_pos[:, 0].abs() - (az_x + fc.active_zone_half_length)
    outside_y = agent_pos[:, 1].abs() - (az_y + fc.active_zone_half_width)
    outside_zone = (outside_x > 0) | (outside_y > 0)
    zone_penalty = outside_zone.float() * rc.out_of_zone_penalty
    reward = reward + zone_penalty
    components["reward/out_of_zone"] = zone_penalty

    # Sparse: ball out of bounds (NOT annealed)
    oob = (ball_pos[:, 0].abs() > fc.half_length) | (ball_pos[:, 1].abs() > fc.half_width)
    oob_penalty = oob.float() * rc.ball_out_of_bounds
    reward = reward + oob_penalty
    components["reward/ball_oob"] = oob_penalty

    # Log the annealing factor for monitoring
    components["reward/dense_scale"] = torch.full((batch_dim,), dense_scale, device=device)

    return reward, components


def _dense_annealing_factor(rc, scenario) -> float:
    """Compute the dense shaping annealing multiplier.

    Returns 1.0 when annealing is disabled (shaping_anneal_end == 0).
    """
    if rc.shaping_anneal_end <= 0:
        return 1.0

    global_frame = getattr(scenario, "global_frame", 0)
    if global_frame < rc.shaping_anneal_start:
        return 1.0

    progress = (global_frame - rc.shaping_anneal_start) / max(rc.shaping_anneal_end - rc.shaping_anneal_start, 1)
    progress = min(progress, 1.0)
    return max(rc.shaping_anneal_min, 1.0 - progress * (1.0 - rc.shaping_anneal_min))


def envy_free_bonus(
    cumulative_attacker_rewards: dict[str, Tensor],
    weight: float,
) -> dict[str, Tensor]:
    """Envy-Free Policy Teaching credit adjustment.

    CRITICAL: Called ONLY at episode termination (inside done()), NOT per-step.
    Avoids breaking GPU vectorization with CPU-bound convex solvers.

    Uses cumulative expected rewards over the episode to compute EF credits.
    Future implementation should use batched torch.linalg or cvxpylayers
    (NOT scipy/CVXPY which would force CPU transfer).

    Placeholder: returns zero adjustment.
    """
    # Stub: no adjustment
    return {name: torch.zeros_like(val) for name, val in cumulative_attacker_rewards.items()}


def deception_penalty(
    defender_pos: Tensor,
    defender_optimal_pos: Tensor,
    scale: float,
) -> Tensor:
    """Imitative Follower Deception: reward attackers for displacing defender.

    Fully differentiable — uses only torch.norm.
    Placeholder: returns zero.
    """
    return torch.zeros(defender_pos.shape[0], device=defender_pos.device)


def displacement_error(
    defender_pos: Tensor,
    ball_pos: Tensor,
    receiver_pos: Tensor,
) -> Tensor:
    """Spatial Displacement Error metric (for evaluation logging).

    Measures L2 distance from defender to the ball-receiver midpoint
    (optimal interception point).

    Fully differentiable — preserves computational graph.
    """
    optimal = (ball_pos + receiver_pos) / 2
    return torch.norm(defender_pos - optimal, dim=-1)
