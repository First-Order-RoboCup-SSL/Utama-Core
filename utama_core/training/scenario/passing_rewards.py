"""ASPAC reward computation for SSL passing drills.

All functions use torch.* ops only — no numpy/scipy — to preserve
differentiability for future gradient-based MARL (e.g., SHAC).
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_passing_reward(
    agent_name: str,
    agent_pos: Tensor,
    ball_pos: Tensor,
    scenario,
) -> Tensor:
    """Per-agent reward using delta-based shaping.

    Delta-based: reward = prev_dist - curr_dist, so standing still = 0.

    Args:
        agent_name: Name of the agent (e.g. "attacker_0").
        agent_pos: Agent position, shape (batch, 2).
        ball_pos: Ball position, shape (batch, 2).
        scenario: PassingScenario instance (for accessing other agents/state).

    Returns:
        Reward tensor, shape (batch,).
    """
    rc = scenario.cfg.rewards
    fc = scenario.cfg.field
    device = agent_pos.device
    batch_dim = agent_pos.shape[0]

    reward = torch.zeros(batch_dim, device=device)

    is_attacker = any(a.name == agent_name for a in scenario.attackers)

    if is_attacker and len(scenario.attackers) >= 2:
        # Determine passer (attacker_0) vs receiver (attacker_1)
        is_passer = agent_name == scenario.attackers[0].name
        receiver = scenario.attackers[1]
        receiver_pos = receiver.state.pos

        if is_passer:
            # Phased passer reward: approach ball → dribble/pass to receiver
            dc = scenario.cfg.dynamics
            passer_to_ball_dist = torch.norm(agent_pos - ball_pos, dim=-1)
            has_ball = passer_to_ball_dist < dc.dribble_dist_threshold
            passer_touched_ball = scenario.last_holder == 0

            # Phase 1: passer approaches ball (before first touch)
            approach_delta = scenario.prev_passer_to_ball_dist - passer_to_ball_dist
            phase1 = rc.passer_to_ball_weight * approach_delta

            # Phase 2: ball→receiver (after touching ball — dribble or kicked)
            ball_to_recv = torch.norm(ball_pos - receiver_pos, dim=-1)
            pass_delta = scenario.prev_ball_to_receiver_dist - ball_to_recv
            phase2 = rc.ball_to_receiver_weight * pass_delta

            in_phase2 = has_ball | passer_touched_ball
            reward = reward + torch.where(in_phase2, phase2, phase1)
        else:
            # Receiver: no dense approach shaping by default (relies on sparse pass reward).
            # Optional residual shaping controlled by receiver_to_ball_weight (default 0).
            if rc.receiver_to_ball_weight > 0:
                curr_dist = torch.norm(agent_pos - ball_pos, dim=-1)
                delta = scenario.prev_receiver_to_ball_dist - curr_dist
                reward = reward + rc.receiver_to_ball_weight * delta

        # Sparse: successful pass
        reward = reward + scenario.pass_completed.float() * rc.successful_pass

    elif not is_attacker:
        # Defender reward: getting closer to ball (interception shaping)
        dist_to_ball = torch.norm(agent_pos - ball_pos, dim=-1)
        # Normalize by field diagonal for scale
        max_dist = (fc.half_length**2 + fc.half_width**2) ** 0.5
        reward = reward + (1.0 - dist_to_ball / max_dist) * 0.5

        # Bonus for interception
        reward = reward + scenario.ball_intercepted.float() * 10.0

    # Out-of-zone penalty (soft constraint)
    az_x = fc.active_zone_center_x
    az_y = fc.active_zone_center_y
    outside_x = agent_pos[:, 0].abs() - (az_x + fc.active_zone_half_length)
    outside_y = agent_pos[:, 1].abs() - (az_y + fc.active_zone_half_width)
    outside_zone = (outside_x > 0) | (outside_y > 0)
    reward = reward + outside_zone.float() * rc.out_of_zone_penalty

    # Sparse: ball out of bounds
    oob = (ball_pos[:, 0].abs() > fc.half_length) | (ball_pos[:, 1].abs() > fc.half_width)
    reward = reward + oob.float() * rc.ball_out_of_bounds

    return reward


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
