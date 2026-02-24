import torch
from vmas.simulator.core import Agent, Landmark, World


def compute_team_reward(
    world: World,
    ball: Landmark,
    blue_agents: list[Agent],
    yellow_agents: list[Agent],
    is_blue: bool,
    cfg,  # SSLScenarioConfig
    goal_scored_blue: torch.Tensor,
    goal_scored_yellow: torch.Tensor,
) -> torch.Tensor:
    """Compute reward for one team. Returns shape (batch_dim,).

    Combines sparse goal rewards with dense shaping rewards.
    All sub-rewards are computed as batch tensors for vectorized evaluation.
    """
    rc = cfg.rewards
    fc = cfg.field_config
    device = world.device
    batch_dim = world.batch_dim

    reward = torch.zeros(batch_dim, device=device)

    ball_pos = ball.state.pos  # (batch, 2)
    ball_vel = ball.state.vel  # (batch, 2)

    # --- Sparse: Goal Scored ---
    if is_blue:
        reward = reward + goal_scored_blue.float() * rc.goal_scored
        reward = reward + goal_scored_yellow.float() * rc.goal_conceded
    else:
        reward = reward + goal_scored_yellow.float() * rc.goal_scored
        reward = reward + goal_scored_blue.float() * rc.goal_conceded

    # --- Dense: Ball to Opponent Goal Distance ---
    if rc.ball_to_goal_weight != 0:
        if is_blue:
            opp_goal = torch.tensor([fc.half_length, 0.0], device=device)
        else:
            opp_goal = torch.tensor([-fc.half_length, 0.0], device=device)

        dist_ball_to_goal = torch.norm(ball_pos - opp_goal.unsqueeze(0), dim=-1)
        max_dist = (fc.half_length**2 + fc.half_width**2) ** 0.5
        reward = reward + rc.ball_to_goal_weight * (1.0 - dist_ball_to_goal / max_dist)

    # --- Dense: Closest Agent to Ball ---
    if rc.agent_to_ball_weight != 0:
        team_agents = blue_agents if is_blue else yellow_agents
        agent_positions = torch.stack([a.state.pos for a in team_agents], dim=1)  # (batch, n, 2)
        dists = torch.norm(agent_positions - ball_pos.unsqueeze(1), dim=-1)  # (batch, n)
        min_dist = dists.min(dim=-1).values  # (batch,)
        max_dist = (fc.half_length**2 + fc.half_width**2) ** 0.5
        reward = reward + rc.agent_to_ball_weight * (1.0 - min_dist / max_dist)

    # --- Dense: Ball Velocity Toward Goal ---
    if rc.ball_vel_to_goal_weight != 0:
        if is_blue:
            goal_dir = torch.tensor([1.0, 0.0], device=device)
        else:
            goal_dir = torch.tensor([-1.0, 0.0], device=device)

        vel_toward_goal = (ball_vel * goal_dir.unsqueeze(0)).sum(dim=-1)
        reward = reward + rc.ball_vel_to_goal_weight * vel_toward_goal.clamp(min=0.0)

    return reward
