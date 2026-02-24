"""Factory function for creating vectorized SSL environments for RL training."""

from typing import Optional

import torch
from vmas import make_env

from utama_core.vmas_simulator.src.Scenario.ssl_scenario import SSLScenario
from utama_core.vmas_simulator.src.Utils.config import SSLScenarioConfig


def make_ssl_env(
    num_envs: int = 32,
    device: str = "cpu",
    max_steps: int = 3000,
    scenario_config: Optional[SSLScenarioConfig] = None,
    wrapper: Optional[str] = None,
    continuous_actions: bool = True,
    seed: Optional[int] = None,
    **kwargs,
):
    """Create a vectorized SSL environment for RL training.

    Args:
        num_envs: Number of parallel environments (VMAS batch dimension).
        device: "cpu" or "cuda" for GPU-accelerated simulation.
        max_steps: Maximum steps per episode.
        scenario_config: SSLScenarioConfig with field, dynamics, reward params.
            If None, uses default Division B 6v6 configuration.
        wrapper: Optional wrapper type:
            - None: returns raw VMAS Environment
            - "gymnasium": wraps with Gymnasium compatibility layer
            - "torchrl": wraps with TorchRL compatibility layer
        continuous_actions: Whether to use continuous action space.
        seed: Random seed for reproducibility.
        **kwargs: Additional kwargs passed to vmas.make_env().

    Returns:
        VMAS Environment (or wrapped version if wrapper specified).

    Example:
        # 1024 parallel envs on GPU
        env = make_ssl_env(num_envs=1024, device="cuda")

        # With Gymnasium wrapper
        env = make_ssl_env(num_envs=32, wrapper="gymnasium")

        # Custom 3v3 configuration
        from utama_core.vmas_simulator import SSLScenarioConfig
        cfg = SSLScenarioConfig(n_blue=3, n_yellow=3, max_steps=1000)
        env = make_ssl_env(num_envs=64, scenario_config=cfg)
    """
    if scenario_config is None:
        scenario_config = SSLScenarioConfig(max_steps=max_steps)

    scenario = SSLScenario()

    env = make_env(
        scenario=scenario,
        num_envs=num_envs,
        device=device,
        continuous_actions=continuous_actions,
        max_steps=scenario_config.max_steps,
        seed=seed,
        wrapper=wrapper,
        scenario_config=scenario_config,
        **kwargs,
    )

    return env
