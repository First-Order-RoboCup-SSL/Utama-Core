"""VMAS scenarios for ASPAC passing drills."""

from utama_core.training.scenario.passing_config import (
    PassingDynamicsConfig,
    PassingFieldConfig,
    PassingRewardConfig,
    PassingScenarioConfig,
)
from utama_core.training.scenario.passing_rewards import (
    compute_passing_reward,
    deception_penalty,
    displacement_error,
    envy_free_bonus,
)
from utama_core.training.scenario.passing_scenario import PassingScenario

__all__ = [
    "PassingDynamicsConfig",
    "PassingFieldConfig",
    "PassingRewardConfig",
    "PassingScenarioConfig",
    "PassingScenario",
    "compute_passing_reward",
    "envy_free_bonus",
    "deception_penalty",
    "displacement_error",
]
