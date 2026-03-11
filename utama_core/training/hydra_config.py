"""Hydra structured config registration for SSL training."""

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import List, Optional

from hydra.core.config_store import ConfigStore

from utama_core.training.scenario.passing_config import (
    PassingDynamicsConfig,
    PassingFieldConfig,
    PassingResetRandomizationConfig,
    PassingRewardConfig,
)


@dataclass
class ScenarioConfig:
    """Structured config for scenario group.

    Mirrors PassingScenarioConfig fields but adds a ``task`` field
    for the task-registry lookup in :class:`SSLTask`.
    """

    task: str = "ssl_2v0_unified"
    n_attackers: int = 2
    n_defenders: int = 0
    max_steps: int = 300
    defender_behavior: str = "fixed"

    field: PassingFieldConfig = dataclass_field(default_factory=PassingFieldConfig)
    dynamics: PassingDynamicsConfig = dataclass_field(default_factory=PassingDynamicsConfig)
    rewards: PassingRewardConfig = dataclass_field(default_factory=PassingRewardConfig)
    reset_randomization: PassingResetRandomizationConfig = dataclass_field(
        default_factory=PassingResetRandomizationConfig
    )


@dataclass
class TrainConfig:
    """Top-level structured config for the training pipeline."""

    seed: int = 0
    max_frames: int = 1_200_000
    n_envs: int = 32
    frames_per_batch: int = 6000
    minibatch_size: int = 400
    n_minibatch_iters: int = 8
    lr: float = 5e-5
    gamma: float = 0.99
    device: str = "auto"
    hidden_sizes: List[int] = dataclass_field(default_factory=lambda: [256, 256, 256])
    wandb_project: Optional[str] = None
    render: bool = False
    evaluation_interval: Optional[int] = None
    evaluation_episodes: int = 10
    resume: Optional[str] = None

    scenario: ScenarioConfig = dataclass_field(default_factory=ScenarioConfig)


def register_configs() -> None:
    """Register structured configs with Hydra's ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(name="train_schema", node=TrainConfig)
    cs.store(group="scenario", name="base_scenario", node=ScenarioConfig)
