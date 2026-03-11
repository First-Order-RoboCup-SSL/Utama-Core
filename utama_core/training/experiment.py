"""Experiment configuration and factory for SSL training."""

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from benchmarl.algorithms import MappoConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

from utama_core.training.scenario.passing_config import PassingScenarioConfig
from utama_core.training.task import SSLTask


def _ceil_to_multiple(x: int, m: int) -> int:
    """Round *x* up to the nearest multiple of *m*."""
    return -(-x // m) * m


@dataclass
class SSLExperimentConfig:
    """High-level config for SSL training experiments."""

    task: str = "ssl_2v0_unified"
    seed: int = 0
    max_frames: int = 1_200_000
    n_envs: int = 32
    frames_per_batch: int = 6000
    minibatch_size: int = 400
    n_minibatch_iters: int = 8
    lr: float = 3e-4
    gamma: float = 0.99
    device: str = "auto"

    # MLP architecture
    hidden_sizes: tuple = (256, 256, 256)

    # WandB
    wandb_project: Optional[str] = None

    # Eval / rendering
    render: bool = False
    evaluation_interval: Optional[int] = None
    evaluation_episodes: int = 10


def create_experiment(
    cfg: SSLExperimentConfig,
    restore_file: Optional[str] = None,
    scenario_config: Optional[PassingScenarioConfig] = None,
) -> Experiment:
    """Create a BenchMARL Experiment from our high-level config.

    Args:
        cfg: High-level experiment configuration.
        restore_file: If provided, resume training from this checkpoint .pt file.
        scenario_config: If provided, override the default scenario config for the task.
    """

    # Resolve device
    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Task
    task = SSLTask.from_name(cfg.task, scenario_config=scenario_config)

    # Algorithm
    algorithm_config = MappoConfig.get_from_yaml()
    algorithm_config.entropy_coef = 0.01  # prevent distribution collapse → NaN
    # Model
    model_config = MlpConfig(
        num_cells=list(cfg.hidden_sizes),
        layer_class=nn.Linear,
        activation_class=nn.Tanh,
    )

    # Round up frames_per_batch to be divisible by n_envs. TorchRL rounds up
    # internally; if we don't match, the collector exhausts total_frames in
    # fewer iterations than BenchMARL expects → StopIteration.
    fpb = _ceil_to_multiple(cfg.frames_per_batch, cfg.n_envs)
    max_frames = _ceil_to_multiple(cfg.max_frames, fpb)

    # Experiment config
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.max_n_frames = max_frames
    experiment_config.on_policy_collected_frames_per_batch = fpb
    experiment_config.on_policy_n_envs_per_worker = cfg.n_envs
    experiment_config.on_policy_minibatch_size = cfg.minibatch_size
    experiment_config.on_policy_n_minibatch_iters = cfg.n_minibatch_iters
    experiment_config.lr = cfg.lr
    experiment_config.gamma = cfg.gamma

    experiment_config.sampling_device = device
    experiment_config.train_device = device
    experiment_config.buffer_device = device

    # Logging
    loggers: List[str] = ["csv"]
    if cfg.wandb_project:
        loggers.append("wandb")
        experiment_config.project_name = cfg.wandb_project
    experiment_config.loggers = loggers

    # Checkpointing
    experiment_config.checkpoint_interval = fpb
    experiment_config.checkpoint_at_end = True
    experiment_config.keep_checkpoints_num = 5

    # Resume from checkpoint
    if restore_file is not None:
        experiment_config.restore_file = restore_file

    # Evaluation / rendering
    if cfg.render and cfg.wandb_project:
        experiment_config.render = True
        eval_interval = cfg.evaluation_interval or fpb
        experiment_config.evaluation_interval = _ceil_to_multiple(eval_interval, fpb)
        experiment_config.evaluation_episodes = cfg.evaluation_episodes
    else:
        experiment_config.render = False
        if cfg.evaluation_interval:
            experiment_config.evaluation_interval = _ceil_to_multiple(cfg.evaluation_interval, fpb)
        else:
            # Align BenchMARL's default to be a multiple of fpb
            experiment_config.evaluation_interval = _ceil_to_multiple(experiment_config.evaluation_interval, fpb)

    return Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        seed=cfg.seed,
        config=experiment_config,
    )
