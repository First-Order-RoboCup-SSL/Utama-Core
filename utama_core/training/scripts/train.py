"""Train MARL agents on SSL scenarios using BenchMARL.

Usage (Hydra-style overrides):
    pixi run -e training train                                  # default (2v0_unified)
    pixi run -e training train scenario=2v1_unified             # pick scenario
    pixi run -e training train max_frames=60000 n_envs=64       # override hyperparams
    pixi run -e training train resume=latest                    # resume latest
    pixi run -e training train resume=/path/to/checkpoint.pt    # resume specific
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from utama_core.training.hydra_config import register_configs

register_configs()


def _resolve_resume_checkpoint(resume_value: str, task: str) -> str:
    """Resolve resume argument to an actual checkpoint path."""
    from utama_core.training.checkpoint_utils import find_latest_experiment_checkpoint

    if resume_value != "latest":
        ckpt = Path(resume_value)
        if not ckpt.exists():
            print(f"Error: checkpoint file not found: {resume_value}", file=sys.stderr)
            sys.exit(1)
        return str(ckpt.resolve())

    ckpt = find_latest_experiment_checkpoint(task_filter=task)
    if ckpt is None:
        print(
            f"Error: no checkpoints found for task '{task}'. "
            "Run a full training first or specify a path with resume=<path>.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[Resume] Found latest checkpoint: {ckpt}")
    return str(ckpt)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    from utama_core.training.checkpoint_utils import print_device_info
    from utama_core.training.experiment import SSLExperimentConfig, create_experiment
    from utama_core.training.scenario.passing_config import PassingScenarioConfig

    print_device_info(cfg.device)

    # --- Resume handling ---
    restore_file = None
    if cfg.resume is not None:
        restore_file = _resolve_resume_checkpoint(cfg.resume, cfg.scenario.task)

    # --- Build SSLExperimentConfig ---
    exp_cfg = SSLExperimentConfig(
        task=cfg.scenario.task,
        seed=cfg.seed,
        max_frames=cfg.max_frames,
        n_envs=cfg.n_envs,
        frames_per_batch=cfg.frames_per_batch,
        minibatch_size=cfg.minibatch_size,
        n_minibatch_iters=cfg.n_minibatch_iters,
        lr=cfg.lr,
        gamma=cfg.gamma,
        device=cfg.device,
        hidden_sizes=tuple(cfg.hidden_sizes),
        wandb_project=cfg.wandb_project,
        render=cfg.render,
        evaluation_interval=cfg.evaluation_interval,
        evaluation_episodes=cfg.evaluation_episodes,
    )

    # --- Build PassingScenarioConfig from Hydra structured config ---
    from utama_core.training.scenario.passing_config import (
        PassingDynamicsConfig,
        PassingFieldConfig,
        PassingResetRandomizationConfig,
        PassingRewardConfig,
    )

    sc = OmegaConf.to_container(cfg.scenario, resolve=True)
    scenario_config = PassingScenarioConfig(
        n_attackers=sc["n_attackers"],
        n_defenders=sc["n_defenders"],
        max_steps=sc["max_steps"],
        defender_behavior=sc["defender_behavior"],
        field=PassingFieldConfig(**sc["field"]) if "field" in sc else PassingFieldConfig(),
        dynamics=PassingDynamicsConfig(**sc["dynamics"]) if "dynamics" in sc else PassingDynamicsConfig(),
        rewards=PassingRewardConfig(**sc["rewards"]) if "rewards" in sc else PassingRewardConfig(),
        reset_randomization=(
            PassingResetRandomizationConfig(**sc["reset_randomization"])
            if "reset_randomization" in sc
            else PassingResetRandomizationConfig()
        ),
    )

    # --- Create and run experiment ---
    experiment = create_experiment(
        exp_cfg,
        restore_file=restore_file,
        scenario_config=scenario_config,
    )
    experiment.run()


if __name__ == "__main__":
    main()
