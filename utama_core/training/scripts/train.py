"""Train MARL agents on SSL scenarios using BenchMARL.

Usage:
    pixi run -e training train --task ssl_2v0 --max-frames 60000 --n-envs 32
    pixi run -e training train --task ssl_2v0 --wandb-project ssl-aspac --render
    pixi run -e training train --resume                         # resume latest
    pixi run -e training train --resume path/to/checkpoint.pt   # resume specific
"""

import argparse
import sys

from utama_core.training.checkpoint_utils import (
    find_latest_experiment_checkpoint,
    print_device_info,
)
from utama_core.training.experiment import SSLExperimentConfig, create_experiment


def parse_args() -> argparse.Namespace:
    _defaults = SSLExperimentConfig()
    parser = argparse.ArgumentParser(description="SSL MARL Training")

    parser.add_argument(
        "--task",
        type=str,
        default=_defaults.task,
        choices=[
            "ssl_2v0",
            "ssl_2v1",
            "ssl_2v2",
            "ssl_2v0_macro",
            "ssl_2v1_macro",
            "ssl_2v2_macro",
            "ssl_2v0_unified",
            "ssl_2v1_unified",
            "ssl_2v2_unified",
        ],
        help=f"Training scenario (default: {_defaults.task})",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=_defaults.max_frames,
        help=f"Total training frames (default: {_defaults.max_frames:_})",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=_defaults.n_envs,
        help=f"Number of parallel environments (default: {_defaults.n_envs})",
    )
    parser.add_argument(
        "--frames-per-batch",
        type=int,
        default=_defaults.frames_per_batch,
        help=f"Frames collected per iteration (default: {_defaults.frames_per_batch})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=_defaults.lr,
        help=f"Learning rate (default: {_defaults.lr})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_defaults.seed,
        help=f"Random seed (default: {_defaults.seed})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=_defaults.device,
        help=f"Device: 'cpu', 'cuda', or 'auto' (default: {_defaults.device})",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="WandB project name (enables WandB logging)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable eval rendering (logged to WandB as videos)",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        metavar="CHECKPOINT",
        help="Resume training. No value or 'latest': find latest checkpoint for --task. "
        "Or provide a path to a specific .pt checkpoint file.",
    )

    return parser.parse_args()


def _resolve_resume_checkpoint(resume_value: str, task: str) -> str:
    """Resolve --resume argument to an actual checkpoint path."""
    from pathlib import Path

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
            "Run a full training first or specify a path with --resume <path>.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[Resume] Found latest checkpoint: {ckpt}")
    return str(ckpt)


def main():
    args = parse_args()

    print_device_info(args.device)

    restore_file = None
    if args.resume is not None:
        restore_file = _resolve_resume_checkpoint(args.resume, args.task)

    cfg = SSLExperimentConfig(
        task=args.task,
        seed=args.seed,
        max_frames=args.max_frames,
        n_envs=args.n_envs,
        frames_per_batch=args.frames_per_batch,
        lr=args.lr,
        device=args.device,
        wandb_project=args.wandb_project,
        render=args.render,
    )

    experiment = create_experiment(cfg, restore_file=restore_file)
    experiment.run()


if __name__ == "__main__":
    main()
