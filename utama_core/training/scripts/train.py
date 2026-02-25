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
    parser = argparse.ArgumentParser(description="SSL MARL Training")

    parser.add_argument(
        "--task",
        type=str,
        default="ssl_2v0",
        choices=["ssl_2v0", "ssl_2v1", "ssl_2v2"],
        help="Training scenario (default: ssl_2v0)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=3_000_000,
        help="Total training frames (default: 3M)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=32,
        help="Number of parallel environments (default: 32)",
    )
    parser.add_argument(
        "--frames-per-batch",
        type=int,
        default=6000,
        help="Frames collected per iteration (default: 6000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'cpu', 'cuda', or 'auto' (default: auto)",
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
