"""Evaluate a trained SSL MARL checkpoint and optionally render to MP4.

Usage:
    pixi run -e training evaluate --checkpoint path/to/checkpoint.pt
    pixi run -e training evaluate --latest                            # newest checkpoint
    pixi run -e training evaluate --latest --task ssl_2v0             # filter by task
    pixi run -e training evaluate --latest --render --output eval.mp4
"""

import argparse
import functools
import sys
from contextlib import contextmanager

import torch
from benchmarl.experiment import Experiment
from torchrl.envs.utils import ExplorationType, set_exploration_type

from utama_core.training.checkpoint_utils import (
    find_latest_experiment_checkpoint,
    print_device_info,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SSL MARL Evaluation")

    ckpt_group = parser.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint file",
    )
    ckpt_group.add_argument(
        "--latest",
        action="store_true",
        help="Automatically find the latest checkpoint (use --task to filter)",
    )

    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Filter experiments by task name when using --latest (e.g. ssl_2v0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'cpu', 'cuda', or 'auto' (default: auto)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render evaluation to MP4 video",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation.mp4",
        help="Output video path (default: evaluation.mp4)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video framerate (default: 30)",
    )

    return parser.parse_args()


@contextmanager
def _force_map_location(device: str):
    """Temporarily patch torch.load to remap tensors to the target device.

    BenchMARL's reload_from_file doesn't pass map_location, so loading a
    GPU-trained checkpoint on a CPU-only machine fails. This patches
    torch.load everywhere it might be referenced.
    """
    import torch.serialization

    orig_torch = torch.load
    orig_serial = torch.serialization.load

    def _patched(*args, **kwargs):
        kwargs["map_location"] = device
        return orig_serial(*args, **kwargs)

    torch.load = _patched
    torch.serialization.load = _patched
    try:
        yield
    finally:
        torch.load = orig_torch
        torch.serialization.load = orig_serial


def main():
    args = parse_args()

    print_device_info(args.device)

    # Resolve checkpoint path
    if args.latest:
        checkpoint = find_latest_experiment_checkpoint(task_filter=args.task)
        if checkpoint is None:
            filter_msg = f" for task '{args.task}'" if args.task else ""
            print(
                f"Error: no checkpoints found{filter_msg}. " "Run training first or use --checkpoint <path>.",
                file=sys.stderr,
            )
            sys.exit(1)
        checkpoint = str(checkpoint)
        print(f"[Eval] Using latest checkpoint: {checkpoint}")
    else:
        checkpoint = args.checkpoint

    device = resolve_device(args.device)

    with _force_map_location(device):
        experiment = Experiment.reload_from_file(
            restore_file=checkpoint,
            experiment_patch={
                "render": False,
                "evaluation_episodes": args.episodes,
                "sampling_device": "cpu",
                "train_device": device,
                "buffer_device": device,
            },
        )

    policy = experiment.policy
    test_env = experiment.test_env
    max_steps = experiment.max_steps
    task = experiment.task

    video_frames = [] if args.render else None

    if args.render:

        def callback(env, td):
            video_frames.append(task.__class__.render_callback(experiment, env, td))

    else:
        callback = None

    print(f"Running {args.episodes} evaluation episode(s)...")

    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        if test_env.batch_size == ():
            rollouts = []
            for ep in range(args.episodes):
                rollout = test_env.rollout(
                    max_steps=max_steps,
                    policy=policy,
                    callback=callback if ep == 0 else None,
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
                rollouts.append(rollout)
                print(f"  Episode {ep + 1}/{args.episodes} done ({rollout.shape[-1]} steps)")
        else:
            rollouts = test_env.rollout(
                max_steps=max_steps,
                policy=policy,
                callback=callback,
                auto_cast_to_device=True,
                break_when_any_done=False,
            )
            rollouts = list(rollouts.unbind(0))
            print(f"  {len(rollouts)} episodes done")

    # Print summary
    for group in experiment.group_map:
        rewards = torch.tensor([r.get(("next", group, "reward")).sum().item() for r in rollouts])
        print(f"  {group} mean reward: {rewards.mean():.3f} (+/- {rewards.std():.3f})")

    # Save video
    if args.render and video_frames:
        import imageio

        frames = [f.cpu().numpy() if isinstance(f, torch.Tensor) else f for f in video_frames]
        imageio.mimsave(args.output, frames, fps=args.fps)
        print(f"Video saved to {args.output} ({len(frames)} frames)")

    test_env.close()

    # Clean up pygame if it was initialized by the renderer
    try:
        import pygame

        if pygame.get_init():
            pygame.quit()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
