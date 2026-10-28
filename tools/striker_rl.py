"""striker_rl.py — PPO for `striker_duel.py`'s attacker against the keepers: each iteration the network
plays `--episodes` duels (`striker_duel play`, on the CPU, half against each keeper) drawing its actions;
then it learns from them on the GPU (the RTX 5090 only): reward 1 for a goal at the episode's last
decision, GAE over the decisions (`--gamma`, `--lam`), PPO's clipped objective with an entropy bonus. It
writes <out>/w<k+1>.npz (for `striker_duel --weights`) and <out>/ckpt.pt; the same command resumes.
Every `--eval-every` iterations the network plays greedily the same `--eval-episodes` starts as the rule
(`striker_duel eval --seed 9999`). <out>/log.jsonl: a line per iteration.

  /workspace/FOR-Engine/.pixi/envs/sglang/bin/python tools/striker_rl.py --out replays/striker/rl1 --iterations 100
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "GPU-5206cf34-8623-f6a9-dace-e0fb425f562f")  # the RTX 5090

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N_OBS, N_ACTIONS = 28, 16  # striker_duel.N_OBS, len(striker_duel.ACTIONS)
EVAL_SEED = 9999


def mlp(n_out: int, hidden: int) -> nn.Sequential:
    layers = [nn.Linear(N_OBS, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, n_out)]
    return nn.Sequential(*layers)


def export(policy: nn.Sequential, path: Path) -> None:
    """As `striker_duel.logits` reads it: w<i> is [in, out]."""
    layers = [m for m in policy if isinstance(m, nn.Linear)]
    arrays = {}
    for i, layer in enumerate(layers):
        arrays[f"w{i}"] = layer.weight.detach().cpu().numpy().T
        arrays[f"b{i}"] = layer.bias.detach().cpu().numpy()
    np.savez(path, **arrays)


def duel(mode: str, args: list[str], log_path: Path) -> None:
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    cmd = ["pixi", "run", "python", "-m", "tools.striker_duel", mode, *args]
    with open(log_path, "a") as log:
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def advantages(data: dict, values: np.ndarray, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """GAE over each episode's decisions; the episode's reward comes after its last one."""
    episode = data["episode"]
    adv = np.zeros(len(episode), dtype=np.float32)
    last = np.r_[episode[1:] != episode[:-1], True]
    gae = 0.0
    for i in range(len(episode) - 1, -1, -1):
        if last[i]:
            delta, gae = data["reward"][episode[i]] - values[i], 0.0
        else:
            delta = gamma * values[i + 1] - values[i]
        gae = delta + gamma * lam * gae
        adv[i] = gae
    return adv, adv + values


def update(policy, value, opt, data: dict, a: argparse.Namespace, device) -> dict:
    t = {k: torch.as_tensor(data[k], device=device) for k in ("obs", "mask", "act", "logp")}
    with torch.no_grad():
        values = value(t["obs"]).squeeze(-1).cpu().numpy()
    adv, ret = advantages(data, values, a.gamma, a.lam)
    adv_t = torch.as_tensor((adv - adv.mean()) / (adv.std() + 1e-8), device=device)
    ret_t = torch.as_tensor(ret, device=device)
    n = len(adv)
    stats = {"entropy": 0.0, "kl": 0.0, "clipped": 0.0, "v_loss": 0.0}
    steps = 0
    for _ in range(a.epochs):
        for idx in torch.randperm(n, device=device).split(a.minibatch):
            logits = policy(t["obs"][idx]).masked_fill(~t["mask"][idx], -1e9)
            logp_all = torch.log_softmax(logits, -1)
            logp = logp_all.gather(1, t["act"][idx, None]).squeeze(1)
            ratio = torch.exp(logp - t["logp"][idx])
            pg = -torch.min(ratio * adv_t[idx], ratio.clamp(1 - a.clip, 1 + a.clip) * adv_t[idx]).mean()
            entropy = -(logp_all.exp() * logp_all).sum(-1).mean()
            v_loss = (value(t["obs"][idx]).squeeze(-1) - ret_t[idx]).pow(2).mean()
            loss = pg - a.entropy * entropy + 0.5 * v_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([*policy.parameters(), *value.parameters()], 0.5)
            opt.step()
            stats["entropy"] += entropy.item()
            stats["kl"] += (t["logp"][idx] - logp).mean().item()
            stats["clipped"] += ((ratio - 1).abs() > a.clip).float().mean().item()
            stats["v_loss"] += v_loss.item()
            steps += 1
    return {k: round(v / steps, 4) for k, v in stats.items()}


def goal_rates(data: dict) -> dict:
    out = {}
    for keeper in sorted(set(data["keeper"].tolist())):
        sel = data["keeper"] == keeper
        out[keeper] = round(float(data["reward"][sel].mean()), 4)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=1024)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=2000)
    a = parser.parse_args()
    out = a.out if a.out.is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    policy, value = mlp(N_ACTIONS, a.hidden).to(device), mlp(1, a.hidden).to(device)
    with torch.no_grad():
        policy[-1].weight.mul_(0.01)  # start near uniform over the legal actions
        policy[-1].bias.zero_()
    opt = torch.optim.Adam([*policy.parameters(), *value.parameters()], lr=a.lr)
    start = 0
    if (out / "ckpt.pt").exists():
        ckpt = torch.load(out / "ckpt.pt", map_location=device)
        policy.load_state_dict(ckpt["policy"])
        value.load_state_dict(ckpt["value"])
        opt.load_state_dict(ckpt["opt"])
        start = ckpt["iteration"]
    if not (out / "rule_eval.json").exists():
        eval_args = ["--episodes", str(a.eval_episodes), "--workers", str(a.workers), "--seed", str(EVAL_SEED)]
        duel("eval", [*eval_args, "--out", str(out / "rule_eval.json")], out / "eval.log")
    rule = json.loads((out / "rule_eval.json").read_text())
    rates = ", ".join(f"{k} {v['goal_rate']:.3f}" for k, v in rule.items() if isinstance(v, dict))
    print(f"device {device}; rule: {rates}")

    for it in range(start, a.iterations):
        t0 = time.time()
        weights = out / f"w{it}.npz"
        export(policy, weights)
        play = out / f"it{it}.npz"
        play_args = ["--weights", str(weights), "--episodes", str(a.episodes), "--seed", str(it)]
        duel("play", [*play_args, "--workers", str(a.workers), "--out", str(play)], out / "play.log")
        with np.load(play) as z:
            data = {k: z[k] for k in z.files}
        t1 = time.time()
        stats = update(policy, value, opt, data, a, device)
        row = {"iteration": it, "decisions": len(data["act"]), "goal_rate": goal_rates(data), **stats}
        row["play_s"], row["learn_s"] = round(t1 - t0), round(time.time() - t1)
        ckpt = {
            "policy": policy.state_dict(),
            "value": value.state_dict(),
            "opt": opt.state_dict(),
            "iteration": it + 1,
        }
        torch.save(ckpt, out / "ckpt.pt")
        play.unlink()  # the decisions are only needed for this update
        if (it + 1) % a.eval_every == 0:
            export(policy, out / f"w{it + 1}.npz")
            ev = out / f"eval{it + 1}.json"
            eval_args = ["--episodes", str(a.eval_episodes), "--workers", str(a.workers), "--seed", str(EVAL_SEED)]
            duel("eval", ["--weights", str(out / f"w{it + 1}.npz"), *eval_args, "--out", str(ev)], out / "eval.log")
            res = json.loads(ev.read_text())
            row["eval"] = {k: v["goal_rate"] for k, v in res.items() if isinstance(v, dict)}
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
