"""striker_e2e_rl.py — the end-to-end attacker (`striker_e2e.py`) against the keepers, or with `--task chase`
the end-to-end ball chaser (`chase_e2e.py`, success: the ball caught; usually from blank): first it imitates
`striker_duel`'s high-level network (`--teacher`) from `striker_e2e record`, then refines that with
`--dagger` rounds (the network drives, the teacher labels where it went), then it learns by PPO, like
`striker_rl.py` but over the robot's own commands, on the GPU (the RTX 5090 only).

The reward from one decision to the next is the change in the shaping potential (`striker_e2e.phi`:
nearer the ball, and the ball nearer their goal), discounted (`--gamma`); the last decision gets the
outcome instead: 1 a goal, `--foul` a stoppage for a foul (out of play is 0, as a save or a timeout).
The first `--value-warmup` iterations fit only the critic, so a random one cannot undo the imitation.
Writes <out>/e<k>.npz (for `striker_e2e --weights`) and <out>/ckpt.pt; the same command resumes. Every
`--eval-every` iterations it plays greedily the same starts as `striker_duel eval --seed 9999`.

  /workspace/FOR-Engine/.pixi/envs/sglang/bin/python tools/striker_e2e_rl.py --out replays/striker/e2e1 \\
      --teacher replays/striker/rl2/w150.npz --iterations 200
  /workspace/FOR-Engine/.pixi/envs/sglang/bin/python tools/striker_e2e_rl.py --task chase --out replays/chase/c1 \
      --std 0.3 --lr 3e-4 --iterations 200
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "GPU-5206cf34-8623-f6a9-dace-e0fb425f562f")  # the RTX 5090

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N_ACT = 5  # striker_e2e.N_ACT
# --task: the environment module, what it shows (its N_OBS) and the outcome that scores 1.
TASKS = {"striker": ("tools.striker_e2e", 42, "goal"), "chase": ("tools.chase_e2e", 23, "caught")}
EVAL_SEED, BC_SEED = 9999, 777


class Policy(nn.Module):
    """As `striker_e2e.forward` reads it: speeds' means tanh of the first three outputs, then the kick
    and dribbler logits; the speeds' spread a learned `log_std`."""

    def __init__(self, n_obs: int, hidden: int, std: float):
        super().__init__()
        layers = [nn.Linear(n_obs, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, N_ACT)]
        self.body = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.full((3,), math.log(std)))

    def heads(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.body(obs)
        return torch.tanh(out[:, :3]), out[:, 3:]

    def log_prob_entropy(self, obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logits = self.heads(obs)
        normal = torch.distributions.Normal(mean, self.log_std.exp())
        coin = torch.distributions.Bernoulli(logits=logits)
        logp = normal.log_prob(act[:, :3]).sum(-1) + coin.log_prob(act[:, 3:]).sum(-1)
        return logp, normal.entropy().sum(-1) + coin.entropy().sum(-1)


def critic(n_obs: int, hidden: int) -> nn.Sequential:
    layers = [nn.Linear(n_obs, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)]
    return nn.Sequential(*layers)


def export(policy: Policy, path: Path) -> None:
    layers = [m for m in policy.body if isinstance(m, nn.Linear)]
    arrays = {f"w{i}": m.weight.detach().cpu().numpy().T for i, m in enumerate(layers)}
    arrays.update({f"b{i}": m.bias.detach().cpu().numpy() for i, m in enumerate(layers)})
    arrays["log_std"] = policy.log_std.detach().cpu().numpy()
    np.savez(path, **arrays)


MODULE, SUCCESS = TASKS["striker"][0], TASKS["striker"][2]  # set from --task in main


def e2e(mode: str, args: list[str], log_path: Path) -> None:
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    cmd = ["pixi", "run", "python", "-m", MODULE, mode, *args]
    with open(log_path, "a") as log:
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def load(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def imitate(policy: Policy, data: dict, a: argparse.Namespace, device, epochs: int) -> None:
    """The speeds' means by squared error, kick and dribbler by cross-entropy (kicks are rare: weighted)."""
    obs = torch.as_tensor(data["obs"], device=device)
    act = torch.as_tensor(data["act"], device=device)
    n = len(obs)
    val = torch.arange(n, device=device) % 20 == 0
    opt = torch.optim.Adam(policy.body.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([a.kick_weight, 1.0], device=device))

    def loss(idx):
        mean, logits = policy.heads(obs[idx])
        return (mean - act[idx, :3]).pow(2).sum(-1).mean() + bce(logits, act[idx, 3:])

    train = torch.nonzero(~val).squeeze(1)
    for epoch in range(epochs):
        for idx in train[torch.randperm(len(train), device=device)].split(4096):
            opt.zero_grad()
            loss(idx).backward()
            opt.step()
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                print(f"imitation epoch {epoch + 1}: held-out loss {loss(torch.nonzero(val).squeeze(1)).item():.4f}")


def rewards(data: dict, gamma: float, foul: float) -> np.ndarray:
    episode, phi = data["episode"], data["phi"]
    last = np.r_[episode[1:] != episode[:-1], True]
    r = np.zeros(len(episode), dtype=np.float32)
    r[:-1] = gamma * phi[1:] - phi[:-1]
    for i in np.nonzero(last)[0]:
        o = str(data["outcome"][episode[i]])
        r[i] = 1.0 if o == SUCCESS else foul if o.startswith("stopped:") and o != "stopped:out_of_bounds" else 0.0
    return r


def advantages(episode: np.ndarray, r: np.ndarray, values: np.ndarray, gamma: float, lam: float):
    last = np.r_[episode[1:] != episode[:-1], True]
    adv = np.zeros(len(r), dtype=np.float32)
    gae = 0.0
    for i in range(len(r) - 1, -1, -1):
        nxt = 0.0 if last[i] else values[i + 1]
        delta = r[i] + gamma * nxt - values[i]
        gae = delta + (0.0 if last[i] else gamma * lam * gae)
        adv[i] = gae
    return adv, adv + values


def update(policy, value, p_opt, v_opt, data, a, device, policy_too: bool) -> dict:
    obs = torch.as_tensor(data["obs"], device=device)
    act = torch.as_tensor(data["act"], device=device)
    old = torch.as_tensor(data["logp"], device=device)
    with torch.no_grad():
        values = value(obs).squeeze(-1).cpu().numpy()
    adv, ret = advantages(data["episode"], rewards(data, a.gamma, a.foul), values, a.gamma, a.lam)
    adv_t = torch.as_tensor((adv - adv.mean()) / (adv.std() + 1e-8), device=device)
    ret_t = torch.as_tensor(ret, device=device)
    stats = {"entropy": 0.0, "kl": 0.0, "clipped": 0.0, "v_loss": 0.0}
    steps = 0
    for _ in range(a.epochs):
        for idx in torch.randperm(len(obs), device=device).split(a.minibatch):
            v_loss = (value(obs[idx]).squeeze(-1) - ret_t[idx]).pow(2).mean()
            v_opt.zero_grad()
            v_loss.backward()
            nn.utils.clip_grad_norm_(value.parameters(), 0.5)
            v_opt.step()
            stats["v_loss"] += v_loss.item()
            if policy_too:
                logp, entropy = policy.log_prob_entropy(obs[idx], act[idx])
                ratio = torch.exp(logp - old[idx])
                surrogate = torch.min(ratio * adv_t[idx], ratio.clamp(1 - a.clip, 1 + a.clip) * adv_t[idx])
                loss = -surrogate.mean() - a.entropy * entropy.mean()
                p_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                p_opt.step()
                stats["entropy"] += entropy.mean().item()
                stats["kl"] += (old[idx] - logp).mean().item()
                stats["clipped"] += ((ratio - 1).abs() > a.clip).float().mean().item()
            steps += 1
    return {k: round(v / steps, 4) for k, v in stats.items()}


def evaluate(path: Path, out_json: Path, a: argparse.Namespace, log: Path) -> dict:
    args = ["--weights", str(path), "--episodes", str(a.eval_episodes), "--seed", str(EVAL_SEED)]
    e2e("eval", [*args, "--workers", str(a.workers), "--out", str(out_json)], log)
    summary = json.loads(out_json.read_text())
    return {k: v.get("success_rate", v.get("goal_rate")) for k, v in summary.items() if isinstance(v, dict)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", choices=sorted(TASKS), default="striker", help="chase: tools/chase_e2e.py")
    parser.add_argument("--teacher", type=Path, help="striker_duel's network to imitate first; none: start blank")
    parser.add_argument("--bc-episodes", type=int, default=6000)
    parser.add_argument("--bc-epochs", type=int, default=30)
    parser.add_argument("--kick-weight", type=float, default=10.0)
    parser.add_argument("--dagger", type=int, default=4, help="DAgger rounds after the first imitation")
    parser.add_argument("--dagger-episodes", type=int, default=3000)
    parser.add_argument("--dagger-epochs", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--std", type=float, default=0.05, help="the speeds' starting spread, of their range")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy", type=float, default=0.001)
    parser.add_argument("--foul", type=float, default=-0.3)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=4096)
    parser.add_argument("--value-warmup", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=4000)
    a = parser.parse_args()
    out = a.out if a.out.is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    global MODULE, SUCCESS
    MODULE, n_obs, SUCCESS = TASKS[a.task]
    policy, value = Policy(n_obs, a.hidden, a.std).to(device), critic(n_obs, a.hidden).to(device)
    p_opt = torch.optim.Adam(policy.parameters(), lr=a.lr)
    v_opt = torch.optim.Adam(value.parameters(), lr=3 * a.lr)
    start = 0
    if (out / "ckpt.pt").exists():
        ckpt = torch.load(out / "ckpt.pt", map_location=device)
        policy.load_state_dict(ckpt["policy"])
        value.load_state_dict(ckpt["value"])
        p_opt.load_state_dict(ckpt["p_opt"])
        v_opt.load_state_dict(ckpt["v_opt"])
        start = ckpt["iteration"]
    elif a.teacher is not None:
        bc = out / "bc.npz"
        if not bc.exists():
            args = ["--teacher", str(a.teacher), "--episodes", str(a.bc_episodes), "--seed", str(BC_SEED)]
            e2e("record", [*args, "--workers", str(a.workers), "--out", str(bc)], out / "record.log")
        data = load(bc)
        print(f"device {device}; imitating {len(data['act'])} decisions of {a.teacher}", flush=True)
        imitate(policy, data, a, device, a.bc_epochs)
        for r in range(1, a.dagger + 1):
            # DAgger: where the network takes the robot, what would the teacher do?
            export(policy, out / f"d{r}.npz")
            more = out / f"dagger{r}.npz"
            if not more.exists():
                args = ["--weights", str(out / f"d{r}.npz"), "--teacher", str(a.teacher), "--seed", str(BC_SEED + r)]
                args += ["--episodes", str(a.dagger_episodes), "--workers", str(a.workers), "--out", str(more)]
                e2e("record", args, out / "record.log")
            extra = load(more)
            data = {k: np.concatenate([data[k], extra[k]]) for k in ("obs", "act")}
            print(f"DAgger round {r}: imitating {len(data['act'])} decisions", flush=True)
            imitate(policy, data, a, device, a.dagger_epochs)
            rates = evaluate(out / f"d{r}.npz", out / f"eval_d{r}.json", a, out / "eval.log")
            print(json.dumps({"dagger": r, "eval_before_round": rates}), flush=True)
        export(policy, out / "e0.npz")
        print(json.dumps({"iteration": 0, "eval": evaluate(out / "e0.npz", out / "eval0.json", a, out / "eval.log")}))

    for it in range(start, a.iterations):
        t0 = time.time()
        weights = out / f"e{it}.npz"
        export(policy, weights)
        play = out / f"it{it}.npz"
        args = ["--weights", str(weights), "--episodes", str(a.episodes), "--seed", str(1000 + it)]
        e2e("play", [*args, "--workers", str(a.workers), "--out", str(play)], out / "play.log")
        data = load(play)
        t1 = time.time()
        stats = update(policy, value, p_opt, v_opt, data, a, device, policy_too=it >= a.value_warmup)
        outcomes = [str(o) for o in data["outcome"]]
        rates = {
            k: round(float(np.mean([o == SUCCESS for o, kk in zip(outcomes, data["keeper"]) if kk == k])), 4)
            for k in sorted(set(data["keeper"].tolist()))
        }
        fouls = sum(o.startswith("stopped:") and o != "stopped:out_of_bounds" for o in outcomes) / len(outcomes)
        row = {"iteration": it, "decisions": len(data["act"]), "goal_rate": rates, "fouls": round(fouls, 4), **stats}
        row["std"] = [round(float(s), 3) for s in policy.log_std.exp().detach().cpu()]
        row["play_s"], row["learn_s"] = round(t1 - t0), round(time.time() - t1)
        ckpt = {
            "policy": policy.state_dict(),
            "value": value.state_dict(),
            "p_opt": p_opt.state_dict(),
            "v_opt": v_opt.state_dict(),
            "iteration": it + 1,
        }
        torch.save(ckpt, out / "ckpt.pt")
        play.unlink()
        if (it + 1) % a.eval_every == 0:
            export(policy, out / f"e{it + 1}.npz")
            row["eval"] = evaluate(out / f"e{it + 1}.npz", out / f"eval{it + 1}.json", a, out / "eval.log")
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
