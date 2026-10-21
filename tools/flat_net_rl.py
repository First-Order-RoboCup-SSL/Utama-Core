"""flat_net_rl.py — self-play reinforcement learning (PPO) for the flat engine's policy network, after
AlphaGo's reinforcement-learning stage: the network plays full matches against itself, against earlier
versions of itself (a league, so it does not only learn to beat its latest self) and, as an anchor,
against the round-robin's pool, drawing its actions (`tools/flat_round.py`, on the CPU); then it learns
from the goals that followed (on the GPU, or `--device cpu` when another job fills it), and again. No
tree search: the simulator cannot snapshot and restore a match, and one 10 s look-ahead costs seconds.

  CUDA_VISIBLE_DEVICES=GPU-5206cf34-8623-f6a9-dace-e0fb425f562f \\
    /workspace/FOR-Engine/.pixi/envs/sglang/bin/python tools/flat_net_rl.py \\
    --init replays/flatnet/bc1.npz --out replays/flatnet/rl2 --iterations 40

Iteration k plays with <out>/w<k>.npz into <out>/it<k>/: itself at `--self-rates` rates x 4 cells (both
teams' questions are training data), `--past` earlier versions x 4 cells and the pool x `--pool-cells`
random cells (0: pure self-play; ours only). Each robot's questions in a match are its own trajectory;
its reward from one question to its next is its team's goal difference in between, discounted per
second of game time (`--gamma`), with a critic of the same shape for the baseline (GAE, `--lam`). The
update is PPO's clipped objective with an entropy bonus and a KL penalty towards the starting network
(`--kl`); it writes <out>/w<k+1>.npz, and the same command resumes from the last weights. Every
`--eval-every` iterations the network plays at its best (no draws) the pool (4 cells at 1 Hz) and the
starting network, the rule's copy (8 matches). <out>/log.jsonl: a line per iteration.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from flat_net_train import FEATURES, Net, export, load_weights

ROOT = Path(__file__).resolve().parents[1]
CELLS = ["Rk", "R", "Lk", "L"]
KEYS = (*FEATURES, "ego", "mask", "act", "logp", "t", "robot", "score_us", "score_them")


def flat_round(args: list[str], log) -> subprocess.Popen:
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    cmd = ["pixi", "run", "python", "-m", "tools.flat_round", *args]
    return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)


def play(out: Path, it: int, a: argparse.Namespace) -> None:
    """Iteration `it`'s matches, three runs at once: itself, earlier versions, the pool."""
    rng = random.Random(it)
    weights = out / f"w{it}.npz"
    rates = [f"{r / 1000:.3f}" for r in rng.sample(range(940, 1061), a.self_rates + 2)]
    past = [f"net:{out / f'w{j}.npz'}" for j in rng.sample(range(it), min(a.past, it))]
    temp = ["--vs-temperature", str(a.temperature)]
    runs = [
        (a.self_rates, [f"net:{weights}", "--rates", ",".join(rates[: a.self_rates]), "--record-vs", *temp]),
        (len(past), [*past, "--rates", rates[-2], *temp]),
        (a.pool_cells, ["pool", "--rates", rates[-1], "--cells", ",".join(rng.sample(CELLS, a.pool_cells))]),
    ]
    common = ["--policy", "net", "--weights", str(weights), "--temperature", str(a.temperature)]
    common += ["--seed", str(it), "--record", "--duration", str(a.duration), "--out", str(out / f"it{it}")]
    with open(out / f"it{it}.log", "a") as log:
        procs = [
            flat_round([*args, *common, "--workers", str(workers)], log)
            for (on, args), workers in zip(runs, (a.self_workers, a.past_workers, a.pool_workers))
            if on
        ]
        for proc in procs:
            proc.wait()


def evaluate(out: Path, k: int, a: argparse.Namespace) -> dict:
    """w<k> at its best (no draws) against the pool, 4 cells at 1 Hz, and against the starting
    network, the rule's copy (4 cells at 2 rates)."""
    common = ["--policy", "net", "--weights", str(out / f"w{k}.npz"), "--out", str(out / f"eval{k}")]
    common += ["--duration", str(a.duration)]
    with open(out / f"eval{k}.log", "a") as log:
        procs = [
            flat_round([*a.eval_opponents, "--rates", "1", "--workers", "24", *common], log),
            flat_round([f"net:{out / 'w0.npz'}", "--rates", "0.97,1.03", "--workers", "4", *common], log),
        ]
        for proc in procs:
            proc.wait()
    results = [json.loads(p.read_text()) for p in (out / f"eval{k}").glob("*.result.json")]
    return {
        "pool": summary(results, lambda r: not r["opponent"].startswith("vs-")),
        "split_shape": summary(results, lambda r: r["opponent"] == "split_shape"),
        "vs_start": summary(results, lambda r: r["opponent"] == "vs-w0"),
    }


def load(it_dir: Path) -> tuple[dict[str, np.ndarray], list[dict], list[dict]]:
    """The iteration's questions; per question its team's index into `teams` (final score from that
    team's side); and the match results."""
    parts: dict[str, list] = {k: [] for k in (*KEYS, "team")}
    teams, results = [], []
    for res_path in sorted(it_dir.glob("*.result.json")):
        res = json.loads(res_path.read_text())
        results.append(res)
        tag = res_path.name[: -len(".result.json")]
        for suffix, gf, ga in ((".npz", res["gf"], res["ga"]), (".b.npz", res["ga"], res["gf"])):
            if not (it_dir / f"{tag}{suffix}").exists():
                continue
            teams.append({"gf": gf, "ga": ga})
            with np.load(it_dir / f"{tag}{suffix}") as z:
                for k in KEYS:
                    parts[k].append(z[k])
                parts["team"].append(np.full(len(z["act"]), len(teams) - 1))
    return {k: np.concatenate(v) for k, v in parts.items()}, teams, results


def rewards(data: dict[str, np.ndarray], teams: list[dict], gamma: float):
    """Per question: its team's goal difference until its robot's next question in that match (to full
    time after its last), the discount to that next question, and the next question's index (-1: last)."""
    n = len(data["act"])
    r, disc, nxt = np.zeros(n), np.zeros(n), np.full(n, -1)
    us, them, t = data["score_us"].astype(float), data["score_them"].astype(float), data["t"]
    for m, team in enumerate(teams):
        mine = data["team"] == m
        for rid in np.unique(data["robot"][mine]):
            idx = np.flatnonzero(mine & (data["robot"] == rid))  # recorded in time order
            cur, after = idx[:-1], idx[1:]
            r[cur] = (us[after] - us[cur]) - (them[after] - them[cur])
            disc[cur] = gamma ** (t[after] - t[cur])
            nxt[cur] = after
            last = idx[-1]
            r[last] = (team["gf"] - us[last]) - (team["ga"] - them[last])
    return r, disc, nxt


def gae(r: np.ndarray, disc: np.ndarray, nxt: np.ndarray, v: np.ndarray, lam: float) -> np.ndarray:
    adv = np.zeros(len(r))
    for j in range(len(r) - 1, -1, -1):  # a question's next one is always recorded after it
        k = nxt[j]
        if k < 0:
            adv[j] = r[j] - v[j]
        else:
            adv[j] = r[j] + disc[j] * v[k] - v[j] + disc[j] * lam * adv[k]
    return adv


def to_dev(data: dict[str, np.ndarray], dev: torch.device) -> dict[str, torch.Tensor]:
    out = {k: torch.from_numpy(data[k]).to(dev) for k in FEATURES}
    out["ego"] = torch.from_numpy(data["ego"].astype(np.int64)).to(dev)
    out["mask"] = torch.from_numpy(data["mask"]).to(dev)
    out["act"] = torch.from_numpy(data["act"].astype(np.int64)).to(dev)
    out["logp"] = torch.from_numpy(data["logp"].astype(np.float32)).to(dev)
    return out


@torch.no_grad()
def values(critic: Net, d: dict[str, torch.Tensor], size: int = 16384) -> np.ndarray:
    n = len(d["act"])
    return np.concatenate(
        [critic.value(*[d[k][s : s + size].float() for k in FEATURES]).cpu().numpy() for s in range(0, n, size)]
    )


def fit_critic(critic: Net, opt, d: dict[str, torch.Tensor], ret: torch.Tensor, epochs: int, batch: int) -> float:
    n, loss = len(ret), torch.tensor(0.0)
    for _ in range(epochs):
        for j in torch.randperm(n, device=ret.device).split(batch):
            loss = ((critic.value(*[d[k][j].float() for k in FEATURES]) - ret[j]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt.step()
    return loss.item()


def ppo(policy: Net, ref: Net, opt, d: dict[str, torch.Tensor], adv: torch.Tensor, a: argparse.Namespace) -> dict:
    n, stats = len(adv), {"kl_old": 0.0, "kl_ref": 0.0, "entropy": 0.0, "clipped": 0.0, "steps": 0}
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    for _ in range(a.epochs):
        for j in torch.randperm(n, device=adv.device).split(a.batch):
            feats, ego, mask = [d[k][j].float() for k in FEATURES], d["ego"][j], d["mask"][j]
            logp_all = torch.log_softmax(policy(*feats, ego).masked_fill(~mask, -1e9) / a.temperature, 1)
            logp = logp_all.gather(1, d["act"][j][:, None])[:, 0]
            ratio = torch.exp(logp - d["logp"][j])
            surr = torch.min(ratio * adv[j], ratio.clamp(1 - a.clip, 1 + a.clip) * adv[j]).mean()
            p = logp_all.exp()
            entropy = -(p * logp_all.masked_fill(~mask, 0.0)).sum(1).mean()
            with torch.no_grad():
                ref_logp = torch.log_softmax(ref(*feats, ego).masked_fill(~mask, -1e9) / a.temperature, 1)
            kl_ref = (p * (logp_all - ref_logp).masked_fill(~mask, 0.0)).sum(1).mean()
            loss = -surr - a.entropy * entropy + a.kl * kl_ref
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()
            with torch.no_grad():
                stats["kl_old"] += float((d["logp"][j] - logp).mean())
                stats["kl_ref"] += float(kl_ref)
                stats["entropy"] += float(entropy)
                stats["clipped"] += float(((ratio - 1).abs() > a.clip).float().mean())
                stats["steps"] += 1
    return {k: v / max(1, stats["steps"]) for k, v in stats.items() if k != "steps"}


def summary(results: list[dict], pick=lambda r: True) -> dict:
    rows = [r for r in results if pick(r)]
    w = sum(r["gf"] > r["ga"] for r in rows)
    d = sum(r["gf"] == r["ga"] for r in rows)
    gf, ga = sum(r["gf"] for r in rows), sum(r["ga"] for r in rows)
    return {"n": len(rows), "w": w, "d": d, "l": len(rows) - w - d, "gf": gf, "ga": ga}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--init", required=True, help="starting weights (the imitation network)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--self-rates", type=int, default=4, help="matches against itself: rates x 4 cells")
    ap.add_argument("--past", type=int, default=3, help="earlier versions played, 4 cells each")
    ap.add_argument("--pool-cells", type=int, default=2, help="cells against each pool strategy (0: none)")
    ap.add_argument("--self-workers", type=int, default=8)
    ap.add_argument("--past-workers", type=int, default=6)
    ap.add_argument("--pool-workers", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.99, help="discount per second of game time")
    ap.add_argument("--lam", type=float, default=0.98)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--entropy", type=float, default=0.003)
    ap.add_argument("--kl", type=float, default=0.02, help="penalty on the KL divergence from the starting network")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--critic-lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--critic-warmup", type=int, default=8, help="critic epochs on the first iteration's returns")
    ap.add_argument("--eval-every", type=int, default=5, help="iterations between greedy checks (0: none)")
    ap.add_argument("--eval-opponents", nargs="*", default=["pool"])
    ap.add_argument("--duration", type=float, default=1200.0, help="sim-time cap (full time ends a match first)")
    ap.add_argument("--device", default="cuda", help="where it learns: cuda, or cpu when another job fills the GPU")
    a = ap.parse_args()

    dev = torch.device(a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "w0.npz").exists():
        shutil.copy(a.init, out / "w0.npz")
    start = max(int(p.stem[1:]) for p in out.glob("w*.npz"))
    with np.load(out / "w0.npz") as z:
        cfg = json.loads(str(z["config"]))
        dims = {k: z[f"in_{k}_w"].shape[0] for k in FEATURES}
        n_fixed = z["fixed_w"].shape[1]
    shape = (dims, n_fixed, cfg["d"], cfg["layers"], cfg["heads"], cfg["ff"], cfg["dk"])
    policy, ref, critic = Net(*shape).to(dev), Net(*shape).to(dev), Net(*shape, value=True).to(dev)
    load_weights(ref, out / "w0.npz")
    load_weights(policy, out / f"w{start}.npz")
    ref.requires_grad_(False)
    opt_pi = torch.optim.Adam(policy.parameters(), lr=a.lr)
    opt_v = torch.optim.Adam(critic.parameters(), lr=a.critic_lr)
    warm = True
    if (out / "critic.pt").exists():
        state = torch.load(out / "critic.pt", map_location=dev)
        critic.load_state_dict(state["critic"])
        opt_v.load_state_dict(state["opt_v"])
        opt_pi.load_state_dict(state["opt_pi"])
        warm = False

    for it in range(start, a.iterations):
        t0 = time.time()
        play(out, it, a)
        t_play = time.time() - t0
        data, teams, results = load(out / f"it{it}")
        if not results:
            raise SystemExit(f"iteration {it}: no match finished, see {out}/it{it}.log")
        r, disc, nxt = rewards(data, teams, a.gamma)
        d = to_dev(data, dev)
        if warm:  # a fresh critic first learns this iteration's plain discounted returns
            mc = gae(r, disc, nxt, np.zeros(len(r)), 1.0)
            fit_critic(critic, opt_v, d, torch.tensor(mc, dtype=torch.float32, device=dev), a.critic_warmup, a.batch)
            warm = False
        v = values(critic, d)
        adv = gae(r, disc, nxt, v, a.lam)
        ret = adv + v
        explained = 1.0 - float(np.var(ret - v) / max(np.var(ret), 1e-8))
        stats = ppo(policy, ref, opt_pi, d, torch.tensor(adv, dtype=torch.float32, device=dev), a)
        v_loss = fit_critic(critic, opt_v, d, torch.tensor(ret, dtype=torch.float32, device=dev), a.epochs, a.batch)
        sample = {k: data[k][:16] for k in (*FEATURES, "ego")}
        export(policy, sample, {**cfg, "rl": vars(a), "iteration": it + 1}, out / f"w{it + 1}.npz")
        torch.save(
            {"critic": critic.state_dict(), "opt_v": opt_v.state_dict(), "opt_pi": opt_pi.state_dict()},
            out / "critic.pt",
        )
        line = {
            "iteration": it,
            "self": summary(results, lambda r: r["opponent"] == f"vs-w{it}"),
            "past": summary(results, lambda r: r["opponent"].startswith("vs-") and r["opponent"] != f"vs-w{it}"),
            "pool": summary(results, lambda r: not r["opponent"].startswith("vs-")),
            "split_shape": summary(results, lambda r: r["opponent"] == "split_shape"),
            "questions": len(r),
            "explained_variance": round(explained, 3),
            "critic_loss": round(v_loss, 4),
            **{k: round(v_, 5) for k, v_ in stats.items()},
            "play_s": round(t_play),
            "learn_s": round(time.time() - t0 - t_play),
        }
        if a.eval_every and (it + 1) % a.eval_every == 0:
            line["greedy"] = evaluate(out, it + 1, a)
            g, s, v = (line["greedy"][k] for k in ("pool", "split_shape", "vs_start"))
            print(
                f"  w{it + 1} at its best: pool {g['w']}-{g['d']}-{g['l']} {g['gf']}-{g['ga']} (split_shape "
                f"{s['w']}-{s['d']}-{s['l']} {s['gf']}-{s['ga']}), vs the rule's copy {v['w']}-{v['d']}-{v['l']} "
                f"{v['gf']}-{v['ga']}",
                flush=True,
            )
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
        wdl = {k: "{w}-{d}-{l} {gf}-{ga}".format(**line[k]) for k in ("self", "past", "pool", "split_shape")}
        print(
            f"it {it}: self {wdl['self']}, past {wdl['past']}, pool {wdl['pool']} (split_shape "
            f"{wdl['split_shape']})  entropy {stats['entropy']:.3f} kl_ref {stats['kl_ref']:.4f} "
            f"kl_old {stats['kl_old']:.4f} clipped {stats['clipped']:.3f} explained {explained:.2f}  "
            f"{line['play_s']}+{line['learn_s']}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
