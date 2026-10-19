"""flat_net_train.py — train the flat engine's policy network on recorded questions
(`tools/flat_round.py --record`) on the GPU, by imitating the scripted rule's pick, and export the weights
`utama_core/tactics/flat_net.py` runs on the CPU.

Imports only numpy and torch. Utama-Core's environment has no torch; any CUDA torch works, e.g.
FOR-Engine's (pin the GPU by UUID):

  CUDA_VISIBLE_DEVICES=GPU-5206cf34-8623-f6a9-dace-e0fb425f562f \\
    /workspace/FOR-Engine/.pixi/envs/sglang/bin/python tools/flat_net_train.py replays/flatnet/data1 \\
    --out replays/flatnet/bc1.npz

Every 10th match file is held out for validation. The exported file also carries this network's logits
for a few held-out questions (`check_*`), which `FlatNet.load` must reproduce in numpy.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SLOTS = 6  # flat_net.SLOTS
FEATURES = ("ours", "theirs", "ball", "glob")


def load(paths: list[Path], target: str) -> dict[str, np.ndarray]:
    parts: dict[str, list] = {k: [] for k in (*FEATURES, "ego", "mask", "y")}
    for p in paths:
        with np.load(p) as z:
            for k in (*FEATURES, "ego", "mask"):
                parts[k].append(z[k])
            parts["y"].append(z[target])
    return {k: np.concatenate(v) for k, v in parts.items()}


class Net(torch.nn.Module):
    """`flat_net.FlatNet` in torch: the same weights, names and (in, out) matrices."""

    def __init__(
        self, dims: dict[str, int], n_fixed: int, d: int, layers: int, heads: int, ff: int, dk: int, value: bool = False
    ):
        """`value`: a value head on the global token instead (`tools/flat_net_rl.py`'s critic)."""
        super().__init__()
        self.layers, self.heads = layers, heads
        self.w = torch.nn.ParameterDict()

        def lin(name: str, i: int, o: int, bias: bool = True) -> None:
            self.w[f"{name}_w"] = torch.nn.Parameter(torch.randn(i, o) / math.sqrt(i))
            if bias:
                self.w[f"{name}_b"] = torch.nn.Parameter(torch.zeros(o))

        for k in FEATURES:
            lin(f"in_{k}", dims[k], d)
        self.w["pos"] = torch.nn.Parameter(torch.randn(2 + 2 * SLOTS, d) * 0.02)
        for i in range(layers):
            for ln in ("ln1", "ln2"):
                self.w[f"l{i}_{ln}_g"] = torch.nn.Parameter(torch.ones(d))
                self.w[f"l{i}_{ln}_b"] = torch.nn.Parameter(torch.zeros(d))
            lin(f"l{i}_qkv", d, 3 * d)
            lin(f"l{i}_out", d, d)
            lin(f"l{i}_ff1", d, ff)
            lin(f"l{i}_ff2", ff, d)
        self.w["ln_f_g"] = torch.nn.Parameter(torch.ones(d))
        self.w["ln_f_b"] = torch.nn.Parameter(torch.zeros(d))
        if value:
            lin("value", d, 1)
            return
        lin("fixed", d, n_fixed)
        for name in ("pass_q", "pass_k", "mark_q", "mark_k"):
            lin(name, d, dk, bias=False)

    def trunk(self, ours, theirs, ball, glob):
        w = self.w
        x = torch.cat(
            [
                (glob @ w["in_glob_w"] + w["in_glob_b"])[:, None],
                ours @ w["in_ours_w"] + w["in_ours_b"],
                theirs @ w["in_theirs_w"] + w["in_theirs_b"],
                (ball @ w["in_ball_w"] + w["in_ball_b"])[:, None],
            ],
            1,
        )
        x = x + w["pos"]
        b, n, d = x.shape
        h_, dh = self.heads, d // self.heads
        one = ours.new_ones(b, 1)
        present = torch.cat([one, ours[:, :, 0], theirs[:, :, 0], one], 1) > 0.5
        bias = torch.zeros(b, 1, 1, n, device=x.device).masked_fill(~present[:, None, None, :], -1e9)
        for i in range(self.layers):
            h = F.layer_norm(x, (d,), w[f"l{i}_ln1_g"], w[f"l{i}_ln1_b"], 1e-5)
            q, k, v = (h @ w[f"l{i}_qkv_w"] + w[f"l{i}_qkv_b"]).split(d, -1)
            q, k, v = (t.reshape(b, n, h_, dh).transpose(1, 2) for t in (q, k, v))
            a = (q @ k.transpose(-1, -2) / math.sqrt(dh) + bias).softmax(-1)
            x = x + (a @ v).transpose(1, 2).reshape(b, n, d) @ w[f"l{i}_out_w"] + w[f"l{i}_out_b"]
            h = F.layer_norm(x, (d,), w[f"l{i}_ln2_g"], w[f"l{i}_ln2_b"], 1e-5)
            x = x + F.relu(h @ w[f"l{i}_ff1_w"] + w[f"l{i}_ff1_b"]) @ w[f"l{i}_ff2_w"] + w[f"l{i}_ff2_b"]
        return F.layer_norm(x, (d,), w["ln_f_g"], w["ln_f_b"], 1e-5)

    def value(self, ours, theirs, ball, glob):
        return (self.trunk(ours, theirs, ball, glob)[:, 0] @ self.w["value_w"] + self.w["value_b"])[:, 0]

    def forward(self, ours, theirs, ball, glob, ego):
        w = self.w
        x = self.trunk(ours, theirs, ball, glob)
        b = x.shape[0]
        me = x[torch.arange(b, device=x.device), 1 + ego]
        dk = w["pass_q_w"].shape[1]
        fixed = me @ w["fixed_w"] + w["fixed_b"]
        passes = torch.einsum("bsk,bk->bs", x[:, 1 : 1 + SLOTS] @ w["pass_k_w"], me @ w["pass_q_w"]) / math.sqrt(dk)
        marks = torch.einsum(
            "bsk,bk->bs", x[:, 1 + SLOTS : 1 + 2 * SLOTS] @ w["mark_k_w"], me @ w["mark_q_w"]
        ) / math.sqrt(dk)
        return torch.cat([fixed, passes, marks], 1)


def menu_loss(logits: torch.Tensor, mask: torch.Tensor, y: torch.Tensor, smooth: float = 0.0) -> torch.Tensor:
    """Cross-entropy over the menu; `smooth` spreads that share of the target evenly over the menu, so the
    network keeps some probability on every option (exploration for `tools/flat_net_rl.py`)."""
    logp = torch.log_softmax(logits.masked_fill(~mask, -1e9), 1)
    nll = -logp.gather(1, y[:, None])[:, 0]
    if smooth <= 0:
        return nll.mean()
    uniform = -(logp.masked_fill(~mask, 0.0)).sum(1) / mask.sum(1)
    return ((1 - smooth) * nll + smooth * uniform).mean()


def export(model: "Net", sample: dict[str, np.ndarray], cfg: dict, out: str | Path) -> None:
    """The weights `FlatNet.load` reads, with this network's own logits for the first questions of
    `sample` (computed on the CPU in float32) for its check."""
    cpu = copy.deepcopy(model).float().cpu().eval()
    k = np.arange(min(16, len(sample["ego"])))
    with torch.no_grad():
        check = cpu(
            *[torch.from_numpy(sample[f][k]).float() for f in FEATURES],
            torch.from_numpy(sample["ego"][k].astype(np.int64)),
        )
    weights = {name: p.detach().numpy().astype(np.float32) for name, p in cpu.w.items()}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        config=np.array(json.dumps(cfg)),
        **weights,
        **{f"check_{f}": sample[f][k].astype(np.float32) for f in FEATURES},
        check_ego=sample["ego"][k],
        check_logits=check.numpy(),
    )


def load_weights(model: "Net", path: str | Path) -> dict:
    """Copy an exported weights file into `model` (same shape); returns its config."""
    with np.load(path) as z:
        for name, p in model.w.items():
            p.data.copy_(torch.from_numpy(z[name]))
        return json.loads(str(z["config"]))


def batches(data: dict[str, torch.Tensor], idx: torch.Tensor, size: int):
    for s in range(0, len(idx), size):
        j = idx[s : s + size]
        feats = [data[k][j].float() for k in FEATURES]
        yield feats, data["ego"][j], data["mask"][j], data["y"][j]


@torch.no_grad()
def evaluate(model: Net, data: dict[str, torch.Tensor], size: int = 16384) -> dict[str, float]:
    model.eval()
    hits, carrier_hits, n_carrier, loss = 0, 0, 0, 0.0
    n = len(data["y"])
    for feats, ego, mask, y in batches(data, torch.arange(n, device=data["y"].device), size):
        logits = model(*feats, ego).masked_fill(~mask, -1e9)
        loss += F.cross_entropy(logits, y, reduction="sum").item()
        ok = logits.argmax(1) == y
        carrier = mask[:, 0]  # only a carrier's menu has shoot
        hits += ok.sum().item()
        carrier_hits += (ok & carrier).sum().item()
        n_carrier += carrier.sum().item()
    model.train()
    return {
        "loss": loss / n,
        "acc": hits / n,
        "acc_carrier": carrier_hits / max(1, n_carrier),
        "acc_off_ball": (hits - carrier_hits) / max(1, n - n_carrier),
        "n": n,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("data", nargs="+", help="directories of recorded matches (*.npz)")
    ap.add_argument("--out", required=True, help="weights file to write (.npz)")
    ap.add_argument(
        "--target", choices=("rule", "act"), default="rule", help="imitate the rule's pick or the action taken"
    )
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ff", type=int, default=256)
    ap.add_argument("--dk", type=int, default=32)
    ap.add_argument("--smooth", type=float, default=0.0, help="label smoothing over the menu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device("cuda")
    files = sorted(p for d in a.data for p in Path(d).glob("*.npz"))
    val_files = files[::10]
    train_files = [p for p in files if p not in set(val_files)]
    t0 = time.time()
    train_np, val_np = load(train_files, a.target), load(val_files, a.target)
    print(
        f"{len(train_files)} train / {len(val_files)} val matches, {len(train_np['y'])} / {len(val_np['y'])} questions "
        f"({time.time() - t0:.0f}s to load)",
        flush=True,
    )

    def to_dev(arrs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        out = {k: torch.from_numpy(arrs[k]).to(dev) for k in FEATURES}  # float16 on the GPU
        out["ego"] = torch.from_numpy(arrs["ego"].astype(np.int64)).to(dev)
        out["mask"] = torch.from_numpy(arrs["mask"]).to(dev)
        out["y"] = torch.from_numpy(arrs["y"].astype(np.int64)).to(dev)
        return out

    train, val = to_dev(train_np), to_dev(val_np)
    dims = {k: train_np[k].shape[-1] for k in FEATURES}
    n_actions = train_np["mask"].shape[1]
    model = Net(dims, n_actions - 2 * SLOTS, a.d, a.layers, a.heads, a.ff, a.dk).to(dev)
    print(f"{sum(p.numel() for p in model.parameters())} parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    n = len(train["y"])
    total = a.epochs * math.ceil(n / a.batch)
    warm = max(1, total // 30)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))
    )
    for epoch in range(a.epochs):
        t1, run_loss, seen = time.time(), 0.0, 0
        for feats, ego, mask, y in batches(train, torch.randperm(n, device=dev), a.batch):
            loss = menu_loss(model(*feats, ego), mask, y, a.smooth)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run_loss += loss.item() * len(y)
            seen += len(y)
        m = evaluate(model, val)
        print(
            f"epoch {epoch + 1}/{a.epochs}  train loss {run_loss / seen:.4f}  val loss {m['loss']:.4f}  "
            f"acc {m['acc']:.4f} (carrier {m['acc_carrier']:.4f}, off the ball {m['acc_off_ball']:.4f})  "
            f"{time.time() - t1:.0f}s",
            flush=True,
        )

    cfg = {
        "layers": a.layers,
        "heads": a.heads,
        "d": a.d,
        "ff": a.ff,
        "dk": a.dk,
        "target": a.target,
        "smooth": a.smooth,
        "data": [str(d) for d in a.data],
        "val": m,
    }
    export(model, val_np, cfg, a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
