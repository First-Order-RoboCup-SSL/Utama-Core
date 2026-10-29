"""Expected goals (xG) fitted on real shots, and the split of goals into shots × shot quality ×
finishing.

- **Model:** logistic regression (numpy IRLS, small L2 on standardised features) on the
  `FEATURES` of each attempt, from the shooter's view.
- **Check** (`cross_validate`): k-fold cross-validation grouped by match, so no match is in both
  train and test; log loss and Brier against a constant rate, AUC, calibration by decile.
- **Decomposition** (`per_match`, `factors`), per team per 10 live minutes: goals from shots =
  shots × mean real xG per shot × finishing (goals from shots ÷ summed xG). A factor's sim ÷
  real ratio says where a goal gap comes from. Goals with no linked shot are counted apart.
"""

from __future__ import annotations

import math

import numpy as np

SET_PIECES = ("free_kick", "kickoff", "penalty")
FEATURES = ("log_dist", "angle", "open_goal", "og_keeper", "no_keeper", "peak", "free_kick", "kickoff", "penalty")
L2 = 1.0
MIN_LIVE_S = 60.0  # matches with less live play are dropped; the 10-goal-rule blowouts are short, keep them


def row(s: dict) -> list[float]:
    return [
        math.log(max(s["distance_m"], 0.2)),
        math.radians(s["angle_deg"]),
        s["open_goal"],
        s["og_keeper"],
        0.0 if s["keeper_in_place"] else 1.0,
        min(s["peak"], 8.0),
        float(s["context"] == "free_kick"),
        float(s["context"] == "kickoff"),
        float(s["context"] == "penalty"),
    ]


def design(matches: list[dict]):
    """(X, y, match index) over every shot of `matches`."""
    X, y, g = [], [], []
    for i, m in enumerate(matches):
        for s in m["shots"]:
            X.append(row(s))
            y.append(float(s["scored"]))
            g.append(i)
    return np.array(X).reshape(-1, len(FEATURES)), np.array(y), np.array(g)


class Logit:
    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        w = np.zeros(Z.shape[1])
        reg = np.full(Z.shape[1], L2)
        reg[0] = 0.0
        for _ in range(100):
            p = 1 / (1 + np.exp(-Z @ w))
            grad = Z.T @ (p - y) + reg * w
            H = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(reg)
            step = np.linalg.solve(H, grad)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.w = w
        return self

    def predict(self, X):
        Z = np.c_[np.ones(len(X)), (np.asarray(X) - self.mu) / self.sd]
        return 1 / (1 + np.exp(-Z @ self.w))

    def to_dict(self) -> dict:
        return {"features": list(FEATURES), "mu": self.mu.tolist(), "sd": self.sd.tolist(), "w": self.w.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Logit":
        if tuple(d["features"]) != FEATURES:
            raise ValueError(f"xG model features {d['features']} differ from {FEATURES}")
        m = cls()
        m.mu, m.sd, m.w = np.array(d["mu"]), np.array(d["sd"]), np.array(d["w"])
        return m


def auc(p, y) -> float:
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logloss(p, y) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def cross_validate(X, y, g, k: int = 5, seed: int = 0) -> dict:
    """Held-out predictions by match fold: log loss, Brier and AUC, each against a constant rate,
    and calibration by decile as (mean xG, observed rate, n)."""
    groups = np.unique(g)
    np.random.default_rng(seed).shuffle(groups)
    p, base = np.empty(len(y)), np.empty(len(y))
    for f in np.array_split(groups, k):
        test = np.isin(g, f)
        p[test] = Logit().fit(X[~test], y[~test]).predict(X[test])
        base[test] = y[~test].mean()
    edges = np.quantile(p, np.linspace(0, 1, 11))
    calibration = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p <= hi)
        calibration.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return {
        "shots": len(y),
        "scored": int(y.sum()),
        "log_loss": logloss(p, y),
        "log_loss_constant": logloss(base, y),
        "brier": float(np.mean((p - y) ** 2)),
        "brier_constant": float(np.mean((base - y) ** 2)),
        "auc": auc(p, y),
        "calibration": calibration,
    }


def per_match(m: dict, model: Logit, kind: str = "all") -> np.ndarray:
    """Per team per 10 live minutes for one match: shots, summed xG, goals from shots, goals with
    no linked shot (counted only for `kind="all"`). `kind`: all, open_play or set_piece."""
    shots = [s for s in m["shots"] if kind == "all" or (s["context"] in SET_PIECES) == (kind == "set_piece")]
    xg = model.predict(np.array([row(s) for s in shots])).sum() if shots else 0.0
    scale = 600.0 / m["live_s"] / 2
    unlinked = sum(m["unshot_goals"].values()) if kind == "all" else 0
    return np.array([len(shots), xg, sum(s["scored"] for s in shots), unlinked]) * scale


def factors(rows: np.ndarray) -> dict:
    shots, xg, goals, unlinked = rows.mean(0)
    return {
        "shots": shots,
        "xg_per_shot": xg / shots if shots else float("nan"),
        "finishing": goals / xg if xg else float("nan"),
        "goals_from_shots": goals,
        "unlinked_goals": unlinked,
    }
