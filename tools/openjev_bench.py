"""openjev_bench.py — latency benchmark for OpenJev served by FOR-Engine.

Measures what one per-tick partition decision would cost against a running
FOR-Engine server (`for-engine serve`, default http://127.0.0.1:3000), before
any Utama integration exists:

  1. boundary   — can a prewarmed *static* state prefix be reused by later
                  requests whose state is `static + dynamic`? Tries a few
                  delimiters between the two parts and checks the server's
                  `partial_hits` counter, since BPE merges at the join can
                  silently break reuse.
  2. sweep      — latency vs dynamic-part size x option count, with the static
                  prefix cached (the planned steady state).
  3. questions  — 1 vs 3 questions per request on the same state.
  4. baselines  — no reuse at all (full prefill every tick) and an exact
                  repeat of the previous state (full hit).

States are synthetic but shaped like the planned encoder output (static
glossary first, bucketed live features last), so token counts are
representative. Stdlib only.

Run:
    pixi run python -m tools.openjev_bench [--url http://127.0.0.1:3000] [--reps 5] [--out results.json]
"""

from __future__ import annotations

import argparse
import http.client
import json
import random
import statistics
import time
from typing import Any
from urllib.parse import urlparse

STATIC_STATE = {
    "role": "You are the coach of the YELLOW team in a 6v6 RoboCup small-size-league robot soccer match. "
    "Each tick you choose how our 5 outfield robots are split across team tactics. Robot 0 is our "
    "goalkeeper and is never part of the split.",
    "pitch": "9m x 6m. We attack towards the goal on the LEFT (-x). Thirds are named from our point of view: "
    "own (nearest our goal), mid, final (nearest their goal). Lanes: left, centre, right as seen facing "
    "the goal we attack.",
    "tactics": {
        "give_and_go": "2+ attackers pass and move in repeated one-twos until a shot opens",
        "decoy_overload": "one attacker drags a marker wide, another attacks the space it leaves",
        "press": "one robot closes down the ball carrier, the rest mark passing outlets",
        "shadow_mark": "first two defenders block the ball-to-goal shot line, the rest man-mark",
        "block_shape": "zone screen in front of our own area, no man-marking",
        "clear_ball": "win the ball deep in our third and kick it out of danger",
    },
    "reading_the_state": "possession is decided by who is clearly closer to the ball; distances are "
    "rounded to 0.1m; counts are precomputed; `recent` lists the latest events, newest first.",
}

ZONES = ["own", "mid", "final"]
LANES = ["left", "centre", "right"]
TACTIC_NAMES = list(STATIC_STATE["tactics"])
EVENTS = [
    "we lost possession",
    "we won possession",
    "shot by us, saved",
    "shot by them, wide",
    "ball went out, their throw-in",
    "our pass completed",
    "their pass intercepted",
]

OPTION_POOL = [
    ("build_up_3_2", "3 give_and_go attackers + 2 shadow_mark defenders"),
    ("press_3_2", "3 press + 2 shadow_mark"),
    ("overload_2_3", "2 decoy_overload attackers + 3 shadow_mark"),
    ("low_block_all", "all 5 block_shape"),
    ("clear_danger", "1-2 clear_ball, rest block_shape"),
    ("all_press", "all 5 press"),
    ("build_up_4_1", "4 give_and_go + 1 shadow_mark"),
    ("counter_3_2", "3 give_and_go + 2 block_shape"),
    ("press_block_3_2", "3 press + 2 block_shape"),
    ("overload_press_2_3", "2 decoy_overload + 3 press"),
    ("mark_all", "all 5 shadow_mark"),
    ("build_up_2_3", "2 give_and_go + 3 shadow_mark"),
    ("final_overload_2_3", "2 decoy_overload + 3 block_shape"),
    ("clear_press", "2 clear_ball + 3 press"),
    ("hold_shape", "keep the current split unchanged"),
    ("other", "none of the above fits"),
]


def _dist() -> float:
    return round(random.uniform(0.1, 4.0), 1)


def dynamic_state(n_history: int, verbose_robots: bool) -> dict[str, Any]:
    """Bucketed live features in the planned encoder's shape; size grows with history/robot detail."""
    ours = []
    for rid in range(1, 6):
        r: dict[str, Any] = {
            "id": rid,
            "zone": random.choice(ZONES),
            "lane": random.choice(LANES),
            "tactic": random.choice(TACTIC_NAMES),
        }
        if verbose_robots:
            r.update(
                dist_to_ball=_dist(),
                nearest_opponent=_dist(),
                open_for_pass=random.choice([True, False]),
                facing="goal" if random.random() < 0.5 else "ball",
            )
        ours.append(r)
    return {
        "score": {"us": random.randint(0, 2), "them": random.randint(0, 2)},
        "time_left_s": random.randint(0, 300),
        "referee": random.choice(["normal_play", "stop", "free_kick_us"]),
        "ball": {
            "zone": random.choice(ZONES),
            "lane": random.choice(LANES),
            "speed": random.choice(["still", "slow", "fast"]),
            "towards_our_goal": random.choice([True, False]),
        },
        "possession": {
            "side": random.choice(["us", "them", "contested"]),
            "held_s": round(random.uniform(0, 10), 1),
            "our_closest": _dist(),
            "their_closest": _dist(),
        },
        "threat": {
            "opponents_in_our_third": random.randint(0, 5),
            "free_shooter": random.choice([True, False]),
        },
        "our_robots": ours,
        "current_split": {"attack": [1, 2, 3], "defense": [4, 5]},
        "split_held_s": round(random.uniform(0, 10), 1),
        "recent": [
            f"{round(random.uniform(0.1, 9.9), 1)}s ago: {random.choice(EVENTS)}"
            for _ in range(n_history)
        ],
    }


def render(static: dict, dynamic: dict, delimiter: str) -> str:
    """State as a pre-rendered string: static block, delimiter, dynamic block (compact JSON each)."""
    s = json.dumps({"static": static}, ensure_ascii=False, separators=(",", ":"))
    d = json.dumps({"live": dynamic}, ensure_ascii=False, separators=(",", ":"))
    return s + delimiter + d


def question(n_options: int, key: str = "split", described: bool = True) -> dict:
    """`described=False` sends option keys only (null descriptions) — the question tail is re-prefilled
    every request (it follows the state), so its length is a per-tick cost the prefix cache can't absorb.
    """
    opts = OPTION_POOL[:n_options]
    return {
        key: {
            "type": "choice",
            "instructions": "Which split of our 5 outfield robots should we use right now?",
            "criteria": {k: (d if described else None) for k, d in opts},
        }
    }


# Static block variant for keys-only options: the option glossary moves into the cached prefix.
STATIC_STATE_WITH_SPLITS = {**STATIC_STATE, "splits": dict(OPTION_POOL)}


class Client:
    def __init__(self, url: str):
        u = urlparse(url)
        self.conn = http.client.HTTPConnection(
            u.hostname or "127.0.0.1", u.port or 80, timeout=600
        )

    def _req(self, method: str, path: str, body: Any = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"} if data else {}
        self.conn.request(method, path, body=data, headers=headers)
        resp = self.conn.getresponse()
        out = json.loads(resp.read() or b"{}")
        if resp.status != 200:
            raise RuntimeError(f"{method} {path} -> {resp.status}: {out}")
        return out

    def status(self) -> dict:
        return self._req("GET", "/v1/status")["prefix_cache"]

    def prewarm(self, state: Any) -> dict:
        return self._req("POST", "/v1/prewarm", {"state": state})

    def decide(self, state: Any, questions: dict) -> tuple[dict, float]:
        t0 = time.perf_counter()
        out = self._req(
            "POST", "/v1/systemone", {"state": state, "questions": questions}
        )
        return out, (time.perf_counter() - t0) * 1000


def delta(before: dict, after: dict) -> dict:
    return {k: after[k] - before[k] for k in ("hits", "partial_hits", "misses")}


def summarize(ms: list[float]) -> dict:
    return {
        "median_ms": round(statistics.median(ms), 1),
        "min_ms": round(min(ms), 1),
        "max_ms": round(max(ms), 1),
        "n": len(ms),
    }


def static_text(static: dict, delim: str) -> str:
    return (
        json.dumps({"static": static}, ensure_ascii=False, separators=(",", ":"))
        + delim
    )


DELIMITERS = {"newline": "\n", "space": " ", "comma": ",", "double_newline": "\n\n"}


def bench_boundary(c: Client, reps: int) -> tuple[dict, str]:
    """1. Which delimiter lets `static + delimiter + dynamic` reuse the prewarmed static prefix?"""
    print("== 1. static-prefix reuse across the static/dynamic join")
    boundary = {}
    for name, delim in DELIMITERS.items():
        pw = c.prewarm(static_text(STATIC_STATE, delim))
        before = c.status()
        ms = [
            c.decide(
                render(STATIC_STATE, dynamic_state(3, False), delim), question(16)
            )[1]
            for _ in range(3)
        ]
        d = delta(before, c.status())
        boundary[name] = {
            "static_prompt_tokens": pw["prompt_tokens"],
            **d,
            **summarize(ms),
        }
        print(
            f"  {name:15s} static={pw['prompt_tokens']:4d} tok  {d}  median={boundary[name]['median_ms']} ms"
        )
    ok = [n for n, b in boundary.items() if b["partial_hits"] >= 3]
    print(
        f"  -> using delimiter {ok[0] if ok else 'newline (no variant reused the prefix!)'}"
    )
    return boundary, DELIMITERS[ok[0] if ok else "newline"]


def bench_sweep(c: Client, reps: int, delim: str) -> list[dict]:
    """2. Latency vs dynamic-part size x option count, static prefix cached."""
    print(
        "\n== 2. latency vs dynamic size x options (1 question, static prefix cached)"
    )
    static_tokens = c.prewarm(static_text(STATIC_STATE, delim))["prompt_tokens"]
    sweep = []
    for n_hist, verbose, label in [
        (1, False, "small"),
        (4, True, "medium"),
        (12, True, "large"),
    ]:
        for n_opt in (8, 16):
            before = c.status()
            ms, toks = [], []
            for _ in range(reps):
                out, t = c.decide(
                    render(STATIC_STATE, dynamic_state(n_hist, verbose), delim),
                    question(n_opt),
                )
                ms.append(t)
                toks.append(out["usage"]["input_tokens"])
            row = {
                "dynamic": label,
                "options": n_opt,
                "prompt_tokens": round(statistics.median(toks)),
                "static_tokens": static_tokens,
                **delta(before, c.status()),
                **summarize(ms),
            }
            sweep.append(row)
            print(
                f"  dyn={label:6s} opts={n_opt:2d}  prompt≈{row['prompt_tokens']:4d} tok "
                f"(static {static_tokens})  median={row['median_ms']:7.1f} ms  "
                f"[{row['min_ms']}-{row['max_ms']}]  partial={row['partial_hits']}/{reps}"
            )
    return sweep


def bench_questions(c: Client, reps: int, delim: str) -> dict:
    """3. 1 vs 3 questions per request on the same state."""
    print("\n== 3. questions per request (medium dynamic, 16 options, static cached)")
    c.prewarm(static_text(STATIC_STATE, delim))
    qres = {}
    for nq in (1, 3):
        qs: dict = {}
        for i in range(nq):
            qs.update(question(16, key=f"q{i}"))
        ms = [
            c.decide(render(STATIC_STATE, dynamic_state(4, True), delim), qs)[1]
            for _ in range(reps)
        ]
        qres[nq] = summarize(ms)
        print(f"  {nq} question(s): median={qres[nq]['median_ms']} ms")
    return qres


def bench_baselines(c: Client, reps: int, delim: str) -> dict:
    """4. No reuse at all (full prefill every tick) vs an exact repeat of the previous state."""
    print("\n== 4. baselines (medium dynamic, 16 options)")
    out = {}
    ms = []
    for _ in range(reps):
        # dict state -> rendered as one JSON object, never extends the cached static text
        state = {
            "static": STATIC_STATE,
            "live": dynamic_state(4, True),
            "nonce": random.random(),
        }
        ms.append(c.decide(state, question(16))[1])
    out["no_reuse"] = summarize(ms)
    print(f"  no reuse (full prefill): median={out['no_reuse']['median_ms']} ms")
    same = render(STATIC_STATE, dynamic_state(4, True), delim)
    c.decide(same, question(16))
    before = c.status()
    ms = [c.decide(same, question(16))[1] for _ in range(reps)]
    out["exact_repeat"] = {**summarize(ms), **delta(before, c.status())}
    print(f"  exact repeat (full hit): median={out['exact_repeat']['median_ms']} ms")
    return out


def bench_tail(c: Client, reps: int, delim: str) -> dict:
    """5. Question-tail size: described options vs keys-only with the glossary in the cached static block."""
    print(
        "\n== 5. question tail: described options vs keys-only + glossary in static prefix (medium dyn)"
    )
    tail = {}
    for label, static, described in [
        ("described", STATIC_STATE, True),
        ("keys_only", STATIC_STATE_WITH_SPLITS, False),
    ]:
        st_tokens = c.prewarm(static_text(static, delim))["prompt_tokens"]
        for n_opt in (8, 16):
            ms, toks = [], []
            for _ in range(reps):
                out, t = c.decide(
                    render(static, dynamic_state(4, True), delim),
                    question(n_opt, described=described),
                )
                ms.append(t)
                toks.append(out["usage"]["input_tokens"])
            row = {
                "static_tokens": st_tokens,
                "prompt_tokens": round(statistics.median(toks)),
                **summarize(ms),
            }
            tail[f"{label}_{n_opt}"] = row
            print(
                f"  {label:9s} opts={n_opt:2d}  static={st_tokens} prompt≈{row['prompt_tokens']} tok  "
                f"median={row['median_ms']} ms"
            )
        same = render(static, dynamic_state(4, True), delim)
        c.decide(same, question(16, described=described))
        ms = [c.decide(same, question(16, described=described))[1] for _ in range(reps)]
        tail[f"{label}_16_exact_repeat"] = summarize(ms)
        print(
            f"  {label:9s} opts=16  exact repeat (tail only): median={tail[f'{label}_16_exact_repeat']['median_ms']} ms"
        )
    return tail


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:3000")
    ap.add_argument("--reps", type=int, default=5, help="ticks per configuration")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--sections", default="1,2,3,4,5", help="comma-separated subset of 1-5 to run"
    )
    ap.add_argument("--out", default=None, help="write full results JSON here")
    args = ap.parse_args()
    random.seed(args.seed)
    run = set(args.sections.split(","))
    c = Client(args.url)
    results: dict[str, Any] = {}

    # warm-up: first MLX call pays kernel compilation
    c.decide(render(STATIC_STATE, dynamic_state(1, False), "\n"), question(8))

    delim = "\n"
    if "1" in run:
        results["boundary"], delim = bench_boundary(c, args.reps)
    if "2" in run:
        results["sweep"] = bench_sweep(c, args.reps, delim)
    if "3" in run:
        results["questions"] = bench_questions(c, args.reps, delim)
    if "4" in run:
        results.update(bench_baselines(c, args.reps, delim))
    if "5" in run:
        results["tail"] = bench_tail(c, args.reps, delim)

    results["server"] = c._req("GET", "/v1/version")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
