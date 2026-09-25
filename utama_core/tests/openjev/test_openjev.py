"""OpenJev decision engine, model-free: split allocation, and full rsim matches driven by a fake
client (the real one needs a FOR-Engine server) — checks the partitioner's wiring, decision
cadence, logging and server-failure fallback without any inference."""

from __future__ import annotations

import functools
import json

import pytest

from tournament_lib import run_match
from utama_core.strategy.openjev import build_openjev_kernel_strategy
from utama_core.strategy.openjev.client import Decision
from utama_core.strategy.openjev.encoder import QUESTION_KEY, STATIC_TEXT
from utama_core.strategy.openjev.splits import SPLIT_KEYS, SPLITS, allocate

ALL_TACTICS = frozenset({"attack", "overload", "press", "mark", "block", "clear"})
FREE = frozenset({1, 2, 3, 4, 5})


@pytest.mark.parametrize("key", SPLIT_KEYS)
def test_every_split_covers_all_free_robots_once(key):
    partition = allocate(key, FREE, ALL_TACTICS)
    assigned = [r for robots in partition.values() for r in robots]
    assert sorted(assigned) == sorted(FREE)
    assert set(partition) <= ALL_TACTICS


def test_inapplicable_slots_fold_into_their_fallback():
    no_press_no_clear = ALL_TACTICS - {"press", "clear"}
    assert allocate("press", FREE, no_press_no_clear) == {"mark": FREE}
    assert allocate("clear_danger", FREE, no_press_no_clear) == {"block": FREE}
    assert allocate("press_block", FREE, no_press_no_clear) == {
        "mark": frozenset({1, 2, 3}),
        "block": frozenset({4, 5}),
    }


def test_counts_respect_fewer_free_robots():
    # committed slots shrink the free pool; the split still assigns what's left, in id order
    assert allocate("build_up", frozenset({4, 5}), ALL_TACTICS) == {
        "attack": frozenset({4, 5})
    }
    assert allocate("overload_support", frozenset({2, 3, 5}), ALL_TACTICS) == {
        "overload": frozenset({2, 3}),
        "attack": frozenset({5}),
    }


def test_split_keys_are_unique_and_in_static_glossary():
    assert len(SPLIT_KEYS) == len(set(SPLIT_KEYS)) == len(SPLITS)
    for key in SPLIT_KEYS:
        assert f'"{key}"' in STATIC_TEXT


class FakeClient:
    """Possession-reactive stand-in: build up when we have the ball, press otherwise."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.states: list[str] = []

    def decide(self, state: str, questions: dict) -> Decision:
        if self.fail:
            raise ConnectionError("no server")
        assert state.startswith(
            STATIC_TEXT
        )  # static block must be a literal prefix (prefix-cache reuse)
        self.states.append(state)
        choice = "build_up" if '"side":"us"' in state else "press"
        probs = {
            k: (0.9 if k == choice else 0.1 / (len(SPLIT_KEYS) - 1)) for k in SPLIT_KEYS
        }
        return Decision(
            answers={
                QUESTION_KEY: {
                    "choice": choice,
                    "probabilities": probs,
                    "confidence": 0.89,
                }
            },
            latency_ms=1.0,
            cached=False,
        )


def _play(tmp_path, client, decision_hz=5.0, seconds=8.0):
    built = []
    log = tmp_path / "decisions.jsonl"
    factory = functools.partial(
        build_openjev_kernel_strategy,
        client=client,
        decision_hz=decision_hz,
        log_path=log,
        on_build=built.append,
    )
    result = run_match(
        "openjev",
        "build_tiki_taka_kernel_strategy",
        duration_seconds=seconds,
        factory_a=factory,
        run_dir=None,
    )
    for p in built:
        p.close()
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    return result, built[0], rows


def test_match_runs_with_fake_client_at_requested_cadence(tmp_path):
    client = FakeClient()
    result, partitioner, rows = _play(tmp_path, client)
    assert result.score_a >= 0 and result.score_b >= 0
    assert partitioner.n_fallback == 0 and rows
    assert all(r["choice"] in SPLIT_KEYS and r["fallback"] is None for r in rows)
    gaps = [b["t"] - a["t"] for a, b in zip(rows, rows[1:])]
    assert (
        min(gaps) >= 0.2 - 1e-6
    )  # 5 Hz: never more often than every 0.2 s of sim time
    assert {"score", "ball", "possession", "threat", "our_robots", "recent"} <= set(
        rows[0]["live"]
    )


def test_server_failure_falls_back_to_scripted_picker(tmp_path):
    _, partitioner, rows = _play(tmp_path, FakeClient(fail=True), seconds=6.0)
    assert partitioner.n_fallback == partitioner.n_decisions == len(rows) > 0
    assert all(r["fallback"].startswith("ConnectionError") for r in rows)
    # a failing server is retried at the decision cadence, not hammered every tick in between
    gaps = [b["t"] - a["t"] for a, b in zip(rows, rows[1:])]
    assert min(gaps) >= 0.2 - 1e-6
    # the scripted picker actually drove robots (partitions are empty only while every robot is committed)
    assert any(r["partition"] for r in rows)
