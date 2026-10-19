# OpenJev

OpenJev is an LLM decision model (served by FOR-Engine) that answers multiple-choice questions about a
text state: one forward pass per question, up to 52 lettered options, no generated text. Two engines
put it in charge of our team; code in `utama_core/strategy/openjev/`, deliberately not in
`kernel_strategy.py`, so a round-robin never picks them up without a running server.

- **split** (`partitioner.py`, `encoder.py`, `splits.py`): OpenJev is the kernel `Partitioner`. Every
  `1 / decision_hz` seconds of sim time it picks one of a fixed set of splits of the 5 outfield robots
  across six tactics (`attack`, `overload`, `press`, `mark`, `block`, `clear`), and the tactics play it.
- **flat** (`flat.py`, `flat_state.py`): no tactics in between. OpenJev picks every outfield robot's
  action itself, one robot at a time; see [The flat engine](#the-flat-engine).

The sim is stepped synchronously, so inference latency costs no game time, only wall time. If the
server errors, a decision falls back to a fixed rule (split: `tiki_taka`'s picker; flat: the scripted
rule below); a result's `openjev.fallback` counts these and must be 0 for the match to count as
OpenJev's.

## Running it

Needs FOR-Engine on `:3000` (in FOR-Engine: `pixi run -e sglang serve-sglang-stack`).

| Script | Purpose |
|---|---|
| `jev_tournament.py` | OpenJev (always yellow) against named strategies, `all` or `pool` (the round-robin's), `--cells 1/2/4` side x kickoff cells (`--cell-tags Rk,L` for a subset, to split an opponent across workers), `--engine split/flat/flat-scripted`; one `<tag>.result.json` per match, resumable |
| `tools/jev_full_round.sh` | The full round below: every catalog strategy, 600 s, 60 Hz, 4 cells, in tmux |
| `tools/jev_summary.py` | Aggregate a run directory's `*.result.json` (correct with several workers) |
| `tools/jev_replay_html.py` | Render a replay plus its `.openjev.jsonl` decision log as HTML |
| `tools/openjev_bench.py` | Server latency benchmark (prefix-cache reuse, state size, question count) |

Tests: `pixi run pytest utama_core/tests/openjev/ --headless` (19 pass, 2026-10-18; no server needed).
