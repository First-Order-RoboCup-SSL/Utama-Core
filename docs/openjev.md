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

## The flat engine

`build_openjev_flat_kernel_strategy` (`flat.py`): one `FlatJevTactic` owns all 5 outfield robots; the
goalkeeper stays on its pinned `GoalkeeperTactic`, and restarts on `RefereeOverride` (with this
engine's own free-kick formations, below). The model decides only in open play.

**One decision, chained questions.** A decision asks one question per robot not busy with a pass, shot
or clearance: the ball carrier first, then the others nearest the ball first. Each question lists what
was already chosen at that moment ("robot 2 -> pass to robot 3"), so the five actions are chosen
jointly, p(a1..a5) = p(a1) p(a2|a1) ..., not independently; all questions share one state text, so
the server's prefix cache serves every question after the first. One question with every combination
as an option is impossible: about 25 options per robot is 25^5 combinations, and a readout takes at
most 52 (FOR-Engine chunks up to 52 x 52; the model was evaluated on at most 26).

**When.** Every `1 / decision_hz` s of sim time (1 Hz), and at once when our carrier, possession or the
ball's cell changes or a pass, shot or clearance ends (at most every 0.1 s). A robot keeps its action
until then.

**The menu** (`flat_state.robot_view`), each option stated with the facts that bear on it (lane open,
free, shot open / narrow / blocked and range: geometry the model can't read from coordinates):

| Robot | Options | Executed by |
|---|---|---|
| ball carrier | `shoot`, `pass:<id>`, `dribble:<cell>` (its own and adjacent cells, while it has carry left), `clear` | `_pass_and_score._score_goal`, `_pass_and_score._pass_exec`, a 1 m-limited carry, a kick long and wide to the nearer touch line |
| others | `get_ball`, `move:<cell>` (9 cells: third x strip, `docs/pitch_zones.md`), `mark:<id>`, `block_shot` (at most two robots), `hold` | `go_to_ball`; against their carrier the nearest chaser holds its shot line (`block_attacker` if alone) and the others tackle; the best open spot in the cell; `mark_target`; `defend_parameter` |

A pass, shot or clearance is locked once chosen until the kick happens, the ball is lost (0.3 s) or it
times out; the receiver is not asked while a pass is on. Ball actions check the dribbler's contact
sensor, not the visual box (which reaches a few cm past contact): from there a robot closes in first.
