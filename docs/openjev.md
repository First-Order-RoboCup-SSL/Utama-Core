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

**Free kicks** (`free_kick_overrides`). Theirs: two robots in a wall 0.65 m from the ball on its line to
our goal (outside the 0.5 m keep-out), the rest marking the opponents nearest our goal; our goalkeeper
is left to its own tactic. Ours: the built-in kicker approach lined up at the widest gap in their goal,
two teammates to open attacking spots, two back. The built-in steps leave defenders frozen where play
stopped and line our kicker up at one of their robots.

**The prompt** (`flat_state.STATIC_STATE`, cached): the role, the pitch, the rules (defense areas, the
1 m carry, aimless kicks), the menu, and four coaching principles. These are the rule that won the
model-free probes below. Each question then gives the moment's coaching line, the roles still unfilled,
and **the coaching's call** for that robot: the pick of that rule (`scripted_choice`), as advice
(`"The coaching's call for robot 3: get_ball."`). Every answer is still the model's own; read alone,
the principles were followed on only 62% of questions (`--engine flat-nocall`, results below). With no
carrier of ours and the ball in our defensive third the moment's line is `deep`: three on the ball, two
blocking, nobody forward; with the ball, three forward and one back, as the rule plays it.

**Scripted control** (`--engine flat-scripted`, `scripted_choice`): the same menu played by a fixed
rule. It is the fallback when the server errors, and a control: it measures the menu and its
execution apart from the model's choices, in minutes instead of the model's half hour a match.
