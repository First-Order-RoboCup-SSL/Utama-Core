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

### Building it: model-free probes (2026-10-18)

Full matches (to full time, capped at 1200 s) of the scripted rule against split_shape, which beat
every other strategy (counter_flow lost all 4 cells to it, 8-20). From the fourth row on, 12 matches:
4 cells x 3 decision rates (+-2%), since one changed line reshuffles every match.

| Step | vs split_shape | What it showed |
|---|---|---|
| first menu: shoot if not blocked, else pass, `get_ball` chases the ball | 0-0-4, 14-35 | |
| `get_ball` challenges their carrier from the goal side (`block_attacker`) | 0-0-4, 10-28 | |
| ball actions on the contact sensor, not the visual box | (with an own-half shooting ban) 0-0-4, 3-25 | robots sat beside loose balls and kicked into air |
| shoot from any range again | 0-1-11, 32-74 | the ban had cut our goals from 10 to 3: long shots score here despite aimless kicks |
| two robots on the ball when they have it | 2-2-8, 53-72 | split_shape never passes, so a second presser beats a marker |
| one of five forward as an outlet instead of a marker | 3-4-5, 43-48 | counter-attacks |
| free-kick formations: a wall for theirs, ours aimed at their goal | 4-0-8, 60-64 | their free kicks converted 25%, ours 7% (then 40%) |
| a blocker with a tackling teammate holds the line | 6-2-4, 59-57 | `block_attacker` lunges inside 0.5 m and opens the line |
| at most two `block_shot`; decisions at 1 Hz plus situation triggers | 6-0-6, 62-61 | as strong as 5 Hz, a third of the questions |
| three on the ball in our defensive third, no outlet (2026-10-19) | 8-0-4, 78-53 | their carrier near our goal beat two; three in the middle third too 7-3-2, 67-58, everywhere 7-0-5, 78-68 |
| three forward with the ball, one back (2026-10-19) | 11-1-0, 83-53 | one forward instead 6-1-5, 80-77; a carrier in our third that shoots only when wide open, else passes out of it or clears 4-2-6, 57-69 |

Tried and dropped: shooting only when wide open (0-0-4, 6-31) or even when blocked (0-0-4, 9-33); passes
kicked at once (0-4-8); one robot forward with the ball (0-2-10); a cautious back line (5-3-4, no better);
2.5 Hz without triggers (4-0-8); the outlet marking instead, in our third (4-2-6, 58-64). On the final
rule, 28 matches each (4 cells x 7 decision rates, 0.94-1.06 Hz) against its 22-3-3, 181-127: no shots
from our own half (4-2-22, 85-139) or only open ones (11-4-13, 131-126), though a miss from there that
crosses their goal line is an aimless kick, their free kick in front of our box; a free-kick wall formed
fast, round the keep-out circle on the tangent and from their ball placement on (17-5-6, 178-151: their
free kicks scored no less within 3 s, and the formation cost elsewhere). Three forward with the ball lost
0-1-11 before a third robot went for the ball in our third; after it, it is the last row.
