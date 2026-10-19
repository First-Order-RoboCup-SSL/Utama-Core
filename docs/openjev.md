# OpenJev

OpenJev is an LLM decision model (served by FOR-Engine) that answers multiple-choice questions about a
text state: one forward pass per question, up to 52 lettered options, no generated text. Two engines
put it in charge of our team; their OpenJev side is in `utama_core/strategy/openjev/`, deliberately not
in `kernel_strategy.py`, so a round-robin never picks them up without a running server. The flat
engine's tactic, menu and policy network are shared code in `utama_core/tactics/` (`flat.py`,
`flat_state.py`, `flat_net.py`); on its scripted rule it is the catalog's `flat_rule`
(`strategy/flat_rule.py`), which a round-robin plays.

- **split** (`partitioner.py`, `encoder.py`, `splits.py`): OpenJev is the kernel `Partitioner`. Every
  `1 / decision_hz` seconds of sim time it picks one of a fixed set of splits of the 5 outfield robots
  across six tactics (`attack`, `overload`, `press`, `mark`, `block`, `clear`), and the tactics play it.
- **flat** (`strategy/openjev/flat.py`, the tactic in `tactics/flat.py`): no tactics in between.
  OpenJev picks every outfield robot's action itself, one robot at a time; see
  [The flat engine](#the-flat-engine).

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
| `tools/flat_round.py` | No server: the flat engine on the rule or the policy network (`--weights`), in parallel on the CPU, 4 cells x 7 rates by default, against strategies or another network (`net:<weights>`); `--record` saves every question |
| `tools/flat_net_train.py` | Train the policy network by imitating the rule (GPU, torch) |
| `tools/flat_net_rl.py` | Improve it by self-play reinforcement learning (PPO): against itself, earlier versions and the pool, CPU matches, GPU updates |

Tests: `pixi run pytest utama_core/tests/openjev/ --headless` (66 pass, 2026-10-19; no server needed).

## The flat engine

`build_openjev_flat_kernel_strategy` (`strategy/openjev/flat.py`): one `FlatJevTactic`
(`tactics/flat.py`) owns all 5 outfield robots; the goalkeeper stays on its pinned `GoalkeeperTactic`,
and restarts on `RefereeOverride` (with this engine's own free-kick formations, below). The model
decides only in open play.

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

### The scripted control against the pool

The final rule (`--engine flat-scripted`, 1 Hz plus triggers), full matches, 4 cells at 7 decision
rates (0.94 to 1.06 Hz), 28 matches an opponent: **310-18-8, 1924-560**, 2.82 points a match, at least 22
wins of 28 and at most 3 losses against every strategy. Over the first 3 rates (12 an opponent), one step
back (two forward with the ball): 135-1-8, 832-227, split_shape 8-0-4; two steps back (also two on the
ball in our third): 131-3-10, 759-218, split_shape 6-0-6, 62-61, while its 1 Hz cells alone read 46-0-2,
split_shape 3-0-1: one rate is a lucky draw, not a measurement.

| Opponent | W-D-L (28) | Goals (28) | Rk | R | Lk | L |
|---|---|---|---|---|---|---|
| split_shape | 22-3-3 | 181-127 | 4-3 | 6-3 | 7-5 | 6-5 |
| counter_flow | 28-0-0 | 124-25 | 4-1 | 6-0 | 3-0 | 7-0 |
| score_aware_counter_flow | 26-1-1 | 125-24 | 1-0 | 6-0 | 4-0 | 5-0 |
| press_and_pass | 26-2-0 | 156-35 | 9-2 | 5-0 | 5-1 | 7-0 |
| clear_press_plus | 24-3-1 | 126-39 | 1-0 | 7-1 | 7-1 | 4-3 |
| high_press | 25-3-0 | 138-36 | 3-0 | 7-1 | 3-2 | 3-1 |
| tiki_taka_plus | 27-0-1 | 205-33 | 9-1 | 3-4 | 6-0 | 8-2 |
| give_and_go_solo | 27-0-1 | 214-49 | 7-4 | 7-1 | 10-0 | 6-1 |
| overload_press | 25-3-0 | 178-62 | 6-3 | 6-4 | 5-2 | 6-2 |
| press_trigger_flow | 28-0-0 | 140-33 | 6-1 | 1-0 | 3-2 | 5-1 |
| decoy_and_overload | 27-1-0 | 194-47 | 8-1 | 4-1 | 8-3 | 6-1 |
| clear_danger | 25-2-1 | 143-50 | 5-3 | 6-3 | 3-1 | 6-1 |

Cells (at 1 Hz): R/L = we play right/left, `k` = we kick off; scores ours first.

### Flat results (2026-10-19)

The model on every robot: full matches (to full time, capped at 1200 s), 4 cells against each of the
round-robin's 12 strategies, openjev-FP8 on SGLang on one RTX 5090, 1 Hz plus the situation triggers.

**Without the call** (`--engine flat-nocall`, and two on the ball everywhere): **30-8-10, 235-161**,
2.04 points a match (that rule alone, at 1 Hz: 2.88). 242,679 questions, none fell back to the rule, 147 ms a question. More
wins than losses against 9 strategies, level with 2 (decoy_and_overload 2-0-2 on goals 25-16,
clear_danger 1-2-1 on 15-14), beaten by split_shape (1-0-3, 25-29).

| Opponent | W-D-L | Goals | Rk | R | Lk | L |
|---|---|---|---|---|---|---|
| split_shape | 1-0-3 | 25-29 | 9-11 | 5-6 | 5-7 | 6-5 |
| counter_flow | 3-0-1 | 16-12 | 6-3 | 5-2 | 3-6 | 2-1 |
| score_aware_counter_flow | 2-2-0 | 14-9 | 3-3 | 3-2 | 6-2 | 2-2 |
| press_and_pass | 2-1-1 | 13-11 | 4-3 | 3-3 | 4-2 | 2-3 |
| clear_press_plus | 3-0-1 | 15-11 | 2-4 | 2-1 | 7-4 | 4-2 |
| high_press | 4-0-0 | 21-3 | 5-1 | 3-1 | 8-0 | 5-1 |
| tiki_taka_plus | 4-0-0 | 27-7 | 8-1 | 5-2 | 6-2 | 8-2 |
| give_and_go_solo | 4-0-0 | 28-20 | 7-6 | 5-4 | 6-3 | 10-7 |
| overload_press | 2-1-1 | 19-15 | 4-2 | 4-5 | 6-3 | 5-5 |
| press_trigger_flow | 2-2-0 | 17-14 | 3-2 | 4-4 | 6-4 | 4-4 |
| decoy_and_overload | 2-0-2 | 25-16 | 5-3 | 3-4 | 14-4 | 3-5 |
| clear_danger | 1-2-1 | 15-14 | 6-3 | 3-3 | 3-3 | 3-5 |

It chose as the coaching's rule would on only 62% of questions. A robot the rule sends for the ball went
53% of the time (the second of the two, 47%) and otherwise went forward or onto the shot line, so their
carrier was often unchallenged: it conceded 3.35 goals a match to that rule's 1.48, the gap mostly at the
end of their long spells on the ball (1.44 a match with no restart or turnover in the 10 s before; the
rule 0.44), after they won the ball in our defensive third (0.85; 0.44) and from their free kicks there
(0.67; 0.38). Restating a principle next to the question did not change this, nor did spelling out the
second chaser's job; stating the rule's pick for the robot did (93%, below).

**With the call, two on the ball everywhere** (v2, stopped after 12 matches when the rule changed): 7-3-2;
split_shape 2-1-1, 22-21 (Rk 4-7, R 8-7, Lk 5-2, L 5-5), clear_danger 2-2-0, 13-5, decoy_and_overload
3-0-0, 31-5, counter_flow's Rk 0-1. The model chose as the rule would on 92-93% of questions: the nearest
robot to the ball went for it on every call, the second nearest (as the second on the ball) on 91-96%; it
turned down calls sent further back (the third nearest and on, after a nearer one declined) four times in
five.

**With the call, three on the ball in our defensive third** (v3, stopped after 2 matches when the rule
changed again): split_shape Rk 4-2, Lk 5-5. The model chose as the rule would on 98% of questions; a third
robot sent for the ball went on 96-99% of calls, where in v2 calls past the second were turned down: the
moment's coaching line now asked for three.

**With the call, three forward with the ball** (v4, stopped after 4 matches): split_shape 1-1-2, 22-25
(Rk 6-9, R 8-8, Lk 6-4, L 2-4), where that rule alone went 22-3-3 over 28. The first robot sent for the
ball went on only 92% of calls, the second on 90%: most refusals were a slow loose ball in our third,
where the question said "we have the ball" (`possession` counts the nearest robot, our keeper included)
and the coaching line for that now sent three forward. On CPU the rule with the first chaser turned
down at random 4, 8 or 16% of the time went 5-3-4, 4-0-8 and 7-3-2 against split_shape (12 each): that
one robot decides the match. Possession is now said from the carrier, as the rule counts it
(`flat_state.side_of`): calls followed 99.8%, the first chaser's 100%.

**With the call, the final rule** (v5, `--engine flat`): (pending)

## The policy network (`policy="net"`)

A small network in the model's place (`tactics/flat_net.py`): the same flat engine, menu, chain and
execution, no LLM and no server. It runs in numpy on the CPU, about 0.2 ms a question (1 ms a
decision).

**What it reads.** 14 tokens, all in our frame (we attack +x and our left is +y, from either side): one
global token (score, time, possession, our carrier, this robot's phase, the roles already chosen in this
decision), our 6 robots and their 6 in id order, and the ball. Each of our robots carries position,
velocity, heading, distance and rank to the ball, the action it is carrying out and the one chosen for it
in this decision, and the facts its menu states: free, lane open from the ball, shot open / narrow /
blocked with the gap's width. A 3-layer transformer (d 128, 4 heads, 434k parameters) mixes them.

**What it answers.** For the asked robot, 35 scores: 23 fixed actions (`shoot`, `clear`, `get_ball`,
`block_shot`, `hold`, `dribble:<cell>` x 9, `move:<cell>` x 9) from that robot's token, and `pass:<k>` /
`mark:<k>` from its token against each teammate's or opponent's. Options not on its menu are masked; in
a match it plays the most likely option, and while learning it draws one (`--temperature`).

**Training.** `tools/flat_round.py --record` saves every question (`flat.RECORD_FIELDS`);
`tools/flat_net_train.py` imitates the rule on them on the GPU (torch, e.g. FOR-Engine's environment, so
Utama-Core's needs none) and exports the weights; `FlatNet.load` checks that numpy reproduces the
trainer's own outputs. `tools/flat_net_rl.py` then improves on it by self-play (below). Data and
weights live under `replays/flatnet/` (ignored).

**Imitation** (`bc1`, 2026-10-19). The rule against all 22 catalog strategies, 4 cells x 2 rates (0.95,
1.05): 176 full matches, 166-7-3, 1039-225, 774k questions to train on and 84k held out (every 10th
match). Twelve epochs with 5% label smoothing over the menu (exploration for later) took 30 s on the RTX
5090: the network picks the rule's option on 99.995% of the held-out questions (the carrier's 99.95%,
everyone else's 100%). Against the pool at 1 Hz, 4 cells, playing its best option: 45 of 48 matches
identical to the rule's, the other three each parted from it after one different choice; 48-0-0,
265-70, where the rule went 47-0-1, 260-71.

**Reinforcement learning by self-play** (`rl2`, running since 2026-10-19 13:09), after AlphaGo's
reinforcement-learning stage: PPO from `bc1`, the network drawing its actions. Each iteration plays 16
full matches against itself (both teams' questions are training data), 12 against three earlier
versions of itself (a league, so it does not only learn to beat its latest self), and the pool as an
anchor, 2 random cells each (self-play alone can drift to beating itself and not the strategies we
have to beat). Each robot's questions are a trajectory whose reward is its team's goal difference until its
next question (discounted 0.99 a second, GAE 0.98), the baseline a critic of the same shape, and a KL
penalty keeps the network near `bc1` until the goals say otherwise. No tree search as in AlphaGo: rsim
cannot snapshot and restore a match, and one 10 s look-ahead costs seconds. Every 5 iterations the
network plays at its best the pool (4 cells, 1 Hz) and `bc1`, the rule's copy (8 matches).

## Split engine results (2026-09-26)

Every match 600 s of game time, decisions at 60 Hz, OpenJev = yellow. Cells: R/L = OpenJev plays
right/left, `k` = OpenJev kicks off. Scores are OpenJev first. Server: FOR-Engine 0.1.0, model
`openjev-FP8`, sglang backend, T = 0.85. Fallback decisions: 0 in every match below.

These were played before `jev-engine` merged v2.0.4 to v2.2.0 (planner and sim speed-ups, referee
fixes, retirements), against the catalog as it was then. rsim is deterministic, but replaying them on
current code will not give the same matches; rerun before comparing with a newer round-robin.

### Full round, coarse timers: 18 W 58 D 12 L, 25-33 (88 matches)

`replays/jev_full_600s_coarse/`, the encoder with event times in coarse buckets (`--coarse-timers`,
since removed). 22 opponents x 4 cells, 7.5 h of match wall time; 94% of decisions were cache hits,
uncached ones took 116 ms on average.

| Opponent | W | D | L | GF-GA | Pts | Rk | R | Lk | L |
|---|---|---|---|---|---|---|---|---|---|
| shadow_switch | 3 | 1 | 0 | 4-0 | 10 | 2-0 | 0-0 | 1-0 | 1-0 |
| clear_danger | 2 | 2 | 0 | 2-0 | 8 | 1-0 | 0-0 | 1-0 | 0-0 |
| decoy_and_overload | 2 | 2 | 0 | 4-1 | 8 | 0-0 | 2-0 | 1-0 | 1-1 |
| switch_of_play | 2 | 2 | 0 | 2-0 | 8 | 0-0 | 1-0 | 0-0 | 1-0 |
| clear_press_plus | 2 | 1 | 1 | 3-2 | 7 | 0-1 | 1-1 | 1-0 | 1-0 |
| high_line_zone | 1 | 3 | 0 | 1-0 | 6 | 0-0 | 0-0 | 1-0 | 0-0 |
| low_block | 1 | 3 | 0 | 1-0 | 6 | 0-0 | 0-0 | 0-0 | 1-0 |
| score_aware_zone_flow | 1 | 3 | 0 | 1-0 | 6 | 0-0 | 1-0 | 0-0 | 0-0 |
| tiki_taka_plus | 1 | 3 | 0 | 1-0 | 6 | 0-0 | 1-0 | 0-0 | 0-0 |
| zone_fluid | 1 | 3 | 0 | 1-0 | 6 | 0-0 | 1-0 | 0-0 | 0-0 |
| counter_press | 1 | 2 | 1 | 1-1 | 5 | 0-0 | 0-0 | 1-0 | 0-1 |
| give_and_go_solo | 1 | 2 | 1 | 1-1 | 5 | 0-0 | 0-0 | 0-1 | 1-0 |
| counter_flow | 0 | 4 | 0 | 0-0 | 4 | 0-0 | 0-0 | 0-0 | 0-0 |
| high_press | 0 | 4 | 0 | 0-0 | 4 | 0-0 | 0-0 | 0-0 | 0-0 |
| overload_flow | 0 | 4 | 0 | 0-0 | 4 | 0-0 | 0-0 | 0-0 | 0-0 |
| press_trigger_flow | 0 | 4 | 0 | 0-0 | 4 | 0-0 | 0-0 | 0-0 | 0-0 |
| three_slot | 0 | 4 | 0 | 0-0 | 4 | 0-0 | 0-0 | 0-0 | 0-0 |
| tiki_taka | 0 | 4 | 0 | 0-0 | 4 | 0-0 | 0-0 | 0-0 | 0-0 |
| press_and_pass | 0 | 3 | 1 | 0-1 | 3 | 0-0 | 0-0 | 0-0 | 0-1 |
| overload_press | 0 | 2 | 2 | 2-6 | 2 | 2-2 | 0-3 | 0-1 | 0-0 |
| score_aware_counter_flow | 0 | 2 | 2 | 0-2 | 2 | 0-1 | 0-1 | 0-0 | 0-0 |
| split_shape | 0 | 0 | 4 | 1-19 | 0 | 0-6 | 0-4 | 0-4 | 1-5 |

Reading it:

- Mostly draws: 58 of 88, 55 of them 0-0. OpenJev beats or holds most of the catalog by small margins
  and loses clearly only to `split_shape` (1-19 over 4 cells), then `overload_press` and
  `score_aware_counter_flow`.
- Many matches stall: 41 of 88 have ball-stuck time (ball under 0.05 m/s for 10 s or more while live),
  10 of them over 300 s. Those 0-0s say little about either side.
- The `L` cells against `zone_fluid`, `score_aware_zone_flow` and `overload_flow` are byte-identical
  replays: the three share their tactics, and at 0-0 without a long possession streak their pickers
  agree, so they count as one match, not three.

### Exact timers (current encoder), retrying coarse's non-wins: 1 W 5 D 6 L, 1-11 (12 matches)

`replays/jev_retry_exact/`, the encoder as it is now (event times to 0.1 s). It replayed the cells the
coarse version did not win, strongest opponents first, and was stopped by hand after 12. The directory
also holds 18 coarse wins copied in, so `tools/jev_summary.py` on it mixes the two versions; the
numbers here are the 12 matches without `coarse_timers` in their result.

| Opponent | Cell | Exact | Coarse |
|---|---|---|---|
| clear_press_plus | Rk | 1-0 W | 0-1 L |
| clear_press_plus | R | 0-0 D | 1-1 D |
| overload_press | Rk | 0-0 D | 2-2 D |
| overload_press | R | 0-1 L | 0-3 L |
| overload_press | Lk | 0-0 D | 0-1 L |
| overload_press | L | 0-1 L | 0-0 D |
| score_aware_counter_flow | Rk | 0-0 D | 0-1 L |
| split_shape | Rk | 0-1 L | 0-6 L |
| split_shape | R | 0-4 L | 0-4 L |
| split_shape | Lk | 0-3 L | 0-4 L |
| split_shape | L | 0-1 L | 1-5 L |
| tiki_taka_plus | Rk | 0-0 D | 0-0 D |

On the same 12 cells exact goes 1 W 5 D 6 L (1-11) against coarse's 0 W 4 D 8 L (4-28): it concedes
far less, mostly against `split_shape`, but still loses all four `split_shape` cells. The other 76
cells have not been played with exact timers, so there is no full-round total for the current encoder.

### Experiments, all reverted

All coarse timers, 600 s, 60 Hz; directories under `replays/`.

| Run | Change | Opponents | Result |
|---|---|---|---|
| `exp_def_press` | force `press` in our third when they have or contest the ball | `split_shape` x 4 | 0 W 1 D 3 L, 0-6 |
| `exp_def_press_block` | force `press_block` there | `split_shape` x 4 | 0 W 0 D 4 L, 0-6 |
| `exp_def_mark_all` | force `mark_all` there | `split_shape` x 4 | 0 W 1 D 3 L, 0-16 |
| `exp_def_low_block` | force `low_block` there | `split_shape` x 4 | 0 W 0 D 4 L, 0-14 |
| `exp2_B1` | 3 s stall-break + own-third `press_block` (stopped early) | 6 opponents, 20 matches | 0 W 16 D 4 L, 0-12 |
| `exp2_B2` | 3 s stall-break + own-third `press` (stopped early) | 6 opponents, 19 matches | 2 W 13 D 4 L, 4-23 |

No forced own-third split fixed `split_shape`: all four still lost to it. The stall-break cut stuck
time on the same cells from 2807 s to 258 s (B1) and 2355 s to 178 s (B2), but the stalls it removed
became 0-0s rather than goals: B1 turned two coarse wins into draws, B2 one win into a draw and one
draw into a win, and both still lost every `split_shape` cell.

A per-match listing of every run above (shots, possession, stuck seconds, decision counts) is in
`replays/jev_all_README.md` (local, not committed; `replays/` is ignored).
