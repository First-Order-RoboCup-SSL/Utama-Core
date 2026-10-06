# Signal report

What each strategy does well and badly, from the signals in [`signals.md`](signals.md), drawn
from one round-robin. Standings are in [`strategies.md`](strategies.md); this shows why.

**Run:** `tournament_20261005_170958` (strategy-guard at `b26f5b37`, 22 configs, 231 matches of
600 s). Regenerate the figures after a new full round-robin, then rewrite the notes:

    pixi run python tools/signal_report.py replays/tournament_<id>

Signals explain results; none is a target. A red cell is a place to look, not a thing to tune
away.

## Every strategy at a glance

![Signals per strategy](img/signals/signal_heatmap.png)

Rows by points per match (the number after each name). Each column is coloured by rank among the
22 strategies, green where the value usually helps; passes are blue because more is neither
better nor worse. The printed number is the real value: per match, or a share.

- **`split_shape` is green across attack**: most shots and entries, a shot after 28% of its
  regains (2.0 s on average) and after 38% of its free kicks, 1.7 m gained per pass. It makes the
  fewest passes (15) and creates the most.
- **The next three win by pressing, not passing.** `high_press`, `give_and_go_solo` and
  `press_and_pass` shoot often after regains, but `high_press` and `press_and_pass` gain under
  0.5 m per pass, and all three face the most open shots and save about half. Their defense is
  where they'd gain.
- **`overload_press` passes the most (61) for the least progress (0.31 m)** and played in 5 of
  the 10 stalled matches: a winning record with a bug in it.
- **The bottom half rarely gets a shot.** `counter_press`, `switch_of_play`, `shadow_switch` and
  `high_line_zone` turn 1–7% of regains into a shot, almost never a free kick, and shoot from
  3 m. Their shots are open (63–68%) and convert well; they just don't get many.
- **`three_slot` enters the final third 15 times a match and shoots 1.4 times**, and commits
  22 out-of-bounds fouls: it gets there and kicks the ball out.

## Attack against defense

![Attack vs defense](img/signals/attack_defense.png)

Left: shots created against seconds the enemy holds the ball in the strategy's defensive third.
Every strategy that wins is in the top left; the ones that lose sit below six shots, most of them
also spending 90+ s a match defending. `high_line_zone` is the extreme: 167 s.

Right: how open the shots a strategy faces are, against its save rate. Every strategy fields the
same keeper, so this is the defense in front of it. The pressing winners (`high_press`,
`overload_press`, `give_and_go_solo`, `press_trigger_flow`, `press_and_pass`) are bottom right:
they leave the lane open and concede half their shots faced. `counter_flow`, `clear_danger` and
`score_aware_counter_flow` win with a defense that blocks (37–47% open); `split_shape` faces
fairly open shots (53%) but few of them, and saves 74%.

## What makes a shot score

![Goals per shot by distance and open goal](img/signals/shot_quality.png)

All 2725 shots of the run. Inside 2 m with at least half the goal open, 81–96% score; past 3.5 m, or
with under a quarter of the goal open, at most 16% do. So shot distance and open goal mouth measure
chance quality, and a strategy's conversion follows from where it shoots, not from luck. Every
shot here is already on target (`MatchStats`' detector), which is why the rates are high.

## How each strategy loses the ball

![Ball losses by kind](img/signals/ball_losses.png)

Real ball losses per match (not won back within 1 s), measured as config_a only: `n` is how many
matches that is, so `tiki_taka_plus` (1), `tiki_taka` (2), `three_slot` (3) and `switch_of_play`
(4) rest on very little, and `zone_fluid` has none.

- **Being tackled is everyone's largest loss**, worst for `switch_of_play`, `counter_press` and
  `high_press` (18–24 a match): carrying into pressure.
- **`three_slot` kicks the ball out 20 times a match**, more than all its other losses together.
- Shots saved or blocked are a loss too, so the strategies that shoot most (`split_shape`,
  `press_trigger_flow`) lose more that way; that one is a cost of attacking.

## Fouls and stalls

![Fouls by rule](img/signals/fouls.png)

Fouls per match by the offending strategy, colour scaled per column, and the stalled matches it
played in (each stall counts for both teams).

- Out of bounds is the most common foul everywhere; `three_slot` (21.9) is the outlier.
- `low_block` dribbles too far (4.1 a match) and `overload_press` does too (2.8); `shadow_switch`
  enters the defense area (3.3).
- `overload_press` was in 5 stalled matches, the most; rerun them with `round_robin.py --pair`
  to see why before trusting its rank.
