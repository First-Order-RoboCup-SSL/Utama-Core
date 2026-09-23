# Turnover breakdown — `tournament_20260923_211717`

231 matches, friendly = `config_a` (yellow). 1990 turnovers + 369 restarts given away = **2359 ball losses** (10.2/match), against 771 completed passes (3.3/match).

Turnover counts match each match's recorded `MatchStats.turnovers`: **yes, all matches**.

## How the ball is lost

| kind | count | share | per match | won back ≤1s | ≤3s after restart | meaning |
|---|---|---|---|---|---|---|
| `tackled` | 1403 | 59% | 6.07 | 61% | 408 | opponent took control while our holder was still within 1.5x possession radius |
| `during_stoppage` | 375 | 16% | 1.62 | 2% |  | possession changed while play was stopped (e.g. opponent placing the ball) |
| `foul` | 221 | 9% | 0.96 |  |  | play stopped with the ball in the field; restart to opponent |
| `loose_ball_lost` | 89 | 4% | 0.39 | 27% | 40 | our holder had drifted off the ball (no kick); opponent reached it first |
| `ball_out_after_kick` | 87 | 4% | 0.38 |  |  | ball left the field after our last kick; restart to opponent |
| `shot_saved_or_blocked` | 74 | 3% | 0.32 | 8% | 37 | released at speed toward the goal mouth; opponent controlled it next |
| `ball_out_other` | 61 | 3% | 0.26 |  |  | ball left the field without a kick from us (dribbled/deflected out) |
| `pass_intercepted` | 49 | 2% | 0.21 | 27% | 22 | released at speed, not goalward; opponent controlled it next |

**Real losses: 1090** (4.7/match) — excluding turnovers won back within 1s (two robots on one ball; the nearest-robot flips) and `during_stoppage` (the opponent handling the ball for a restart already counted as `foul`/`ball_out_*`).

## Which rule the `foul` restarts were for

| rule (referee status message) | count | top tactics |
|---|---|---|
| Excessive dribbling | 138 | `GiveAndGoTactic` 51, `DecoyOverloadTactic` 41, `GoalkeeperTactic` 32 |
| Double touch | 69 | `GiveAndGoTactic` 43, `PressAndContainTactic` 12, `SwitchOfPlayTactic` 4 |
| Yellow attacker in blue defense area | 8 | `GiveAndGoTactic` 5, `BlockShapeTactic` 2, `DecoyOverloadTactic` 1 |
| Extra yellow defender touched ball inside own defense area | 6 | `GoalkeeperTactic` 6 |

## Which tactic had the ball (real losses only)

| tactic | losses | share | `tackled` | `foul` | `ball_out_after_kick` | `shot_saved_or_blocked` | `loose_ball_lost` | `ball_out_other` | `pass_intercepted` |
|---|---|---|---|---|---|---|---|---|---|
| `PressAndContainTactic` | 318 | 29% | 266 | 13 | 4 | 15 | 14 | 2 | 4 |
| `GiveAndGoTactic` | 278 | 26% | 79 | 99 | 39 | 22 | 7 | 21 | 11 |
| `DecoyOverloadTactic` | 136 | 12% | 53 | 43 | 16 | 6 | 6 | 6 | 6 |
| `BlockShapeTactic` | 66 | 6% | 47 | 6 | 0 | 0 | 5 | 8 | 0 |
| `PassAndShootTactic` | 63 | 6% | 35 | 3 | 0 | 0 | 19 | 5 | 1 |
| `SwitchOfPlayTactic` | 57 | 5% | 16 | 5 | 15 | 2 | 2 | 11 | 6 |
| `ShadowAndMarkTactic` | 55 | 5% | 27 | 13 | 2 | 7 | 4 | 0 | 2 |
| `GoalkeeperTactic` | 48 | 4% | 0 | 38 | 0 | 2 | 3 | 4 | 1 |
| `LeadAndSupportTactic` | 40 | 4% | 13 | 1 | 10 | 11 | 2 | 3 | 0 |
| `ClearBallTactic` | 11 | 1% | 8 | 0 | 1 | 0 | 0 | 1 | 1 |
| `DefenseTactic` | 8 | 1% | 5 | 0 | 0 | 1 | 1 | 0 | 1 |
| `unassigned` | 7 | 1% | 3 | 0 | 0 | 1 | 2 | 0 | 1 |
| `restart override` | 3 | 0% | 0 | 0 | 0 | 1 | 0 | 0 | 2 |

`restart override` = the robot was driven by `RefereeOverride` (restart positioning), not a tactic.
Ball-out and foul losses are attributed to the tactic of our last robot on the ball.
`unassigned` = no tactic slot held the robot at that moment.
