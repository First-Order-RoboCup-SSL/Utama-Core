# Turnover breakdown — `tournament_20260923_223656`

231 matches, friendly = `config_a` (yellow). 2129 turnovers + 328 restarts given away = **2457 ball losses** (10.6/match), against 975 completed passes (4.2/match).

Turnover counts match each match's recorded `MatchStats.turnovers`: **yes, all matches**.

## How the ball is lost

| kind | count | share | per match | won back ≤1s | ≤3s after restart | meaning |
|---|---|---|---|---|---|---|
| `tackled` | 1501 | 61% | 6.50 | 62% | 322 | opponent took control while our holder was still within 1.5x possession radius |
| `during_stoppage` | 340 | 14% | 1.47 | 2% |  | possession changed while play was stopped (e.g. opponent placing the ball) |
| `foul` | 146 | 6% | 0.63 |  |  | play stopped with the ball in the field; restart to opponent |
| `shot_saved_or_blocked` | 119 | 5% | 0.52 | 20% | 33 | released at speed toward the goal mouth; opponent controlled it next |
| `ball_out_after_kick` | 113 | 5% | 0.49 |  |  | ball left the field after our last kick; restart to opponent |
| `loose_ball_lost` | 104 | 4% | 0.45 | 31% | 33 | our holder had drifted off the ball (no kick); opponent reached it first |
| `ball_out_other` | 69 | 3% | 0.30 |  |  | ball left the field without a kick from us (dribbled/deflected out) |
| `pass_intercepted` | 65 | 3% | 0.28 | 31% | 24 | released at speed, not goalward; opponent controlled it next |

**Real losses: 1104** (4.8/match) — excluding turnovers won back within 1s (two robots on one ball; the nearest-robot flips) and `during_stoppage` (the opponent handling the ball for a restart already counted as `foul`/`ball_out_*`).

## Which rule the `foul` restarts were for

| rule (referee status message) | count | top tactics |
|---|---|---|
| Double touch | 66 | `GiveAndGoTactic` 38, `PressAndContainTactic` 10, `PassAndShootTactic` 5 |
| Excessive dribbling | 62 | `DecoyOverloadTactic` 32, `ShadowAndMarkTactic` 14, `GiveAndGoTactic` 10 |
| Yellow attacker in blue defense area | 9 | `GiveAndGoTactic` 7, `PassAndShootTactic` 1, `SwitchOfPlayTactic` 1 |
| Extra yellow defender touched ball inside own defense area | 5 | `BlockShapeTactic` 2, `DefenseTactic` 2, `PassAndShootTactic` 1 |
| Ball held in yellow defense area over 10s | 4 | `GoalkeeperTactic` 4 |

## Which tactic had the ball (real losses only)

| tactic | losses | share | `tackled` | `foul` | `ball_out_after_kick` | `shot_saved_or_blocked` | `loose_ball_lost` | `ball_out_other` | `pass_intercepted` |
|---|---|---|---|---|---|---|---|---|---|
| `PressAndContainTactic` | 309 | 28% | 263 | 10 | 4 | 12 | 15 | 1 | 4 |
| `GiveAndGoTactic` | 289 | 26% | 73 | 55 | 68 | 43 | 11 | 25 | 14 |
| `DecoyOverloadTactic` | 144 | 13% | 64 | 33 | 12 | 12 | 3 | 12 | 8 |
| `BlockShapeTactic` | 74 | 7% | 54 | 6 | 0 | 0 | 8 | 6 | 0 |
| `PassAndShootTactic` | 73 | 7% | 37 | 8 | 0 | 0 | 23 | 4 | 1 |
| `ShadowAndMarkTactic` | 53 | 5% | 20 | 17 | 2 | 8 | 4 | 0 | 2 |
| `SwitchOfPlayTactic` | 52 | 5% | 15 | 7 | 11 | 1 | 3 | 9 | 6 |
| `LeadAndSupportTactic` | 47 | 4% | 23 | 3 | 7 | 8 | 2 | 4 | 0 |
| `GoalkeeperTactic` | 26 | 2% | 0 | 4 | 1 | 9 | 2 | 5 | 5 |
| `ClearBallTactic` | 17 | 2% | 9 | 0 | 4 | 0 | 0 | 3 | 1 |
| `DefenseTactic` | 11 | 1% | 5 | 2 | 2 | 0 | 1 | 0 | 1 |
| `unassigned` | 5 | 0% | 1 | 1 | 2 | 0 | 0 | 0 | 1 |
| `restart override` | 4 | 0% | 0 | 0 | 0 | 2 | 0 | 0 | 2 |

`restart override` = the robot was driven by `RefereeOverride` (restart positioning), not a tactic.
Ball-out and foul losses are attributed to the tactic of our last robot on the ball.
`unassigned` = no tactic slot held the robot at that moment.
