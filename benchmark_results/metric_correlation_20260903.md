# Metric-correlation study: proxy metrics vs. match outcome

Generated 2026-09-03 16:58 UTC by `tools/metric_correlation.py`.

Runs analysed: tournament_20260903_101521, tournament_20260903_112025, tournament_20260903_115838 (693 matches total).

Side convention: `config_a` is always yellow and plays right (`tournament.py::run_match` hardcodes `my_team_is_yellow=True, my_team_is_right=True`); `friendly` in stats/frame metrics always means `config_a`. All metrics below are reported as `a - b` differentials. Because config_a is right on literally every match in this corpus, side cannot be fit as a genuine covariate (it is a constant column) -- see part A.

## A. Per-match: Spearman correlation with goal diff, AUC for decisive matches

Spearman correlates each metric's `a-b` differential with `score_a - score_b` over ALL matches (including draws, since goal diff is still informative at 0-0 for many metrics). AUC is computed only over the 22-26 decisive matches per run (config_a win=1 vs config_b win=0), predicting winner from the raw differential's rank. A logistic fit (intercept + standardized slope) is also reported per metric; a genuine side covariate could not be added (see note above), so this is intercept+slope only, not the originally intended 3-parameter model.

| metric | n (corr) | spearman rho | p | n (AUC) | AUC | logistic slope/unit |
|---|---:|---:|---:|---:|---:|---:|
| possession_pct | 693 | 0.160 | 0.000 | 64 | 0.786 | 2.5679 |
| shots | 693 | 0.380 | 0.000 | 64 | 0.854 | 17.4649 |
| zone_time_pct_attacking | 693 | 0.148 | 0.000 | 64 | 0.786 | 13.8251 |
| robot_motion_pct | 693 | -0.195 | 0.000 | 64 | 0.156 | -11.8898 |
| shots_on_target | 693 | 0.439 | 0.000 | 64 | 0.776 | 3.0593 |
| attacking_third_entries | 693 | 0.268 | 0.000 | 64 | 0.982 | 17.4797 |
| completed_passes | 693 | 0.208 | 0.000 | 64 | 0.735 | 0.1791 |
| turnovers † | 693 | 0.009 | 0.821 | 64 | 0.643 | 1.2891 |
| possession_under_pressure_s | 693 | 0.097 | 0.011 | 64 | 0.464 | 0.3004 |
| mean_ball_x_towards_opponent_goal | 693 | -0.055 | 0.151 | 64 | 0.838 | 0.5751 |
| defensive_third_time_pct † | 693 | -0.093 | 0.015 | 64 | 0.125 | -2.4115 |
| restart_to_first_entry_s † | 38 | 0.169 | 0.311 | 8 | 0.286 | -0.2247 |

`ball_travel_m` (from `stats.json`) and `shot_backed_up_rate` are not in the table above: `MatchStats` records `ball_travel_m` as one match-level total with no side split, so it has no `a-b` differential -- see below for a match-level check against `|goal diff|` instead. `shot_backed_up_rate` requires a side to have taken >=1 shot on target, and with only a handful of `shots_on_target` events per run, both sides rarely take one in the *same* match -- its per-match differential is `None` almost everywhere, so it's analysed only in parts B/C as a pooled per-strategy rate instead (`pooled_shot_backed_up_rate_diff`).

`ball_travel_m` (match total) vs. `|goal diff|`: spearman rho=0.296, p=0.000, n=693. A more-decisive match plausibly involves more end-to-end ball movement (attacks that go somewhere) rather than a stalemate, hence testing against |goal diff| rather than the signed value.

## B. Per-strategy: mean differential vs. points/goal-diff per match

Each strategy's mean `own - opponent` differential per metric, aggregated across every match it played in the three 101521/112025/115838 runs (63 matches per strategy, 21 per run x 3 runs), correlated (Spearman) against that strategy's points-per-match (3/1/0) and goal-diff-per-match across the 22 strategies.

Full ranking (by |rho vs points-per-match|):

| metric | n strategies | rho vs points/match | p | rho vs goal-diff/match | p |
|---|---:|---:|---:|---:|---:|
| shots_on_target | 22 | 0.791 | 0.000 | 0.811 | 0.000 |
| shots | 22 | 0.754 | 0.000 | 0.779 | 0.000 |
| turnovers † | 22 | 0.730 | 0.000 | 0.726 | 0.000 |
| completed_passes | 22 | 0.696 | 0.000 | 0.641 | 0.001 |
| attacking_third_entries | 22 | 0.653 | 0.001 | 0.659 | 0.001 |
| mean_ball_x_towards_opponent_goal | 22 | 0.630 | 0.002 | 0.623 | 0.002 |
| zone_time_pct_attacking | 22 | 0.475 | 0.025 | 0.451 | 0.035 |
| defensive_third_time_pct † | 22 | -0.466 | 0.029 | -0.452 | 0.035 |
| possession_under_pressure_s | 22 | 0.270 | 0.224 | 0.288 | 0.194 |
| possession_pct | 22 | 0.163 | 0.468 | 0.174 | 0.439 |
| robot_motion_pct | 22 | -0.123 | 0.587 | -0.139 | 0.536 |
| restart_to_first_entry_s † | 22 | -0.014 | 0.950 | 0.018 | 0.938 |
| shot_backed_up_rate (pooled) | 11 | n/a | n/a | n/a | n/a |

### Top five metrics by |rho vs points-per-match|

| rank | metric | rho vs points/match | p | rho vs goal-diff/match | p |
|---:|---|---:|---:|---:|---:|
| 1 | shots_on_target | 0.791 | 0.000 | 0.811 | 0.000 |
| 2 | shots | 0.754 | 0.000 | 0.779 | 0.000 |
| 3 | turnovers | 0.730 | 0.000 | 0.726 | 0.000 |
| 4 | completed_passes | 0.696 | 0.000 | 0.641 | 0.001 |
| 5 | attacking_third_entries | 0.653 | 0.001 | 0.659 | 0.001 |

## C. Reliability: per-strategy metric value, run 101521 vs run 115838

**Caveat: code changed between these runs** (stall/deadlock fixes landed on this branch between 101521, 112025, and 115838 -- see the git log). Any reliability number below is therefore a *lower bound* on true metric reliability: some run-to-run disagreement here is genuine strategy-vs-strategy noise, but some is the runner behaving differently, not the metric being noisy.

Comparing 22 strategies present in both `tournament_20260903_101521` and `tournament_20260903_115838`.

| metric | n strategies | spearman rho (run-to-run) | p |
|---|---:|---:|---:|
| possession_pct | 22 | 0.660 | 0.001 |
| shots | 22 | 0.175 | 0.437 |
| zone_time_pct_attacking | 22 | 0.823 | 0.000 |
| robot_motion_pct | 22 | 0.687 | 0.000 |
| shots_on_target | 22 | 0.187 | 0.405 |
| attacking_third_entries | 22 | 0.829 | 0.000 |
| completed_passes | 22 | 0.792 | 0.000 |
| turnovers † | 22 | 0.855 | 0.000 |
| possession_under_pressure_s | 22 | 0.032 | 0.888 |
| mean_ball_x_towards_opponent_goal | 22 | 0.808 | 0.000 |
| defensive_third_time_pct † | 22 | 0.661 | 0.001 |
| restart_to_first_entry_s † | 7 | -0.143 | 0.760 |
| shot_backed_up_rate (pooled) | 1 | n/a | n/a |

### Reliability of the top-five B metrics

| metric | validity rho (points/match) | reliability rho (run-to-run) | verdict |
|---|---:|---:|---|
| shots_on_target | 0.791 | 0.187 | valid but unreliable -- likely noise, deprioritize |
| shots | 0.754 | 0.175 | valid but unreliable -- likely noise, deprioritize |
| turnovers † | 0.730 | 0.855 | valid & reliable -- good instrumentation candidate |
| completed_passes | 0.696 | 0.792 | valid & reliable -- good instrumentation candidate |
| attacking_third_entries | 0.653 | 0.829 | valid & reliable -- good instrumentation candidate |

## D. Recommendation

**Honesty check first:** matches are 65 s and only 64/693 (9%) are decisive across the analysed runs. Per-match AUCs in part A are computed on ~22-26 matches per run; per-strategy correlations in part B have n=22 (one point per strategy) and n=9 for the tiny reliability spot-check runs. None of these sample sizes support strong causal claims -- treat every number here as "consistent with" rather than "proves". The cleanest signal is the per-strategy aggregate (63 matches/strategy pools out a lot of single-match noise), which is why part D's weights are fit there, not on individual matches.

Weighted proxy score (OLS, goal-diff-per-match ~ intercept + weighted metrics; n=22 strategies, R^2=0.752):

`proxy_score = 0.000`
  `+ (0.05436) * mean_diff_turnovers`
  `+ (0.02760) * mean_diff_completed_passes`
  `+ (0.27900) * mean_diff_attacking_third_entries`
  `+ (-0.06549) * mean_diff_mean_ball_x_towards_opponent_goal`
  `+ (-0.06636) * mean_diff_zone_time_pct_attacking`
  `+ (-0.05405) * mean_diff_defensive_third_time_pct`

Units: `proxy_score` is in goals per match; each weight is per one raw unit of that metric's `a-b` differential (e.g. per 1.0 of possession_pct differential, or per 1 extra shot_on_target). R^2=0.752 on n=22 strategies -- with this few points, treat the specific weight values as directionally suggestive, not a tuned production model.

### Which metrics to instrument live in MatchStats

Combined score = `|validity rho (B, vs points/match)| x |reliability rho (C, run-to-run)|`, over every metric analysed in B (including the pooled `shot_backed_up_rate`) -- rewards metrics that are both predictive of winning and stable run-to-run, rather than gating on a hard top-five-only cutoff:

| metric | validity rho | reliability rho | combined score |
|---|---:|---:|---:|
| turnovers † | 0.730 | 0.855 | 0.624 |
| completed_passes | 0.696 | 0.792 | 0.551 |
| attacking_third_entries | 0.653 | 0.829 | 0.542 |
| mean_ball_x_towards_opponent_goal | 0.630 | 0.808 | 0.509 |
| zone_time_pct_attacking | 0.475 | 0.823 | 0.391 |
| defensive_third_time_pct † | -0.466 | 0.661 | 0.308 |
| shots_on_target | 0.791 | 0.187 | 0.148 |
| shots | 0.754 | 0.175 | 0.132 |
| possession_pct | 0.163 | 0.660 | 0.108 |
| robot_motion_pct | -0.123 | 0.687 | 0.084 |
| possession_under_pressure_s | 0.270 | 0.032 | 0.009 |
| restart_to_first_entry_s † | -0.014 | -0.143 | 0.002 |
| shot_backed_up_rate (pooled) | n/a | n/a | n/a |

**Instrument these 6 live in `MatchStats`:**

- `turnovers`
- `completed_passes`
- `attacking_third_entries`
- `mean_ball_x_towards_opponent_goal`
- `zone_time_pct_attacking`
- `defensive_third_time_pct`

**Drop or deprioritize:**

- `shots_on_target`
- `shots`
- `possession_pct`
- `robot_motion_pct`
- `possession_under_pressure_s`
- `restart_to_first_entry_s`
- `shot_backed_up_rate (pooled)`

Note `possession_pct`, `robot_motion_pct`, and `possession_under_pressure_s` land in the drop list despite `possession_pct`/`robot_motion_pct` already being free (already computed by `MatchStats` today) -- "drop" here means "don't treat as a proxy for winning / don't weight in the proxy score", not "remove from `MatchStats`"; they may still be useful for other diagnostic purposes.

**Runtime:** full analysis over 693 matches took 0.1s.
