# Metric-correlation study: proxy metrics vs. match outcome

Generated 2026-09-04 22:38 UTC by `tools/metric_correlation.py`.

Runs analysed: tournament_20260904_202318, tournament_20260904_221937 (462 matches total).

Side convention: `config_a` is always yellow and plays right (`tournament.py::run_match` hardcodes `my_team_is_yellow=True, my_team_is_right=True`); `friendly` in stats/frame metrics always means `config_a`. All metrics below are reported as `a - b` differentials. Because config_a is right on literally every match in this corpus, side cannot be fit as a genuine covariate (it is a constant column) -- see part A.

## A. Per-match: Spearman correlation with goal diff, AUC for decisive matches

Spearman correlates each metric's `a-b` differential with `score_a - score_b` over ALL matches (including draws, since goal diff is still informative at 0-0 for many metrics). AUC is computed only over the 22-26 decisive matches per run (config_a win=1 vs config_b win=0), predicting winner from the raw differential's rank. A logistic fit (intercept + standardized slope) is also reported per metric; a genuine side covariate could not be added (see note above), so this is intercept+slope only, not the originally intended 3-parameter model.

| metric | n (corr) | spearman rho | p | n (AUC) | AUC | logistic slope/unit |
|---|---:|---:|---:|---:|---:|---:|
| possession_pct | 462 | 0.029 | 0.538 | 42 | 0.560 | 1.4399 |
| shots | 462 | 0.528 | 0.000 | 42 | 0.996 | 17.1793 |
| zone_time_pct_attacking | 462 | 0.226 | 0.000 | 42 | 0.915 | 9.2070 |
| robot_motion_pct | 462 | -0.036 | 0.435 | 42 | 0.433 | -3.1296 |
| shots_on_target | 462 | 0.525 | 0.000 | 42 | 0.936 | 3.3269 |
| attacking_third_entries | 462 | 0.355 | 0.000 | 42 | 0.989 | 16.6588 |
| completed_passes | 462 | 0.075 | 0.105 | 42 | 0.673 | 0.1323 |
| turnovers † | 462 | -0.011 | 0.811 | 42 | 0.456 | -0.3650 |
| possession_under_pressure_s | 462 | 0.024 | 0.603 | 42 | 0.568 | 0.0208 |
| mean_ball_x_towards_opponent_goal | 462 | 0.203 | 0.000 | 42 | 0.934 | 1.1330 |
| defensive_third_time_pct † | 462 | -0.241 | 0.000 | 42 | 0.085 | -8.8485 |
| restart_to_first_entry_s † | 67 | 0.140 | 0.257 | 11 | 0.679 | 0.2399 |
| near_stall_events † | 462 | -0.017 | 0.718 | 42 | 0.453 | -0.0885 |
| kicker_identity_thrash † | 462 | 0.006 | 0.901 | 42 | 0.529 | 0.0041 |
| retreat_reapproach_oscillations † | 462 | 0.055 | 0.239 | 42 | 0.591 | 0.2302 |
| ball_carrier_hold_max_s † | 462 | -0.031 | 0.507 | 42 | 0.424 | -0.0263 |

`ball_travel_m` (from `stats.json`) and `shot_backed_up_rate` are not in the table above: `MatchStats` records `ball_travel_m` as one match-level total with no side split, so it has no `a-b` differential -- see below for a match-level check against `|goal diff|` instead. `shot_backed_up_rate` requires a side to have taken >=1 shot on target, and with only a handful of `shots_on_target` events per run, both sides rarely take one in the *same* match -- its per-match differential is `None` almost everywhere, so it's analysed only in parts B/C as a pooled per-strategy rate instead (`pooled_shot_backed_up_rate_diff`).

`ball_travel_m` (match total) vs. `|goal diff|`: spearman rho=0.348, p=0.000, n=462. A more-decisive match plausibly involves more end-to-end ball movement (attacks that go somewhere) rather than a stalemate, hence testing against |goal diff| rather than the signed value.

`total_fouls` (sum of `rule_event_counts`, match total) vs. `|goal diff|`: spearman rho=0.060, p=0.198, n=462.

## B. Per-strategy: mean differential vs. points/goal-diff per match

Each strategy's mean `own - opponent` differential per metric, aggregated across every match it played in the three 101521/112025/115838 runs (63 matches per strategy, 21 per run x 3 runs), correlated (Spearman) against that strategy's points-per-match (3/1/0) and goal-diff-per-match across the 22 strategies.

Full ranking (by |rho vs points-per-match|):

| metric | n strategies | rho vs points/match | p | rho vs goal-diff/match | p |
|---|---:|---:|---:|---:|---:|
| shots | 22 | 0.839 | 0.000 | 0.673 | 0.001 |
| shots_on_target | 22 | 0.754 | 0.000 | 0.677 | 0.001 |
| attacking_third_entries | 22 | 0.707 | 0.000 | 0.547 | 0.008 |
| mean_ball_x_towards_opponent_goal | 22 | 0.631 | 0.002 | 0.538 | 0.010 |
| zone_time_pct_attacking | 22 | 0.554 | 0.007 | 0.427 | 0.048 |
| defensive_third_time_pct † | 22 | -0.530 | 0.011 | -0.486 | 0.022 |
| restart_to_first_entry_s † | 21 | -0.354 | 0.115 | -0.347 | 0.124 |
| completed_passes | 22 | 0.213 | 0.341 | 0.090 | 0.692 |
| possession_pct | 22 | 0.121 | 0.590 | 0.018 | 0.935 |
| possession_under_pressure_s | 22 | -0.089 | 0.694 | -0.223 | 0.319 |
| robot_motion_pct | 22 | -0.080 | 0.723 | -0.189 | 0.399 |
| near_stall_events † | 22 | 0.067 | 0.769 | 0.211 | 0.346 |
| ball_carrier_hold_max_s † | 22 | 0.052 | 0.820 | -0.110 | 0.625 |
| turnovers † | 22 | 0.036 | 0.872 | -0.090 | 0.691 |
| retreat_reapproach_oscillations † | 22 | -0.029 | 0.898 | -0.050 | 0.824 |
| kicker_identity_thrash † | 22 | -0.017 | 0.940 | -0.102 | 0.650 |
| shot_backed_up_rate (pooled) | 16 | n/a | n/a | n/a | n/a |

### Top five metrics by |rho vs points-per-match|

| rank | metric | rho vs points/match | p | rho vs goal-diff/match | p |
|---:|---|---:|---:|---:|---:|
| 1 | shots | 0.839 | 0.000 | 0.673 | 0.001 |
| 2 | shots_on_target | 0.754 | 0.000 | 0.677 | 0.001 |
| 3 | attacking_third_entries | 0.707 | 0.000 | 0.547 | 0.008 |
| 4 | mean_ball_x_towards_opponent_goal | 0.631 | 0.002 | 0.538 | 0.010 |
| 5 | zone_time_pct_attacking | 0.554 | 0.007 | 0.427 | 0.048 |

## C. Reliability: per-strategy metric value, run 101521 vs run 115838

**Caveat: code changed between these runs** (stall/deadlock fixes landed on this branch between 101521, 112025, and 115838 -- see the git log). Any reliability number below is therefore a *lower bound* on true metric reliability: some run-to-run disagreement here is genuine strategy-vs-strategy noise, but some is the runner behaving differently, not the metric being noisy.

Comparing 22 strategies present in both `tournament_20260904_202318` and `tournament_20260904_221937`.

| metric | n strategies | spearman rho (run-to-run) | p |
|---|---:|---:|---:|
| possession_pct | 22 | 0.930 | 0.000 |
| shots | 22 | 0.478 | 0.024 |
| zone_time_pct_attacking | 22 | 0.778 | 0.000 |
| robot_motion_pct | 22 | 0.932 | 0.000 |
| shots_on_target | 22 | 0.309 | 0.162 |
| attacking_third_entries | 22 | 0.866 | 0.000 |
| completed_passes | 22 | 0.957 | 0.000 |
| turnovers † | 22 | 0.895 | 0.000 |
| possession_under_pressure_s | 22 | 0.718 | 0.000 |
| mean_ball_x_towards_opponent_goal | 22 | 0.811 | 0.000 |
| defensive_third_time_pct † | 22 | 0.574 | 0.005 |
| restart_to_first_entry_s † | 15 | -0.628 | 0.012 |
| near_stall_events † | 22 | 0.742 | 0.000 |
| kicker_identity_thrash † | 22 | 0.458 | 0.032 |
| retreat_reapproach_oscillations † | 22 | 0.494 | 0.020 |
| ball_carrier_hold_max_s † | 22 | 0.678 | 0.001 |
| shot_backed_up_rate (pooled) | 6 | n/a | n/a |

### Reliability of the top-five B metrics

| metric | validity rho (points/match) | reliability rho (run-to-run) | verdict |
|---|---:|---:|---|
| shots | 0.839 | 0.478 | valid but unreliable -- likely noise, deprioritize |
| shots_on_target | 0.754 | 0.309 | valid but unreliable -- likely noise, deprioritize |
| attacking_third_entries | 0.707 | 0.866 | valid & reliable -- good instrumentation candidate |
| mean_ball_x_towards_opponent_goal | 0.631 | 0.811 | valid & reliable -- good instrumentation candidate |
| zone_time_pct_attacking | 0.554 | 0.778 | valid & reliable -- good instrumentation candidate |

## D. Recommendation

**Honesty check first:** matches are 65 s and only 42/462 (9%) are decisive across the analysed runs. Per-match AUCs in part A are computed on ~22-26 matches per run; per-strategy correlations in part B have n=22 (one point per strategy) and n=9 for the tiny reliability spot-check runs. None of these sample sizes support strong causal claims -- treat every number here as "consistent with" rather than "proves". The cleanest signal is the per-strategy aggregate (63 matches/strategy pools out a lot of single-match noise), which is why part D's weights are fit there, not on individual matches.

Weighted proxy score (OLS, goal-diff-per-match ~ intercept + weighted metrics; n=22 strategies, R^2=0.931):

`proxy_score = 0.000`
  `+ (-0.00547) * mean_diff_attacking_third_entries`
  `+ (-0.02666) * mean_diff_mean_ball_x_towards_opponent_goal`
  `+ (-0.07280) * mean_diff_zone_time_pct_attacking`
  `+ (0.32618) * mean_diff_shots`
  `+ (-0.32017) * mean_diff_defensive_third_time_pct`

Units: `proxy_score` is in goals per match; each weight is per one raw unit of that metric's `a-b` differential (e.g. per 1.0 of possession_pct differential, or per 1 extra shot_on_target). R^2=0.931 on n=22 strategies -- with this few points, treat the specific weight values as directionally suggestive, not a tuned production model.

### Which metrics to instrument live in MatchStats

Combined score = `|validity rho (B, vs points/match)| x |reliability rho (C, run-to-run)|`, over every metric analysed in B (including the pooled `shot_backed_up_rate`) -- rewards metrics that are both predictive of winning and stable run-to-run, rather than gating on a hard top-five-only cutoff:

| metric | validity rho | reliability rho | combined score |
|---|---:|---:|---:|
| attacking_third_entries | 0.707 | 0.866 | 0.612 |
| mean_ball_x_towards_opponent_goal | 0.631 | 0.811 | 0.512 |
| zone_time_pct_attacking | 0.554 | 0.778 | 0.431 |
| shots | 0.839 | 0.478 | 0.401 |
| defensive_third_time_pct † | -0.530 | 0.574 | 0.304 |
| shots_on_target | 0.754 | 0.309 | 0.233 |
| restart_to_first_entry_s † | -0.354 | -0.628 | 0.222 |
| completed_passes | 0.213 | 0.957 | 0.204 |
| possession_pct | 0.121 | 0.930 | 0.113 |
| robot_motion_pct | -0.080 | 0.932 | 0.075 |
| possession_under_pressure_s | -0.089 | 0.718 | 0.064 |
| near_stall_events † | 0.067 | 0.742 | 0.049 |
| ball_carrier_hold_max_s † | 0.052 | 0.678 | 0.035 |
| turnovers † | 0.036 | 0.895 | 0.033 |
| retreat_reapproach_oscillations † | -0.029 | 0.494 | 0.014 |
| kicker_identity_thrash † | -0.017 | 0.458 | 0.008 |
| shot_backed_up_rate (pooled) | n/a | n/a | n/a |

**Instrument these 5 live in `MatchStats`:**

- `attacking_third_entries`
- `mean_ball_x_towards_opponent_goal`
- `zone_time_pct_attacking`
- `shots`
- `defensive_third_time_pct`

**Drop or deprioritize:**

- `shots_on_target`
- `restart_to_first_entry_s`
- `completed_passes`
- `possession_pct`
- `robot_motion_pct`
- `possession_under_pressure_s`
- `near_stall_events`
- `ball_carrier_hold_max_s`
- `turnovers`
- `retreat_reapproach_oscillations`
- `kicker_identity_thrash`
- `shot_backed_up_rate (pooled)`

Note `possession_pct`, `robot_motion_pct`, and `possession_under_pressure_s` land in the drop list despite `possession_pct`/`robot_motion_pct` already being free (already computed by `MatchStats` today) -- "drop" here means "don't treat as a proxy for winning / don't weight in the proxy score", not "remove from `MatchStats`"; they may still be useful for other diagnostic purposes.

**Runtime:** full analysis over 462 matches took 4.2s.
