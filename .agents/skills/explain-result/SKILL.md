---
name: explain-result
description: Explain how a strategy plays and why it won, lost or came out undecided, from the replays of an evaluate.py run (or an older round robin). Use after an evaluation, when asked "why did it lose?", "how does X play?", "where do its goals come from?", or before changing a strategy to see what to change. Never use it to rank strategies; goals in evaluate.py's verdict are the only ranking.
---

# Explain a result

Every tool here reads a **round folder**: `replays/eval_<id>/round_<k>/` (older round robins:
`replays/tournament_<id>/`). A round folder holds all of its matches as `<a>_vs_<b>.*`: those
the run played, and links to those it reused from the match results store. `config_a` (the
name before `_vs_`) is the friendly side, the only side with an intentions log.

Nothing here judges. A signal explains a result; fewer ball losses or more shots is not
"better" on its own (`docs/signals.md`, "Ranking"). Report what the numbers show and what
they suggest changing, never a new ranking.

## Steps, cheapest first

1. **Read the run's own report** (no tool to run).
   `replays/eval_<id>/report.json`: per candidate the verdict, the stage that decided it
   with its mean diff ± SE and p (`screen`, `confirm`), and
   `per_opponent` (candidate's and reference's goal difference against each opponent, same seed
   and colour). Which opponents moved the margin? `round_1/summary.json` has every signal and
   the stall coverage. Look up any signal in `docs/signals.md`: where it lives, its reference
   range, and what an off value usually means.

2. **Is something broken?** `pixi run python tools/debug/find_bugs.py ROUND_DIR --strategy NAME`
   (about 20 s a match; `--restarts` adds restart geometry, a minute a match). Open
   `find_bugs/index.md` first: the worst matches, findings by kind, a picture each. Our fouls by
   rule: `--fouls --by strategy,rule`. A stall, thrash or foul is a bug, not a style: hand it to
   the `debug-incident` skill.

3. **Where do goals come from?**
   `pixi run python -m utama_core.analysis.goal_sources ROUND_DIR --out ROUND_DIR/goal_sources.json`.
   Goals per match = Σ over situation types (restart or ball won, by third) of situations per
   match × P(goal). Tells whether a change should create more situations, convert better, or
   allow fewer. Situation names: AGENTS.md glossary and `docs/signals.md`, "Where do goals come
   from?".

4. **How does it play, on the pitch?**
   - One strategy against the field: `pixi run python tools/debug/pitch_maps.py ROUND_DIR --out DIR`
     (positions with and without the ball, shots, fouls; every panel turned so the strategy
     attacks right).
   - A candidate against the champion, side by side on one scale:
     `pixi run python tools/evaluation/explain.py ROUND_DIR CANDIDATE REFERENCE --out DIR`
     (`positions.png`, `shots.png`, `fouls.png`, `summary.json`). REFERENCE is the name as
     `report.json` gives it (a frozen file is named by its stem, e.g. `split_shape@cffe3a28`).
   Zones: use the names in `docs/signals.md ("Pitch zones")` (defensive / middle / attacking third, wings, box).

5. **Write it up** in a few lines: the verdict and margin; which opponents decided it; the
   two or three signals or pictures that explain it, with numbers; any bug found (handed to
   `debug-incident`); what to try next. Look at every picture before citing it.

## Traps

- A round has 15 matches (more with `--all`); one strategy is config_a in only some of them, so
  friendly-side analyses (intentions, stalls) cover those only. `summary.json`'s
  `stalls_checked` says how many.
- Matches from before 2026-11-12, or whose original run folder was deleted, have no replay to
  link: tools then see fewer matches than the round lists. Say so when numbers look thin.
- Catch rate moves ±4–8 points between identical runs: never a reason on its own.
- `ball_travel` is not a quality signal.
- Don't start new matches to explain a result; if more data is needed, ask first.
