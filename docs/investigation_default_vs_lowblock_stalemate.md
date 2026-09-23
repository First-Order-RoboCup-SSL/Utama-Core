# Investigation — `default` vs `low_block` stays 0-0

2026-08-19. **Status: partly fixed, still 0-0.** Both configs are baselines
(`docs/strategies.md`), so the remainder isn't worth fixing for its own sake. Full original
writeup: `git log -p -- docs/investigation_default_vs_lowblock_stalemate.md`.

## Finding

In a 60s headless 6v6 rsim match the two teams' `PassAndShootTactic` passers (robot 1 on each
side) pinned the ball between them near the centre circle for the whole match: both stayed in
`setup` for all 3600 ticks, no kick ever fired (max ball speed ~1 m/s), ball travel was 6.8m,
and there were zero referee events. It is a tactic-level deadlock, not a referee, planner or
rsim-physics problem.

Root-cause chain:

1. **Symmetric race.** Mirrored formation spots, both passers 3.43m from the ball, and the sim
   started in `FORCE_START` with no kickoff, so both arrived at the same moment.
2. **`go_to_ball` had no concept of an opponent on the ball.** Both drove to the ball's exact
   position from opposite sides and wedged against each other at pin distance.
3. **`PassAndShootTactic`'s setup is opponent-blind.** `run_setup_phase` exits only when the
   passer reaches a target ~2.5m away *while holding the ball*; the longest possession either
   side managed was 0.9s.
4. **The 12s setup timeout (`3a17afc`) only restarts the same dance**: same pair, same pinned
   ball, target still unreachable.
5. **`default` commands only 2 of its 5 outfield robots.** `PassAndShootTactic` only commands
   `robot_ids[0:2]`, so robots 3-5 never move and nobody can break the 1v1.

## Fix candidates and status

1. **Opponent-aware approach in `go_to_ball`** — done 2026-08-20 (approach from the far side of
   a contesting enemy). Partial: possession went from a near-total pin to 55%/44% and ball
   travel to 8.8m, but the score stays 0-0.
2. **Setup-phase kick-out** (holding the ball with an opponent within ~0.4m for N ticks → push
   it into space) — not implemented. See `low_block`'s setup bug in `docs/strategies.md`.
3. **Stop handing 5 robots to a 2-robot tactic in `default`** — not implemented (open bug in
   `docs/strategies.md`).
4. **Kickoff ceremony for sim matches** — done: `tournament_lib.py` starts matches with a real
   `PREPARE_KICKOFF_*`. `StrategyRunner` itself still defaults to `FORCE_START` in sim modes.

Reproduce with `pixi run python arena_tournament.py default low_block` (per-tick instrumented
dump) and render windows with `utama_core/replay/render_window.py`.
