"""Draws match_data_flow.png for docs/architecture.md: one rsim match's files under replays/, the stores that
index them, and the analyses, tools and ratings that read them back.

pixi run python docs/img/architecture/make_match_data_flow.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagram_style import Diagram  # noqa: E402

# Every name a box shows; utama_core/tests/tools/test_diagram_names.py checks they still exist.
NAMES = [
    "StrategyRunner",
    "tools/evaluation/evaluate.py",
    "tools/realism/convert_ssl_log.py",
    "utama_core/run/strategy_runner.py",
    "utama_core/replay/ssl_log.py",
    "utama_core/replay/columnar_writer.py",
    "utama_core/replay/columnar_reader.py",
    "utama_core/engine/match_log.py",
    "utama_core/engine/match_stats.py",
    "utama_core/replay/match_results.py",
    "utama_core/replay/fingerprint.py",
    "utama_core/analysis/goal_sources.py",
    "utama_core/analysis/ratings.py",
    "utama_core/analysis/chances.py",
    "utama_core/analysis/turnover_breakdown.py",
    "utama_core/analysis/restart_outcomes.py",
    "utama_core/analysis/stuck_detector.py",
    "tools/debug/moments.py",
    "tools/debug/replay_trace.py",
    "tools/debug/repro_from_replay.py",
]

d = Diagram(
    184,
    124,
    "Match data flow",
    "Every rsim match writes its files under replays/. The stores index them; the analyses, tools and ratings read "
    "them back.",
    figsize=(15, 10.8),
)

# --- drivers and sources --------------------------------------------------------------------------
d.box(
    22,
    80,
    36,
    20,
    "Drivers",
    "tools/evaluation/evaluate.py\nfixtures (replay/match_results.py)\npicks which matches to play",
    kind="decision",
)
d.box(22, 104, 38, 12, "Official SSL logs", "*.log.gz\nconvert_ssl_log.py, replay/ssl_log.py", kind="source")
d.box(72, 80, 32, 16, "StrategyRunner", "run/strategy_runner.py\nplays each match", kind="decision")
d.arrow((40, 80), (56, 80))
d.arrow((40, 104), (98, 100), "converted logs: same types", lx=1.0, ly=2.0)

# --- files written per match ----------------------------------------------------------------------
d.box(
    120,
    98,
    44,
    14,
    "A_vs_B.npz",
    "+ A_vs_B.sparse_referee.pkl\nreplay/columnar_writer.py",
    kind="state",
)
d.box(
    120,
    80,
    44,
    16,
    "A_vs_B.intentions.jsonl",
    "+ A_vs_B.stats.json\nengine/match_log.py\nengine/match_stats.py",
    kind="state",
)
d.arrow((88, 84), (98, 96))
d.arrow((88, 78), (98, 80))

# --- stores ---------------------------------------------------------------------------------------
d.box(72, 26, 32, 12, "Match results", "replays/match_results/\nreplay/match_results.py", kind="state")
d.arrow((40, 71), (56, 30), "store; stored matches taken, not played", lx=-1.0, ha="right")
d.box(118, 60, 36, 12, "summary.json", "+ ball_losses.md, one per run", kind="state")
d.arrow((40, 73), (100, 62), "evaluate.py", lx=0.0, ly=2.0)

# --- readers --------------------------------------------------------------------------------------
d.box(165, 92, 34, 12, "goal_sources.py", "goal_sources.json", kind="output")
d.box(
    165,
    70,
    34,
    16,
    "analysis/",
    "chances, turnover_breakdown,\nrestart_outcomes, stuck_detector",
    kind="output",
)
d.box(165, 48, 34, 18, "tools/debug/", "moments.py,\nreplay_trace.py,\nrepro_from_replay.py", kind="output")
d.box(165, 26, 34, 12, "ratings.py", "the verdict's test", kind="output")
d.arrow((142, 98), (148, 92))
d.arrow((142, 94), (148, 72))
d.arrow((142, 80), (148, 70))
d.arrow((142, 76), (148, 48))
d.arrow((88, 26), (148, 26), "every stored match", lx=0.0, ly=2.0)

d.note(
    4,
    10,
    "evaluate.py keeps each round's replays in replays/eval_<id>/round_<k>/;\nresults also go to the match results store.",
)

d.key(
    104,
    48,
    [
        ("source", "outside the process: logs"),
        ("decision", "drivers and the strategy runner"),
        ("state", "files and stores read back later"),
        ("output", "analyses, tools, ratings"),
    ],
)
print(d.save(__file__, "match_data_flow.png"))
