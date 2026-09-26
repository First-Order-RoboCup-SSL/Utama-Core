#!/usr/bin/env bash
# OpenJev vs every catalog strategy, full_match_tournament.py methodology: 600 s per match, all 4
# side x kickoff cells, 60 Hz decisions. Opponents run strongest-first, so if the night runs out the
# competitive tier is what's finished. Resumable: re-run the same command and finished matches skip.
#
#   bash tools/jev_full_round.sh [workers=1] [out_dir=replays/jev_full_600s]
#
# 1 worker by default: sglang is already ~73% busy with one (pure prefill, no batching gain), so more
# workers mostly queue and make results non-reproducible.
#
# Needs FOR-Engine on :3000 (FOR-Engine: `pixi run -e sglang serve-sglang-stack`). Each worker gets its
# own tmux window in session `jev-full`; results: `pixi run python -m tools.jev_summary <out_dir>`.
set -euo pipefail
cd "$(dirname "$0")/.."

WORKERS="${1:-1}"
OUT="${2:-replays/jev_full_600s}"
SESSION=jev-full

# strongest-first per docs/strategies.md: competitive tier, then baselines/experimental/parked
OPPONENTS=(
  clear_press_plus tiki_taka_plus counter_flow tiki_taka overload_flow press_trigger_flow
  zone_fluid shadow_switch score_aware_counter_flow score_aware_zone_flow counter_press
  press_and_pass split_shape high_press give_and_go_solo three_slot decoy_and_overload
  switch_of_play low_block overload_press high_line_zone clear_danger
)

if ! curl -sf -m 3 http://127.0.0.1:3000/v1/version >/dev/null; then
  echo "FOR-Engine not reachable on :3000. Start it first:" >&2
  echo "  cd /workspace/FOR-Engine && CUDA_VISIBLE_DEVICES=0 pixi run -e sglang serve-sglang-stack" >&2
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists (attach: tmux attach -t $SESSION)" >&2
  exit 1
fi

mkdir -p "$OUT"
# Reuse matches already finished elsewhere under the identical protocol (600 s, 60 Hz): same tag = same match.
pixi run python - "$OUT" <<'EOF'
import json, shutil, sys
from pathlib import Path
out = Path(sys.argv[1]).resolve()
for run in Path("replays").glob("*/run.json"):
    src = run.parent
    if src.resolve() == out:
        continue
    a = json.loads(run.read_text()).get("args", {})
    if a.get("duration") != 600.0 or a.get("decision_hz") != 60.0:
        continue
    # runs made with since-removed experimental flags (coarse timers, rules, ...) are a different jev
    if any(a.get(k) for k in ("coarse_timers", "rules", "stall_break", "fuzz_seed")):
        continue
    for res in src.glob("*.result.json"):
        tag = res.name.removesuffix(".result.json")
        if (out / res.name).exists():
            continue
        for f in src.glob(f"{tag}.*"):
            shutil.copy2(f, out / f.name)
        print(f"reused {tag} from {src}")
EOF

tmux new-session -d -s "$SESSION" -n info "echo 'workers: tmux select-window -t $SESSION:w<N>'; exec bash"
for ((w = 0; w < WORKERS; w++)); do
  mine=()
  for ((i = w; i < ${#OPPONENTS[@]}; i += WORKERS)); do mine+=("${OPPONENTS[i]}"); done
  tmux new-window -t "$SESSION" -n "w$w" \
    "pixi run python jev_tournament.py ${mine[*]} --duration 600 --decision-hz 60 --cells 4 --out-dir $OUT; echo; echo 'worker $w finished'; exec bash"
  echo "worker $w: ${mine[*]}"
done
echo
echo "started. attach:   tmux attach -t $SESSION   (Ctrl-b n = next window, Ctrl-b d = detach)"
echo "summary anytime:  pixi run python -m tools.jev_summary $OUT"
