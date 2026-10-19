#!/usr/bin/env bash
# Flat OpenJev (it picks every robot's action) vs the round-robin's pool: full matches (to full time,
# capped at 1200 s), all 4 side x kickoff cells. Resumable: re-run the same command and finished
# matches skip.
#
#   bash tools/jev_flat_round.sh [workers=2] [out_dir=replays/jevflat_pool] [engine=flat]
#
# The GPU is the bottleneck: a question takes ~0.12 s alone and ~0.5 s with four workers, so more than
# two workers barely adds throughput. engine=flat-scripted plays the model-free control (no server).
#
# Needs FOR-Engine on :3000 for engine=flat (FOR-Engine: `pixi run -e sglang serve-sglang-stack`).
# Results: `pixi run python -m tools.jev_summary <out_dir>`.
set -euo pipefail
cd "$(dirname "$0")/.."

WORKERS="${1:-2}"
OUT="${2:-replays/jevflat_pool}"
ENGINE="${3:-flat}"
SESSION="jevflat-pool-$(basename "$OUT")"

# the round-robin's pool (tools/evaluation/round_robin.py: every config not RETIRED), strongest first
POOL=(
  split_shape counter_flow score_aware_counter_flow press_and_pass clear_press_plus high_press
  tiki_taka_plus give_and_go_solo overload_press press_trigger_flow decoy_and_overload clear_danger
)

if [[ "$ENGINE" != "flat-scripted" ]] && ! curl -sf -m 3 http://127.0.0.1:3000/v1/version >/dev/null; then
  echo "FOR-Engine not reachable on :3000. Start it first, e.g. on one GPU:" >&2
  echo "  cd /workspace/FOR-Engine && CUDA_VISIBLE_DEVICES=0 pixi run -e sglang serve-sglang-stack" >&2
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists (attach: tmux attach -t $SESSION)" >&2
  exit 1
fi

mkdir -p "$OUT"
tmux new-session -d -s "$SESSION" -n info "echo 'workers: tmux select-window -t $SESSION:w<N>'; exec bash"
for ((w = 0; w < WORKERS; w++)); do
  mine=()
  for ((i = w; i < ${#POOL[@]}; i += WORKERS)); do mine+=("${POOL[i]}"); done
  tmux new-window -t "$SESSION" -n "w$w" \
    "pixi run python jev_tournament.py ${mine[*]} --engine $ENGINE --duration 1200 --cells 4 --out-dir $OUT; echo; echo 'worker $w finished'; exec bash"
  echo "worker $w: ${mine[*]}"
done
echo
echo "started. attach:   tmux attach -t $SESSION"
echo "summary anytime:  pixi run python -m tools.jev_summary $OUT"
