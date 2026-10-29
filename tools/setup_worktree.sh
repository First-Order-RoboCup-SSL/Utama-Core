#!/usr/bin/env bash
# setup_worktree.sh — prepare a git worktree of this repo to run tests and tools.
#
# Run inside the worktree (not the main checkout). It makes the worktree usable without a
# fresh `pixi install` or planner build:
#   - symlinks .pixi to the main checkout's .pixi (the environment);
#   - copies the gitignored compiled extensions (_fastpathplanner, _robosim .so files) from
#     the main checkout. Without them the planner falls back to Python and results differ.
# Idempotent: re-running changes nothing that is already in place. Prints one line per action.
#
# Usage: tools/setup_worktree.sh
set -euo pipefail

wt_root="$(git rev-parse --show-toplevel)"
main_root="$(git worktree list --porcelain | sed -n '1s/^worktree //p')"

if [ "$wt_root" = "$main_root" ]; then
    echo "skip: $wt_root is the main checkout; run this inside a worktree"
    exit 0
fi

link="$wt_root/.pixi"
target="$main_root/.pixi"
if [ -L "$link" ] && [ "$(readlink "$link")" = "$target" ]; then
    echo "ok: .pixi already links to $target"
elif [ -e "$link" ]; then
    echo "skip: $link exists and is not the link to $target (remove it to relink)"
else
    ln -s "$target" "$link"
    echo "linked: .pixi -> $target"
fi

for rel in \
    utama_core/motion_planning/src/fastpathplanning \
    utama_core/rsoccer_simulator/src/Simulators/robosim; do
    for src in "$main_root/$rel"/_*.cpython-*.so; do
        [ -e "$src" ] || continue
        name="$(basename "$src")"
        dst="$wt_root/$rel/$name"
        if [ -e "$dst" ] && cmp -s "$src" "$dst"; then
            echo "ok: $rel/$name"
        else
            cp -p "$src" "$dst"
            echo "copied: $rel/$name"
        fi
    done
done
