#!/usr/bin/env bash
# tools/dev/setup_worktree.sh — prepare a git worktree of this repo to run tests and tools.
#
# Run inside the worktree (not the main checkout). It makes the worktree usable without a
# planner or simulator build:
#   - installs the worktree's own pixi environment (`pixi install`, about 2 min and 120 MB: the
#     packages are hardlinks into pixi's cache). It can't share the main checkout's: pixi refuses
#     an environment reached from another directory ("the environment directory has moved",
#     pixi 0.83), so an older setup's .pixi symlink is replaced;
#   - links `replays` to the main checkout's, so matches are stored once (the match results
#     store) and evaluations in different worktrees reuse each other's matches;
#   - copies the gitignored compiled extensions (_fastpathplanner, _robosim .so files) from
#     the main checkout. Without them the planner falls back to Python and results differ.
# Idempotent: re-running changes nothing that is already in place. Prints one line per action.
#
# Usage: tools/dev/setup_worktree.sh
set -euo pipefail

wt_root="$(git rev-parse --show-toplevel)"
main_root="$(git worktree list --porcelain | sed -n '1s/^worktree //p')"

if [ "$wt_root" = "$main_root" ]; then
    echo "skip: $wt_root is the main checkout; run this inside a worktree"
    exit 0
fi

env="$wt_root/.pixi"
if [ -L "$env" ]; then
    rm "$env"
    echo "removed: the .pixi symlink (pixi can't use a shared environment)"
fi
if [ -d "$env/envs/default" ]; then
    echo "ok: .pixi has its own environment"
else
    if ! out="$(cd "$wt_root" && pixi install 2>&1)"; then
        echo "$out"
        exit 1
    fi
    echo "installed: .pixi (the worktree's own environment)"
fi

link="$wt_root/replays"
target="$main_root/replays"
if [ -L "$link" ] && [ "$(readlink "$link")" = "$target" ]; then
    echo "ok: replays links to $target"
elif [ -e "$link" ]; then
    echo "skip: $link exists and is not the link to $target"
else
    ln -s "$target" "$link"
    echo "linked: replays -> $target"
fi

for rel in \
    utama_core/motion_planning/fastpathplanning \
    utama_core/simulator/simulators/robosim; do
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
