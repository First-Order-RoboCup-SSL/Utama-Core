#!/bin/bash
#
# start_test_env.sh — launch the three external processes a grSim (simulated)
# or real-hardware test session needs, and tear them all down on Ctrl+C.
#
# Usage:
#     ./start_test_env.sh          # then Ctrl+C to stop everything
#
# This starts nothing from this repo. It is purely a convenience launcher for
# the external SSL toolchain that `main.py` / the grsim-mode tests talk to over
# the network; run your own strategy separately once these are up. Nothing in
# this repo starts or manages these processes otherwise, which is also why the
# grsim tests are excluded in CI (`--ignore-glob "**/*grsim*"`).
#
# What it launches, in order:
#   1. grSim                     — the official SSL simulator (vision + robot
#                                  command UDP). Must be on PATH; see README's
#                                  "Setup grSim".
#   2. ssl-game-controller/      — the official referee GameController. Its web
#                                  UI is the http://localhost:8081/#/match the
#                                  script reminds you to open; it is not served
#                                  by this repo (our own dashboard is :8080).
#   3. AutoReferee/ (./gradlew run) — TIGERs Mannheim's automatic referee, which
#                                  watches vision and feeds decisions to the
#                                  GameController. See README's "Setup
#                                  AutoReferee".
#
# Prerequisites (none of these directories are tracked here — both are
# gitignored, you clone/download them yourself per the README):
#   - `grSim` callable from the terminal
#   - `./ssl-game-controller/` containing the GameController binary
#   - `./AutoReferee/` containing the AutoReferee checkout (gradle wrapper)
#
# Note: the CustomReferee (`utama_core/custom_referee/`) is an in-process
# replacement for items 2 and 3 and needs none of this — it works identically
# in rsim/grsim/real with no network dependency. Use this script only when you
# specifically want the official GameController/AutoReferee in the loop, e.g.
# validating against real competition software. See docs/custom_referee.md.
#
# Known rough edges (documented rather than silently changed — the naming one
# needs whoever actually runs this to say which spelling is correct):
#   - README's "Setup AutoReferee" step 4 says to rename the downloaded
#     GameController binary to `ssl_game_controller` (underscore), but line 49
#     below executes `./ssl-game-controller` (hyphen, same as the directory).
#     One of the two is wrong; following the README literally makes this step
#     fail.
#   - The `if [ $? -ne 0 ]` checks after each `&` test whether the shell
#     managed to background the job, not whether the program actually started,
#     so a missing binary or a crash-on-startup is not caught here — it shows
#     up as a silently absent process.
#   - Output from all three is sent to /dev/null, so startup errors are
#     invisible; drop the `> /dev/null 2>&1` on whichever line you are
#     debugging.

# Function to handle cleanup on script exit
cleanup() {
    echo "Caught SIGINT signal! Cleaning up..."

    # Kill grSim, game controller, and AutoReferee processes if they exist
    if [ ! -z "$GRSIM_PID" ]; then
        echo "Stopping grSim..."
        kill $GRSIM_PID 2>/dev/null
    fi

    if [ ! -z "$GAME_CONTROLLER_PID" ]; then
        echo "Stopping game controller..."
        kill $GAME_CONTROLLER_PID 2>/dev/null
    fi

    if [ ! -z "$AUTOREFEREE_PID" ]; then
        echo "Stopping AutoReferee..."
        kill $AUTOREFEREE_PID 2>/dev/null
    fi

    echo "Cleanup complete. Exiting."
    exit
}

# Trap SIGINT (Ctrl+C) and call the cleanup function
trap cleanup SIGINT

# Output reminder to open the website manually
echo "Reminder: Please open the following website in your browser:"
echo "http://localhost:8081/#/match"
echo "Once the website is opened, the script will continue..."

# Start grSim in the background, suppressing output
echo "Starting grSim..."
grSim > /dev/null 2>&1 &
GRSIM_PID=$!

# Check if grSim started successfully
if [ $? -ne 0 ]; then
    echo "Failed to start grSim. Exiting."
    exit 1
fi

# Change to the ssl-game-controller directory and run the game controller, suppressing output
echo "Starting game controller..."
cd ssl-game-controller/
./ssl-game-controller > /dev/null 2>&1 &
GAME_CONTROLLER_PID=$!
cd ..

# Check if the game controller started successfully
if [ $? -ne 0 ]; then
    echo "Failed to start game controller. Exiting."
    cleanup
fi

# Change to the AutoReferee directory and run the run.sh script, suppressing output
echo "Starting AutoReferee..."
cd AutoReferee/
./gradlew run > /dev/null 2>&1 &
AUTOREFEREE_PID=$!
cd ..

# Check if AutoReferee started successfully
if [ $? -ne 0 ]; then
    echo "Failed to start AutoReferee. Exiting."
    cleanup
fi

# Wait for all background processes to finish
wait
