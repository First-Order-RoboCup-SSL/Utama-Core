#!/bin/bash

# Function to handle cleanup on script exit
cleanup() {
    echo "Caught SIGINT signal! Cleaning up..."
    kill $GRSIM_PID $GAME_CONTROLLER_PID 2>/dev/null
    pkill -f "AutoReferee"
    exit
}

# Trap SIGINT (Ctrl+C) and call the cleanup function
trap cleanup SIGINT

# Start grSim in the background
grSim &
GRSIM_PID=$!

# Change to the ssl-game-controller directory and run the game controller
cd ssl-game-controller/
./ssl-game-controller_v3.12.7_linux_amd64 &
GAME_CONTROLLER_PID=$!
cd ..

# Change to the AutoReferee directory and run the run.sh script
cd AutoReferee/
./run.sh -a &
AUTOREFEREE_PID=$!
cd ..

# Wait for all background processes to finish
wait
