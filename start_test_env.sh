#!/bin/bash

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
./ssl-game-controller_v3.12.7_linux_amd64 > /dev/null 2>&1 &
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
