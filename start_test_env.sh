#!/bin/bash

grSim &

cd ssl-game-controller/
./ssl-game-controller_v3.12.7_linux_amd64 &
cd ..

cd AutoReferee/
./run.sh &
cd ..

wait