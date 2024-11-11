# Welcome to the Team Controller Wiki!
This page contains an overview of how this repository interacts with other components of Robocup SSL and a brief explanation of what is contained in each folder in src.
# Requirements
## Required Dependencies
 - Python==3.10.12
 - protobuf==3.20.3
 - numpy (newest version)
## Usage
For this repository to be fully implemented, you should run GrSim, SSL_Game_Controller, and an AutoReferee simultaneously in different terminals.
> Note: GrSim Defaults to broadcasting vision data from the wrong port (we want 10006) and defaults to DivA.
# Data flow
This is the data flow diagram of what will be received and sent by the Team Controller:
![Dataflow Diagram](https://github.com/ICRS-RoboCup-SSL/Team_Controller/blob/main/docs/SSL_dataflow%20diagram.jpg)
From the image above the Vision and Referee are UDP multicast addresses and there is more information from the [SSL Robocup official page](https://ssl.robocup.org/league-software/#:~:text=Simulation%20Protocol.-,Standard%20Network%20Parameters,-Protocol)
# Codebase
This will provide an overview of the current repository and what each file will do.
> Note: The [PID folder](https://github.com/ICRS-RoboCup-SSL/Team_Controller/tree/main/src/pid) along with the [math_utils.py](https://github.com/ICRS-RoboCup-SSL/Team_Controller/blob/main/src/utils/math_utils.py) file will be moved elsewhere.

> Note: The [robot_startup_controller.py](https://github.com/ICRS-RoboCup-SSL/Team_Controller/blob/main/src/controllers/robot_startup_controller.py) will also be moved elsewhere
## Config
This folder contains the [settings.py](https://github.com/ICRS-RoboCup-SSL/Team_Controller/blob/main/src/config/settings.py) file which acts as a config file with all the preset parameters.
> Note: This is planned to change to a .yaml file using the Hydra framework
## Controllers
This folder contains all the different controller classes for communication with both sim & real components, this includes:
* Simulation Control (sim)
* Robot Control (real & real)
* Game Control (sim & real)
> Note: Game Control seems to be used for very limited cases so it won't be implemented until later
## Data
This folder contains all the files which perform the initial processing of data being received, this includes:
* Vision Data
* Referee Data
## Utils
This file contains files which manage network communication via a UDP socket for sending and receiving data:
* Network Manager
* Network Utils
## Tests
This file contains all the test files for this repository
## Main
Ignore it for now as there is no functionality for this Python file.
