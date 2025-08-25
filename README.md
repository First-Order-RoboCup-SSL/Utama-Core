# Utama

## Setup Guidelines

### Setup Utama

1. Install `pixi` package manager with `curl -fsSL https://pixi.sh/install.sh | sh` or click here for Windows installaton [Pixi installation](https://pixi.sh/latest/#__tabbed_1_1)
2. With pixi: just run `pixi install` in the base folder and you're all setup.
3. Note that this also installs all modules with `__init__.py` (so you need to run it again when you add an `__init__.py`)
4. In order to go into the `pixi` venv, run `pixi shell`. You can also run any of the tasks in the `pixi.toml` without first being in a pixi shell. See [Pixi Tasks](#pixi-tasks).
5. Finally, run `pixi run precommit-install`. This will ensure that ruff linting is done before you commit.

#### Pixi Tasks
`pixi run {task_name}` is the generic way to run a task. Some of the main tasks you can run:
1. `pixi run main` runs main.py
2. `pixi run lint` runs the ruff checker.
3. `pixi run test` runs the pytest over the `test/` folder
4. `pixi run precommit` runs the precommit steps before you commit and push (mainly just ruff linting). To install precommit, run `pixi run precommit-install`.

### Setup Autoreferee

1. Make sure `grSim` is setup properly and can be called through terminal.
2. `git clone` from [AutoReferee repo](https://github.com/TIGERs-Mannheim/AutoReferee) in a folder named `/AutoReferee` in root directory.
3. Change `DIV_A` in `/AutoReferee/config/moduli/moduli.xml` to `DIV_B`.

```xml
    <globalConfiguration>
        <environment>ROBOCUP</environment>
        <geometry>DIV_B</geometry>
    </globalConfiguration>
```

4. Get the latest [compiled game controller](https://github.com/RoboCup-SSL/ssl-game-controller/releases/) and rename it to `ssl_game_controller`. Save it in `/ssl-game-controller` directory.

### Field Guide

![field_guide](assets/images/field_guide.jpg)

1. All coordinates and velocities will be in meters or meters per second.
2. All angular properties will be in radians or radians per second, normalised between [pi, -pi]. A heading of radian 0 indicates a robot facing towards the positive x-axis (ie left to right).
3. Unless otherwise stated, the coordinate system is aligned such that blue robots are on the left and yellow are on the right.

### Setup SSL Vision for real testing

1. Connect to a external hotspot and connect the vison linux laoptop and you own personal laptop to the same network
2. Allow Inbound UDP packets to allow packets through the port you set, run this with adim privaleges:
<pre>
New-NetFirewallRule -DisplayName "Allow Multicast UDP 10006" -Direction Inbound -Protocol UDP -LocalPort 10006 -Action Allow
</pre>
3. paste "%USERPROFILE%" into "Windows + R" then add a .wslconfig file ensure that the file type properties are WSLCONFIG file.
<pre>
[wsl2]
networkingMode=mirrored
</pre>
4. restart wsl using --shutdown then check using the cmd:
<pre>
sudo tcpdump -i eth1 -n host 224.5.23.2 and udp port 10006
</pre>
if you see UDP packets everything is working

## Guidelines

#### Folder Hierarchy

1. `decision_maker`: higher level control from above roles to plays and tactics [**No other folder should be importing from this folder**]
2. `robot_control`: lower level control for individual robots spanning skills to roles [**utility folder for decision_maker**]
3. `motion_planning`: control algorithms for movement and path planning [**utility folder for robot_control and other folders**]
4. `team_controller`: interacing with vision (including processing) and robots [**No other folder should be importing from this folder**]
5. `vision_processing`: data processing for vision related data [**utility folder for team_controller**]
6. `global_utils`: store utility functions that can be shared across all folders [**this folder should not be importing from any other folder**]
7. `entities`: store classes for building field, robot, data entities etc. [**this folder should not be importing from any other folder**]
8. `rsoccer_simulator`: Lightweight rSoccer simulator for testing [**import this folder for testing**]
9. [TODO] `replay`: replay system for storing played games in a .pkl file that can be reconstructed in rSoccer sim [**imports from rsoccer**]

#### Code Writing

1. Use typing for all functions.
2. Document your code on the subfolder's `README.md`.
3. Download and install `Black Formatter` for code formatting

   1. For VScode, go to View > Command Palette and search `Open User Settings (JSON)`
   2. Find the `"[python]"` field and add the following lines:

   ```yaml
   "[python]": {
       "editor.defaultFormatter": "ms-python.black-formatter", # add this
       "editor.formatOnSave": true, # and add this
     }
   ```

### CI
In CI we are using:
 - Pytest to run unit tests throughout the whole project
 - Ruff Linter to check linting

#### Push and Commit

1. Each team should be working within your own branch of the repository. Do not push to main branch on your own.
2. Ensure that you have run `pixi run precommit-install` at least once. This ensure that the pre-commit steps are always done to clean up your code.

## Milestones

- 2024 November 20 - First goal in grSim (featuring Ray casting)
