# Utama

## Setup Guidelines

1. cd to root folder `/Utama` and type `pip install -e .` to install all dependencies.
2. Note that this also installs all modules with `__init__.py` (so you need to run it again when you add an `__init__.py`)

If you are still struggling with local import errors (ie importing our local modules), you can add `export PYTHONPATH=/path/to/folder/Utama` to your `~/.bashrc`. To do this, run `sudo nano ~/.bashrc` and then add the export line at the bottom of the document. Finally, close and save.

###### Warning: the above workaround is quite hackish and may result in source conflicts if you are working on other projects in the same Linux machine. Be advised.

### Field Guide

![field_guide](assets/images/field_guide.jpg)

1. All coordinates and velocities will be in meters or meters per second.
2. All angular properties will be in radians or radians per second, normalised between [pi, -pi]. A heading of radian 0 indicates a robot facing towards the positive x-axis (ie left to right).
3. Unless otherwise stated, the coordinate system is aligned such that blue robots are on the left and yellow are on the right.

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
9. `replay`: replay system for storing played games in a .pkl file that can be reconstructed in rSoccer sim [**imports from rsoccer**]

#### Code Writing

1. Use typing for all functions.
2. Please please document your code on the subfolder's `README.md`.
3. Download and install `Black Formatter` for code formatting

   1. For VScode, go to View > Command Palette and search `Open User Settings (JSON)`
   2. Find the `"[python]"` field and add the following lines:

   ```yaml
   "[python]": {
       "editor.defaultFormatter": "ms-python.black-formatter", # add this
       "editor.formatOnSave": true, # and add this
     }
   ```

#### Push and Commit

1. Each team should be working within your own branch of the repository.
2. Inform your lead when ready to push to main.
3. We aim to merge at different releases, so that it is easier for version control.

## Milestones

- 2024 November 20 - First goal in grSim (featuring Ray casting)
