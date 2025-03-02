
# RSim Test guide

Game is now immutable so any modification to game will return a new game object. This means that the structure of RSim tests will change now. 

RSim is interfaced via the RSimController class. When the RSim environment steps it used to update the game passed by reference. However, this is not possible anymore. To do this, we instead expose access to the game via the RSimRobotController. Therefore use of game in RSim tests should be of the following structure.

## Initialisation

    game = Game(...)


    env = SSLStandardEnv(...)
    env.reset()

    env_controller = RSimController(env)

    sim_robot_controller = RSimRobotController(
        is_team_yellow=is_yellow, env=env, game_obj=game
    )


    # To use the game in your test

    friendly_robots = sim_robot_controller.game.friendly_robots

    # Note that you can extract the game at the start of a loop iteration

    temp_game = sim_robot_controller.game

    print(temp_game.friendly_robots)

    # But when RSim steps, this temp_game object does NOT update
    # so it must be discarded.
## Behaviour Trees
TODO