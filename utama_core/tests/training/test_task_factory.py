"""Task factory regression tests."""

from utama_core.training.task import SSLTask


def test_ssl_task_env_factory_returns_fresh_scenario_instances():
    task = SSLTask.from_name("ssl_2v0_unified")
    env_fn = task.get_env_fun(num_envs=1, continuous_actions=True, seed=0, device="cpu")

    env1 = env_fn()
    env2 = env_fn()
    try:
        assert env1._env.scenario is not env2._env.scenario
        assert env1._env.scenario.cfg is not env2._env.scenario.cfg
    finally:
        env1.close()
        env2.close()
