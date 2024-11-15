from gymnasium.envs.registration import register

register(
    id="VSS-v0", entry_point="rsoccer_gym.vss.env_vss:VSSEnv", max_episode_steps=1200
)

register(
    id="SSLStaticDefenders-v0",
    entry_point="rsoccer_gym.ssl.ssl_hw_challenge.static_defenders:SSLHWStaticDefendersEnv",
    kwargs={"field_type": 2},
    max_episode_steps=1000,
)

register(
    id="SSLDribbling-v0",
    entry_point="rsoccer_gym.ssl.ssl_hw_challenge.dribbling:SSLHWDribblingEnv",
    max_episode_steps=4800,
)

register(
    id="SSLContestedPossession-v0",
    entry_point="rsoccer_gym.ssl.ssl_hw_challenge.contested_possession:SSLContestedPossessionEnv",
    max_episode_steps=1200,
)

register(
    id="SSLPassEndurance-v0",
    entry_point="rsoccer_gym.ssl.ssl_hw_challenge:SSLPassEnduranceEnv",
    max_episode_steps=1200,
)

register(
    id="SSLStandard-v0",
    entry_point="rsoccer_gym.ssl.ssl_hw_challenge.standard_ssl:SSLStandardEnv",
    kwargs={"field_type": 2, "n_robots_blue": 6, "n_robots_yellow": 6, "time_step": 0.025},
    max_episode_steps=1000,
)