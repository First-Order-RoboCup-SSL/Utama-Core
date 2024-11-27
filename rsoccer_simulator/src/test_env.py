import gymnasium as gym

# importing this as this test is run within the rsoccer_simulator.src package (to perform the __init__.py registration)
import rsoccer_simulator.src

# making environment
env = gym.make("SSLStandard-v0")

env.reset()
# Run for 1 episode and print reward at the end
for i in range(10000):
    terminated = False
    truncated = False
    while not (terminated or truncated):
        # Step using random actions
        action = env.action_space.sample()
        print(f"Action: {action}\n\n")
        # reward_shaping is for info purposes only
        next_state, reward, terminated, truncated, reward_shaping = env.step(action)
        print(f"Next state: {next_state}\n\nReward_shape: {reward_shaping}\n\n")
        # time.sleep(0.1)
