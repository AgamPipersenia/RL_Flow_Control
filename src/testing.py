from flow_env import FlowControlEnv

env = FlowControlEnv()
obs, _ = env.reset()
for step in range(1000):
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    print(f"Step: {step}, Reward: {reward}, Done: {done}")
    if done:
        print("Episode finished.")
        obs, _ = env.reset()