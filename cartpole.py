import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="rgb_array")
observation, info = env.reset()

for step in range(200):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    print(f"Step {step+1} | Action: {action} | Reward: {reward} | Done: {terminated}")

    if terminated or truncated:
        print("--- Pole fell! Restarting ---")
        observation, info = env.reset()

env.close()
print("Done!")