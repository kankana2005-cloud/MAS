import gymnasium as gym
from stable_baselines3 import PPO

# Create the environment
env = gym.make("CartPole-v1")

# Create the agent and train it
print("Training the agent... please wait!")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

print("\nTraining done! Now lets see how it performs...")

# Test the trained agent
env2 = gym.make("CartPole-v1", render_mode="rgb_array")
observation, info = env2.reset()

total_reward = 0
for step in range(500):
    action, _ = model.predict(observation)
    observation, reward, terminated, truncated, info = env2.step(action)
    total_reward += reward
    print(f"Step {step+1} | Action: {action} | Reward: {reward} | Total: {total_reward}")
    
    if terminated or truncated:
        print(f"--- Pole fell after {step+1} steps! Total reward: {total_reward} ---")
        break

env2.close()
print(f"\nFinal total reward: {total_reward}")