import gym
from stable_baselines3 import DQN

# Create the environment
env = gym.make('LunarLander-v2')

# Create the agent
model = DQN('MlpPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Save the trained agent
model.save("dqn_lunar")

# Load the trained agent
model = DQN.load("dqn_lunar")

# Use the trained agent to play an episode
done = False
obs = env.reset()
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()

env.close()
