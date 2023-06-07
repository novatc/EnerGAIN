
from stable_baselines3 import PPO
from enviroment import EnergyMarketEnv

# Create the environment
env = EnergyMarketEnv()

# Create the agent

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=30000)

# Test the trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()
