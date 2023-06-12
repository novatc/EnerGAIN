
from stable_baselines3 import PPO
from enviroment import EnergyMarketEnv
from stable_baselines3.common.env_checker import check_env

# Create the environment
env = EnergyMarketEnv()
check_env(env, warn=True)

# Create the agent

model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=1_000_00)
# print the sum of env.price_history
env.render()

#Test the trained agent
# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     if done:
#       obs = env.reset()
# env.render()