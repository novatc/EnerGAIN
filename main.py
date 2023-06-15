import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from enviroment import EnergyMarketEnv
from stable_baselines3.common.env_checker import check_env

# Create the environment
env = EnergyMarketEnv()
check_env(env, warn=True)

# Create the agent

model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=50_000)
# print the sum of env.price_history
env.render()
trades = env.get_trade_log()
trades.to_csv('trades.csv')




















# Test the trained agent
obs = env.reset()
# for i in range(8700):
#     action, _states = model.predict(obs)
#     obs, reward, done, info = env.step(action)
#     if done:
#       obs = env.reset()
# env.render()
