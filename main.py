from stable_baselines3 import PPO
from enviroment import EnergyMarketEnv
from stable_baselines3.common.env_checker import check_env

# Create the environment
env = EnergyMarketEnv()
check_env(env, warn=True)

# Create the agent

model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=74_544)
# print the sum of env.price_history
env.render()
trades = env.get_trade_log()
trades.to_csv('trades.csv')

# plot the trades dataframe with the columns: price', 'amount', 'trade_type', 'date', 'market_price', 'reward
trades.plot(x='date', y='price', kind='scatter')
trades.plot(x='date', y='amount', kind='scatter')
trades.plot(x='date', y='reward', kind='scatter')
trades.plot(x='date', y='market_price', kind='scatter')

















# Test the trained agent
obs = env.reset()
# for i in range(8700):
#     action, _states = model.predict(obs)
#     obs, reward, done, info = env.step(action)
#     if done:
#       obs = env.reset()
# env.render()
