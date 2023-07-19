import joblib
import numpy as np
from gymnasium import register, make
from matplotlib import pyplot as plt
from stable_baselines3 import PPO, SAC

try:
    model = SAC.load("agents/sac")
except Exception as e:
    print("Error loading model: ", e)
    exit()

register(
    id='energy-validation-v0',
    entry_point='environment:EnergyEnv',
    kwargs={'data_path': "data/clean/test_set.csv"}
)

try:
    eval_env = make('energy-validation-v0')
except Exception as e:
    print("Error creating environment: ", e)
    exit()

obs, _ = eval_env.reset()

ep_length = eval_env.dataframe.shape[0]

# Evaluate the agent
episode_rewards = []
num_episodes = 15

for _ in range(num_episodes):
    episode_reward = 0
    done = False
    for _ in range(ep_length):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        episode_reward += reward
    episode_rewards.append(episode_reward)
    obs, _ = eval_env.reset()

trades = eval_env.get_trades()
# save trades to file
with open("trades.txt", "w") as f:
    for trade in trades:
        f.write(str(trade) + "\n")

# count how many times a buy or sell action was taken
buy_count = 0
sell_count = 0
for trade in trades:
    if trade[2] > 0:
        buy_count += 1
    elif trade[2] < 0:
        sell_count += 1

print("Total reward:", sum(episode_rewards))
print("Total trades:", len(trades))
print("Buy count:", buy_count)
print("Sell count:", sell_count)
print("Average reward:", sum(episode_rewards) / len(trades))
print("Average price:", sum([trade[1] for trade in trades]) / len(trades))
print("Average amount:", sum([trade[2] for trade in trades]) / len(trades))

eval_env.render()
