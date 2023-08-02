import pandas as pd
from gymnasium import register, make
from stable_baselines3 import SAC
from envs.assets import env_utilities as utilities


try:
    model = SAC.load("agents/sac_trend_env")
except Exception as e:
    print("Error loading model: ", e)
    exit()

register(
    id='trend-v0',
    entry_point='envs.trend_env:TrendEnv',
    kwargs={'data_path': "data/in-use/eval_data.csv"}
)

try:
    eval_env = make('trend-v0')
except Exception as e:
    print("Error creating environment: ", e)
    exit()

obs, _ = eval_env.reset()

ep_length = eval_env.dataframe.shape[0]

# Evaluate the agent
episode_rewards = []
num_episodes = 1

for _ in range(num_episodes):
    episode_reward = 0
    done = False
    for _ in range(ep_length):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        print(info)
        episode_reward += reward
    episode_rewards.append(episode_reward)
    obs, _ = eval_env.reset()

trades = eval_env.get_trades()
# list of tuples (step, price, amount, trade_type) to dataframe
df_trades = pd.DataFrame(trades, columns=['step', 'price', 'amount', 'trade_type'])
df_trades.to_csv("trades.csv", index=False)


# count how many times a buy or sell action was taken
buy_count = 0
sell_count = 0
for trade in trades:
    if trade[2] > 0:
        buy_count += 1
    elif trade[2] < 0:
        sell_count += 1

avg_price = sum([trade[1] for trade in trades]) / len(trades)
rescaled_avg_price = utilities.rescale_value_price(avg_price)

print("Total reward:", sum(episode_rewards))
print("Total trades:", len(trades))
print("Buy count:", buy_count)
print("Sell count:", sell_count)
print("Average reward:", sum(episode_rewards) / len(trades))
print("Average price:", rescaled_avg_price)
print("Average amount:", sum([trade[2] for trade in trades]) / len(trades))

eval_env.render()
