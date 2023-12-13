import argparse

import numpy as np
import pandas as pd
from gymnasium import register, make
from stable_baselines3 import SAC

from cutsom_wrappers.custom_wrappers import CustomNormalizeObservation
from envs.assets import env_utilities as utilities
import warnings
import json

# Define and parse command-line arguments
parser = argparse.ArgumentParser(description='Evaluate a SAC model.')
parser.add_argument('--env',
                    choices=['base', 'trend', 'no_savings', 'base_prl', 'multi', 'multi_no_savings',
                             'multi_trend', 'reward_boosting'],
                    default="base",
                    required=True,
                    help='Environment to use.')
parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run.')
parser.add_argument('--plot', action='store_true', help='Plot the results.')
parser.add_argument('--month', type=int, default=5, help='Month to use for validation.')
args = parser.parse_args()

validation_da_data_path = f'data/in-use/eval_data/month_{args.month}_data_da.csv'
validation_prl_data_path = f'data/in-use/eval_data/month_{args.month}_data_prl.csv'

if args.month == 0:
    validation_da_data_path = f'data/in-use/eval_data/average_da_year.csv'
    validation_prl_data_path = f'data/in-use/eval_data/average_prl_year.csv'

# Define environment parameters
env_params = {
    'base': {'id': 'base_env-v0', 'entry_point': 'envs.base_env:BaseEnv',
             'data_path_da': validation_da_data_path},

    'trend': {'id': 'trend_env-v0', 'entry_point': 'envs.trend_env:TrendEnv',
              'data_path_da': validation_da_data_path},

    'no_savings': {'id': 'no_savings_env-v0', 'entry_point': 'envs.no_savings_env:NoSavingsEnv',
                   'data_path_da': validation_da_data_path},

    'base_prl': {'id': 'base_prl-v0', 'entry_point': 'envs.base_prl:BasePRL',
                 'data_path_prl': validation_prl_data_path,
                 'data_path_da': validation_da_data_path},

    'multi': {'id': 'multi-v0', 'entry_point': 'envs.multi_market:MultiMarket',
              'data_path_prl': validation_prl_data_path,
              'data_path_da': validation_da_data_path},

    'multi_no_savings': {'id': 'multi_no_savings-v0', 'entry_point': 'envs.multi_no_savings:MultiNoSavings',
                         'data_path_prl': validation_prl_data_path,
                         'data_path_da': validation_da_data_path},

    'multi_trend': {'id': 'multi_trend-v0', 'entry_point': 'envs.multi_trend:MultiTrend',
                    'data_path_prl': validation_prl_data_path,
                    'data_path_da': validation_da_data_path},

    'reward_boosting': {'id': 'reward_boosting-v0', 'entry_point': 'envs.reward_boosting:RewardBoosting',
                        'data_path_prl': validation_prl_data_path,
                        'data_path_da': validation_da_data_path}
}

# Check if chosen environment is valid
if args.env not in env_params:
    raise ValueError(
        f"Invalid environment '{args.env}'. "
        f"Choices are 'base', 'trend', 'savings_reward', 'unscaled' and 'no_savings'.")

# Set chosen environment parameters
env_id = env_params[args.env]['id']
entry_point = env_params[args.env]['entry_point']
data_path_da = env_params[args.env]['data_path_da']
data_path_prl = env_params["base_prl"]['data_path_prl']

# suppress any warnings
warnings.filterwarnings("ignore")

# Load the model
try:
    # find the model that name starts with sac_{args.env}
    model_name = [name for name in utilities.get_model_names() if name.startswith(f"sac_{args.env}_")][0]
    print(f"Loading model {model_name}")
    model = SAC.load(f"agents/{model_name}")
except Exception as e:
    print("Error loading model: ", e)
    exit()

# Register and make the environment
# Register and make the environment
if (args.env == 'base_prl' or args.env == 'multi' or args.env == 'multi_no_savings'
        or args.env == 'multi_trend' or args.env == 'reward_boosting'):
    register(id=env_id, entry_point=entry_point,
             kwargs={'da_data_path': data_path_da, 'prl_data_path': data_path_prl, 'validation': True})
else:
    print(f"Registering {env_id} with {entry_point}")
    register(id=env_id, entry_point=entry_point,
             kwargs={'da_data_path': data_path_da, 'validation': True})
try:
    eval_env = make(env_id)
    eval_env = CustomNormalizeObservation(eval_env)
except Exception as e:
    print("Error creating environment: ", e)
    exit()

ep_length = eval_env.da_dataframe.shape[0]

# Initialize variables for trade statistics
buy_count = sell_count = reserve_count = 0
total_buy_price = total_sell_price = 0
buy_price_differences = []  # List to store buy price differences
sell_price_differences = []  # List to store sell price differences

# Evaluate the agent and gather trade data
episode_rewards = []
obs, _ = eval_env.reset()
for _ in range(args.episodes):
    episode_reward = 0
    done = False
    for _ in range(ep_length - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        episode_reward += reward
    episode_rewards.append(episode_reward)
    obs, _ = eval_env.reset()

trades = eval_env.get_trades()

trades_log = pd.DataFrame(trades, columns=["step", "type", "market price", "offered_price", "amount", "reward", "case"])
# write trades to csv
trades_log.to_csv(f"trade_logs/{model_name}_trades.csv", index=False)

invalid_trades = eval_env.get_invalid_trades()
invalid_trades_log = pd.DataFrame(invalid_trades, columns=["step", "type", "market price", "offered_price", "amount",
                                                           "reward", "case"])
# write invalid trades to csv
invalid_trades_log.to_csv(f"trade_logs/invalid/{model_name}_invalid_trades.csv", index=False)

# count how many times a buy or sell action was taken


# Process trades
trades = eval_env.get_trades()
for trade in trades:
    trade_type, offered_price, market_price = trade[1], trade[3], trade[2]
    if trade_type == 'buy':
        buy_count += 1
        total_buy_price += offered_price
        # Calculate difference for each buy trade and add to list
        buy_price_differences.append(abs(offered_price - market_price))
    elif trade_type == 'sell':
        sell_count += 1
        total_sell_price += offered_price
        # Calculate difference for each sell trade and add to list
        sell_price_differences.append(abs(offered_price - market_price))
    elif trade_type == 'reserve':
        reserve_count += 1

try:
    avg_price = sum([trade[2] for trade in trades]) / len(trades)
except ZeroDivisionError:
    avg_price = 0
try:
    avg_amount = sum([abs(trade[4]) for trade in trades]) / len(trades)
except ZeroDivisionError:
    avg_amount = 0

# Calculate average statistics
avg_buy_price = total_buy_price / buy_count if buy_count else 0
avg_sell_price = total_sell_price / sell_count if sell_count else 0
avg_buy_amount = sum([abs(trade[4]) for trade in trades if trade[1] == 'buy']) / buy_count if buy_count else 0
avg_sell_amount = sum([abs(trade[4]) for trade in trades if trade[1] == 'sell']) / sell_count if sell_count else 0

# Calculate average price differences
avg_buy_price_difference = np.mean(buy_price_differences) if buy_price_differences else 0
avg_sell_price_difference = np.mean(sell_price_differences) if sell_price_differences else 0

# Display statistics
print(f"Average Buy Price: {avg_buy_price:.2f}")
print(f"Average Sell Price: {avg_sell_price:.2f}")
print(f"Average Price Difference from Market for Buy Trades: {avg_buy_price_difference:.2f}")
print(f"Average Price Difference from Market for Sell Trades: {avg_sell_price_difference:.2f}")
print(f"Average price: {avg_price:.2f}")
print(f"Average amount: {avg_amount:.2f}")
print(f"Number of trades: {len(trades)}")
print(f"Buy count: {buy_count}")
print(f"Sell count: {sell_count}")
print(f"Reserve count: {reserve_count}")
print(f"Number of Hold actions:{len(eval_env.get_holdings())}")
print(f"Number of invalid trades: {len(invalid_trades)}")
print(f"Average reward: {(episode_rewards[0] / len(trades)):.2f}")
print(f"Total reward: {np.sum(episode_rewards):.2f}")
print(f"Total profit: {eval_env.savings_log[-1]:.2f}")

# write these states to a json file
stats = {
    "avg_price": avg_price,
    "avg_amount": avg_amount,
    "buy_count": buy_count,
    "sell_count": sell_count,
    "reserve_count": reserve_count,
    "num_holdings": len(eval_env.get_holdings()),
    "num_invalid_trades": len(invalid_trades),
    "avg_buy_price": avg_buy_price,
    "avg_sell_price": avg_sell_price,
    "avg_buy_amount": avg_buy_amount,
    "avg_sell_amount": avg_sell_amount,
    "avg_buy_price_difference": avg_buy_price_difference,
    "avg_sell_price_difference": avg_sell_price_difference,
    "avg_reward": episode_rewards[0] / len(trades),
    "total_reward": np.sum(episode_rewards),
    "total_profit": eval_env.savings_log[-1]
}

with open(f"agent_data/{model_name}_stats.json", "w") as f:
    json.dump(stats, f, indent=4)

if args.plot:
    eval_env.render()
