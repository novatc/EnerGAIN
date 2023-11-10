import argparse

import numpy as np
import pandas as pd
from gymnasium import register, make
from gymnasium.wrappers import RescaleAction
from stable_baselines3 import SAC

from cutsom_wrappers.custom_wrappers import CustomNormalizeObservation
from envs.assets import env_utilities as utilities
import warnings

# Define and parse command-line arguments
parser = argparse.ArgumentParser(description='Evaluate a SAC model.')
parser.add_argument('--env', choices=['base', 'trend', 'no_savings', 'base_prl', 'multi'], default="base",
                    required=True,
                    help='Environment to use.')
parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run.')
parser.add_argument('--plot', action='store_true', help='Plot the results.')
args = parser.parse_args()

# Define environment parameters
env_params = {
    'base': {'id': 'base_env-v0', 'entry_point': 'envs.base_env:BaseEnv',
             'data_path_da': 'data/in-use/unscaled_eval_data.csv'},
    'trend': {'id': 'trend_env-v0', 'entry_point': 'envs.trend_env:TrendEnv',
              'data_path_da': 'data/in-use/unscaled_eval_data.csv'},
    'no_savings': {'id': 'no_savings_env-v0', 'entry_point': 'envs.no_savings_env:NoSavingsEnv',
                   'data_path_da': 'data/in-use/unscaled_eval_data.csv'},
    'base_prl': {'id': 'base_prl-v0', 'entry_point': 'envs.base_prl:BasePRL',
                 'data_path_prl': 'data/prm/preprocessed_prl.csv',
                 'data_path_da': 'data/in-use/unscaled_train_data.csv'},
    'multi': {'id': 'multi-v0', 'entry_point': 'envs.multi_market:MultiMarket',
              'data_path_prl': 'data/prm/preprocessed_prl.csv',
              'data_path_da': 'data/in-use/unscaled_train_data.csv'},
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
if args.env == 'base_prl' or args.env == 'multi':
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

ep_length = 24 * 30

# Evaluate the agent
episode_rewards = []
num_episodes = args.episodes
obs, _ = eval_env.reset()
for _ in range(num_episodes):
    episode_reward = 0
    done = False
    for _ in range(ep_length - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        episode_reward += reward
    episode_rewards.append(episode_reward)
    obs, _ = eval_env.reset()

trades = eval_env.get_trades()
# list of tuples (step, price, amount, trade_type) to dataframe
trades_log = pd.DataFrame(trades, columns=["step", "price", "amount", "trade_type", "reward"])
# write trades to csv
trades_log.to_csv(f"trade_logs/{model_name}_trades.csv", index=False)

# count how many times a buy or sell action was taken
buy_count = 0
sell_count = 0
reserve_count = 0

for trade in trades:
    if trade[3] == 'buy':
        buy_count += 1
    if trade[3] == 'sell':
        sell_count += 1
    if trade[3] == 'reserve':
        reserve_count += 1

try:
    avg_price = sum([trade[1] for trade in trades]) / len(trades)
except ZeroDivisionError:
    avg_price = 0
try:
    avg_amount = sum([trade[2] for trade in trades]) / len(trades)
except ZeroDivisionError:
    avg_amount = 0

# could you pretty print these stats?
print(f"Average reward: {np.mean(episode_rewards)}")
print(f"Average price: {avg_price}")
print(f"Average amount: {avg_amount}")
print(f"Buy count: {buy_count}")
print(f"Sell count: {sell_count}")
print(f"Reserve count: {reserve_count}")

if args.plot:
    eval_env.render()
