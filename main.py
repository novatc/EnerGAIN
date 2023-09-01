import os
import time
import argparse

import numpy as np
from stable_baselines3.common.env_checker import check_env
from gymnasium import register
from gymnasium import make
from cutsom_wrappers.custom_wrappers import CustomNormalizeObservation
from gymnasium.wrappers import RescaleAction

from stable_baselines3 import SAC

# Define and parse command-line arguments
parser = argparse.ArgumentParser(description='Train a SAC model.')
parser.add_argument('--training_steps', type=int, required=True, default=100, help='Number of training steps.')
parser.add_argument('--env', choices=['base', 'trend', 'no_savings', 'savings_reward', 'unscaled'], default="unscaled",
                    required=True,
                    help='Environment to use.')
parser.add_argument('--save', action='store_true',
                    help='Save the model.')  # q: how to not save the model? a: don't use this flag
args = parser.parse_args()

# Define environment parameters
env_params = {
    'base': {'id': 'base_env-v0', 'entry_point': 'envs.base_env:BaseEnv',
             'data_path': 'data/in-use/unscaled_train_data.csv'},
    'trend': {'id': 'trend_env-v0', 'entry_point': 'envs.trend_env:TrendEnv',
              'data_path': 'data/in-use/unscaled_train_data.csv'},
    'no_savings': {'id': 'no_savings_env-v0', 'entry_point': 'envs.no_savings_env:NoSavingsEnv',
                   'data_path': 'data/in-use/unscaled_train_data.csv'},
    'savings_reward': {'id': 'savings_reward_env-v0', 'entry_point': 'envs.savings_reward:SavingsRewardEnv',
                       'data_path': 'data/in-use/unscaled_train_data.csv'},
    'unscaled': {'id': 'unscaled_env-v0', 'entry_point': 'envs.unscaled_env:UnscaledEnv',
                 'data_path': 'data/in-use/unscaled_train_data.csv'}
}

# Check if chosen environment is valid
if args.env not in env_params:
    raise ValueError(
        f"Invalid environment '{args.env}'. Choices are 'base', 'trend', 'savings_reward', 'unscaled' and 'no_savings'.")

# Set chosen environment parameters
env_id = env_params[args.env]['id']
entry_point = env_params[args.env]['entry_point']
data_path = env_params[args.env]['data_path']

os.makedirs('logging', exist_ok=True)

# Register and make the environment
register(id=env_id, entry_point=entry_point, kwargs={'data_path': data_path, 'validation': False})
try:
    eval_env = make(env_id)
    env = CustomNormalizeObservation(eval_env)
except Exception as e:
    print("Error creating environment: ", e)
    exit()

start_time = time.time()  # Get the current time

# Create and train model
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=args.training_steps)
now = time.strftime("%Y%m%d-%H%M%S")
if args.save:
    model.save(f"agents/sac_{args.env}_{now}")

end_time = time.time()  # Get the current time after running the model

print(f'Total runtime: {(end_time - start_time) / 60} minutes')  # Print the difference, which is the total runtime
