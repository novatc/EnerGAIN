import os
import time
import argparse
from typing import Callable

from stable_baselines3.common.env_checker import check_env
from gymnasium import register
from gymnasium import make

from stable_baselines3 import SAC

# Define and parse command-line arguments
parser = argparse.ArgumentParser(description='Train a SAC model.')
parser.add_argument('--training_steps', type=int, required=True, default=100_00, help='Number of training steps.')
parser.add_argument('--env', choices=['base', 'trend', 'savings'], default="base", required=True,
                    help='Environment to use.')
args = parser.parse_args()

# Define environment parameters
env_params = {
    'base': {'id': 'base_env-v0', 'entry_point': 'envs.base_env:BaseEnergyEnv',
             'data_path': 'data/in-use/train_data.csv'},
    'trend': {'id': 'trend_env-v0', 'entry_point': 'envs.trend_env:TrendEnv',
              'data_path': 'data/in-use/train_data.csv'},
    'no_savings': {'id': 'no_savings_env-v0', 'entry_point': 'envs.no_savings_env:NoSavingsEnv',
                   'data_path': 'data/in-use/train_data.csv'}
}

# Check if chosen environment is valid
if args.env not in env_params:
    raise ValueError(f"Invalid environment '{args.env}'. Choices are 'base', 'trend', and 'savings'.")

# Set chosen environment parameters
env_id = env_params[args.env]['id']
entry_point = env_params[args.env]['entry_point']
data_path = env_params[args.env]['data_path']

os.makedirs('logging', exist_ok=True)

# Register and make environment
register(id=env_id, entry_point=entry_point, kwargs={'data_path': data_path})
env = make(env_id)
check_env(env)

start_time = time.time()  # Get the current time


# Create and train model
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=args.training_steps)
now = time.strftime("%Y%m%d-%H%M%S")
model.save(f"agents/sac_{args.env}_{now}")

end_time = time.time()  # Get the current time after running the model

print(f'Total runtime: {(end_time - start_time) / 60} minutes')  # Print the difference, which is the total runtime
