import os
import random
import time
import argparse

import numpy as np
from gymnasium import register
from gymnasium import make
from stable_baselines3.common.noise import NormalActionNoise

from callbacks.summary_writer import SummaryWriterCallback
from cutsom_wrappers.custom_wrappers import CustomNormalizeObservation
from gymnasium.wrappers import NormalizeReward

from stable_baselines3 import SAC

# Define and parse command-line arguments
parser = argparse.ArgumentParser(description='Train a SAC model.')
parser.add_argument('--training_steps', type=int, required=True, default=100,
                    help='Number of training steps.')
parser.add_argument('--env',
                    choices=['base', 'trend', 'no_savings', 'base_prl', 'multi', 'multi_no_savings',
                             'multi_trend', 'base_state', 'trend_compact', 'multi_compact',
                             'base_state_ext', 'trend_compact_ext', 'multi_compact_ext'],
                    default="base",
                    required=True,
                    help='Environment to use.')
parser.add_argument('--seed', type=int, default=None,
                    help='Seed for the stdlib, numpy and SAC RNGs. The market simulators draw their '
                         'random-walk start from the global random module, so without this a run is '
                         'not reproducible even if the env is seeded.')
parser.add_argument('--save', action='store_true',
                    help='Save the model.')  # q: how to not save the model? a: don't use this flag
args = parser.parse_args()

# Define environment parameters
env_params = {
    'base_state': {'id': 'base_state-v0', 'entry_point': 'envs.base_state:BaseState',
                   'data_path_da': 'data/in-use/unscaled_train_data.csv'},

    'trend_compact': {'id': 'trend_compact-v0', 'entry_point': 'envs.trend_compact:TrendCompact',
                      'data_path_da': 'data/in-use/unscaled_train_data.csv'},

    'multi_compact': {'id': 'multi_compact-v0', 'entry_point': 'envs.multi_compact:MultiCompact',
                      'data_path_prl': 'data/prm/preprocessed_prl.csv',
                      'data_path_da': 'data/in-use/unscaled_train_data.csv'},

    'base_state_ext': {'id': 'base_state_ext-v0', 'entry_point': 'envs.base_state:BaseState',
                       'data_path_da': 'data/in-use/unscaled_train_data_ext.csv'},

    'trend_compact_ext': {'id': 'trend_compact_ext-v0', 'entry_point': 'envs.trend_compact:TrendCompact',
                          'data_path_da': 'data/in-use/unscaled_train_data_ext.csv'},

    'multi_compact_ext': {'id': 'multi_compact_ext-v0', 'entry_point': 'envs.multi_compact:MultiCompact',
                          'data_path_prl': 'data/prm/preprocessed_prl_ext.csv',
                          'data_path_da': 'data/in-use/unscaled_train_data_ext.csv'},

    'base': {'id': 'base_env-v0', 'entry_point': 'envs.base_env:BaseEnv',
             'data_path_da': 'data/in-use/unscaled_train_data.csv'},

    'trend': {'id': 'trend_env-v0', 'entry_point': 'envs.trend_env:TrendEnv',
              'data_path_da': 'data/in-use/unscaled_train_data.csv'},

    'no_savings': {'id': 'no_savings_env-v0', 'entry_point': 'envs.no_savings_env:NoSavingsEnv',
                   'data_path_da': 'data/in-use/unscaled_train_data.csv'},

    'base_prl': {'id': 'base_prl-v0', 'entry_point': 'envs.base_prl:BasePRL',
                 'data_path_prl': 'data/prm/preprocessed_prl.csv',
                 'data_path_da': 'data/in-use/unscaled_train_data.csv'},

    'multi': {'id': 'multi-v0', 'entry_point': 'envs.multi_market:MultiMarket',
              'data_path_prl': 'data/prm/preprocessed_prl.csv',
              'data_path_da': 'data/in-use/unscaled_train_data.csv'},

    'multi_no_savings': {'id': 'multi_no_savings-v0', 'entry_point': 'envs.multi_no_savings:MultiNoSavings',
                         'data_path_prl': 'data/prm/preprocessed_prl.csv',
                         'data_path_da': 'data/in-use/unscaled_train_data.csv'},

    'multi_trend': {'id': 'multi_trend-v0', 'entry_point': 'envs.multi_trend:MultiTrend',
                    'data_path_prl': 'data/prm/preprocessed_prl.csv',
                    'data_path_da': 'data/in-use/unscaled_train_data.csv'},

}

# Check if chosen environment is valid
if args.env not in env_params:
    raise ValueError(
        f"Invalid environment '{args.env}'. Choices are 'base', 'trend', 'no_savings', 'base_prl', "
        f"'multi', 'multi_no_savings' and 'multi_trend'.")

# Set chosen environment parameters
env_id = env_params[args.env]['id']
entry_point = env_params[args.env]['entry_point']
data_path_da = env_params[args.env]['data_path_da']
data_path_prl = env_params[args.env].get('data_path_prl', env_params["base_prl"]['data_path_prl'])

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)

os.makedirs('logging', exist_ok=True)

# Register and make the environment
if args.env in ('base_prl', 'multi', 'multi_no_savings', 'multi_trend',
                'multi_compact', 'multi_compact_ext'):
    register(id=env_id, entry_point=entry_point,
             kwargs={'da_data_path': data_path_da, 'prl_data_path': data_path_prl, 'validation': False})
else:
    register(id=env_id, entry_point=entry_point,
             kwargs={'da_data_path': data_path_da, 'validation': False})

eval_env = make(env_id)
env = CustomNormalizeObservation(eval_env)
try:
    eval_env = make(env_id)
    env = CustomNormalizeObservation(eval_env)
    env = NormalizeReward(env)
except Exception as e:
    print("Error creating environment: ", e)
    exit()

start_time = time.time()  # Get the current time

# Create and train model
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
model = SAC("MlpPolicy", env, verbose=0, device='auto', action_noise=action_noise, seed=args.seed,
            tensorboard_log='logging/tensorboard_logs/{}/'.format(args.env))

print(f'Training device: {model.device}')
model.learn(total_timesteps=args.training_steps, progress_bar=True)
now = time.strftime("%d.%m-%H-%M")
if args.save:
    model.save(f"agents/sac_{args.env}_{args.training_steps / 1000}k_{now}.zip")

end_time = time.time()  # Get the current time after running the model

print(f'Total runtime: {(end_time - start_time) / 60} minutes')  # Print the difference, which is the total runtime
