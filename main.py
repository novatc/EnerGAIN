import os
from typing import Callable

from stable_baselines3.common.env_checker import check_env
from gymnasium import register
from gymnasium import make

from stable_baselines3 import SAC

os.makedirs('logging', exist_ok=True)

register(
    id='no_savings-v0',
    entry_point='envs.no_savings_env:NoSavingsEnv',
    kwargs={'data_path': "data/clean/env_data.csv"}
)

env = make('no_savings-v0')
check_env(env)

model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("agents/sac_no_savings")
