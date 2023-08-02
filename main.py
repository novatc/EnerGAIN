import os
from typing import Callable

from stable_baselines3.common.env_checker import check_env
from gymnasium import register
from gymnasium import make

from stable_baselines3 import SAC

os.makedirs('logging', exist_ok=True)

register(
    id='trend_env-v0',
    entry_point='envs.trend_env:TrendEnv',
    kwargs={'data_path': "data/in-use/train_data.csv"}
)

env = make('trend_env-v0')
check_env(env)

model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("agents/sac_trend_env")
