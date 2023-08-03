import os
import time
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

start_time = time.time()  # Get the current time

model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500_000)
model.save("agents/sac_trend_env")

end_time = time.time()  # Get the current time after running the model

print(f'Total runtime: {(end_time - start_time)/60} minutes')  # Print the difference, which is the total runtime
