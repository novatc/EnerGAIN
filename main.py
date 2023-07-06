import os
from typing import Callable

import numpy as np
from stable_baselines3.common.env_checker import check_env
from gymnasium import register
from gymnasium import make
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

os.makedirs('logging', exist_ok=True)


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


register(
    id='energy-v0',
    entry_point='environment:EnergyEnv',
    kwargs={'data_path': "data/clean/env_data.csv"}
)

env = make('energy-v0')
check_env(env)

model = SAC("MlpPolicy", env, verbose=0, tensorboard_log="logging/")
model.learn(total_timesteps=500_000, log_interval=4)
model.save("agents/sac_energy_testing")
env.render()
