import os
from typing import Callable

import numpy as np
from stable_baselines3.common.env_checker import check_env
from gymnasium import register
from gymnasium import make
from gymnasium.wrappers import NormalizeObservation
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from callbacks.tensorboard_callback import LoggingCallback
from cutsom_wrappers.cast_wrapper import CastObservation

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
env = Monitor(env, filename="logging/", allow_early_resets=True)
norm_obs_env = NormalizeObservation(env)
cast_obs_env = CastObservation(norm_obs_env)
check_env(cast_obs_env)

model = PPO("MlpPolicy", cast_obs_env, verbose=0, tensorboard_log="logging/",
            device="auto")
model.learn(total_timesteps=10_000, callback=LoggingCallback())
model.save("agents/ppo_energy_testing")
env.render()
