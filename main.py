from typing import Callable

import numpy as np
from stable_baselines3.common.env_checker import check_env
from gymnasium import register
from gymnasium import make
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import ProgressBarCallback

from callbacks.tensorboard_callback import LoggingCallback
from custom_wrappers import NormalizeObservation


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
    max_episode_steps=37272,
    kwargs={'data_path': "data/clean/env_data.csv"}
)

env = make('energy-v0')
env = Monitor(env, filename="logging/", allow_early_resets=True)
env = NormalizeObservation(env, max_savings=1, max_charge=1)
check_env(env)

model = PPO("MlpPolicy", env, verbose=0, tensorboard_log="logging/",
            device="auto")
model.learn(total_timesteps=2_000)
model.save("agents/ppo_energy_testing")
env.render()
