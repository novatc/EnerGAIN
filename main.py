from typing import Callable

import numpy as np
from stable_baselines3.common.env_checker import check_env
from gymnasium import register
from gymnasium import make
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import ProgressBarCallback

from callbacks.tensorboard_callback import LoggingCallback


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
check_env(env)

model = PPO("MlpPolicy", env, verbose=0, tensorboard_log="logging/", learning_rate=linear_schedule(0.001))
model.learn(total_timesteps=500_000)
model.save("agents/ppo_energy")
env.render()

# Testing phase

register(
    id='energy_test-v1',
    entry_point='environment:EnergyEnv',
    kwargs={'data_path': "data/clean/test_set.csv"}
)
num_test_episodes = 100
total_rewards = []
test_env = make('energy_test-v1')
for _ in range(num_test_episodes):
    obs = test_env.reset()
    done = False
    episode_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        episode_reward += reward
        total_rewards.append(episode_reward)

print(f"Mean reward over {num_test_episodes} episodes: {np.mean(total_rewards)}")
