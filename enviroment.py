from typing import Tuple

import numpy as np
import pandas as pd
import gym
from gym import spaces
from gym.core import ActType, ObsType


class EnergyMarkets(gym.Env):

    def __init__(self, data: pd.DataFrame):
        super(EnergyMarkets, self).__init__()

        self.data = data
        self.current_step = 0

        self.action_space = spaces.Discrete(3)  # 0: buy, 1: sell, 2: hold
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.data.columns),), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return self.data.iloc[self.current_step].values

    def step(self, action):
        """
        Take a step in the environment
        :param action:
        :return: returns the next state, reward, done, and info
        """
        self.current_step += 1

        state = self.data.iloc[self.current_step].values
        reward = self.calculate_reward(action)

        done = self.current_step == len(self.data) - 1

        return state, reward, done, {}

    def calculate_reward(self, action):
        """
        Calculate the reward for the given action
        :param action:
        :return: reward
        """
        reward = 0
        return reward
