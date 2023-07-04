import numpy as np
from gymnasium import Wrapper
from gymnasium.vector.utils import spaces


class CastObservation(Wrapper):
    def __init__(self, env):
        super(CastObservation, self).__init__(env)

        # Change the dtype of the observation space to float32
        self.observation_space = spaces.Box(
            low=self.observation_space.low.astype(np.float32),
            high=self.observation_space.high.astype(np.float32),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        observation, info = self.env.reset()
        return observation.astype(np.float32), info

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        return observation.astype(np.float32), reward, done, info
