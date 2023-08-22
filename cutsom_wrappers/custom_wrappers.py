import numpy as np
from gymnasium import Wrapper
from gymnasium.wrappers import NormalizeObservation


class CustomNormalizeObservation(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = NormalizeObservation(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # make sure the observation is a numpy array
        if isinstance(obs, tuple):
            obs = np.array(obs)
        obs = obs.astype(np.float32)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs.astype(np.float32), reward, terminated, truncated, info
