import numpy as np
import gymnasium as gym
from sklearn.preprocessing import MinMaxScaler


class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env, max_savings, max_charge):
        super().__init__(env)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.max_savings = max_savings
        self.max_charge = max_charge

    def observation(self, obs):
        # Normalize the market data for each feature independently
        market_data = obs[:-2].reshape(1, -1)  # Reshape for the scaler
        scaled_market_data = self.scaler.transform(market_data)[0]  # Apply the scaler
        # Normalize savings and charge
        normalized_savings = obs[-2] / self.max_savings
        normalized_charge = obs[-1] / self.max_charge
        # Scale savings and charge to -1 to 1
        scaled_savings = 2.0 * normalized_savings - 1.0
        scaled_charge = 2.0 * normalized_charge - 1.0
        # Return the scaled observation
        return np.concatenate((scaled_market_data, [scaled_savings, scaled_charge]))
