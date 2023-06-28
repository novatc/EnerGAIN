import gymnasium as gym
import pandas as pd
from gymnasium import spaces
import numpy as np
from matplotlib import pyplot as plt

from market import Market


class EnergyEnv(gym.Env):
    def __init__(self, data_path):
        super(EnergyEnv, self).__init__()
        self.dataframe = pd.read_csv(data_path)
        self.market = Market(self.dataframe)
        self.savings = 0
        self.charge = 0
        self.max_battery_charge = 1
        self.current_step = 0
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.dataframe.shape[1] + 2,))
        self.charge_log = []
        self.savings_log = []
        self.trade_log = []

        self.rewards = []

    def step(self, action):
        self.current_step += 1
        self.market.step()

        done = False

        if self.current_step >= len(self.dataframe):
            self.current_step = 0
            done = True

        price = float(action[0].item())
        amount = float(action[1].item())

        truncated = False
        info = {}

        reward = -1  # default reward

        if amount > 0:  # buy
            reward = self.buy(price, amount)
        elif amount < 0:  # sell
            reward = self.sell(price, amount)
        else:  # if amount is 0
            reward = 0

        self.rewards.append(reward)
        return self.get_observation().astype(np.float32), reward, done, truncated, info

    def buy(self, price, amount):
        # Check if the agent has enough money to buy
        if price > self.savings or amount > self.max_battery_charge - self.charge or amount <= 0:
            return -1

        if self.market.accept_offer(price, 'buy'):
            self.charge += abs(amount)
            self.savings -= price
            self.charge_log.append(self.charge)
            self.savings_log.append(self.savings)
            self.trade_log.append((self.current_step, self.current_step, price, amount, 'buy'))
            return abs(float(self.market.get_current_price()) * amount)

        return -1

    def sell(self, price, amount):
        if amount < -self.charge or price <= 0:  # Check if the agent has enough energy to sell
            return -1

        if self.market.accept_offer(price, 'sell'):
            self.savings += price
            self.charge -= abs(amount)
            self.charge_log.append(self.charge)
            self.savings_log.append(self.savings)
            self.trade_log.append((self.current_step, self.current_step, price, amount, 'sell'))
            return abs(float(self.market.get_current_price()) * amount)

        return -1  # adjust this as needed for your specific case

    def get_observation(self):
        # Return the current state of the environment as a numpy array
        return np.concatenate((self.dataframe.iloc[self.current_step].to_numpy(), [self.savings, self.charge]))

    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # Reset the state of the environment to an initial state
        super().reset(seed=seed, options=options)
        self.current_step = 0
        self.market.reset()
        return self.get_observation().astype(np.float32), {}

    def render(self, mode='human'):
        # calculate the average reward over 100 steps and plot it
        avg_rewards = []
        scaler = 100
        for i in range(0, len(self.rewards), scaler):
            avg_rewards.append(sum(self.rewards[i:i + scaler]) / scaler)
        plt.plot(avg_rewards)
        plt.ylabel('Average Reward')
        plt.xlabel(f'Number of Steps (/ {scaler})')
        plt.show()

    def get_trades(self):
        return self.trade_log

    def get_savings(self):
        return self.savings_log

    def get_charge(self):
        return self.charge_log
