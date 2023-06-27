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
        self.savings = 0.5
        self.charge = 0.5
        self.max_battery_charge = 1
        self.current_step = 0
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.dataframe.shape[1] + 2,))
        self.latest_price_buy = 0
        self.latest_price_sell = 0

        self.trade_log = []

        self.rewards = []

    def step(self, action):
        self.current_step += 1
        self.market.step()

        if self.current_step >= len(self.dataframe):
            self.current_step = 0

        price = float(action[0].item())
        amount = float(action[1].item())

        done = False
        truncated = False
        info = {}

        if amount > 0:  # buy
            if price > self.savings or amount > self.max_battery_charge - self.charge or amount <= 0:
                reward = -1
                self.rewards.append(reward)
                return self.get_observation().astype(np.float32), reward, done, truncated, info

            if self.market.accept_offer(price, 'buy'):
                # print("Buying with price: ", price, " and amount: ", amount)
                self.charge += abs(amount)
                self.savings -= price
                self.latest_price_buy = price
                reward = abs(float(self.market.get_current_price()) * amount)
                self.rewards.append(reward)
                self.trade_log.append([self.current_step, self.market.get_current_price(), amount, 'buy'])
                return self.get_observation().astype(np.float32), reward, done, truncated, info
            else:
                reward = -1  # adjust this as needed for your specific case
                self.rewards.append(reward)
                return self.get_observation().astype(np.float32), reward, done, truncated, info

        elif amount < 0:  # sell
            if amount < -self.charge or price <= 0:  # Check if the agent has enough energy to sell
                reward = -1
                self.rewards.append(reward)
                self.trade_log.append([self.current_step, self.market.get_current_price(), amount, 'sell'])
                return self.get_observation().astype(np.float32), reward, done, truncated, info

            if self.market.accept_offer(price, 'sell'):
                # print("Selling with price: ", price, " and amount: ", amount)
                self.savings += price
                self.charge -= abs(amount)
                self.latest_price_sell = price
                reward = abs(float(self.market.get_current_price()) * amount)
                self.rewards.append(reward)
                return self.get_observation().astype(np.float32), reward, done, truncated, info
            else:
                reward = -1  # adjust this as needed for your specific case
                self.rewards.append(reward)
                return self.get_observation().astype(np.float32), reward, done, truncated, info

        else:  # if amount is 0
            reward = 0
            self.rewards.append(reward)
            return self.get_observation().astype(np.float32), reward, done, truncated, info

        # catch-all return statement
        print(f"Warning: step function reached an unexpected state. Price: {price}, Amount: {amount}, "
              f"Savings: {self.savings}, Charge: {self.charge}")
        reward = -1
        self.rewards.append(reward)
        return self.get_observation().astype(np.float32), reward, done, truncated, info

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
        """

        :param mode:
        :return:
        """
        # calculate the average reward over 100 steps and plot it
        avg_rewards = []
        scaler = 100
        for i in range(0, len(self.rewards), scaler):
            avg_rewards.append(sum(self.rewards[i:i + scaler]) / scaler)
        plt.plot(avg_rewards)
        plt.ylabel('Average Reward')
        plt.xlabel('Number of Episodes')
        plt.show()

    def get_trades(self):
        return self.trade_log
