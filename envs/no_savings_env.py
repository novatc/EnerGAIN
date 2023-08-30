import gymnasium as gym
import joblib
import pandas as pd
from gymnasium import spaces
import numpy as np
from matplotlib import pyplot as plt

from envs.assets.market import Market


class NoSavingsEnv(gym.Env):
    def __init__(self, data_path):
        super(NoSavingsEnv, self).__init__()
        self.dataframe = pd.read_csv(data_path)
        self.market = Market(self.dataframe)
        self.savings = 0.5
        self.charge = 0.5
        self.max_battery_charge = 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.dataframe.shape[1] + 1,))
        self.charge_log = []

        self.trade_log = []

        self.rewards = []
        self.reward_log = []

    def step(self, action):
        """
        Execute one step in the environment.

        Args:
            action (tuple): A 2-element tuple representing the action to take. The first element is the price and the
            second is the amount.

        Returns:
            A 5-element tuple containing the new observation, the reward, two booleans indicating whether the episode
            is done or truncated, and a dictionary with additional info.
        """
        self.market.random_walk()

        price = action[0].item()
        amount = action[1].item()

        terminated = False  # Whether the agent reaches the terminal state
        truncated = False  # this can be Fasle all the time since there is no failure condition the agent could trigger
        info = {'current_price': self.market.get_current_price(),
                'current_step': self.market.get_current_step(),
                'charge': self.charge,
                'action_price': price,
                'action_amount': amount,
                }

        if amount > 0:  # buy
            reward = self.trade(price, amount, 'buy')
        elif amount < 0:  # sell
            reward = self.trade(price, amount, 'sell')
        else:  # if amount is 0
            reward = 0

        self.rewards.append(reward)
        self.reward_log.append(
            (self.reward_log[-1] + reward) if self.reward_log else reward)  # to keep track of the reward over time
        # Return the current state of the environment as a numpy array, the reward,
        return self.get_observation().astype(np.float32), reward, terminated, truncated, info

    def validation_step(self, action):
        self.market.step()

        price = action[0].item()
        amount = action[1].item()

        terminated = False  # Whether the agent reaches the terminal state
        truncated = False  # this can be Fasle all the time since there is no failure condition the agent could trigger
        info = {'current_price': self.market.get_current_price(),
                'current_step': self.market.get_current_step(),
                'charge': self.charge,
                'action_price': price,
                'action_amount': amount,
                }

        if amount > 0:  # buy
            reward = self.trade(price, amount, 'buy')
        elif amount < 0:  # sell
            reward = self.trade(price, amount, 'sell')
        else:  # if amount is 0
            reward = 0

        self.rewards.append(reward)
        self.reward_log.append(
            (self.reward_log[-1] + reward) if self.reward_log else reward)  # to keep track of the reward over time
        # Return the current state of the environment as a numpy array, the reward,
        return self.get_observation().astype(np.float32), reward, terminated, truncated, info

    def trade(self, price, amount, trade_type):
        if trade_type == 'buy':
            if amount > self.max_battery_charge - self.charge:
                return -1
        elif trade_type == 'sell':
            if amount < -self.charge:
                return -1
        else:
            raise ValueError(f"Invalid trade type: {trade_type}")
        if self.market.accept_offer(price, trade_type):
            # this works for both buy and sell because amount is negative for sale and + and - cancel out and fot buy
            # amount is positive
            self.charge += amount
            # the same applies here for savings

            self.charge_log.append(self.charge)
            self.trade_log.append((self.market.get_current_step(), price, amount, trade_type))
        else:
            return -1

        return abs(float(self.market.get_current_price()) * amount)

    def get_observation(self):
        # Return the current state of the environment as a numpy array
        return np.concatenate(
            (self.dataframe.iloc[self.market.get_current_step()].to_numpy(), [self.charge]))

    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # Reset the state of the environment to an initial state
        super().reset(seed=seed, options=options)
        self.charge = 0.5
        self.market.reset()
        return self.get_observation().astype(np.float32), {}

    def render(self, mode='human'):
        # Calculate the average reward over 100 steps and plot it
        avg_rewards = []
        scaler = 1
        for i in range(0, len(self.rewards), scaler):
            avg_rewards.append(sum(self.rewards[i:i + scaler]) / scaler)
        plt.figure(figsize=(10, 6))
        plt.subplot(3, 1, 1)
        plt.plot(avg_rewards)
        plt.ylabel('Average Reward')
        plt.xlabel(f'Number of Steps (/ {scaler})')

        # Plot the history of trades
        buys = [trade for trade in self.trade_log if trade[3] == 'buy']
        sells = [trade for trade in self.trade_log if trade[3] == 'sell']

        if buys:
            buy_steps, buy_prices, buy_amounts, _ = zip(*buys)
            # Rescale the prices and amounts using the scaler objects
            plt.subplot(3, 1, 2)
            plt.scatter(buy_steps, buy_prices, c='green', label='Buy', alpha=0.6)
            plt.ylabel('Trade Price (€/MWh)')
            plt.subplot(3, 1, 3)
            plt.scatter(buy_steps, buy_amounts, color='green', label='Buy', alpha=0.6)
            plt.ylabel('Trade Amount (MWh)')
        if sells:
            sell_steps, sell_prices, sell_amounts, _ = zip(*sells)
            # Rescale the prices and amounts using the scaler objects
            plt.subplot(3, 1, 2)
            plt.scatter(sell_steps, sell_prices, c='red', label='Sell', alpha=0.6)
            plt.subplot(3, 1, 3)
            plt.scatter(sell_steps, sell_amounts, color='red', label='Sell', alpha=0.6)

        plt.xlabel('Steps')
        plt.legend()
        plt.tight_layout()
        plt.show()
        # self.plot_charge()
        self.plot_reward_log()

    def get_trades(self):
        # list of trades: (step, price, amount, trade_type)

        return self.trade_log

    def get_charge(self):
        return self.charge_log

    def get_real_charge(self):
        amount_scaler = joblib.load('amount_scaler.pkl')
        return amount_scaler.inverse_transform(np.array(self.charge_log).reshape(-1, 1))

    def plot_charge(self):
        # Load the scaler
        amount_scaler = joblib.load('amount_scaler.pkl')

        # Get the original charge values
        charge_original = amount_scaler.inverse_transform(np.array(self.charge_log).reshape(-1, 1))

        plt.figure(figsize=(10, 6))
        plt.plot(charge_original / 10)
        plt.title('Charge Over Time')
        plt.xlabel('Step')
        plt.ylabel('Charge')
        plt.show()

    def plot_reward_log(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.reward_log)
        plt.title('Reward Over Time')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.show()
