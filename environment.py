import gymnasium as gym
import joblib
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
        self.savings = None
        self.charge = None
        self.max_battery_charge = 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.dataframe.shape[1] + 2,))
        self.charge_log = []
        self.savings_log = []
        self.trade_log = []

        self.rewards = []

    def step(self, action):
        self.market.week_walk()

        price = action[0].item()
        amount = action[1].item()

        terminated = False  # Whether the agent reaches the terminal state
        truncated = False  # this can be Fasle all the time since there is no failure condition the agent could trigger
        info = {}

        reward = -1  # default reward

        if amount > 0:  # buy
            reward = self.trade(price, amount, 'buy')
        elif amount < 0:  # sell
            reward = self.trade(price, amount, 'sell')
        else:  # if amount is 0
            reward = 0

        self.rewards.append(reward)
        # Return the current state of the environment as a numpy array, the reward,
        return self.get_observation().astype(np.float32), reward, terminated, truncated, info

    def trade(self, price, amount, trade_type):
        if trade_type == 'buy':
            if price > self.savings or amount > self.max_battery_charge - self.charge or amount <= 0:
                return -1
            if self.market.accept_offer(price, 'buy'):
                self.charge += abs(amount)
                self.savings -= self.market.get_current_price() * amount
        elif trade_type == 'sell':
            if amount < -self.charge or price <= 0:
                return -1
            if self.market.accept_offer(price, 'sell'):
                self.savings += self.market.get_current_price() * amount
                self.charge -= abs(amount)
        else:
            raise ValueError(f"Invalid trade type: {trade_type}")

        self.charge_log.append(self.charge)
        self.savings_log.append(self.savings)
        self.trade_log.append((self.market.get_current_step(), price, amount, trade_type))

        return abs(float(self.market.get_current_price()) * amount)

    def get_observation(self):
        # Return the current state of the environment as a numpy array
        return np.concatenate(
            (self.dataframe.iloc[self.market.get_current_step()].to_numpy(), [self.savings, self.charge]))

    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # Reset the state of the environment to an initial state
        super().reset(seed=seed, options=options)
        self.savings = 0
        self.charge = 0
        self.market.reset()
        return self.get_observation().astype(np.float32), {}

    def render(self, mode='human'):
        price_scaler = joblib.load('price_scaler.pkl')
        amount_scaler = joblib.load('amount_scaler.pkl')

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
            buy_prices = price_scaler.inverse_transform(np.array(buy_prices).reshape(-1, 1))
            buy_amounts = amount_scaler.inverse_transform(np.array(buy_amounts).reshape(-1, 1))
            plt.subplot(3, 1, 2)
            plt.scatter(buy_steps, buy_prices / 10, c='green', label='Buy', alpha=0.6)
            plt.ylabel('Trade Price (€/MWh)')
            plt.subplot(3, 1, 3)
            plt.scatter(buy_steps, buy_amounts / 100, color='green', label='Buy', alpha=0.6)
            plt.ylabel('Trade Amount (MWh)')
        if sells:
            sell_steps, sell_prices, sell_amounts, _ = zip(*sells)
            # Rescale the prices and amounts using the scaler objects
            sell_prices = price_scaler.inverse_transform(np.array(sell_prices).reshape(-1, 1))
            sell_amounts = amount_scaler.inverse_transform(np.array(sell_amounts).reshape(-1, 1))
            plt.subplot(3, 1, 2)
            plt.scatter(sell_steps, sell_prices / 10, c='red', label='Sell', alpha=0.6)
            plt.subplot(3, 1, 3)
            plt.scatter(sell_steps, sell_amounts / 100, color='red', label='Sell', alpha=0.6)

        plt.xlabel('Steps')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def get_trades(self):
        return self.trade_log

    def get_savings(self):
        return self.savings_log

    def get_charge(self):
        return self.charge_log

    def plot_charge(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.charge_log)
        plt.title('Battery Charge Over Time')
        plt.xlabel('Step')
        plt.ylabel('Charge')
        plt.show()

    def plot_savings(self):
        # Load the scaler
        price_scaler = joblib.load('price_scaler.pkl')

        # Get the original savings values
        savings_original = price_scaler.inverse_transform(np.array(self.savings_log).reshape(-1, 1))

        plt.figure(figsize=(10, 6))
        plt.plot(savings_original / 10)
        plt.title('Savings Over Time')
        plt.xlabel('Step')
        plt.ylabel('Savings')
        plt.show()
