import gymnasium as gym
import pandas as pd
from gymnasium import spaces
import numpy as np
from matplotlib import pyplot as plt

from envs.assets.env_utilities import moving_average
from envs.assets.market import Market


class NoSavingsEnv(gym.Env):
    def __init__(self, data_path: str, validation=False):
        super(NoSavingsEnv, self).__init__()
        self.dataframe = pd.read_csv(data_path)

        low_boundary = self.dataframe.min().values
        low_boundary = np.append(low_boundary, [0.0])  # set the lower boundary of charge to 0
        high_boundary = self.dataframe.max().values
        high_boundary = np.append(high_boundary,
                                  [1000.0])  # set the upper boundary of charge to 1000
        action_low = np.array([-1.0, -10000.0])
        action_high = np.array([1.0, 1000.0])
        self.action_space = spaces.Box(low=action_low, high=action_high, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=low_boundary, high=high_boundary, shape=(self.dataframe.shape[1] + 1,))

        self.market = Market(self.dataframe)
        self.charge = 500  # kWh
        self.max_battery_charge = 1000  # kWh

        self.charge_log = []

        self.trade_log = []

        self.rewards = []
        self.reward_log = []
        self.window_size = 20

        self.validation = validation

    def step(self, action):
        if self.validation:
            self.market.step()
        else:
            self.market.random_walk()

        price = action[0].item()
        amount = action[1].item()

        terminated = False  # Whether the agent reaches the terminal state
        truncated = False  # this can be false all the time since there is no failure condition the agent could trigger
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
        # print(self.get_observation().astype(np.float32).dtype)
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
            self.trade_log.append((self.market.get_current_step(), price, amount, trade_type,
                                   abs(float(self.market.get_current_price()) * amount)))
        else:
            return -1

        return abs(float(self.market.get_current_price()) * amount)

    def get_observation(self):
        # Return the current state of the environment as a numpy array
        observation = np.concatenate(
            (self.dataframe.iloc[self.market.get_current_step()].to_numpy(), [self.charge])
        )
        return observation

    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # Reset the state of the environment to an initial state
        super().reset(seed=seed, options=options)
        self.charge = 500
        self.market.reset()
        observation = self.get_observation().astype(np.float32)
        return observation, {}

    def render(self, mode='human'):
        # Calculate the average reward over 100 steps and plot it
        avg_rewards = []
        scaler = 10
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
            buy_steps, buy_prices, buy_amounts, _, _ = zip(*buys)
            buy_prices = abs(np.array(buy_prices))
            # Rescale the prices and amounts using the scaler objects
            plt.subplot(3, 1, 2)
            plt.scatter(buy_steps, buy_prices, c='green', label='Buy', alpha=0.6)
            plt.ylabel('Trade Price (€/MWh)')
            plt.subplot(3, 1, 3)
            plt.scatter(buy_steps, buy_amounts, color='green', label='Buy', alpha=0.6)
            plt.ylabel('Trade Amount (MWh)')
        if sells:
            sell_steps, sell_prices, sell_amounts, _, _ = zip(*sells)
            sell_prices = abs(np.array(sell_prices))
            # Rescale the prices and amounts using the scaler objects
            plt.subplot(3, 1, 2)
            plt.scatter(sell_steps, sell_prices, c='red', label='Sell', alpha=0.6)
            plt.subplot(3, 1, 3)
            plt.scatter(sell_steps, sell_amounts, color='red', label='Sell', alpha=0.6)

        plt.xlabel('Steps')
        plt.legend()
        plt.tight_layout()
        plt.show()
        self.plot_charge()
        self.plot_reward_log()
        self.plot_price_comparison()

    def get_trades(self):
        # list of trades: (step, price, amount, trade_type)

        return self.trade_log

    def get_charge(self):
        return self.charge_log

    def plot_charge(self):
        plt.figure(figsize=(10, 6))

        # Original data
        plt.plot(self.charge_log, label='Original', alpha=0.5)

        # Smoothed data
        smoothed_data = moving_average(self.charge_log, self.window_size)
        smoothed_steps = np.arange(self.window_size - 1,
                                   len(self.charge_log))  # Adjust the x-axis for the smoothed data

        plt.plot(smoothed_steps, smoothed_data, label=f'Smoothed (window size = {self.window_size})')

        plt.title('Charge Over Time')
        plt.xlabel('Step')
        plt.ylabel('Charge')
        plt.legend()
        plt.show()

    def plot_reward_log(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.reward_log, label='Original', alpha=0.5)
        smoothed_data = moving_average(self.reward_log, self.window_size)
        smoothed_steps = np.arange(self.window_size - 1, len(self.reward_log))
        plt.plot(smoothed_steps, smoothed_data, label=f'Smoothed (window size = {self.window_size})')
        plt.title('Reward Over Time')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.legend()
        plt.show()

    def plot_price_comparison(self):
        plt.figure(figsize=(10, 6))

        # Get the buy and sell trades from the trade log
        buys = [trade for trade in self.trade_log if trade[3] == 'buy']
        sells = [trade for trade in self.trade_log if trade[3] == 'sell']

        # Check if there are any buy or sell trades to plot
        if not buys and not sells:
            print("No trades to plot.")
            return

        # Collect and sort market prices
        market_prices_list = []

        # Plot buy data if available
        if buys:
            buy_steps, buy_prices, buy_amounts, _, _ = zip(*buys)
            buy_prices = abs(np.array(buy_prices))
            market_prices_buy = [self.market.get_price_at_step(step) for step in buy_steps]
            plt.scatter(buy_steps, buy_prices, c='green', marker='o', label='Buy', alpha=0.6)

            # Add to market prices list
            market_prices_list.extend(zip(buy_steps, market_prices_buy))

        # Plot sell data if available
        if sells:
            sell_steps, sell_prices, sell_amounts, _, _ = zip(*sells)
            market_prices_sell = [self.market.get_price_at_step(step) for step in sell_steps]
            plt.scatter(sell_steps, sell_prices, c='red', marker='x', label='Sell', alpha=0.6)

            # Add to market prices list
            market_prices_list.extend(zip(sell_steps, market_prices_sell))

        # Sort market prices list by steps
        market_prices_list.sort(key=lambda x: x[0])
        sorted_steps, sorted_market_prices = zip(*market_prices_list)

        plt.plot(sorted_steps, sorted_market_prices, color='blue', label='Market Price', alpha=0.6)

        plt.ylabel('Trade Price (€/kWh)')
        plt.xlabel('Steps')
        plt.legend()
        plt.show()
