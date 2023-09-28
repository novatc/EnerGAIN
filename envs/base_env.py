import gymnasium as gym
import pandas as pd
from gymnasium import spaces
import numpy as np
from matplotlib import pyplot as plt

from envs.assets.battery import Battery
from envs.assets.env_utilities import moving_average
from envs.assets.dayahead import DayAhead


class BaseEnv(gym.Env):
    def __init__(self, data_path: str, validation=False):
        super(BaseEnv, self).__init__()
        self.dataframe = pd.read_csv(data_path)

        low_boundary = self.dataframe.min().values

        high_boundary = self.dataframe.max().values

        action_low = np.array([-1.0, -500.0])
        action_high = np.array([1.0, 500.0])
        self.action_space = spaces.Box(low=action_low, high=action_high, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=low_boundary, high=high_boundary, shape=(self.dataframe.shape[1],))

        self.day_ahead = DayAhead(self.dataframe)
        self.battery = Battery(1000, 500)
        self.savings = 50  # €
        self.max_battery_charge = 1000  # kWh

        self.savings_log = []

        self.trade_log = []
        self.invalid_trades = []
        self.holding = []

        self.rewards = []
        self.reward_log = []
        self.window_size = 20

        self.validation = validation

    def step(self, action):
        if self.validation:
            self.day_ahead.step()
        else:
            self.day_ahead.random_walk()

        price = action[0].item()
        amount = action[1].item()

        terminated = False  # Whether the agent reaches the terminal state
        truncated = False  # this can be false all the time since there is no failure condition the agent could trigger
        info = {'current_price': self.day_ahead.get_current_price(),
                'current_step': self.day_ahead.get_current_step(),
                'savings': self.savings,
                'charge': self.battery.get_soc(),
                'action_price': price,
                'action_amount': amount,
                }

        if amount > 0:  # buy
            reward = self.trade(price, amount, 'buy')
        elif amount < 0:  # sell
            reward = self.trade(price, amount, 'sell')
        else:  # if amount is 0
            reward = 5
            self.holding.append((self.day_ahead.get_current_step(), 'hold'))

        self.rewards.append(reward)
        self.reward_log.append(
            (self.reward_log[-1] + reward) if self.reward_log else reward)  # to keep track of the reward over time
        return self.get_observation().astype(np.float32), reward, terminated, truncated, info

    def trade(self, price, amount, trade_type):
        if trade_type == 'buy':
            if price * amount > self.savings or self.savings <= 0 or self.battery.can_charge(amount) is False:
                return -10
        elif trade_type == 'sell':
            if self.battery.can_discharge(amount) is False:
                return -10
        else:
            raise ValueError(f"Invalid trade type: {trade_type}")
        if self.day_ahead.accept_offer(price, trade_type):
            # this works for both buy and sell because amount is negative for sale and + and - cancel out and fot buy
            # amount is positive
            self.battery.charge(amount)
            # the same applies here for savings
            self.savings -= self.day_ahead.get_current_price() * amount

            self.battery.add_charge_log(self.battery.get_soc())
            self.savings_log.append(self.savings)
            self.trade_log.append((self.day_ahead.get_current_step(), price, amount, trade_type,
                                   abs(float(self.day_ahead.get_current_price()) * amount)))
        else:
            self.invalid_trades.append((self.day_ahead.get_current_step(), price, amount, trade_type,
                                        abs(float(self.day_ahead.get_current_price()) * amount)))
            return -10

        return abs(float(self.day_ahead.get_current_price()) * amount)

    def get_observation(self):
        # Return the current state of the environment as a numpy array
        observation = self.dataframe.iloc[self.day_ahead.get_current_step()].to_numpy()
        return observation

    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # Reset the state of the environment to an initial state
        super().reset(seed=seed, options=options)
        self.savings = 50
        self.battery.reset()
        self.day_ahead.reset()
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
        self.plot_savings()
        self.plot_charge()
        self.plot_reward_log()
        self.plot_trades_timeline("trade_log", "Trades vs. Real Market Price", "green", "red",
                                  'img/base_price_comparison_full_timeline.png')

        # For invalid trades:
        self.plot_trades_timeline("invalid_trades", "Invalid Trades", "black", "orange",
                                  'img/base_invalid_trades_full_timeline.png')
        self.plot_holding()

    def get_trades(self):
        # list of trades: (step, price, amount, trade_type)

        return self.trade_log

    def plot_charge(self):
        plt.figure(figsize=(10, 6))
        charge_log = self.battery.get_charge_log()

        # Original data
        plt.plot(charge_log, label='Original', alpha=0.5)

        # Smoothed data
        smoothed_data = moving_average(charge_log, self.window_size)
        smoothed_steps = np.arange(self.window_size - 1,
                                   len(charge_log))  # Adjust the x-axis for the smoothed data

        plt.plot(smoothed_steps, smoothed_data, label=f'Smoothed (window size = {self.window_size})')

        plt.title('Charge Over Time')
        plt.xlabel('Number of trades')
        plt.ylabel('Charge (kWh)')
        plt.legend()
        plt.savefig('img/base_charge.png', dpi=400)
        plt.show()

    def plot_savings(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.savings_log, label='Original', alpha=0.5)
        smoothed_data = moving_average(self.savings_log, self.window_size)
        smoothed_steps = np.arange(self.window_size - 1, len(self.savings_log))
        plt.plot(smoothed_steps, smoothed_data, label=f'Smoothed (window size = {self.window_size})')
        plt.title('Savings Over Time')
        plt.xlabel('Number of trades')
        plt.ylabel('Savings (€)')
        plt.legend()
        plt.savefig('img/base_savings.png', dpi=400)

        plt.show()

    def plot_reward_log(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.reward_log, label='Original', alpha=0.5)
        smoothed_data = moving_average(self.reward_log, self.window_size)
        smoothed_steps = np.arange(self.window_size - 1, len(self.reward_log))
        plt.plot(smoothed_steps, smoothed_data, label=f'Smoothed (window size = {self.window_size})')
        plt.title('Reward Over Time')
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        plt.legend()
        plt.savefig('img/base_reward.png', dpi=400)

        plt.show()

    def plot_trades_timeline(self, trade_source, title, buy_color, sell_color, save_path):
        plt.figure(figsize=(10, 6))
        trade_log = getattr(self, trade_source)
        eval_data_df = pd.read_csv('data/in-use/unscaled_eval_data.csv')
        total_trades = len(trade_log)

        # Get the buy and sell trades from the trade log
        buys = [trade for trade in trade_log if trade[3] == 'buy']
        sells = [trade for trade in trade_log if trade[3] == 'sell']

        # Check if there are any buy or sell trades to plot
        if not buys and not sells:
            print("No trades to plot.")
            return

        # Plot real market prices from evaluation dataset
        plt.plot(eval_data_df.index, eval_data_df['price'], color='blue', label='Real Market Price', alpha=0.6)

        # Plot buy data if available
        if buys:
            buy_steps, buy_prices, _, _, _ = zip(*buys)
            plt.scatter(buy_steps, buy_prices, c=buy_color, marker='o', label='Buy', alpha=0.6, s=10)

        # Plot sell data if available
        if sells:
            sell_steps, sell_prices, _, _, _ = zip(*sells)
            plt.scatter(sell_steps, sell_prices, c=sell_color, marker='x', label='Sell', alpha=0.6, s=10)

        plt.title(title + f' ({total_trades} trades)')
        plt.ylabel('Price (€/kWh)')
        plt.xlabel('Step')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=400)
        plt.show()

    def plot_holding(self):
        if not self.holding:
            print("No trades to plot.")
            return
        plt.figure(figsize=(10, 6))
        eval_data_df = pd.read_csv('data/in-use/unscaled_eval_data.csv')
        plt.plot(eval_data_df.index, eval_data_df['price'], color='blue', label='Real Market Price', alpha=0.6)
        steps, _, _, _, _ = zip(*self.holding)
        plt.scatter(steps, [eval_data_df['price'][step] for step in steps], c='black', marker='o', label='Hold',
                    alpha=0.6, s=10)
        plt.title('Hold')
        plt.ylabel('Price (€/kWh)')
        plt.xlabel('Step')
        plt.legend()
        plt.tight_layout()
        plt.savefig('img/base_hold.png', dpi=400)
        plt.show()
