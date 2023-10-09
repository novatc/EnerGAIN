import gymnasium as gym
import pandas as pd
from gymnasium import spaces
import numpy as np

from envs.assets.battery import Battery
from envs.assets.dayahead import DayAhead
from envs.assets.plot_engien import plot_reward, plot_savings, plot_charge, plot_trades_timeline, plot_holding, \
    kernel_density_estimation


class TrendEnv(gym.Env):
    def __init__(self, da_data_path, validation=False):
        super(TrendEnv, self).__init__()
        self.dataframe = pd.read_csv(da_data_path)
        self.trend_horizon = 4

        low_boundary = self.dataframe.min().values

        high_boundary = self.dataframe.max().values

        low_boundary = np.tile(low_boundary, self.trend_horizon)  # repeat the array 4 times to match the obs space
        high_boundary = np.tile(high_boundary, self.trend_horizon)

        # add 10 to each value in the high boundary to make sure the agent can't reach the upper boundary
        high_boundary += 10

        action_low = np.array([-1.0, -500.0])
        action_high = np.array([1.0, 500.0])

        self.action_space = spaces.Box(low=action_low, high=action_high, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-low_boundary, high=high_boundary,
                                            shape=((self.dataframe.shape[1]) * self.trend_horizon,))

        self.day_ahead = DayAhead(self.dataframe)
        self.savings = 50  # €
        self.battery = Battery(1000, 500)
        self.battery.add_charge_log(self.battery.get_soc())

        self.savings_log = []
        self.savings_log.append(self.savings)

        self.trade_log = []
        self.invalid_trades = []
        self.holding = []

        self.rewards = []
        self.reward_log = []
        self.window_size = 20
        self.penalty = -5

        self.validation = validation

    def step(self, action):
        if self.validation:
            self.day_ahead.step()
        else:
            self.day_ahead.random_walk()

        price = action[0].item()
        amount = action[1].item()

        terminated = False  # Whether the agent reaches the terminal state
        truncated = False  # this can be Fasle all the time since there is no failure condition the agent could trigger
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
            reward = 0
            self.holding.append((self.day_ahead.get_current_step(), 'hold'))

        self.rewards.append(reward)
        self.reward_log.append(
            (self.reward_log[-1] + reward) if self.reward_log else reward)  # to keep track of the reward over time
        # Return the current state of the environment as a numpy array, the reward,
        return self.get_observation().astype(np.float32), reward, terminated, truncated, info

    def trade(self, price, amount, trade_type):
        """
        Execute a trade (buy/sell) and update the battery status and logs accordingly.

        :param price: (float) The price at which the trade is attempted.
        :param amount: (float) The amount of energy to be traded. Positive values indicate buying or charging,
                       and negative values indicate selling or discharging.
        :param trade_type: (str) Type of trade to execute, accepted values are 'buy' or 'sell'.

        :return: (float) Absolute value of the traded energy amount multiplied by the current market price.
                  Returns a penalty (self.penalty) if the trade is invalid or not accepted.

        :raises ValueError: If `trade_type` is neither 'buy' nor 'sell'.
        """
        if trade_type == 'buy':
            if price * amount > self.savings or self.savings <= 0 or self.battery.can_charge(amount) is False:
                return self.penalty
        elif trade_type == 'sell':
            if self.battery.can_discharge(amount) is False:
                return self.penalty
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
            return self.penalty

        return abs(float(self.day_ahead.get_current_price()) * amount)

    def get_observation(self):
        # Return the current state of the environment as a numpy array
        trend_data = self.day_ahead.previous_hours(self.trend_horizon)
        return trend_data

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
        return self.get_observation().astype(np.float32), {}

    def render(self, mode='human'):
        """
        Render the environment to the screen
        :param mode:
        :return:
        """
        plot_reward(self.reward_log, self.window_size, 'trend')
        plot_savings(self.savings_log, self.window_size, 'trend')
        plot_charge(self.window_size, self.battery, 'trend')
        plot_trades_timeline(trade_source=self.trade_log, title='Trades', buy_color='green', sell_color='red',
                             model_name='trend')
        plot_trades_timeline(trade_source=self.invalid_trades, title='Invalid Trades', buy_color='black',
                             sell_color='brown', model_name='trend')
        plot_holding(self.holding, 'trend')
        kernel_density_estimation(self.trade_log)

    def get_trades(self):
        """
        Returns the trade log
        :return: the trade log as a list of tuples
        """
        return self.trade_log
