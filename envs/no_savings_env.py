import gymnasium as gym
import pandas as pd
from gymnasium import spaces
import numpy as np

from envs.assets.battery import Battery
from envs.assets.dayahead import DayAhead
from envs.assets.plot_engien import plot_reward, plot_charge, plot_trades_timeline, plot_holding, \
    kernel_density_estimation, plot_savings


class NoSavingsEnv(gym.Env):
    def __init__(self, da_data_path: str, validation=False):
        super(NoSavingsEnv, self).__init__()
        self.da_dataframe = pd.read_csv(da_data_path)

        low_boundary = self.da_dataframe.min().values

        high_boundary = self.da_dataframe.max().values

        action_low = np.array([0.0, -1000.0])  # price, amount
        action_high = np.array([1.0, 1000.0])  # price, amount
        self.action_space = spaces.Box(low=action_low, high=action_high, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=low_boundary, high=high_boundary, shape=(self.da_dataframe.shape[1],))

        self.day_ahead = DayAhead(self.da_dataframe)
        self.battery = Battery(1000, 500)
        self.savings = 50  # €
        self.savings_log = []

        self.trade_log = []
        self.invalid_trades = []
        self.holding = []

        self.reward_log = []
        self.window_size = 5
        self.penalty = -30

        self.trade_threshold = 10

        self.validation = validation

    def step(self, action):
        should_truncated = False
        if self.validation:
            self.day_ahead.step()
        else:
            should_truncated = self.day_ahead.random_walk(24 * 7)

        price, amount = action

        reward = 0

        terminated = False  # Whether the agent reaches the terminal state
        truncated = should_truncated

        # Handle DA trade or holding
        if -self.trade_threshold < amount < self.trade_threshold:
            reward += self.handle_holding()

        reward += self.perform_da_trade(energy_amount=amount, market_price=price)

        self.reward_log.append((self.reward_log[-1] + reward) if self.reward_log else reward)

        info = {'current_price': self.day_ahead.get_current_price(),
                'current_step': self.day_ahead.get_current_step(),
                'savings': self.savings,
                'charge': self.battery.get_soc(),
                'action_price': price,
                'action_amount': amount,
                }
        return self.get_observation().astype(np.float32), reward, terminated, truncated, info

    def is_trade_valid(self, price, amount, trade_type):
        """
        Check if a trade is valid, i.e. if the battery can handle the trade and if the agent has enough savings.

        :param price: (float) The price at which the trade is attempted.
        :param amount: (float) The amount of energy to be traded. Positive values indicate buying or charging,
                       and negative values indicate selling or discharging.
        :param trade_type: (str) Type of trade to execute, accepted values are 'buy' or 'sell'.

        :return: (bool) True if the trade is valid, False otherwise.
        """
        if trade_type == 'buy':
            if self.battery.can_charge(amount) is False:
                self.log_trades(False, 'buy', price, amount, self.penalty,
                                'savings' if self.savings <= 0 else 'battery')
                return False
        elif trade_type == 'sell':
            if self.battery.can_discharge(amount) is False:
                self.log_trades(False, 'sell', price, amount, self.penalty, 'battery')
                return False
        else:
            raise ValueError(f"Invalid trade type: {trade_type}")

        return True

    def perform_da_trade(self, energy_amount: float, market_price: float) -> float:
        """
        Perform a trade on the day-ahead market.

        :param energy_amount: Energy to be traded (positive for buying, negative for selling).
        :param market_price: Price at which the trade is attempted.
        :return: Reward based on the trade outcome.
        """
        return self.trade(market_price, energy_amount, 'buy' if energy_amount > 0 else 'sell')

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
        if trade_type not in ['buy', 'sell']:
            raise ValueError(f"Invalid trade type: {trade_type}")

        # Check if the trade is valid
        if not self.is_trade_valid(price, amount, trade_type):
            return self.penalty

        return self.execute_trade(price, amount, trade_type)

    def execute_trade(self, price, amount, trade_type):
        """
        Execute the validated trade and update the system status.

        :param price: Price at which the trade is executed.
        :param amount: Amount of energy involved in the trade.
        :param trade_type: 'buy' or 'sell'.
        :return: Profit (or loss) from the trade.
        """
        current_price = self.day_ahead.get_current_price()
        profit = 0
        if self.day_ahead.accept_offer(price, trade_type):
            if trade_type == 'buy':
                self.battery.charge(amount)  # Charge battery for buy trades
                self.savings -= current_price * amount  # Update savings
                profit = -current_price * amount  # Negative profit for buying

            elif trade_type == 'sell':
                self.battery.charge(amount)  # Discharge battery for sell trades
                self.savings += current_price * abs(amount)  # Update savings
                profit = current_price * abs(amount)  # Positive profit for selling
        else:
            self.log_trades(False, trade_type, price, amount, self.penalty, 'market rejected')
            return self.penalty

        # Logging the trade details
        self.battery.add_charge_log(self.battery.get_soc())
        self.savings_log.append(self.savings)
        self.log_trades(True, trade_type, price, amount, profit, 'accepted')

        return profit

    def handle_holding(self):
        # Logic for handling the holding scenario
        self.holding.append((self.day_ahead.get_current_step(), 'hold'))
        return 5

    def get_observation(self):
        """
        Returns the current state of the environment
        :return: the current state of the environment as a numpy array
        """
        # Return the current state of the environment as a numpy array
        observation = self.da_dataframe.iloc[self.day_ahead.get_current_step()].to_numpy()
        return observation

    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # Reset the state of the environment to an initial state
        super().reset(seed=seed, options=options)
        self.battery.reset()
        self.day_ahead.reset()
        observation = self.get_observation().astype(np.float32)
        return observation, {}

    def render(self, mode='human'):
        """
        Render the environment to the screen
        :param mode:
        :return:
        """
        plot_reward(self.reward_log, self.window_size, 'no_savings')
        plot_savings(self.savings_log, self.window_size, 'no_savings')
        plot_charge(self.window_size, self.battery, 'no_savings')
        plot_trades_timeline(trade_source=self.trade_log, title='Trades', buy_color='green', sell_color='red',
                             model_name='no_savings', data=self.da_dataframe)
        plot_trades_timeline(trade_source=self.invalid_trades, title='Invalid Trades', buy_color='black',
                             sell_color='brown', model_name='no_savings', data=self.da_dataframe)
        plot_holding(self.holding, 'no_savings', da_data=self.da_dataframe)
        kernel_density_estimation(self.trade_log, model_name='no_savings', da_data=self.da_dataframe)

    def get_trades(self):
        """
        Returns the trade log
        :return: the trade log as a list of tuples
        """
        return self.trade_log

    def get_invalid_trades(self):
        return self.invalid_trades

    def get_holdings(self):
        return self.holding

    def log_trades(self, valid: bool, type: str, offered_price: float, amount: float, reward: float,
                   case: str) -> None:
        """
        Log the trades
        :param type: the trade type
        :param valid: if the trade was accepted or not
        :param offered_price: the offered price
        :param amount: the trade amount
        :param reward: the reward based on the offer
        :param case: if not valid, why
        :return: None
        """
        if valid:
            self.trade_log.append(
                (self.day_ahead.get_current_step(), type, self.day_ahead.get_current_price(), offered_price, amount,
                 reward, case))
        else:
            self.invalid_trades.append(
                (self.day_ahead.get_current_step(), type, self.day_ahead.get_current_price(), offered_price, amount,
                 reward, case))
