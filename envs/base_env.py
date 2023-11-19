import gymnasium as gym
from gymnasium import spaces

from envs.assets.battery import Battery
from envs.assets.dayahead import DayAhead
from envs.assets.plot_engien import *


class BaseEnv(gym.Env):
    def __init__(self, da_data_path: str, validation=False):
        super(BaseEnv, self).__init__()
        self.da_dataframe = pd.read_csv(da_data_path)

        low_boundary = self.da_dataframe.min().values

        high_boundary = self.da_dataframe.max().values

        action_low = np.array([-1.0, -500.0])
        action_high = np.array([1.0, 500.0])
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
        self.window_size = 20
        self.penalty = -5

        self.trade_threshold = 50


        self.validation = validation

    def step(self, action):
        if self.validation:
            self.day_ahead.step()
        else:
            self.day_ahead.random_walk()

        price, amount = action

        reward = 0

        terminated = False  # Whether the agent reaches the terminal state
        truncated = False  # this can be false all the time since there is no failure condition the agent could trigger

        # Handle DA trade or holding
        if -self.trade_threshold < amount < self.trade_threshold:
            reward += self.handle_holding()

        reward += self.perform_da_trade(amount, price)

        self.reward_log.append((self.reward_log[-1] + reward) if self.reward_log else reward)
        info = {'current_price': self.day_ahead.get_current_price(),
                'current_step': self.day_ahead.get_current_step(),
                'savings': self.savings,
                'charge': self.battery.get_soc(),
                'action_price': price,
                'action_amount': amount,
                }
        return self.get_observation().astype(np.float32), reward, terminated, truncated, info

    def perform_da_trade(self, amount_da: float, price_da: float) -> float:
        """
        Perform a trade on the day ahead market.
        :param amount_da: the amount of energy to be traded
        :param price_da: the price at which the trade is attempted
        :return: reward for the agent based on the trade outcome
        """
        reward = 0
        if amount_da > 0:  # buy
            reward = self.trade(price_da, amount_da, 'buy')

        if amount_da < 0:  # sell
            reward = self.trade(price_da, amount_da, 'sell')

        return reward

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
                case = 'savings' if self.savings <= 0 else 'battery'
                self.log_trades(False, 'buy', price, amount, self.penalty, case)
                return self.penalty
        elif trade_type == 'sell':
            if self.battery.can_discharge(amount) is False:
                self.log_trades(False, 'sell', price, amount, self.penalty, 'battery')
                return self.penalty
        else:
            raise ValueError(f"Invalid trade type: {trade_type}")

        if self.day_ahead.accept_offer(price, trade_type):
            # this works for both buy and sell because amount is negative for sale and + and - cancel out and for buy
            # amount is positive
            self.battery.charge(amount)
            # the same applies here for savings
            self.savings -= self.day_ahead.get_current_price() * amount
            self.battery.add_charge_log(self.battery.get_soc())
            self.savings_log.append(self.savings)
            self.log_trades(True, trade_type, price, amount, self.day_ahead.get_current_price() * amount, 'accepted')
        else:
            self.log_trades(False, trade_type, price, amount, self.penalty, 'market rejected')
            return self.penalty

        return float(self.day_ahead.get_current_price()) * amount

    def handle_holding(self):
        # Logic for handling the holding scenario
        self.holding.append((self.day_ahead.get_current_step(), 'hold'))
        return 1

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
        self.savings = 50
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
        plot_reward(self.reward_log, self.window_size, 'base')
        plot_savings(self.savings_log, self.window_size, 'base')
        plot_charge(self.window_size, self.battery, 'base')
        plot_trades_timeline(trade_source=self.trade_log, title='Trades', buy_color='green', sell_color='red',
                             model_name='base', data=self.da_dataframe)
        plot_trades_timeline(trade_source=self.invalid_trades, title='Invalid Trades', buy_color='black',
                             sell_color='brown', model_name='base', data=self.da_dataframe)
        plot_holding(self.holding, 'base', da_data=self.da_dataframe)
        kernel_density_estimation(self.trade_log, model_name='base', da_data=self.da_dataframe)

    def get_trades(self):
        """
        Returns the trade log
        :return: the trade log as a list of tuples
        """
        return self.trade_log

    def get_invalid_trades(self):
        return self.invalid_trades

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
