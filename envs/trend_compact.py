import gymnasium as gym
from gymnasium import spaces

from envs.assets.battery import Battery
from envs.assets.dayahead import DayAhead
from envs.assets.features import (DEFAULT_LAGS, clip_to_battery, clip_to_budget,
                                  price_feature_count, price_features)
from envs.assets.plot_engien import *


class TrendCompact(gym.Env):
    """
    BaseState plus a compact set of price-history features.

    TrendEnv tiles the whole 9-column observation row over 8 hours, so 48 of its 72 dimensions
    are cyclical time columns repeated almost unchanged, and the window is too short to see the
    daily (lag 24 h, r = 0.91) or weekly (lag 168 h, r = 0.84) structure. This variant keeps the
    current row and adds targeted lags, a rolling mean and standard deviation, the ratio of the
    current price to that mean, and an exponentially weighted average - ten extra dimensions
    instead of sixty-three.
    """

    def __init__(self, da_data_path: str, validation=False, lags=DEFAULT_LAGS):
        super(TrendCompact, self).__init__()
        self.lags = tuple(lags)
        self.da_dataframe = pd.read_csv(da_data_path)

        low_boundary = self.da_dataframe.min().values
        high_boundary = self.da_dataframe.max().values

        action_low = np.array([0.0, -1000.0])  # price, amount
        action_high = np.array([1.0, 1000.0])  # price, amount
        self.action_space = spaces.Box(low=action_low, high=action_high, shape=(2,), dtype=np.float32)

        self.battery = Battery(1000, 500)
        self.savings = 50  # €
        self.initial_savings = self.savings

        # price-history block, then +2 for the agent's own state: soc and savings
        n_feat = price_feature_count(self.lags)
        obs_low = np.concatenate((low_boundary, np.full(n_feat, -np.inf), [0.0, -np.inf]))
        obs_high = np.concatenate((high_boundary, np.full(n_feat, np.inf),
                                   [self.battery.capacity, np.inf]))
        self.observation_space = spaces.Box(low=obs_low, high=obs_high,
                                            shape=(self.da_dataframe.shape[1] + n_feat + 2,))

        self.day_ahead = DayAhead(self.da_dataframe)
        self.savings_log = []

        self.trade_log = []
        self.invalid_trades = []
        self.holding = []

        self.reward_log = []
        self.penalty = -10  # penalty for invalid trades and breaking the rules

        self.trade_threshold = 10  # kWh, if the trade is within this threshold, it is considered as holding

        self.validation = validation

    def step(self, action):
        should_truncated = False
        if self.validation:
            self.day_ahead.step()
        else:
            should_truncated = self.day_ahead.random_walk(24 * 30)
            if should_truncated:
                self.reset()

        price, amount = action

        reward = 0

        terminated = False  # Whether the agent reaches the terminal state
        truncated = should_truncated

        # Clip first, then decide: an amount the battery cannot absorb becomes a smaller trade or
        # a hold instead of a guaranteed penalty.
        amount = clip_to_battery(amount, self.battery, 'buy' if amount > 0 else 'sell')
        amount = clip_to_budget(amount, price, self.savings)

        if -self.trade_threshold < amount < self.trade_threshold:
            reward += self.handle_holding()
        else:
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
            if price * amount > self.savings or self.savings <= 0 or self.battery.can_charge(amount) is False:
                # Report the actual cause. BaseEnv labels an unaffordable buy 'battery'
                # whenever savings is still positive, which conflates the two.
                affordable = price * amount <= self.savings and self.savings > 0
                self.log_trades(False, 'buy', price, amount, self.penalty,
                                'battery' if affordable else 'savings')
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
        :param amount: (float) The amount of energy to be traded.
        :param trade_type: (str) Type of trade to execute, accepted values are 'buy' or 'sell'.

        :return: (float) The profit from the trade, or the penalty if it was invalid or rejected.

        :raises ValueError: If `trade_type` is neither 'buy' nor 'sell'.
        """
        if trade_type not in ['buy', 'sell']:
            raise ValueError(f"Invalid trade type: {trade_type}")

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
        if trade_type == 'buy' and price < current_price or trade_type == 'sell' and price > current_price:
            profit = self.penalty

        if self.day_ahead.accept_offer(price, trade_type):
            if trade_type == 'buy':
                self.battery.charge(amount)
                self.savings -= current_price * amount
                profit = -current_price * amount
            elif trade_type == 'sell':
                self.battery.charge(amount)
                self.savings += current_price * abs(amount)
                profit = current_price * abs(amount)
        else:
            self.log_trades(False, trade_type, price, amount, self.penalty, 'market rejected')
            return self.penalty

        self.battery.add_charge_log(self.battery.get_soc())
        self.savings_log.append(self.savings)
        self.log_trades(True, trade_type, price, amount, profit, 'accepted')

        return profit

    def handle_holding(self):
        # Logic for handling the holding scenario
        self.holding.append((self.day_ahead.get_current_step(), 'hold'))
        return 1

    def get_observation(self):
        """
        Returns the current state: market row, price-history features, and the agent's own state.

        :return: the current state of the environment as a numpy array
        """
        step = self.day_ahead.get_current_step()
        observation = self.da_dataframe.iloc[step].to_numpy(dtype=float)
        history = price_features(self.da_dataframe, step, lags=self.lags)
        return np.concatenate((observation, history, [self.battery.get_soc(), self.savings]))

    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)
        self.savings = self.initial_savings
        self.battery.reset()
        observation = self.get_observation().astype(np.float32)
        return observation, {}

    def render(self, mode='human'):
        """
        Render the environment to the screen
        :param mode:
        :return:
        """
        plot_savings(self.trade_log, 'trend_compact')
        plot_savings_on_trade_steps(trade_log=self.trade_log, total_steps=self.da_dataframe.shape[0],
                                    model_name='trend_compact')
        plot_charge(self.battery, 'trend_compact')
        plot_trades_timeline(trade_source=self.trade_log, title='Trades', buy_color='green', sell_color='red',
                             model_name='trend_compact', data=self.da_dataframe, plot_name='trades')
        plot_trades_timeline(trade_source=self.invalid_trades, title='Invalid Trades', buy_color='black',
                             sell_color='brown', model_name='trend_compact', data=self.da_dataframe,
                             plot_name='invalid_trades')
        plot_holding(self.holding, 'trend_compact', da_data=self.da_dataframe)
        kernel_density_estimation(self.trade_log, 'trend_compact', da_data=self.da_dataframe)

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
                 reward, case, self.battery.get_soc(), self.savings))
        else:
            self.invalid_trades.append(
                (self.day_ahead.get_current_step(), type, self.day_ahead.get_current_price(), offered_price, amount,
                 reward, case, self.battery.get_soc(), self.savings))
