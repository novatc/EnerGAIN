import gymnasium as gym
from gymnasium import spaces

from envs.assets.battery import Battery
from envs.assets.dayahead import DayAhead
from envs.assets.frequency_containment_reserve import FrequencyContainmentReserve
from envs.assets.plot_engien import *


class MultiTrend(gym.Env):
    def __init__(self, da_data_path: str, prl_data_path: str, validation=True):
        super(MultiTrend, self).__init__()
        self.da_dataframe = pd.read_csv(da_data_path)
        self.prl_dataframe = pd.read_csv(prl_data_path)

        self.trend_horizon = 8  # number of hours to look back for the trend

        da_low_boundary = self.da_dataframe.min().values
        da_high_boundary = self.da_dataframe.max().values

        da_low_boundary = np.tile(da_low_boundary,
                                  self.trend_horizon)  # repeat the array 4 times to match the obs space
        da_high_boundary = np.tile(da_high_boundary, self.trend_horizon)

        prl_low_boundary = self.prl_dataframe.min().values
        prl_high_boundary = self.prl_dataframe.max().values

        prl_low_boundary = np.tile(prl_low_boundary,
                                   self.trend_horizon)  # repeat the array 4 times to match the obs space
        prl_high_boundary = np.tile(prl_high_boundary, self.trend_horizon)

        # add 10 to each value in the high boundary to make sure the agent can't reach the upper boundary
        # without it, it somehow results in an error
        da_high_boundary += 10
        prl_high_boundary += 10

        min_array = np.concatenate((da_low_boundary, prl_low_boundary))
        max_array = np.concatenate((da_high_boundary, prl_high_boundary))

        # adding one more entry for the prl cooldown. The cooldown is the number of steps the agent has to wait until
        # it can participate in the PRL market again, representing the 4-hour block per trade
        observation_high = np.append(max_array, [4])
        observation_low = np.append(min_array, [0])

        obs_shape = (((self.da_dataframe.shape[1] + self.prl_dataframe.shape[1]) * self.trend_horizon + 1),)

        action_low = np.array([0.001, 0, 0, -1000.0])  # prl choice, prl price, prl amount, da price, da amount
        action_high = np.array([0.5, 1000, 1, 1000.0])  # prl choice, prl price, prl amount, da price, da amount

        self.action_space = spaces.Box(low=action_low, high=action_high, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=observation_low, high=observation_high,
                                            shape=obs_shape)

        self.day_ahead = DayAhead(self.da_dataframe)
        self.prl = FrequencyContainmentReserve(self.prl_dataframe)
        self.battery = Battery(1000, 500)
        self.savings = 50  # €
        self.savings_log = []

        self.trade_log = []
        self.invalid_trades = []
        self.holding = []
        self.prl_trades = []

        self.soc_log = []  # To keep track of SOC values over time
        self.upper_bound_log = []  # To keep track of upper boundaries over time
        self.lower_bound_log = []

        self.rewards = []
        self.reward_log = []
        self.penalty = -10  # Penalty for invalid trades and breaking constraints

        self.validation = validation

        # this indicates, if the agent is in a 4-hour block or not. A normal step will decrease it by 1,
        # participation in the PRL market will set it to 4
        self.prl_cooldown = 0
        # The upper and lower boundaries for the SOC. They are set based on the amount of energy the agent offers in the
        # PRL market
        self.upper_bound = self.battery.capacity
        self.lower_bound = 0

        self.trade_threshold = 10

    def step(self, action):
        """
        Execute a step in the environment, making trade decisions based on the provided action,
        and updating relevant state and logs.

        :param action: (list of floats) A list containing action variables which might include
                       decisions on participating in markets, prices, and amounts to trade.
                       Expected order: [prl_choice, price_prl, amount_prl, price_da, amount_da]

        :return: (np.array, float, bool, bool, dict) A tuple containing:
                 - The new state observation as a NumPy array.
                 - The reward obtained in this step as a float.
                 - A boolean indicating if the environment has reached a terminal state.
                 - A boolean indicating if the environment has been truncated.
                 - A dictionary containing additional information about the current state.

        """
        should_truncated = False
        if self.validation:
            self.day_ahead.step()
            self.prl.step()
        else:
            # make sure the two markets are always in sync
            should_truncated = self.prl.random_walk(24 * 30)
            current_step = self.prl.get_current_step()
            self.day_ahead.set_step(current_step)
            if should_truncated:
                self.reset()

        price_prl, amount_prl, price_da, amount_da = action

        reward = 0

        terminated = False  # Whether the agent reaches the terminal state
        truncated = should_truncated

        # Reset boundaries if PRL cooldown has expired
        if self.prl_cooldown == 0:
            self.upper_bound = self.battery.capacity
            self.lower_bound = 0

        amount_prl = min(amount_prl, self.battery.get_soc())

        # agent chooses to participate in the PRL market. The cooldown checks, if a new 4-hour block is ready
        if self.check_prl_constraints() and self.battery.can_discharge(amount_prl):
            if -self.trade_threshold < amount_prl < self.trade_threshold:
                reward += self.handle_holding()
            else:
                reward += self.perform_prl_trade(price_prl, amount_prl)

        # Handle DA trade or holding
        amount_da = self.clip_trade_amount(amount_da, 'buy' if amount_da > 0 else 'sell')
        if self.check_boundaries(amount_da):
            # Clip the amount to ensure that the battery state of charge remains within the bounds and decide based on
            # the new amount if the agent should hold or trade
            if -self.trade_threshold < amount_da < self.trade_threshold:
                reward += self.handle_holding()
            else:
                reward += self.perform_da_trade(amount_da, price_da)

        self.reward_log.append((self.reward_log[-1] + reward) if self.reward_log else reward)
        self.prl_cooldown = max(0, self.prl_cooldown - 1)  # Ensure it doesn't go below 0

        self.log_step(reward)

        info = {'current_price': self.day_ahead.get_current_price(),
                'current_step': self.day_ahead.get_current_step(),
                'savings': self.savings,
                'charge': self.battery.get_soc(),
                'prl price': price_prl,
                'prl amount': amount_prl,
                'da price': price_da,
                'da amount': amount_da,
                'reward': reward
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

    def clip_trade_amount(self, amount, trade_type):
        """
        Clips the trade amount to ensure that the state of charge remains within the bounds.

        :param amount: (float) The amount of energy to be traded.
        :param trade_type: (str) Type of trade to execute, accepted values are 'buy' or 'sell'.
        :return: (float) The clipped amount of energy that can be safely traded.
        """
        new_amount = amount
        if trade_type == 'buy':
            potential_soc = self.battery.get_soc() + amount
            if not (self.lower_bound < potential_soc < self.upper_bound):
                new_amount = min(amount, self.upper_bound - self.battery.get_soc())
                # print(f"Clipped buy amount from {amount} to {new_amount}")
        elif trade_type == 'sell':
            potential_soc = self.battery.get_soc() - amount
            if not (self.lower_bound < potential_soc < self.upper_bound):
                new_amount = max(amount, self.battery.get_soc() - self.upper_bound)
                # print(f"Clipped sell amount from {amount} to {new_amount}")
        else:
            raise ValueError(f"Invalid trade type: {trade_type}")

        return new_amount

    def set_boundaries(self, amount_prl):
        """Set boundaries based on PRL amount."""
        self.upper_bound = ((self.battery.capacity - 0.5 * amount_prl) / self.battery.capacity) * 1000
        self.lower_bound = ((0.5 * amount_prl) / self.battery.capacity) * 1000

    def check_boundaries(self, amount):
        """Check if the battery can charge or discharge the given amount of energy."""
        if self.lower_bound < self.battery.get_soc() + amount < self.upper_bound:
            return True
        return False

    def check_prl_constraints(self):
        """Check if all constraints for PRL participation are met."""
        if self.prl_cooldown <= 0 == self.prl.get_current_step() % 4:
            return True
        return False

    def perform_da_trade(self, energy_amount: float, market_price: float) -> float:
        """
        Perform a trade on the day-ahead market.

        :param energy_amount: Energy to be traded (positive for buying, negative for selling).
        :param market_price: Price at which the trade is attempted.
        :return: Reward based on the trade outcome.
        """
        energy_amount = self.clip_trade_amount(energy_amount, 'buy' if energy_amount > 0 else 'sell')

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
        if trade_type == 'buy' and price < current_price or trade_type == 'sell' and price > current_price:
            profit = self.penalty

        if self.day_ahead.accept_offer(price, trade_type):
            if trade_type == 'buy':
                self.battery.charge(amount)  # Charge battery for buy trades
                profit = -current_price * amount  # Negative profit for buying

            elif trade_type == 'sell':
                self.battery.discharge(amount)  # Discharge battery for sell trades
                profit = current_price * abs(amount)  # Positive profit for selling
        else:
            self.log_trades(False, trade_type, price, amount, self.penalty, 'market rejected')
            return self.penalty

        # Logging the trade details
        self.battery.add_charge_log(self.battery.get_soc())
        self.savings = self.savings + profit
        self.log_trades(True, trade_type, price, amount, profit, 'accepted')

        return profit

    def perform_prl_trade(self, price, amount) -> float:
        """
        Attempt to participate in the PRL market. It's a pay as bid market,
         so the agent will always get the price it offered.
        :param price: Offer price
        :param amount: Amount of energy being offered
        """
        # Check if the offer is accepted by the prl market and the battery can adhere to prl constraints
        if self.prl.accept_offer(price):
            # Update savings based on the transaction in prl market
            profit = float((price * amount) * 4)
            self.battery.charge_log.append(self.battery.get_soc())
            self.savings += profit
            # add the next four hours to the trade log. They should be equal to each other and just differ from the
            # step value
            for i in range(4):
                trade_info = (
                    self.prl.get_current_step() + i,
                    'reserve',
                    self.prl.get_current_price(),
                    price,
                    amount,
                    profit,
                    'prl accepted',
                    self.battery.get_soc(),
                    self.savings
                )
                self.trade_log.append(trade_info)

            self.set_boundaries(amount)
            self.prl_cooldown = 4

            return float(price * amount) * 4
        else:
            # return penalty if the offer was not accepted
            return self.penalty

    def handle_holding(self):
        # Logic for handling the holding scenario
        self.holding.append((self.day_ahead.get_current_step(), 'hold'))
        return 1

    def get_observation(self) -> np.array:
        """
        Get the current state of the environment.
        :return: np.array containing the current state of the environment
        """
        da_trend = self.day_ahead.previous_hours(self.trend_horizon)
        prl_trend = self.prl.previous_hours(self.trend_horizon)
        # Return the current state of the environment as a numpy array

        # add the prl cooldown to the observation
        prl_trend = np.append(prl_trend, [self.prl_cooldown])
        return np.concatenate((da_trend, prl_trend))

    def reset(self, seed=None, options=None) -> np.array:
        """
        Important: the observation must be a numpy array
        :return: np.array containing the initial state of the environment
        """
        # Reset the state of the environment to an initial state
        super().reset(seed=seed, options=options)
        self.savings = 50
        self.battery.reset()
        observation = self.get_observation().astype(np.float32)
        return observation, {}

    def log_step(self, reward):
        """
        Update the logs for the current step.

        :param reward: The reward obtained in this step.
        """
        self.rewards.append(reward)
        self.reward_log.append((self.reward_log[-1] + reward) if self.reward_log else reward)
        self.upper_bound_log.append(self.upper_bound)
        self.lower_bound_log.append(self.lower_bound)
        self.soc_log.append(self.battery.get_soc())

    def render(self, mode='human'):
        """
        Render the environment to the screen
        :param mode:
        :return:
        """
        plot_savings(self.trade_log, 'multi_trend')
        plot_savings_on_trade_steps(trade_log=self.trade_log, total_steps=self.da_dataframe.shape[0],
                                    model_name='multi_trend')
        plot_charge(self.battery, 'multi_trend')
        plot_trades_timeline(trade_source=self.trade_log, title='Trades', buy_color='green', sell_color='red',
                             model_name='multi_trend', data=self.da_dataframe, plot_name='trades')
        plot_trades_timeline(trade_source=self.invalid_trades, title='Invalid Trades', buy_color='black',
                             sell_color='brown', model_name='multi_trend', data=self.da_dataframe,
                             plot_name='invalid_trades')
        plot_holding(self.holding, 'multi_trend', da_data=self.da_dataframe)
        plot_soc_and_boundaries(self.soc_log, self.upper_bound_log, self.lower_bound_log, 'multi_trend')
        kernel_density_estimation(self.trade_log, 'multi_trend', da_data=self.da_dataframe)

    def get_trades(self) -> list:
        """
        Returns the trade log
        :return: list of tuples
        """
        return self.trade_log

    def get_invalid_trades(self) -> list:
        """
        Returns the trade log
        :return: list of tuples
        """
        return self.invalid_trades

    def get_prl_trades(self) -> list:
        """
        Returns the trade log for the PRL market
        :return: list of tuples
        """
        return self.prl_trades

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
