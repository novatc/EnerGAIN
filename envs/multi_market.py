import gymnasium as gym
from gymnasium import spaces

from envs.assets.battery import Battery
from envs.assets.dayahead import DayAhead
from envs.assets.frequency_containment_reserve import FrequencyContainmentReserve
from envs.assets.plot_engien import *


class MultiMarket(gym.Env):
    def __init__(self, da_data_path: str, prl_data_path: str, validation):
        super(MultiMarket, self).__init__()
        self.da_dataframe = pd.read_csv(da_data_path)
        self.prl_dataframe = pd.read_csv(prl_data_path)

        da_low_boundary = self.da_dataframe.min().values
        da_high_boundary = self.da_dataframe.max().values

        prl_low_boundary = self.prl_dataframe.min().values
        prl_high_boundary = self.prl_dataframe.max().values

        min_array = np.concatenate((da_low_boundary, prl_low_boundary))
        max_array = np.concatenate((da_high_boundary, prl_high_boundary))

        # adding one more entry for the prl cooldown. The cooldown is the number of steps the agent has to wait until
        # it can participate in the PRL market again, representing the 4-hour block per trade
        observation_high = np.append(max_array, [4])
        observation_low = np.append(min_array, [0])

        # +3 for prl cooldown, upper & lower bounds
        obs_shape = (self.da_dataframe.shape[1] + self.prl_dataframe.shape[1] + 3,)

        action_low = np.array([-1, 0, 0, 0, -1000.0])  # prl choice, prl price, prl amount, da price, da amount
        action_high = np.array([1, 1.0, 1000, 1, 1000.0])  # prl choice, prl price, prl amount, da price, da amount

        self.action_space = spaces.Box(low=action_low, high=action_high, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.append(observation_low, [0, 0]),  # Add 0 for lower and upper bounds
                                            high=np.append(observation_high, [1000, 1000]),
                                            # Assuming 1000 is the max bound
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
        self.window_size = 5
        self.penalty = -30

        self.validation = validation

        # this indicates, if the agent is in a 4-hour block or not. A normal step will decrease it by 1,
        # participation in the PRL market will set it to 4
        self.prl_cooldown = 0
        self.reserve_amount = 0
        self.upper_bound = self.battery.capacity
        self.lower_bound = 0

        self.trade_threshold = 50

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
            should_truncated = self.prl.random_walk(24 * 7)
            current_step = self.prl.get_current_step()
            self.day_ahead.set_step(current_step)

        prl_choice, price_prl, amount_prl, price_da, amount_da = action

        reward = 0

        terminated = False  # Whether the agent reaches the terminal state
        truncated = should_truncated  # this will be true if the agent does a time jump

        # Reset boundaries if PRL cooldown has expired
        if self.prl_cooldown == 0:
            self.upper_bound = self.battery.capacity
            self.lower_bound = 0
            self.battery.charge(self.reserve_amount)
            self.reserve_amount = 0

        # Reward for staying within battery bounds
        if not self.lower_bound < self.battery.get_soc() < self.upper_bound:
            reward += self.penalty

        # Handle PRL trade if constraints are met
        if self.check_prl_constraints(prl_choice):
            reward += self.perform_prl_trade(price_prl, amount_prl)

        # Handle DA trade or holding
        # clip the amount for the day ahead market to ensure that the battery can handle the trade
        amount_da = self.clip_trade_amount(amount_da, 'buy' if amount_da > 0 else 'sell')
        if -self.trade_threshold < amount_da < self.trade_threshold:
            reward += self.handle_holding()
        elif self.check_boundaries(amount_da):
            reward += self.perform_da_trade(amount_da, price_da)
        else:
            reward += self.penalty  # Apply penalty if DA boundaries are violated

        # Decrement PRL cooldown
        self.prl_cooldown = max(0, self.prl_cooldown - 1)

        self.log_step(reward)

        info = {'current_price': self.day_ahead.get_current_price(),
                'current_step': self.day_ahead.get_current_step(),
                'savings': self.savings,
                'charge': self.battery.get_soc(),
                'prl choice': prl_choice,
                'prl price': price_prl,
                'prl amount': amount_prl,
                'da price': price_da,
                'da amount': amount_da,
                'reward': reward
                }

        obs = self.get_observation()

        return obs, reward, terminated, truncated, info

    def set_boundaries(self, amount_prl):
        """Set boundaries based on PRL amount."""
        self.upper_bound = ((self.battery.capacity - 0.5 * amount_prl) / self.battery.capacity) * 1000
        self.lower_bound = ((0.5 * amount_prl) / self.battery.capacity) * 1000

    def check_boundaries(self, amount):
        """Check if the battery can charge or discharge the given amount of energy."""
        if self.lower_bound < self.battery.get_soc() + amount < self.upper_bound:
            return True
        return False

    def check_prl_constraints(self, choice_value):
        """Check if all constraints for PRL participation are met."""
        if choice_value > 0 >= self.prl_cooldown and self.prl.get_current_step() % 4 == 0:
            return True
        return False

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
            self.savings += (price * amount)
            self.battery.charge_log.append(self.battery.get_soc())
            self.savings_log.append(self.savings)
            self.battery.discharge(amount)
            self.reserve_amount = amount
            # add the next four hours to the trade log. They should be equal to each other and just differ from the
            # step value
            for i in range(4):
                trade_info = (
                    self.prl.get_current_step() + i,
                    'reserve',
                    self.prl.get_current_price(),
                    price,
                    amount,
                    price * amount,
                    'prl accepted'
                )
                self.trade_log.append(trade_info)

            self.set_boundaries(amount)
            self.prl_cooldown = 4

            return float(price * amount)
        else:
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
        # Return the current state of the environment as a numpy array
        observation = np.concatenate((self.da_dataframe.iloc[self.day_ahead.get_current_step()].to_numpy(dtype=float),
                                      self.prl_dataframe.iloc[self.prl.get_current_step()].to_numpy(dtype=float)))

        # Append the current SOC boundaries to the observation
        return np.append(observation, [self.prl_cooldown, self.lower_bound, self.upper_bound])

    def reset(self, seed=None, options=None) -> np.array:
        """
        Important: the observation must be a numpy array
        :return: np.array containing the initial state of the environment
        """
        # Reset the state of the environment to an initial state
        super().reset(seed=seed, options=options)
        self.savings = 50
        self.battery.reset()
        self.day_ahead.reset()
        self.prl.reset()
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
        plot_reward(self.reward_log, self.window_size, 'multi')
        plot_savings(self.savings_log, self.window_size, 'multi')
        plot_charge(self.window_size, self.battery, 'multi')
        plot_trades_timeline(trade_source=self.trade_log, title='Trades', buy_color='green', sell_color='red',
                             model_name='multi', data=self.da_dataframe)
        plot_trades_timeline(trade_source=self.invalid_trades, title='Invalid Trades', buy_color='black',
                             sell_color='brown', model_name='multi', data=self.da_dataframe)
        plot_holding(self.holding, 'multi', da_data=self.da_dataframe)
        plot_soc_and_boundaries(self.soc_log, self.upper_bound_log, self.lower_bound_log, 'multi')
        kernel_density_estimation(self.trade_log, 'multi', da_data=self.da_dataframe)

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
