import gymnasium as gym
from gymnasium import spaces

from envs.assets.battery import Battery
from envs.assets.dayahead import DayAhead
from envs.assets.frequency_containment_reserve import FrequencyContainmentReserve
from envs.assets.plot_engien import *


class MultiMarket(gym.Env):
    def __init__(self, da_data_path: str, prl_data_path: str, validation=False):
        super(MultiMarket, self).__init__()
        self.da_dataframe = pd.read_csv(da_data_path)
        self.prl_dataframe = pd.read_csv(prl_data_path)

        da_low_boundary = self.da_dataframe.min().values
        da_high_boundary = self.da_dataframe.max().values

        prl_low_boundary = self.prl_dataframe.min().values
        prl_high_boundary = self.prl_dataframe.max().values

        min_array = np.concatenate((da_low_boundary, prl_low_boundary))
        max_array = np.concatenate((da_high_boundary, prl_high_boundary))

        action_low = np.array([-1, 0, 0, 0, -500.0])  # prl choice, prl price, prl amount, da price, da amount
        action_high = np.array([1, 1.0, 500, 1, 500.0])  # prl choice, prl price, prl amount, da price, da amount

        self.action_space = spaces.Box(low=action_low, high=action_high, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(low=min_array, high=max_array,
                                            shape=(self.da_dataframe.shape[1] + self.prl_dataframe.shape[1],))

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
        self.window_size = 20
        self.penalty = - 10

        self.validation = validation

        # this indicates, if the agent is in a 4-hour block or not. A normal step will decrease it by 1,
        # participation in the PRL market will set it to 4
        self.prl_cooldown = 0
        self.reserve_amount = 0
        self.upper_bound = self.battery.capacity
        self.lower_bound = 0

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
        if self.validation:
            self.day_ahead.step()
            self.prl.step()
        else:
            # make sure the two markets are always in sync
            self.prl.random_walk()
            current_step = self.prl.get_current_step()
            self.day_ahead.set_step(current_step)

        prl_choice, price_prl, amount_prl, price_da, amount_da = action

        reward = 0

        terminated = False  # Whether the agent reaches the terminal state
        truncated = False  # this can be false all the time since there is no failure condition the agent could trigger

        prl_criteria = (self.battery.capacity / amount_prl) > 1
        # Reset boundaries if PRL cooldown has expired
        if self.prl_cooldown == 0:
            self.upper_bound = self.battery.capacity
            self.lower_bound = 0

        # Perform PRL trade if constraints are met
        if self.check_prl_constraints(prl_choice) and prl_criteria and self.check_boundaries(amount_prl):
            reward = self.perform_prl_trade(price_prl, amount_prl)

        # Perform DA trade if boundaries allow
        if self.check_boundaries(amount_da):
            reward += self.perform_da_trade(amount_da, price_da)
        else:
            # Apply penalty if boundaries are violated
            reward += self.penalty

        # Decrement PRL cooldown
        self.prl_cooldown = max(0, self.prl_cooldown - 1)  # Ensure it doesn't go below 0

        self.log_step(reward, prl_choice, price_prl, amount_prl, price_da, amount_da)

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

        return self.get_observation().astype(np.float32), reward, terminated, truncated, info

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

        elif amount_da < 0:  # sell
            reward = self.trade(price_da, amount_da, 'sell')

        elif amount_da == 0:  # if amount is 0
            reward = 5
            self.holding.append((self.day_ahead.get_current_step(), 'hold'))

        return reward

    def trade(self, price, amount, trade_type) -> float:
        """
        Execute a trade in the market.

        :param price: Price at which trade is attempted
        :param amount: Amount of energy being traded
        :param trade_type: Type of trade - 'buy' or 'sell'
        :return: Reward for the agent based on trade outcome
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

            # updating the logs
            self.battery.add_charge_log(self.battery.get_soc())
            self.savings_log.append(self.savings)
            self.trade_log.append((self.day_ahead.get_current_step(), price, amount, trade_type,
                                   abs(float(self.day_ahead.get_current_price()) * amount)))
        else:
            self.invalid_trades.append((self.day_ahead.get_current_step(), price, amount, trade_type,
                                        abs(float(self.day_ahead.get_current_price()) * amount)))
            return self.penalty

        return abs(float(self.day_ahead.get_current_price()) * amount)

    def perform_prl_trade(self, price, amount) -> float:
        """
        Attempt to participate in the PRL market.
        :param price: Offer price
        :param amount: Amount of energy being offered
        """
        # Check if the offer is accepted by the prl market and the battery can adhere to prl constraints
        if self.prl.accept_offer(price):
            # Update savings based on the transaction in prl market
            self.savings += (self.prl.get_current_price() * amount) * 4
            # Set cooldown and reserve amount since participation was successful
            self.prl_cooldown = 4
            # Immediately update boundaries after a successful PRL trade
            self.set_boundaries(amount)
            self.battery.charge_log.append(self.battery.get_soc())
            self.savings_log.append(self.savings)
            # add the next four hours to the trade look. They should be equal to each other and just differ from the
            # step value
            for i in range(4):
                trade_info = (
                    self.prl.get_current_step() + i,
                    price,
                    amount,
                    'reserve',
                    abs(float((self.prl.get_current_price() * amount) * 4))
                )
                self.trade_log.append(trade_info)
            return (self.prl.get_current_price() * amount) * 4
        else:
            return self.penalty

    def get_observation(self) -> np.array:
        """
        Get the current state of the environment.
        :return: np.array containing the current state of the environment
        """
        # Return the current state of the environment as a numpy array
        observation = np.concatenate((self.da_dataframe.iloc[self.day_ahead.get_current_step()].to_numpy(dtype=float),
                                      self.prl_dataframe.iloc[self.prl.get_current_step()].to_numpy(dtype=float)))

        return observation

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

    def log_step(self, reward, prl_choice, price_prl, amount_prl, price_da, amount_da):
        """
        Update the logs for the current step.

        :param reward: The reward obtained in this step.
        :param prl_choice: The PRL choice made in this step.
        :param price_prl: The PRL price set in this step.
        :param amount_prl: The PRL amount set in this step.
        :param price_da: The DA price set in this step.
        :param amount_da: The DA amount set in this step.
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
        kernel_density_estimation(self.trade_log)
        plot_reward(self.reward_log, self.window_size, 'multi')
        plot_savings(self.savings_log, self.window_size, 'multi')
        plot_charge(self.window_size, self.battery, 'multi')
        plot_trades_timeline(trade_source=self.trade_log, title='Trades', buy_color='green', sell_color='red',
                             model_name='multi')
        plot_trades_timeline(trade_source=self.invalid_trades, title='Invalid Trades', buy_color='black',
                             sell_color='brown', model_name='multi')
        plot_holding(self.holding, 'multi')
        plot_soc_and_boundaries(self.soc_log, self.upper_bound_log, self.lower_bound_log, 'multi')

    def get_trades(self) -> list:
        """
        Returns the trade log
        :return: list of tuples
        """
        return self.trade_log

    def get_prl_trades(self) -> list:
        """
        Returns the trade log for the PRL market
        :return: list of tuples
        """
        return self.prl_trades