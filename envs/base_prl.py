import gymnasium as gym
from gymnasium import spaces

from envs.assets.battery import Battery
from envs.assets.dayahead import DayAhead
from envs.assets.frequency_containment_reserve import FrequencyContainmentReserve
from envs.assets.plot_engien import *


class BasePRL(gym.Env):
    def __init__(self, da_data_path: str, prl_data_path: str, validation=False):
        super(BasePRL, self).__init__()
        self.da_dataframe = pd.read_csv(da_data_path)
        self.prl_dataframe = pd.read_csv(prl_data_path)

        da_low_boundary = self.da_dataframe.min().values
        da_high_boundary = self.da_dataframe.max().values

        prl_low_boundary = self.prl_dataframe.min().values
        prl_high_boundary = self.prl_dataframe.max().values

        min_array = np.concatenate((da_low_boundary, prl_low_boundary))
        max_array = np.concatenate((da_high_boundary, prl_high_boundary))

        action_low = np.array([-1, 0, 0, 0, -500.0])  # prl choice, prl price, prl amount, da price, da amount
        action_high = np.array([1, 1.0, 250, 1, 500.0])  # prl choice, prl price, prl amount, da price, da amount

        self.action_space = spaces.Box(low=action_low, high=action_high, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(low=min_array, high=max_array,
                                            shape=(self.da_dataframe.shape[1] + self.prl_dataframe.shape[1],))

        self.day_ahead = DayAhead(self.da_dataframe)
        self.prl = FrequencyContainmentReserve(self.prl_dataframe)
        self.battery = Battery(1000, 0)
        self.savings = 50  # €
        self.savings_log = []

        self.trade_log = []
        self.invalid_trades = []
        self.holding = []
        self.prl_trades = []

        self.rewards = []
        self.reward_log = []
        self.window_size = 20
        self.penalty = -5

        self.validation = validation

        # this indicates, if the agent is in a 4-hour block or not. A normal step will decrease it by 1,
        # participation in the PRL market will set it to 4
        self.prl_cooldown = 0
        self.reserve_amount = 0

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

        prl_choice = action[0].item()
        price_prl = action[1].item()
        amount_prl = action[2].item()
        price_da = action[3].item()
        amount_da = action[4].item()

        reward = 0

        terminated = False  # Whether the agent reaches the terminal state
        truncated = False  # this can be false all the time since there is no failure condition the agent could trigger
        info = {'current_price': self.day_ahead.get_current_price(),
                'current_step': self.day_ahead.get_current_step(),
                'savings': self.savings,
                'charge': self.battery.get_soc(),
                'prl choice': prl_choice,
                'prl price': price_prl,
                'prl amount': amount_prl,
                'da price': price_da,
                'da amount': amount_da,
                }
        # agent chooses to participate in the PRL market. The cooldown checks, if a new 4-hour block is ready
        if prl_choice > 0 >= self.prl_cooldown and self.prl.get_current_step() % 4 == 0:
            reward = self.participate_in_prl(price_prl, amount_prl)

        # the agent chooses to trade on the DA market outside the 4-hour block
        if prl_choice < 0 and self.prl_cooldown <= 0:
            reward = self.perform_da_trade(amount_da=amount_da, price_da=price_da)

        self.rewards.append(reward)
        self.reward_log.append((self.reward_log[-1] + reward) if self.reward_log else reward)
        self.prl_cooldown -= 1
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

    def participate_in_prl(self, price, amount) -> float:
        """
        Attempt to participate in the PRL market.
        :param price: Offer price
        :param amount: Amount of energy being offered
        """
        # Check if the offer is accepted by the prl market and the battery can adhere to prl constraints
        if self.prl.accept_offer(price) and self.battery.check_prl_constraints(amount):
            # Update savings based on the transaction in prl market
            self.savings += (self.prl.get_current_price() * amount) * 4
            # Set cooldown and reserve amount since participation was successful
            self.prl_cooldown = 4
            self.reserve_amount = amount
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
            return -1

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

    def render(self, mode='human'):
        """
        Render the environment to the screen
        :param mode:
        :return:
        """
        kernel_density_estimation(self.trade_log)
        plot_reward(self.reward_log, self.window_size, 'base')
        plot_savings(self.savings_log, self.window_size, 'base')
        plot_charge(self.window_size, self.battery, 'base')
        plot_trades_timeline(trade_source=self.trade_log, title='Trades', buy_color='green', sell_color='red',
                             model_name='base')
        plot_trades_timeline(trade_source=self.invalid_trades, title='Invalid Trades', buy_color='black',
                             sell_color='brown', model_name='base')
        plot_holding(self.holding, 'base')

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
