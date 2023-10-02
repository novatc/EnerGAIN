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

        action_low = np.array([-1, -1.0, -1.0, 0, -500.0])  # prl choice, prl price, prl amount, da price, da amount
        action_high = np.array([1, 1.0, 1.0, 500.0, 500.0])  # prl choice, prl price, prl amount, da price, da amount

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

        self.rewards = []
        self.reward_log = []
        self.window_size = 20

        self.validation = validation

        # this indicates, if the agent is in a 4-hour block or not. A normal step will decrease it by 1,
        # participation in the PRL market will set it to 4
        self.prl_cooldown = 0

    def step(self, action):
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

        if prl_choice > 0 and self.prl_cooldown == 0:  # agent chooses to participate in the prl market
            self.participate_in_prl(price_prl, amount_prl)
            self.prl_cooldown = 4

        # TODO: if agent participates in prl market, all its constrains must be applied to the da market for the next
        #  4 steps

        # check if the battery still can cover the amount when the agent is in one 4-hour block
        if self.prl_cooldown > 0:
            if self.battery.check_prl_constraints(amount_da) is True:
                if amount_da > 0:  # buy
                    reward = self.trade(price_da, amount_da, 'buy')
                elif amount_da < 0:  # sell
                    reward = self.trade(price_da, amount_da, 'sell')
                else:  # if amount is 0
                    self.holding.append((self.day_ahead.get_current_step(), 'hold'))
                    reward = 5
            else:
                reward = -10

        if self.prl_cooldown <= 0:
            if amount_da > 0:  # buy
                reward = self.trade(price_da, amount_da, 'buy')
            elif amount_da < 0:  # sell
                reward = self.trade(price_da, amount_da, 'sell')
            else:  # if amount is 0
                self.holding.append((self.day_ahead.get_current_step(), 'hold'))

        self.prl_cooldown -= 1
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

    def participate_in_prl(self, price, amount):
        if self.prl.accept_offer(price) and self.battery.check_prl_constraints(amount):
            self.savings += (self.prl.get_current_price() * amount) * 4
            self.prl_cooldown = 4

    def get_observation(self):
        # Return the current state of the environment as a numpy array
        observation = np.concatenate((self.da_dataframe.iloc[self.day_ahead.get_current_step()].to_numpy(dtype=float),
                                      self.prl_dataframe.iloc[self.prl.get_current_step()].to_numpy(dtype=float)))

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
        self.prl.reset()
        observation = self.get_observation().astype(np.float32)
        return observation, {}

    def render(self, mode='human'):
        plot_reward(self.reward_log, self.window_size, 'base')
        plot_savings(self.savings_log, self.window_size, 'base')
        plot_charge(self.window_size, self.battery, 'base')
        plot_trades_timeline(trade_source=self.trade_log, title='Trades', buy_color='green', sell_color='red',
                             model_name='base')
        plot_trades_timeline(trade_source=self.invalid_trades, title='Invalid Trades', buy_color='black',
                             sell_color='brown', model_name='base')
        plot_holding(self.holding, 'base')

    def get_trades(self):
        return self.trade_log
