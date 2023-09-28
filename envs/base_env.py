import gymnasium as gym
from gymnasium import spaces

from envs.assets.battery import Battery
from envs.assets.dayahead import DayAhead
from envs.assets.plot_engien import *


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
            self.holding.append((self.day_ahead.get_current_step(), 'hold'))
            reward = 5

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
