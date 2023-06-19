import gym
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from market import Market


class EnergyMarketEnv(gym.Env):
    def __init__(self):
        super(EnergyMarketEnv, self).__init__()

        # Initialize state
        self.trade_log = []
        self.current_step = 0
        self.current_savings = 0.5  # Current profit in €
        self.current_charge = 0.5  # Current battery charge
        self.max_battery_charge = 1000  # kWh
        self.reward_history = []  # History of rewards

        # Load the dataset
        self.df = pd.read_csv("data/clean/env_data.csv")

        self.max_steps = 37272
        self.market = Market(self.df)

        # Define action with two continuous variables
        self.action_space = gym.spaces.box.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # assuming the original shape of df is (n, m)
        self.observation_space = gym.spaces.box.Box(low=-1, high=1, shape=(self.df.shape[1] + 2,))

    def step(self, action):
        # Execute one time step within the environment
        self.current_step += 1
        self.market.step()

        if self.current_step >= len(self.df):
            # If it does, reset the current step to 0
            self.current_step = 0

        # Extract the components of the action
        price = float(action[0].item())
        amount = float(action[1].item())

        # Initialize the reward to 0
        reward = 0

        if amount > 0:  # buy
            # Check if the agent has enough savings to buy the amount of energy
            if price > self.current_savings or amount > self.max_battery_charge - self.current_charge or amount <= 0:
                reward = -1
                return self.get_observation().astype(np.float32), reward, False, {}
            if self.market.accept_offer(price, 'buy'):
                self.current_charge += abs(amount)
                self.current_savings -= price
                reward = abs(float(self.market.get_current_price()) * amount)
                self.log_trade(price, abs(amount), 'buy', self.current_step, reward, self.current_savings)
            else:
                reward = -1

        if amount < 0:  # sell
            # Check if the agent has enough energy to sell
            if amount < -self.current_charge or price <= 0:
                reward = -1
                return self.get_observation().astype(np.float32), reward, False, {}
            if self.market.accept_offer(price, 'sell'):
                self.current_savings += price
                self.current_charge -= abs(amount)
                reward = abs(float(self.market.get_current_price()) * amount)
                self.log_trade(price, abs(amount), 'sell', self.current_step, reward, self.current_savings)
            else:
                reward = -1

        if amount == 0:
            reward = 0

        # Append the current charge level to the history
        self.reward_history.append(reward)

        done = self.current_step >= self.max_steps - 1
        if done:
            print("Episode finished after {} timesteps".format(self.current_step + 1))

        return self.get_observation().astype(np.float32), reward, done, {}

    def get_observation(self):
        # Return the current state as an observation
        return np.concatenate([self.df.iloc[self.current_step].values, [self.current_charge], [self.current_savings]])

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        self.current_charge = 0.5
        self.current_savings = 0.5
        self.reward_history = [0]

        # Return the initial observation
        initial_observation = self.get_observation()
        return initial_observation.astype(np.float32), {}  # return a tuple with observation and an empty dictionary

    def render(self, mode='human'):
        # calculate the average reward over 100 steps and plot it
        plt.plot(self.reward_history)
        plt.xlabel('Steps')
        plt.ylabel('Average Reward')

        plt.show()

    def log_trade(self, price, amount, trade_type, date, reward, savings):
        self.trade_log.append([price, amount, trade_type, date, self.market.get_current_price(), reward, savings])

    def get_trade_log(self):
        # return the trade log as a pandas dataframe
        return pd.DataFrame(self.trade_log,
                            columns=['price', 'amount', 'trade_type', 'date', 'market_price', 'reward', 'savings'])
