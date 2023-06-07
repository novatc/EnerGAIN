import gym
from gym import spaces
import pandas as pd
import numpy as np

from market import Market


class EnergyMarketEnv(gym.Env):
    def __init__(self):
        super(EnergyMarketEnv, self).__init__()

        # Load the dataset
        self.df = pd.read_csv('data/clean/dataset_01102018_01012023.csv')
        self.df['date'] = pd.to_datetime(self.df['date']).apply(lambda x: x.timestamp())
        # only keep the date and the price
        self.df = self.df[['date', 'price']]
        self.max_steps = len(self.df)
        self.market = Market(self.df)

        # Define action and observation space
        self.action_space = spaces.Box(low=0, high=100000, shape=(6,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.df.shape[1] + 1,))

        # Initialize state
        self.current_step = 0
        self.battery_charge = 0
        self.max_battery_charge = 1000  # kWh

    def step(self, action):
        # Execute one time step within the environment
        self.market.step()
        self.current_step += 1

        # Extract the components of the action
        buy_sell = action[0]
        offer_amount = action[1]
        offer_price = action[2]
        market = action[3]
        charge_discharge = action[4]

        # Check if the market accepts the offer
        if self.market.accept_offer(offer_price):
            # The offer was accepted, so the reward is equal to the price times the amount
            reward = offer_price * offer_amount

            # Update the battery charge level based on the amount in the offer
            if charge_discharge == 0:  # discharge
                self.battery_charge = max(0, self.battery_charge - offer_amount)
            else:  # charge
                self.battery_charge = min(self.max_battery_charge, self.battery_charge + offer_amount)
        else:
            # The offer was not accepted, so the reward is 0
            reward = 0

        done = self.current_step >= self.max_steps

        return self.get_observation(), reward, done, {}

    def get_observation(self):
        # Return the current state as an observation
        return np.concatenate([self.df.iloc[self.current_step].values, [self.battery_charge]])

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        self.battery_charge = 0
        return self.get_observation()

    def render(self, mode='human'):
        pass

    def get_reward(self, action):
        # Define how the agent is rewarded
        # This is just a placeholder, you'll need to define this based on your specific use case
        return np.random.rand()
