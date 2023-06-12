import gym
from gym import spaces
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from market import Market


class EnergyMarketEnv(gym.Env):
    def __init__(self):
        super(EnergyMarketEnv, self).__init__()

        # Initialize state
        self.current_step = 0
        self.current_savings = 50  # Current profit in €
        self.current_charge = 500  # Current battery charge
        self.battery_charge_history = [500]  # History of battery charge levels
        self.max_battery_charge = 1000  # kWh
        self.price_history = []  # History of market prices
        self.reward_history = []  # History of rewards
        self.savings_history = []  # History of savings
        self.charge_percentages = [50]  # History of charge percentages

        # Load the dataset
        self.df = pd.read_csv('data/clean/dataset_01102018_01012023.csv')
        self.df['date'] = pd.to_datetime(self.df['date']).apply(lambda x: x.timestamp())
        # only keep the date and the price
        self.df = self.df[['date', 'price', 'consumption', 'prediction']]
        # normalize everything except the date
        self.df[['price', 'consumption', 'prediction']] = self.df[['price', 'consumption', 'prediction']].apply(
            lambda x: (x - x.min()) / (x.max() - x.min()))
        # set the date as index
        self.df = self.df.set_index('date')

        self.max_steps = len(self.df)
        self.market = Market(self.df)

        # Define action and observation space
        self.action_space = spaces.Box(low=np.array([-1000, 0, 0]), high=np.array([1000, 1, self.max_battery_charge]),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.df.shape[1] + 1,))

    def step(self, action):
        # Execute one time step within the environment
        self.current_step += 1

        if self.current_step >= len(self.df):
            # If it does, reset the current step to 0
            self.current_step = 0

        # Extract the components of the action
        price = action[0]
        charge_discharge = action[1]
        amount = action[2]

        # Initialize the reward to 0
        reward = 0

        if charge_discharge == 0:  # buy
            if price > self.current_savings or amount > self.max_battery_charge - self.current_charge:
                reward = -100
            elif self.market.accept_offer(price):
                if self.current_savings - price < 0:  # Check if the agent would lose too much savings
                    reward = -1000  # Hard penalty
                else:
                    self.current_savings -= price
                    self.current_charge += amount
                    reward = price * amount * (1 - self.current_charge / self.max_battery_charge)  # Less reward for buying when battery is charged
                    if amount > 0.5 * self.max_battery_charge:  # Bonus for large amounts
                        reward += 50
                    if self.current_charge > 0.9 * self.max_battery_charge:  # Penalty for buying when battery is highly charged
                        reward -= 50
                    if self.current_charge < 0.1 * self.max_battery_charge:  # Bonus for buying when battery is low
                        reward += 50

        elif charge_discharge == 1:  # sell
            if self.current_charge - amount < 0:  # check if the battery can be discharged
                reward = -100
            elif self.market.accept_offer(price):
                self.current_savings += price
                self.current_charge -= amount
                reward = price * amount * (self.current_charge / self.max_battery_charge)  # More reward for selling when battery is charged
                if amount > 0.5 * self.max_battery_charge:  # Bonus for large amounts
                    reward += 50
                if self.current_charge > 0.9 * self.max_battery_charge:  # Bonus for selling when battery is highly charged
                    reward += 50
                if self.current_charge < 0.1 * self.max_battery_charge:  # Penalty for selling when battery is low
                    reward -= 50

        elif charge_discharge == 2:  # do nothing
            reward = 0  # No reward for doing nothing

        # Append the current charge level to the history
        self.battery_charge_history.append(self.current_charge)
        self.reward_history.append(reward)
        self.savings_history.append(self.current_savings)
        self.calculate_charge_percentage()

        done = self.current_step > self.max_steps

        observation = self.get_observation()
        return observation.astype(np.float32), reward, done, {}

    def get_observation(self):
        # Return the current state as an observation
        return np.concatenate([self.df.iloc[self.current_step].values, [self.current_charge]])

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        self.current_charge = 500
        self.current_savings = 50
        self.reward_history = [0]
        self.savings_history = [50]
        self.battery_charge_history = [50]
        self.charge_percentages = []

        # Return the initial observation
        initial_observation = self.get_observation()
        return initial_observation.astype(np.float32)

    def render(self, mode='human'):
        # Create a figure with two subplots
        fig, axs = plt.subplots(4)

        # Plot the battery charge level in the second subplot
        axs[0].plot(self.battery_charge_history[:self.current_step])
        axs[0].set_xlabel('Time step')
        axs[0].set_ylabel('Battery Charge Level')
        axs[0].set_title('Battery Charge Level Over Time')

        # Plot the reward in the second subplot
        axs[1].plot(self.reward_history)
        axs[1].set_xlabel('Time step')
        axs[1].set_ylabel('Reward')
        axs[1].set_title('Reward Over Time')

        # Plot the savings in the third subplot
        axs[2].plot(self.savings_history)
        axs[2].set_xlabel('Time step')
        axs[2].set_ylabel('Savings')
        axs[2].set_title('Savings Over Time')

        # Plot the charge percentage in the fourth subplot
        axs[3].plot(self.charge_percentages)
        axs[3].set_xlabel('Time step')
        axs[3].set_ylabel('Charge Percentage')
        axs[3].set_title('Charge Percentage Over Time')

        # make more space between the subplots and the figure bigger
        plt.tight_layout()
        fig.set_size_inches(18.5, 10.5)

        # Show the figure
        plt.show()

    def calculate_charge_percentage(self):
        self.charge_percentages.append(self.current_charge / self.max_battery_charge * 100)
