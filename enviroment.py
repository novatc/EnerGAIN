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
        self.trade_log = []
        self.current_step = 0
        self.current_savings = 50  # Current profit in €
        self.current_charge = 500  # Current battery charge
        self.battery_charge_history = [500]  # History of battery charge levels
        self.max_battery_charge = 1000  # MWh
        self.price_history = []  # History of market prices
        self.reward_history = []  # History of rewards
        self.savings_history = []  # History of savings
        self.buy_sell_history = []  # History of buy/sell actions
        self.charge_percentages = [50]  # History of charge percentages

        # Load the dataset
        self.df = pd.read_csv('data/clean/dataset_01102018_01012023.csv')
        self.df['date'] = pd.to_datetime(self.df['date']).apply(lambda x: x.timestamp())
        # only keep the date and the price
        self.df = self.df[['date', 'price', 'consumption', 'prediction', 'Einstrahlung auf die Horizontale (kWh/m²)',
                           'Diffusstrahlung auf die Horizontale (kWh/m²)']]
        # normalize everything except the date
        self.df.iloc[:, 1:] = (self.df.iloc[:, 1:] - self.df.iloc[:, 1:].mean()) / self.df.iloc[:, 1:].std()

        self.max_steps = len(self.df)
        self.market = Market(self.df)

        # Define action and observation space
        self.action_space = spaces.Box(low=np.array([-1000, 0, 0]), high=np.array([1000, 1, self.max_battery_charge]),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.df.shape[1] + 2,))

    def step(self, action):
        # Execute one time step within the environment
        self.current_step += 1
        self.market.step()

        if self.current_step >= len(self.df):
            # If it does, reset the current step to 0
            self.current_step = 0

        # Extract the components of the action
        price = action[0]
        charge_discharge = action[1]
        amount = action[2]

        # Initialize the reward to 0
        reward = 0

        # Implement the logic of the environment
        # How the state is updated depends on the action taken
        # Here, we implement the logic of buying, selling or doing nothing
        if charge_discharge == 0:  # buy
            if amount <= 10:
                reward = -100
            if price > self.current_savings or amount > self.max_battery_charge - self.current_charge:
                reward = -100
            elif self.market.accept_offer(price, 'buy'):
                self.log_trade(price, amount, 'buy', self.current_step)
                self.buy_sell_history.append(-1)
                if self.current_savings - price < 0:  # Check if the agent would lose too many savings
                    reward = -100  # Hard penalty
                else:
                    self.current_savings -= price * amount
                    self.current_charge += amount
                    reward = price * amount  # Reward based on the profit

        elif charge_discharge == 1:
            if amount <= 0:
                reward = -100
            # sell
            if self.current_charge - amount < 0:  # check if the battery can be discharged
                reward = -100
            elif self.market.accept_offer(price, 'sell'):
                self.log_trade(price, amount, 'sell', self.current_step)
                self.buy_sell_history.append(1)
                self.current_savings += price * amount
                self.current_charge -= amount
                reward = price * amount  # Reward based on the profit

        elif charge_discharge == 2:  # do nothing
            reward = 0  # No reward for doing nothing

        # Append the current charge level to the history
        self.battery_charge_history.append(self.current_charge)
        self.reward_history.append(reward)
        self.savings_history.append(self.current_savings)
        self.calculate_charge_percentage()

        done = self.current_step > self.max_steps
        if done:
            self.reset()
        if self.current_charge == 0 and self.current_savings == 0:
            reward = -1000
            self.reset()

        observation = self.get_observation()
        return observation.astype(np.float32), reward, done, {}

    def get_observation(self):
        # Return the current state as an observation
        return np.concatenate([self.df.iloc[self.current_step].values, [self.current_charge], [self.current_savings]])

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        self.current_charge = 500
        self.current_savings = 0
        self.reward_history = [0]
        self.savings_history = [0]
        self.battery_charge_history = [50]
        self.charge_percentages = []
        self.buy_sell_history = []

        # Return the initial observation
        initial_observation = self.get_observation()
        return initial_observation.astype(np.float32)

    def render(self, mode='human'):
        # Create a figure with two subplots
        fig, axs = plt.subplots(3)

        # Plot the battery charge level in the second subplot
        axs[0].plot(self.battery_charge_history[:self.current_step])
        axs[0].set_xlabel('Time step')
        axs[0].set_ylabel('Battery Charge Level')
        axs[0].set_title('Battery Charge Level Over Time')

        # calculate the average reward over 100 steps and plot it
        avg_reward = []
        for i in range(len(self.reward_history)):
            if i % 100 == 0:
                avg_reward.append(np.mean(self.reward_history[i:i + 100]))
        axs[1].plot(avg_reward)
        axs[1].set_xlabel('Time step')
        axs[1].set_ylabel('Average Reward')
        axs[1].set_title('Average Reward Over Time')



        # Plot the buy/sell history in the third subplot as a barl plot starting from 0 and positive for buying and
        # negative for selling also use color green for buying and red for selling
        axs[2].bar(range(len(self.buy_sell_history)), self.buy_sell_history,
                   color=['red' if x < 0 else 'green' for x in self.buy_sell_history])
        axs[2].set_xlabel('Time step')
        axs[2].set_ylabel('Buy/Sell')
        axs[2].set_title('Buy/Sell Over Time')

        plt.tight_layout()
        fig.set_size_inches(18.5, 10.5)
        plt.show()

        fig, axs = plt.subplots(2)
        # Plot the savings in the third subplot
        axs[0].plot(self.savings_history)
        axs[0].set_xlabel('Time step')
        axs[0].set_ylabel('Savings')
        axs[0].set_title('Savings Over Time')

        # Plot the charge percentage in the fourth subplot
        axs[1].plot(self.charge_percentages)
        axs[1].set_xlabel('Time step')
        axs[1].set_ylabel('Charge Percentage')
        axs[1].set_title('Charge Percentage Over Time')

        # make more space between the subplots and the figure bigger
        plt.tight_layout()
        fig.set_size_inches(18.5, 10.5)

        # Show the figure
        plt.show()

    def calculate_charge_percentage(self):
        self.charge_percentages.append(self.current_charge / self.max_battery_charge * 100)

    def log_trade(self, price, amount, trade_type, date):
        self.trade_log.append([price, amount, trade_type, date, self.market.get_current_price()])
    def get_trade_log(self):
        # return the trade log as a pandas dataframe
        return pd.DataFrame(self.trade_log, columns=['price', 'amount', 'trade_type', 'date', 'market_price'])

    def extra_reward_step(self, action):

        # Execute one time step within the environment
        self.current_step += 1
        self.market.step()

        if self.current_step >= len(self.df):
            # If it does, reset the current step to 0
            self.current_step = 0

        # Extract the components of the action
        price = action[0]
        charge_discharge = action[1]
        amount = action[2]

        # Initialize the reward to 0
        reward = 0

        # Here, we implement the logic of buying, selling or doing nothing
        if charge_discharge == 0:  # buy
            if price > self.current_savings or amount > self.max_battery_charge - self.current_charge:
                reward = -10
            elif self.market.accept_offer(price):
                self.buy_sell_history.append(-1)
                if self.current_savings - price < 0:  # Check if the agent would lose too much savings
                    reward = -20  # Hard penalty
                else:
                    self.current_savings -= price
                    self.current_charge += amount
                    # Less reward for buying when battery is charged
                    reward = price * amount * (1 - self.current_charge / self.max_battery_charge)
                    if amount > 0.5 * self.max_battery_charge:  # Bonus for large amounts
                        reward += 10
                        # Penalty for buying when battery is highly charged
                    if self.current_charge > 0.9 * self.max_battery_charge:
                        reward -= 10
                    if self.current_charge < 0.1 * self.max_battery_charge:  # Bonus for buying when battery is low
                        reward += 10

        elif charge_discharge == 1:  # sell
            if self.current_charge - amount < 0:  # check if the battery can be discharged
                reward = -10
            elif self.market.accept_offer(price):
                self.buy_sell_history.append(1)
                self.current_savings += price
                self.current_charge -= amount
                # More reward for selling when battery is charged
                reward = price * amount * (self.current_charge / self.max_battery_charge)
                if amount > 0.5 * self.max_battery_charge:  # Bonus for large amounts
                    reward += 10
                    # Bonus for selling when battery is highly charged
                if self.current_charge > 0.9 * self.max_battery_charge:
                    reward += 10
                if self.current_charge < 0.1 * self.max_battery_charge:  # Penalty for selling when battery is low
                    reward -= 10

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
