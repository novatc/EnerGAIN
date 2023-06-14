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
        self.df = pd.read_csv('data/clean/env_data.csv')

        self.max_steps = 37272
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

        # Here, we implement the logic of buying, selling or doing nothing
        if charge_discharge == 0:  # buy
            if price > self.current_savings or amount > self.max_battery_charge - self.current_charge:
                reward = -100
                return self.get_observation().astype(np.float32), reward, False, {}

            elif self.market.accept_offer(price, 'buy'):
                self.buy_sell_history.append(-1)
                if self.current_savings - price < 0:  # Check if the agent would lose too many savings
                    reward = -100  # Hard penalty
                else:
                    self.current_savings -= price * amount
                    self.current_charge += amount
                    reward = (price * amount) + 10  # Reward based on the profit
                    self.log_trade(price, amount, 'buy', self.current_step, reward)

        elif charge_discharge == 1:
            if amount <= 0 or price < 0:
                reward = -100
                return self.get_observation().astype(np.float32), reward, False, {}

            # sell
            if self.current_charge - amount < 0:  # check if the battery can be discharged
                reward = -100
                return self.get_observation().astype(np.float32), reward, False, {}

            elif self.market.accept_offer(price, 'sell'):
                self.buy_sell_history.append(1)
                self.current_savings += price * amount
                self.current_charge -= amount
                reward = (price * amount) + 10  # Reward based on the profit
                self.log_trade(price, amount, 'sell', self.current_step, reward)


        elif charge_discharge == 2:  # do nothing
            reward = 0  # No reward for doing nothing

        # Append the current charge level to the history
        self.battery_charge_history.append(self.current_charge)
        self.reward_history.append(reward)
        self.savings_history.append(self.current_savings)
        self.calculate_charge_percentage()

        done = self.current_step >= self.max_steps -1
        if done:
            print("Episode finished after {} timesteps".format(self.current_step + 1))
            self.render()


        observation = self.get_observation()
        return observation.astype(np.float32), reward, done, {}

    def get_observation(self):
        # Return the current state as an observation
        return np.concatenate([self.df.iloc[self.current_step].values, [self.current_charge], [self.current_savings]])

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        self.current_charge = 500
        self.current_savings = 50
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
        axs[0].plot(self.battery_charge_history)
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

        # Plot the buy/sell history in the third subplot as a barl plot starting from 0 and negative for buying and
        # positive for selling also use color green for buying and red for selling
        axs[2].bar(range(len(self.buy_sell_history)), self.buy_sell_history,
                   color=['green' if x < 0 else 'red' for x in self.buy_sell_history])
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

    def log_trade(self, price, amount, trade_type, date, reward):
        self.trade_log.append([price, amount, trade_type, date, self.market.get_current_price(), reward])

    def get_trade_log(self):
        # return the trade log as a pandas dataframe
        return pd.DataFrame(self.trade_log, columns=['price', 'amount', 'trade_type', 'date', 'market_price', 'reward'])
