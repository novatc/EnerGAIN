import numpy as np
import pandas as pd

from envs.assets.market import Market


class MovingAvgBot:
    def __init__(self, market, initial_money=50.0, initial_inventory=0, max_inventory=1000,
                 unit_buy_sell=500, short_window=5, long_window=20):
        "Initialize the DummyBot with given parameters."
        self.market = market
        self.money = initial_money
        self.inventory = initial_inventory
        self.max_inventory = max_inventory
        self.unit_buy_sell = unit_buy_sell
        self.short_window = short_window
        self.long_window = long_window
        self.money_over_time = [self.money]
        self.inventory_over_time = [self.inventory]
        self.short_moving_avg = []
        self.long_moving_avg = []

    def calculate_moving_averages(self):
        """Calculate the short-term and long-term moving averages."""
        prices = np.array(self.market.dataset['price'][:self.market.current_step + 1])
        if len(prices) >= self.long_window:
            self.short_moving_avg.append(prices[-self.short_window:].mean())
            self.long_moving_avg.append(prices[-self.long_window:].mean())
        else:
            self.short_moving_avg.append(None)
            self.long_moving_avg.append(None)

    def trade(self):
        """Perform a trade based on the moving average crossover strategy."""
        self.calculate_moving_averages()

        # Ensure enough data points are available for both moving averages
        if (self.short_moving_avg[-1] is not None and self.long_moving_avg[-1] is not None and
                self.short_moving_avg[-2] is not None and self.long_moving_avg[-2] is not None):
            buy_signal = self.short_moving_avg[-1] > self.long_moving_avg[-1] and self.short_moving_avg[-2] <= \
                         self.long_moving_avg[-2]
            sell_signal = self.short_moving_avg[-1] < self.long_moving_avg[-1] and self.short_moving_avg[-2] >= \
                          self.long_moving_avg[-2]

            # Check if we should buy
            if buy_signal and self.money >= self.unit_buy_sell * self.short_moving_avg[-1]:
                self.money -= self.unit_buy_sell * self.short_moving_avg[-1]
                self.inventory += self.unit_buy_sell
                if self.inventory > self.max_inventory:
                    excess_units = self.inventory - self.max_inventory
                    self.inventory = self.max_inventory
                    self.money += excess_units * self.short_moving_avg[-1]

            # Check if we should sell
            elif sell_signal and self.inventory >= self.unit_buy_sell:
                self.money += self.unit_buy_sell * self.short_moving_avg[-1]
                self.inventory -= self.unit_buy_sell

        # Record the state for this time
        self.money_over_time.append(self.money)
        self.inventory_over_time.append(self.inventory)
        self.market.current_step += 1  # Move to the next time step

    def run_simulation(self):
        """Run a trading simulation based on the moving average crossover strategy."""
        for _ in self.market.dataset['price']:
            self.trade()
        return self.money, self.inventory


market = Market(pd.read_csv('../data/in-use/unscaled_eval_data.csv'))
bot = MovingAvgBot(market, initial_money=50.0, initial_inventory=500, max_inventory=1000)
money, inventory = bot.run_simulation()
print(f"Money: {money}, Inventory: {inventory}")