import numpy as np
import pandas as pd
from collections import deque

from envs.assets.dayahead import DayAhead


class SimpleThresholdBot:
    def __init__(self, market_sim, buy_threshold, sell_threshold, moving_average_window,
                 initial_money=50.0, initial_inventory=500, max_inventory=1000, unit_buy_sell=50):
        """ Initialize the SimpleThresholdBot with given parameters. """
        self.market = market_sim
        self.money = initial_money
        self.inventory = initial_inventory
        self.max_inventory = max_inventory
        self.unit_buy_sell = unit_buy_sell
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.moving_average_window = moving_average_window
        self.prices_window = deque(maxlen=moving_average_window)
        self.money_over_time = [self.money]
        self.inventory_over_time = [self.inventory]
        self.trade_log = []

    def log_trade(self, step, market_price, offer_amount, trade_type, reason):
        if trade_type == 'buy':
            profit = -1 * (float(market_price) * offer_amount)
        else:
            profit = float(market_price) * offer_amount
        self.trade_log.append((step, trade_type, self.market.get_current_price(), market_price, offer_amount,
                               profit, reason))

    def update_moving_average(self, current_price):
        """ Update the moving average with the current price. """
        self.prices_window.append(current_price)
        return sum(self.prices_window) / len(self.prices_window)

    def trade(self):
        """ Perform a trade based on the moving average. """
        step = self.market.current_step
        current_price = self.market.get_current_price()
        moving_avg_price = self.update_moving_average(current_price)

        # Buy logic based on moving average
        if (moving_avg_price <= self.buy_threshold and self.money >= self.unit_buy_sell * current_price
                and self.inventory + self.unit_buy_sell <= self.max_inventory):
            adjusted_buy_price = current_price * 1.1  # 10% higher to ensure market acceptance
            if self.market.accept_offer(adjusted_buy_price, 'buy'):
                self.money -= self.unit_buy_sell * current_price
                self.inventory += self.unit_buy_sell
                self.log_trade(step, current_price, self.unit_buy_sell, 'buy', 'moving_avg')

        # Sell logic based on moving average
        elif (moving_avg_price >= self.sell_threshold and self.inventory >= self.unit_buy_sell):
            adjusted_sell_price = current_price * 0.9  # 10% lower to ensure market acceptance
            if self.market.accept_offer(adjusted_sell_price, 'sell'):
                self.money += self.unit_buy_sell * current_price
                self.inventory -= self.unit_buy_sell
                self.log_trade(step, adjusted_sell_price, self.unit_buy_sell, 'sell', 'moving_avg')

        self.money_over_time.append(self.money)
        self.inventory_over_time.append(self.inventory)
        self.market.current_step += 1

    def run_simulation(self):
        """ Run a trading simulation. """
        for _ in range(len(self.market.dataset)):
            self.trade()
        return self.money, self.inventory

    def get_trades(self):
        """ Get a log of all trades. """
        return self.trade_log


# Example usage
# Assuming DayAhead is defined elsewhere and provides market simulation
market = DayAhead(pd.read_csv('../data/in-use/average_da_year.csv'))
moving_average_window = 5  # For example, a 5-step moving average

bot = SimpleThresholdBot(market, buy_threshold=0.067847, sell_threshold=0.126536,
                         moving_average_window=moving_average_window, initial_money=50.0,
                         initial_inventory=500, max_inventory=1000, unit_buy_sell=50)

money, inventory = bot.run_simulation()
print(f"Money: {money}, Inventory: {inventory}")

trades = bot.get_trades()
trades_log = pd.DataFrame(trades, columns=["step", "type", "market price", "offered_price", "amount", "reward", "case"])
trades_log.to_csv("../trade_logs/moving_average_bot_trades.csv", index=False)
