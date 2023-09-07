import numpy as np
import pandas as pd

from envs.assets.market import Market


class SimpleThresholdBot:
    def __init__(self, market_sim, buy_threshold, sell_threshold, initial_money=50.0,
                 initial_inventory=0, max_inventory=1000, unit_buy_sell=500):
        "Initialize the SimpleThresholdBot with given parameters."
        self.market = market_sim
        self.money = initial_money
        self.inventory = initial_inventory
        self.max_inventory = max_inventory
        self.unit_buy_sell = unit_buy_sell
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.money_over_time = [self.money]
        self.inventory_over_time = [self.inventory]
        self.trade_log = []

    def log_trade(self, step, offer_price, offer_amount, trade_type):
        self.trade_log.append((step, offer_price, offer_amount, trade_type,
                               abs(float(self.market.get_current_price()) * offer_amount)))

    def trade(self):
        "Perform a trade based on the set thresholds."
        step = self.market.current_step
        current_price = self.market.get_current_price()

        # Check if we should buy
        if current_price <= self.buy_threshold and self.money >= self.unit_buy_sell * current_price:
            adjusted_buy_price = current_price * 1.01  # 1% higher to ensure market acceptance
            if self.market.accept_offer(adjusted_buy_price, 'buy'):
                self.money -= self.unit_buy_sell * current_price  # Pay the actual current price
                self.inventory += self.unit_buy_sell
                if self.inventory > self.max_inventory:
                    excess_units = self.inventory - self.max_inventory
                    self.inventory = self.max_inventory
                    self.money += excess_units * current_price
                self.log_trade(step, current_price, self.unit_buy_sell, 'buy')

        # Check if we should sell
        elif current_price >= self.sell_threshold and self.inventory >= self.unit_buy_sell:
            adjusted_sell_price = current_price * 0.99  # 1% lower to ensure market acceptance
            if self.market.accept_offer(adjusted_sell_price, 'sell'):
                self.money += self.unit_buy_sell * current_price  # Receive the actual current price
                self.inventory -= self.unit_buy_sell
                self.log_trade(step, current_price, self.unit_buy_sell, 'sell')

        # Record the state for this time
        self.money_over_time.append(self.money)
        self.inventory_over_time.append(self.inventory)
        self.market.current_step += 1  # Move to the next time step

    def run_simulation(self):
        "Run a trading simulation based on the set thresholds."
        for _ in self.market.dataset['price']:
            self.trade()
        return self.money, self.inventory

    def get_trades(self):
        return self.trade_log


# Example usage
#print current working directory
import os
print(os.getcwd())

market = Market(pd.read_csv('data/in-use/unscaled_eval_data.csv'))
bot = SimpleThresholdBot(market, buy_threshold=10.0, sell_threshold=15.0, initial_money=50.0,
                         initial_inventory=500, max_inventory=1000, unit_buy_sell=50)
money, inventory = bot.run_simulation()
print(f"Money: {money}, Inventory: {inventory}")

trades = bot.get_trades()
trades_log = pd.DataFrame(trades, columns=["step", "price", "amount", "trade_type", "reward"])
trades_log.to_csv(f"trade_logs/simple_threshold_bot_trades.csv", index=False)
