import pandas as pd

from market import Market


class TradingBot:
    def __init__(self, market, buy_threshold, sell_threshold):
        self.market = market
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.successful_trades = []
        self.current_charge = 0.5
        self.max_amount = 1
        self.min_amount = 0
        self.profit = 0

    def check_constrains(self, intended_amount, trade_intend) -> bool:
        if trade_intend == 'sell' and intended_amount > self.current_charge:
            return False
        if trade_intend == 'buy' and intended_amount + self.current_charge > self.max_amount:
            return False
        return True

    def get_signal(self):
        current_price = self.market.get_current_price()

        amount = None  # Initialize amount to None

        if current_price < self.buy_threshold:
            intent = 'buy'
            current_price += 0.01
            buy_amount = 0.5
            if self.check_constrains(buy_amount, trade_intend='buy'):
                if self.market.accept_offer(current_price, intent):
                    self.current_charge += abs(buy_amount)
                    self.profit -= current_price * abs(buy_amount)
                    self.successful_trades.append((current_price, buy_amount, intent))

        if current_price > self.sell_threshold:
            intent = 'sell'
            current_price -= 0.01
            sell_amount = 0.5
            # make sure the bot doesn't sell more than it has
            if self.check_constrains(sell_amount, trade_intend='sell'):
                if self.market.accept_offer(current_price, 'sell'):
                    self.current_charge -= abs(sell_amount)
                    self.profit += current_price * abs(sell_amount)
                    self.successful_trades.append((current_price, sell_amount, intent))
        else:
            intent = 'hold'

        return intent, amount

    def get_mock_signal(self):
        current_price = self.market.get_current_mock_price()

        amount = None  # Initialize amount to None

        if current_price < self.buy_threshold:
            intent = 'buy'
            current_price += 0.01
            buy_amount = 0.5
            if self.check_constrains(buy_amount, trade_intend='buy'):
                if self.market.accept_offer(current_price, intent):
                    self.current_charge += abs(buy_amount)
                    self.profit -= current_price * abs(buy_amount)
                    self.successful_trades.append((current_price, buy_amount, intent))

        elif current_price > self.sell_threshold:
            intent = 'sell'
            current_price -= 0.01
            sell_amount = 0.5
            # make sure the bot doesn't sell more than it has
            if self.check_constrains(sell_amount, trade_intend='sell'):
                if self.market.accept_offer(current_price, 'sell'):
                    self.current_charge -= abs(sell_amount)
                    self.profit += current_price * abs(sell_amount)
                    self.successful_trades.append((current_price, sell_amount, intent))
        else:
            intent = 'hold'

        return intent, amount

    def get_successful_trades(self):
        return self.successful_trades


# Initialize the Market and the bot
sell_threshold = -0.70
buy_threshold = -0.78

data = pd.read_csv("data/clean/test_set.csv")
market = Market(data)
bot = TradingBot(market, sell_threshold=sell_threshold, buy_threshold=buy_threshold)

# Test the bot with the first 100 steps of data
for _ in range(data.shape[0]):
    signal, amount = bot.get_signal()
    market.step()  # move the market to the next step
print(bot.get_successful_trades())
print(bot.profit)
