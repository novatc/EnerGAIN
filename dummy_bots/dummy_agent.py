import pandas as pd

from envs.assets.market import Market


class TradingBot:
    def __init__(self, market, buy_threshold, sell_threshold):
        self.market = market
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.successful_trades = []
        self.current_charge = 500
        self.max_amount = 1000
        self.min_amount = 0
        self.profit = 0

    def check_constrains(self, intended_amount, trade_intend) -> bool:
        """
        Check if the bot can trade the intended amount
        :param intended_amount: the bot intends to trade
        :param trade_intend: the bot intends to buy or sell
        :return:
        """
        if trade_intend == 'sell' and intended_amount > self.current_charge:
            return False
        if trade_intend == 'buy' and intended_amount + self.current_charge > self.max_amount:
            return False
        return True

    def get_signal(self):
        """
        Get the signal from the bot based on the current price
        :return: the signal and the amount, if the bot intends to trade
        """
        current_price = self.market.get_current_price()
        return_amount = 0
        if current_price < self.buy_threshold:
            intent = 'buy'
            current_price += 0.01
            buy_amount = 500
            if self.check_constrains(buy_amount, trade_intend='buy') and self.market.accept_offer(current_price, intent):
                self.current_charge += abs(buy_amount)
                self.profit -= current_price * abs(buy_amount)
                return_amount = buy_amount
                self.successful_trades.append((current_price, buy_amount, intent))

        if current_price > self.sell_threshold:
            intent = 'sell'
            current_price -= 0.01
            sell_amount = 500
            # make sure the bot doesn't sell more than it has
            if self.check_constrains(sell_amount, trade_intend='sell') and self.market.accept_offer(current_price, 'sell'):
                self.current_charge -= abs(sell_amount)
                self.profit += current_price * abs(sell_amount)
                return_amount = sell_amount
                self.successful_trades.append((current_price, sell_amount, intent))
        else:
            intent = 'hold'

        return intent, return_amount

    def get_successful_trades(self):
        return self.successful_trades


# Initialize the Market and the bot
#     Mean Price: ≈0.195
#     Median Price: ≈0.167
#     Standard Deviation: ≈0.123
#     Minimum Price: ≈−0.004
#     Maximum Price: ≈0.665

sell_threshold = 0.318
buy_threshold = 0.0720

data = pd.read_csv("../data/in-use/unscaled_eval_data.csv")
market = Market(data)
bot = TradingBot(market, sell_threshold=sell_threshold, buy_threshold=buy_threshold)

for _ in range(data.shape[0]):
    signal, amount = bot.get_signal()
    print(signal, amount)
    market.step()  # move the market to the next step
print(bot.get_successful_trades())
print(bot.profit)
