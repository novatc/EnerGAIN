import pandas as pd

from market import Market


class TradingBot:
    def __init__(self, market, buy_threshold, sell_threshold):
        self.market = market
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.past_consumption = []
        self.max_consumption_history = 5
        self.successful_trades = []
        self.current_charge = 0.5
        self.max_amount = 1
        self.min_amount = 0
        self.profit = 0

    def update_state(self, current_prediction):
        self.past_consumption.append(current_prediction)

        if len(self.past_consumption) > self.max_consumption_history:
            self.past_consumption.pop(0)

    def check_constrains(self, intended_amount, trade_intend) -> bool:
        if trade_intend == 'sell' and intended_amount > self.current_charge:
            print('Selling more than the batterie has')
            return False
        if trade_intend == 'buy' and intended_amount + self.current_charge > self.max_amount:
            print('Buying more than the batterie can hold')
            return False
        return True

    def get_signal(self):
        current_price = self.market.get_current_price()
        current_prediction = self.market.dataset.iloc[self.market.get_current_step()]['consumption']
        self.update_state(current_prediction)

        amount = None  # Initialize amount to None

        if current_price < self.buy_threshold:
            print(self.current_charge)
            intent = 'buy'
            current_price += 0.01
            buy_amount = sum(self.past_consumption) / len(self.past_consumption)
            if self.check_constrains(buy_amount, trade_intend='buy'):
                if self.market.accept_offer(current_price, intent):
                    self.current_charge += abs(buy_amount)
                    self.profit -= current_price * abs(buy_amount)
                    self.successful_trades.append((current_price, buy_amount, intent))

        elif current_price > self.sell_threshold and len(self.past_consumption) >= self.max_consumption_history:
            print(self.current_charge)
            intent = 'sell'
            current_price -= 0.01
            sell_amount = sum(self.past_consumption) / len(self.past_consumption)
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
sell_threshold = 0.3
buy_threshold = -0.3

data = pd.read_csv("data/clean/test_set.csv")
market = Market(data)
bot = TradingBot(market, sell_threshold=sell_threshold, buy_threshold=buy_threshold)

# Test the bot with the first 100 steps of data
for _ in range(data.shape[0]):
    signal, amount = bot.get_signal()
    market.step()  # move the market to the next step
print(bot.get_successful_trades())
print(bot.profit)
