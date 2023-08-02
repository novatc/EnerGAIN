import unittest

import pandas as pd

from dummy_agent import TradingBot
from envs.assets.market import Market


class TestTradingBot(unittest.TestCase):
    def setUp(self):
        dataset = pd.read_csv('../data/clean/test_set.csv')
        self.market = Market(dataset)
        self.bot = TradingBot(self.market, sell_threshold=0.8, buy_threshold=-0.8)

    def test_update_state(self):
        self.bot.update_state(0.5)
        self.assertEqual(self.bot.past_consumption[-1], 0.5)
        self.bot.update_state(0.6)
        self.assertEqual(self.bot.past_consumption[-1], 0.6)

    def test_check_constrains(self):
        self.assertTrue(self.bot.check_constrains(0.4, 'buy'))
        self.assertFalse(self.bot.check_constrains(1.2, 'buy'))
        self.assertTrue(self.bot.check_constrains(0.4, 'sell'))
        self.assertFalse(self.bot.check_constrains(1.2, 'sell'))

    def test_buy_signal(self):
        # Set the current market price to a value below the buy threshold
        self.market.set_current_price(-0.9)
        # Call the get_signal method
        intent, _ = self.bot.get_signal()
        # Check if the bot intends to buy
        self.assertEqual(intent, 'buy')

    def test_sell_signal(self):
        # Set the current market price to a value above the sell threshold
        self.market.set_current_price(0.9)
        # Call the get_signal method
        intent, _ = self.bot.get_signal()
        # Check if the bot intends to sell
        self.assertEqual(intent, 'sell')

    def test_hold_signal(self):
        # Set the current market price to a value between the buy and sell thresholds
        self.market.set_current_price(0)
        # Call the get_signal method
        intent, _ = self.bot.get_signal()
        # Check if the bot intends to hold
        self.assertEqual(intent, 'hold')

if __name__ == '__main__':
    unittest.main()
