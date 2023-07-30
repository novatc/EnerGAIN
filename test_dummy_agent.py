import unittest
from unittest.mock import Mock

import pandas as pd

from dummy_agent import TradingBot
from market import Market


class TestTradingBot(unittest.TestCase):
    def setUp(self):
        dataset = pd.read_csv('data/clean/test_set.csv')
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

    def test_get_signal_buy(self):
        self.market.set_current_price(-0.89)
        self.assertEqual(self.bot.get_signal()[0], 'buy')

    def test_get_signal_sell(self):
        self.market.get_current_price.return_value = 0.9
        self.assertEqual(self.bot.get_signal()[0], 'sell')

    def test_get_signal_hold(self):
        self.market.get_current_price.return_value = 0
        self.assertEqual(self.bot.get_signal()[0], 'hold')

if __name__ == '__main__':
    unittest.main()
