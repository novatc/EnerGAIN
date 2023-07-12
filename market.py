import random
import numpy as np


class Market:
    def __init__(self, dataset):
        """
        Initialize the Market with a given dataset.

        :param dataset: the given data for the market prices.
        """
        self.successful_trades = []  # List to store successful trades
        self.dataset = dataset
        self.current_step = 0
        self.steps_since_last_random_start = 0

    def get_current_price(self):
        """
        Get the current market price based on the current step.

        :return: the current market price
        """
        return self.dataset.iloc[self.current_step]['price']

    def reset(self):
        """
        Reset the current step to 0.
        """
        self.current_step = 0

    def step(self):
        """
        Increment the current step by 1.
        """
        self.current_step += 1

    def get_current_step(self):
        """
        Get the current step.

        :return: the current step
        """
        return self.current_step

    def week_walk(self):
        """
        Choose a random starting position and increment the current step by 1200 (5 days) from that position.
        After completing the 1200 steps, select a new random starting position.
        """
        # If this is the first call or 120 steps have been taken since the last random start,
        # choose a new random starting position
        if self.current_step == 0 or self.steps_since_last_random_start >= 120:
            dataset_length = len(self.dataset)
            self.current_step = random.randint(0, dataset_length - 1)
            self.steps_since_last_random_start = 0  # Reset the step counter

        # Increment the current step by 120 starting from the current position
        for _ in range(24 * 5):
            self.current_step = (self.current_step + 1) % len(self.dataset)
            self.steps_since_last_random_start += 1  # Increment the step counter

    def accept_offer(self, offer_price, intent):
        """
        Accept or reject a buy/sell offer based on the offer price and current market price.
        If the offer is accepted, the trade details are logged into the successful_trades list.

        :param offer_price: the offer price for the trade.
        :param intent: the intent of the trade, either 'buy' or 'sell'.
        :return: True if the offer is accepted, False otherwise.
        """
        current_price = self.get_current_price()
        if intent == 'buy' and offer_price > current_price:
            self.successful_trades.append((self.current_step, 'buy', offer_price))  # log the trade
            return True
        elif intent == 'sell' and offer_price < current_price:
            self.successful_trades.append((self.current_step, 'sell', offer_price))  # log the trade
            return True
        return False
