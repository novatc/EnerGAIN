import random
import numpy as np
import pandas as pd

from collections import deque


class DayAhead:
    def __init__(self, dataset):
        """
        Initialize the Market with a given dataset.

        :param dataset: the given data for the market prices.
        """
        self.dataset = dataset
        self.current_step = 0
        self.steps_since_last_random_start = 0
        self.current_price = 0
        self.price_history = deque(maxlen=10)  # 'n' is the number of steps for averaging

    def get_current_price(self):
        """
        Get the current market price based on the current step.

        :return: the current market price
        """
        return self.dataset.iloc[self.current_step]['price']

    def get_current_mock_price(self):
        """
        Get the current market price

        :return: the current mocked market price
        """
        return self.current_price

    def reset(self):
        """
        Reset the current step to 0.
        """
        self.current_step = 0

    def step(self):
        """
        Increment the current step by 1. If the current step is equal to the length of the dataset, reset the first step
        """
        self.current_step = (self.current_step + 1) % len(self.dataset)
        current_price = self.get_current_price()
        self.price_history.append(current_price)

    def set_step(self, step):
        """
        Set the current step to the given step.

        :param step: the given step.
        """
        self.current_step = step

    def get_current_step(self):
        """
        Get the current step.

        :return: the current step
        """
        return self.current_step

    def get_price_at_step(self, step):
        """
        Get the market price at a given step.

        :param step: the given step.
        :return: the market price at the given step.
        """
        return self.dataset.iloc[step]['price']

    def get_average_price(self):
        """
        Calculate the exponentially weighted average price.

        The most recent price has the highest weight and the weight decreases
        exponentially for older prices.

        :return: The exponentially weighted average price.
        """
        if not self.price_history:
            return 0

        weights = np.exp(np.linspace(-1, 0, len(self.price_history)))
        weights /= weights.sum()

        weighted_average = np.dot(np.array(self.price_history), weights)
        return weighted_average

    def random_walk(self, sequence_length: int):
        """
        Choose a random starting position and increment the current step by sequence_length (90 days) from
        that position.
        After completing the sequence_length (90 days) steps, select a new random starting position.

        :param sequence_length: the length of the sequence to increment the current step by.
        """
        # If this is the first call or 120 steps have been taken since the last random start,
        # choose a new random starting position
        truncated = False

        if self.current_step == 0 or self.steps_since_last_random_start >= sequence_length:
            truncated = True
            dataset_length = len(self.dataset)
            self.current_step = random.randint(0, dataset_length - 1)
            self.steps_since_last_random_start = 0  # Reset the step counter

        # Increment the current step by 120 starting from the current position

        self.current_step = (self.current_step + 1) % len(self.dataset)
        self.steps_since_last_random_start += 1  # Increment the step counter
        return truncated

    def previous_hours(self, hours: int) -> np.array:
        """
        Get the previous hours of data from the current step.

        :param hours: the number of hours to get the data from.
        :return: the indexes of the previous hours of data.
        """
        index = self.current_step

        if index < hours:
            # If n is larger than the random index, create a DataFrame with n - idx rows filled with zeros
            zero_data = pd.DataFrame(np.zeros((hours - index, len(self.dataset.columns))), columns=self.dataset.columns)

            # Select the min(hours, index) rows before the current index in reverse order
            selected_data = self.dataset.iloc[:index][::-1].copy()

            # Concatenate the zero data and selected data into one dara
            selected_data = pd.concat([zero_data, selected_data])
        else:
            # Select the n rows before the current index, in reverse order
            selected_data = self.dataset.iloc[index - hours:index][::-1].copy()

        # put everything together
        concatenated_row = pd.concat([selected_data.iloc[i] for i in range(hours)])

        # just return the values
        return concatenated_row.values

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
            return True
        elif intent == 'sell' and offer_price < current_price:
            return True
        return False

    def set_current_price(self, price):
        self.current_price = price
