import random

import numpy as np
import pandas as pd


class FrequencyContainmentReserve:
    def __init__(self, dataset: pd.DataFrame):
        """
        Create a new Frequency Containment Reserve.
        :param dataset: the given data for the market data.
        """

        self.dataset = dataset
        self.current_step = 0
        self.steps_since_last_random_start = 0
        self.current_price = 0

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
        Increment the current step by 1. If the current step is equal to the length of the dataset, reset the first step
        """
        self.current_step = (self.current_step + 1) % len(self.dataset)

    def get_current_step(self):
        """
        Get the current step.

        :return: the current step
        """
        return self.current_step

    def random_walk(self, sequence_length=24 * 30):
        """
        Choose a random starting position and increment the current step by sequence_length from
        that position.
        After completing the sequence_length steps, select a new random starting position.

        :param sequence_length: the length of the sequence to increment the current step by.
        """
        truncated = False
        # If this is the first call or 120 steps have been taken since the last random start,
        # choose a new random starting position
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

    def accept_offer(self, offer_price):
        """
        Accept or reject an offer based on the price and current market price.

        :param offer_price: the offer price for the trade.
        :return: True if the offer is accepted, False otherwise.
        """
        current_price = self.get_current_price()
        if offer_price > current_price:
            return True
        else:
            return False


