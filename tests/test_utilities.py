import unittest
import pandas as pd
import numpy as np
from envs.assets import env_utilities as utilities


class TestUtilities(unittest.TestCase):
    def test_rescale_value_price(self):
        random_index = np.random.randint(0, 120)
        og_price = pd.read_csv("../data/in-use/test_data.csv")
        og_price_value = og_price['price'].iloc[random_index]
        scaled_price = pd.read_csv("../data/in-use/eval_data.csv")
        scaled_price_value = scaled_price['price'].iloc[random_index]
        rescaled_price_value = utilities.rescale_value_price(scaled_price_value, test_mode=True)
        self.assertAlmostEqual(og_price_value, rescaled_price_value, places=5)

    def test_rescale_list_price(self):
        og_price = pd.read_csv("../data/in-use/test_data.csv")
        scaled_price = pd.read_csv("../data/in-use/eval_data.csv")
        rescaled_price = utilities.rescale_list_price(scaled_price['price'].values, test_mode=True)

        # pick a couple random indices to test
        random_indices = np.random.randint(0, 120, 5)
        for index in random_indices:
            og_price_value = og_price['price'].iloc[index]
            rescaled_price_value = rescaled_price[index]
            self.assertAlmostEqual(og_price_value, rescaled_price_value, places=5)

    def test_rescale_value_amount(self):
        og_amount = pd.read_csv("../data/in-use/test_data.csv")
        scaled_amount = pd.read_csv("../data/in-use/eval_data.csv")
        rescaled_amount = utilities.rescale_value_amount(scaled_amount['consumption'].iloc[0])
        self.assertAlmostEqual(og_amount['consumption'].iloc[0], rescaled_amount, places=5)

    def test_rescale_list_amount(self):
        og_amount = pd.read_csv("../data/in-use/test_data.csv")
        scaled_amount = pd.read_csv("../data/in-use/eval_data.csv")
        rescaled_amount = utilities.rescale_list_amount(scaled_amount['consumption'].values)

        random_indices = np.random.randint(0, 120, 5)
        for index in random_indices:
            og_amount_value = og_amount['consumption'].iloc[index]
            rescaled_amount_value = rescaled_amount[index]
            self.assertAlmostEqual(og_amount_value, rescaled_amount_value, places=5)


if __name__ == '__main__':
    unittest.main()
