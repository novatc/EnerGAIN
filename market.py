class Market:
    def __init__(self, dataset):
        self.dataset = dataset
        self.current_step = 0

    def get_current_price(self):
        # Get the price at the current step
        # If the current step exceeds the length of the dataset, reset the current step to 0
        if self.current_step >= len(self.dataset):
            self.current_step = 0
        return self.dataset.iloc[self.current_step]['price']

    def reset(self):
        self.current_step = 0

    def step(self):
        self.current_step += 1

    def accept_offer(self, offer_price, intent):
        """
        A buy bid is accepted when its price is not below the market
        price for its hour. A sell bid is accepted when its price is not above the market price for its hour.
        :param intent: buy or sell
        :param offer_price:
        :return:
        """
        # Get the current market price
        current_price = self.get_current_price()
        if intent == 'buy':
            return offer_price > current_price
        elif intent == 'sell':
            return offer_price < current_price
