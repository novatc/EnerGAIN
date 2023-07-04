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
        :param offer_price: the price of the offer
        :return: True if the offer was accepted, False otherwise
        """
        # Get the current market price
        current_price = self.get_current_price()
        if intent == 'buy':
            # print("The buy offer was accepted because the offer price was lower than the current price: ", offer_price,
            #     current_price)
            return offer_price > current_price
        elif intent == 'sell':
            # print("The buy offer was accepted because the offer price was lower than the current price: ", offer_price,
            #     current_price)
            return offer_price < current_price
        # print("The offer was not accepted because the price was not lower than the current price: ", offer_price,
        #     current_price)
        return False
