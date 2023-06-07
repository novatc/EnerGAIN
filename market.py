class Market:
    def __init__(self, dataset):
        self.dataset = dataset
        self.current_step = 0

    def get_current_price(self):
        # Assuming 'price' is a column in your dataset
        return self.dataset.iloc[self.current_step]['price']

    def step(self):
        self.current_step += 1

    def accept_offer(self, offer_price):
        # Accept the offer if the offer price is less than or equal to the current market price
        return offer_price <= self.get_current_price()
