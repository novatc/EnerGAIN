class Battery:
    def __init__(self, capacity, soc):
        """
        Create a new battery with the given capacity and state of charge.

        :param capacity: the maximum amount of energy the battery can store
        :param soc: the current state of charge of the battery
        """
        self.capacity = capacity
        self.soc = soc
        self.charge_rate = 0.925
        self.discharge_rate = 0.925
        self.charge_log = []

    def charge(self, amount: float):
        """
        Charge the battery with the given amount of energy.

        :param amount: Amount of energy to charge the battery with.
        """
        self.soc = self.soc + amount * self.charge_rate

    def discharge(self, amount: float):
        """
        Discharge the battery with the given amount of energy.

        :param amount: Amount of energy to discharge the battery with.
        """
        self.soc = max(0, self.soc - amount * self.discharge_rate)

    def get_soc(self):
        """
        Get the current state of charge of the battery.
        :return: soc
        """
        return self.soc

    def can_charge(self, amount):
        """
        Check if the battery can charge or discharge the given amount of energy.

        :param amount: The amount of energy to charge or discharge.
        :return: True if the battery can charge or discharge the given amount of energy, False otherwise.
        """
        if amount > self.capacity - self.soc:
            return False
        return True

    def can_discharge(self, amount):
        """
        Check if the battery can discharge the given amount of energy.

        :param amount: The amount of energy to discharge.
        :return: True if the battery can discharge the given amount of energy, False otherwise.
        """
        if amount < -self.soc:
            return False
        return True

    def check_prl_constraints_for_da(self, amount, reserve_amount):
        """
        Check if the battery as enough energy to cover the amount and also if there is enough room left in the battery
        to charge the amount when the promised prl amount is deducted.
        :param reserve_amount:
        :param amount: amount to be offered in the prl market
        :return: true when both criteria are met, false otherwise
        """
        if amount > self.capacity - self.soc - reserve_amount:
            return False
        # check if there is enough room in the battery after the promised prl amount is added
        if amount + reserve_amount > self.capacity - self.soc:
            return False

        return True

    def check_prl_constraints(self, amount):
        """
        Check if the battery as enough energy to cover the amount and also if there is enough room left in the battery
        to charge the amount.
        :param amount: amount to be offered in the prl market
        :return: true when both criteria are met, false otherwise
        """
        # prevent the agent from offering small amounts since the official limit is 1MW
        if amount < 200:
            return False
        # check if there is enough capacity in the battery to offer such an amount
        if amount > self.capacity - self.soc:
            return False
        # check if the battery could also charge the offered amount
        if amount + self.soc > self.capacity:
            return False

        return True

    def reset(self):
        """
        Reset the battery to its initial state.
        """
        self.soc = 0

    def get_charge_log(self):
        """
        Retrieve the charging log for the battery.

        :return: A list containing the amounts with which the battery has been charged.
        """
        return self.charge_log

    def add_charge_log(self, amount):
        """
        Add an amount to the charge log.

        :param amount: The amount of energy to add to the charge log.
        """
        self.charge_log.append(amount)
