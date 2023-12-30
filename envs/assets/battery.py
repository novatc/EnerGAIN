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
        self.soc = min(self.capacity, self.soc + amount * self.charge_rate)

    def discharge(self, amount: float):
        """
        Discharge the battery with the given amount of energy.

        :param amount: Amount of energy to discharge the battery with.
        """
        amount = abs(amount)
        self.soc = max(0, self.soc - amount * self.discharge_rate)

    def get_soc(self):
        """
        Get the current state of charge of the battery.
        :return: soc
        """
        return self.soc

    def can_charge(self, amount):
        """
        Check if the battery can charge the given amount of energy.

        :param amount: The amount of energy to charge.
        :return: True if the battery can charge or discharge the given amount of energy, False otherwise.
        """
        if amount * self.charge_rate > self.capacity - self.soc:
            return False

        return True

    def can_discharge(self, amount):
        """
        Check if the battery can discharge the given amount of energy.

        :param amount: The amount of energy to discharge.
        :return: True if the battery can discharge the given amount of energy, False otherwise.
        """
        # abs() because amount is negative when selling. The check would have always returned True otherwise because
        # a negative amount is always smaller than the soc which is always positive.
        amount = abs(amount)
        if amount * self.discharge_rate > self.soc:
            return False
        return True

    def reset(self):
        """
        Reset the battery to its initial state.
        """
        self.soc = 500

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
