import gymnasium as gym
from gymnasium import spaces

from envs.assets.battery import Battery
from envs.assets.dayahead import DayAhead
from envs.assets.features import clip_to_budget, price_feature_count, price_features
from envs.assets.frequency_containment_reserve import FrequencyContainmentReserve
from envs.assets.plot_engien import *


class MultiCompact(gym.Env):
    """
    MultiMarket with compact price-history features and the agent's own state.

    MultiMarket observes both market rows plus [prl_cooldown, lower_bound, upper_bound] - the
    flexibility band, but not the SOC that band constrains, and no price history at all. This
    variant adds the day-ahead lag/rolling block and [soc, savings]. Point it at the *_ext data
    to also pick up the solar columns and the PRL time features, both of which the committed
    preprocessing computes and then discards.
    """

    def __init__(self, da_data_path: str, prl_data_path: str, validation, boundary_penalty=True):
        super(MultiCompact, self).__init__()
        # Whether to charge for a day-ahead request the flexibility band had to cut down.
        self.boundary_penalty = boundary_penalty
        self.da_dataframe = pd.read_csv(da_data_path)
        self.prl_dataframe = pd.read_csv(prl_data_path)

        da_low_boundary = self.da_dataframe.min().values
        da_high_boundary = self.da_dataframe.max().values

        prl_low_boundary = self.prl_dataframe.min().values
        prl_high_boundary = self.prl_dataframe.max().values

        min_array = np.concatenate((da_low_boundary, prl_low_boundary))
        max_array = np.concatenate((da_high_boundary, prl_high_boundary))

        # day-ahead price history, then the agent's own state and the PRL flexibility band:
        # [soc, savings, prl_cooldown, lower_bound, upper_bound]
        n_feat = price_feature_count()
        observation_low = np.concatenate((min_array, np.full(n_feat, -np.inf),
                                          [0.0, -np.inf, 0.0, 0.0, 0.0]))
        observation_high = np.concatenate((max_array, np.full(n_feat, np.inf),
                                           [1000.0, np.inf, 4.0, 1000.0, 1000.0]))

        obs_shape = (self.da_dataframe.shape[1] + self.prl_dataframe.shape[1] + n_feat + 5,)

        action_low = np.array([0.001, 0, 0, -1000.0])  # prl price, prl amount, da price, da amount
        action_high = np.array([0.5, 1000, 1, 1000.0])  # prl price, prl amount, da price, da amount

        self.action_space = spaces.Box(low=action_low, high=action_high, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=observation_low, high=observation_high,
                                            shape=obs_shape)

        self.day_ahead = DayAhead(self.da_dataframe)
        self.prl = FrequencyContainmentReserve(self.prl_dataframe)
        self.battery = Battery(1000, 500)
        self.savings = 50  # €

        self.trade_log = []
        self.invalid_trades = []
        self.holding = []
        self.prl_trades = []

        self.soc_log = []  # To keep track of SOC values over time
        self.upper_bound_log = []  # To keep track of upper boundaries over time
        self.lower_bound_log = []

        self.rewards = []
        self.reward_log = []
        self.penalty = -10  # Penalty for invalid trades and breaking constraints

        self.validation = validation

        # this indicates, if the agent is in a 4-hour block or not. A normal step will decrease it by 1,
        # participation in the PRL market will set it to 4
        self.prl_cooldown = 0   # 0 means the agent can participate in the PRL market
        # The upper and lower boundaries for the SOC. They are set based on the amount of energy the agent offers in the
        # PRL market
        self.upper_bound = self.battery.capacity
        self.lower_bound = 0

        self.trade_threshold = 10

    def step(self, action):
        """
        Execute a step in the environment, making trade decisions based on the provided action,
        and updating relevant state and logs.

        :param action: (list of floats) A list containing action variables which might include
                       decisions on participating in markets, prices, and amounts to trade.
                       Expected order: [prl_choice, price_prl, amount_prl, price_da, amount_da]

        :return: (np.array, float, bool, bool, dict) A tuple containing:
                 - The new state observation as a NumPy array.
                 - The reward obtained in this step as a float.
                 - A boolean indicating if the environment has reached a terminal state.
                 - A boolean indicating if the environment has been truncated.
                 - A dictionary containing additional information about the current state.

        """
        should_truncated = False
        if self.validation:
            self.day_ahead.step()
            self.prl.step()
        else:
            # make sure the two markets are always in sync
            should_truncated = self.prl.random_walk(24 * 30)
            current_step = self.prl.get_current_step()
            self.day_ahead.set_step(current_step)
            if should_truncated:
                self.reset()

        price_prl, amount_prl, price_da, amount_da = action

        reward = 0

        terminated = False  # Whether the agent reaches the terminal state
        truncated = should_truncated

        # Reset boundaries if PRL cooldown has expired
        if self.prl_cooldown == 0:
            self.upper_bound = self.battery.capacity
            self.lower_bound = 0

        amount_prl = min(amount_prl, self.battery.get_soc())
        amount_prl = self.clamp_prl_to_band(amount_prl)
        # agent chooses to participate in the PRL market. The cooldown checks, if a new 4-hour block is ready
        if self.check_prl_constraints():
            if -self.trade_threshold < amount_prl < self.trade_threshold:
                reward += self.handle_holding()
            else:
                reward += self.perform_prl_trade(price_prl, amount_prl)

        # Handle DA trade or holding
        requested_da = amount_da
        amount_da = self.clip_trade_amount(amount_da, 'buy' if amount_da > 0 else 'sell')
        reward += self.boundary_cost(requested_da, amount_da)
        amount_da = clip_to_budget(amount_da, price_da, self.savings)

        if self.check_boundaries(amount_da):
            # Clip the amount to ensure that the battery state of charge remains within the bounds and decide based on
            # the new amount if the agent should hold or trade

            if -self.trade_threshold < amount_da < self.trade_threshold:
                reward += self.handle_holding()
            else:
                reward += self.perform_da_trade(amount_da, price_da)
        else:
            # MultiMarket silently does nothing here. Record it so the refusal is visible in the
            # trade log; boundary_cost() has already charged for it.
            self.log_trades(False, 'buy' if amount_da > 0 else 'sell', price_da, amount_da,
                            self.penalty, 'boundary')

        self.prl_cooldown = max(0, self.prl_cooldown - 1)  # Ensure it doesn't go below 0

        self.log_step(reward)

        info = {'current_price': self.day_ahead.get_current_price(),
                'current_step': self.day_ahead.get_current_step(),
                'savings': self.savings,
                'charge': self.battery.get_soc(),
                'prl price': price_prl,
                'prl amount': amount_prl,
                'da price': price_da,
                'da amount': amount_da,
                'reward': reward
                }
        return self.get_observation().astype(np.float32), reward, terminated, truncated, info

    def is_trade_valid(self, price, amount, trade_type):
        """
        Check if a trade is valid, i.e. if the battery can handle the trade and if the agent has enough savings.

        :param price: (float) The price at which the trade is attempted.
        :param amount: (float) The amount of energy to be traded. Positive values indicate buying or charging,
                       and negative values indicate selling or discharging.
        :param trade_type: (str) Type of trade to execute, accepted values are 'buy' or 'sell'.

        :return: (bool) True if the trade is valid, False otherwise.
        """
        if trade_type == 'buy':
            if price * amount > self.savings or self.savings <= 0 or self.battery.can_charge(amount) is False:
                # Report the actual cause. MultiMarket labels an unaffordable buy 'battery'
                # whenever savings is still positive, which conflates the two.
                affordable = price * amount <= self.savings and self.savings > 0
                self.log_trades(False, 'buy', price, amount, self.penalty,
                                'battery' if affordable else 'savings')
                return False
        elif trade_type == 'sell':
            if self.battery.can_discharge(amount) is False:
                self.log_trades(False, 'sell', price, amount, self.penalty, 'battery')
                return False
        else:
            raise ValueError(f"Invalid trade type: {trade_type}")

        return True

    def clip_trade_amount(self, amount, trade_type):
        """
        Clip a trade so the state of charge stays inside the flexibility band.

        MultiMarket's version can invert the trade: with the SOC above the upper bound,
        min(amount, upper - soc) returns a negative number for a buy, and perform_da_trade() then
        classifies it as a sell. On the average year that flipped 403 buys into sells and 177
        sells into buys. Its sell branch also computes soc - amount with amount already negative,
        which is the wrong direction for a discharge.

        :param amount: (float) The amount of energy to be traded.
        :param trade_type: (str) 'buy' or 'sell'.
        :return: (float) The clipped amount, never sign-flipped.
        """
        soc = self.battery.get_soc()
        # Shave a hair off so the resulting SOC lands strictly inside the band: check_boundaries()
        # compares with a strict <, so clipping to exactly the bound would still be refused and
        # logged as a violation, on what is really a routine capacity clip.
        margin = 1.0 - 1e-9
        if trade_type == 'buy':
            return float(min(amount, max(0.0, self.upper_bound - soc) * margin))
        if trade_type == 'sell':
            return float(max(amount, -max(0.0, soc - self.lower_bound) * margin))
        raise ValueError(f"Invalid trade type: {trade_type}")

    def set_boundaries(self, amount_prl):
        """
        Set the flexibility band a PRL commitment reserves.

        MultiMarket scales by a hardcoded 1000 rather than the capacity, which is only correct
        for Battery(1000, ...).

        :param amount_prl: the committed reserve amount in kW.
        :return: None, the bounds are set in place.
        """
        half = 0.5 * amount_prl
        self.upper_bound = self.battery.capacity - half
        self.lower_bound = half

    def clamp_prl_to_band(self, amount_prl):
        """
        Cap a PRL offer at what the current state of charge can actually back.

        Committing amount_prl reserves the band [0.5a, capacity - 0.5a], so the SOC has to sit
        inside it for the commitment to be deliverable in both directions. set_boundaries()
        derives the band from the offer alone and never looks at the SOC, so an offer made at a
        high SOC put the SOC outside its own band: the committed multi model spent 580 steps
        (6.6 %) of the average year above the upper bound, by up to 400 kWh, selling reserve it
        could not have delivered. Feasibility requires a < 2 * min(soc, capacity - soc).

        :param amount_prl: the offered reserve amount in kW.
        :return: the offer capped so the band strictly contains the current SOC.
        """
        soc = self.battery.get_soc()
        feasible = 2.0 * min(soc, self.battery.capacity - soc)
        return max(0.0, min(amount_prl, feasible * (1.0 - 1e-9)))

    def boundary_cost(self, requested, granted):
        """
        Penalty proportional to how much of a day-ahead request the band refused.

        Clipping on its own corrects an infeasible request silently, so the policy never learns
        that it asked for something the band could not take - and when check_boundaries() refuses
        outright, MultiMarket adds no reward at all, neither penalty nor hold bonus. A flat
        penalty would fire on nearly every step and drown out the trade rewards, so this scales
        with the fraction clipped away: zero when the request survives untouched, the full
        penalty when none of it does.

        :param requested: the amount the agent asked for.
        :param granted: what is left after clipping to the band.
        :return: the penalty contribution, always <= 0.
        """
        if not self.boundary_penalty or abs(requested) < 1e-9:
            return 0.0
        # Only charge when a PRL commitment has actually narrowed the band. With no commitment
        # the band is the whole battery range, and clipping there is ordinary capacity limiting -
        # which the day-ahead envs handle silently via clip_to_battery. Charging for it made the
        # penalty fire on 50 % of steps at a mean of -3.26, a near-constant tax rather than a
        # signal about the reserve obligation.
        if self.lower_bound <= 0.0 and self.upper_bound >= self.battery.capacity:
            return 0.0
        removed = abs(requested) - abs(granted)
        if removed <= 0:
            return 0.0
        return self.penalty * min(1.0, removed / abs(requested))

    def check_boundaries(self, amount):
        """Check if the battery can charge or discharge the given amount of energy."""
        if self.lower_bound < self.battery.get_soc() + amount < self.upper_bound:
            return True
        return False

    def check_prl_constraints(self):
        """Check if all constraints for PRL participation are met."""
        if self.prl_cooldown <= 0 == self.prl.get_current_step() % 4:
            return True
        return False

    def perform_da_trade(self, energy_amount: float, market_price: float) -> float:
        """
        Perform a trade on the day-ahead market.

        :param energy_amount: Energy to be traded (positive for buying, negative for selling).
        :param market_price: Price at which the trade is attempted.
        :return: Reward based on the trade outcome.
        """

        return self.trade(market_price, energy_amount, 'buy' if energy_amount > 0 else 'sell')

    def trade(self, price, amount, trade_type):
        """
        Execute a trade (buy/sell) and update the battery status and logs accordingly.

        :param price: (float) The price at which the trade is attempted.
        :param amount: (float) The amount of energy to be traded. Positive values indicate buying or charging,
                       and negative values indicate selling or discharging.
        :param trade_type: (str) Type of trade to execute, accepted values are 'buy' or 'sell'.

        :return: (float) Absolute value of the traded energy amount multiplied by the current market price.
                  Returns a penalty (self.penalty) if the trade is invalid or not accepted.

        :raises ValueError: If `trade_type` is neither 'buy' nor 'sell'.
        """
        if trade_type not in ['buy', 'sell']:
            raise ValueError(f"Invalid trade type: {trade_type}")

        # Check if the trade is valid
        if not self.is_trade_valid(price, amount, trade_type):
            return self.penalty

        return self.execute_trade(price, amount, trade_type)

    def execute_trade(self, price, amount, trade_type):
        """
        Execute the validated trade and update the system status.

        :param price: Price at which the trade is executed.
        :param amount: Amount of energy involved in the trade.
        :param trade_type: 'buy' or 'sell'.
        :return: Profit (or loss) from the trade.
        """
        current_price = self.day_ahead.get_current_price()
        profit = 0
        if trade_type == 'buy' and price < current_price or trade_type == 'sell' and price > current_price:
            profit = self.penalty

        if self.day_ahead.accept_offer(price, trade_type):
            if trade_type == 'buy':
                self.battery.charge(amount)  # Charge battery for buy trades
                profit = -current_price * amount  # Negative profit for buying

            elif trade_type == 'sell':
                self.battery.discharge(amount)  # Discharge battery for sell trades
                profit = current_price * abs(amount)  # Positive profit for selling
        else:
            self.log_trades(False, trade_type, price, amount, self.penalty, 'market rejected')
            return self.penalty

        # Logging the trade details
        self.battery.add_charge_log(self.battery.get_soc())
        self.savings = self.savings + profit
        self.log_trades(True, trade_type, price, amount, profit, 'accepted')

        return profit

    def perform_prl_trade(self, price, amount) -> float:
        """
        Attempt to participate in the PRL market. It's a pay as bid market,
         so the agent will always get the price it offered.
        :param price: Offer price
        :param amount: Amount of energy being offered
        """
        # Check if the offer is accepted by the prl market and the battery can adhere to prl constraints
        if self.prl.accept_offer(price):
            # Update savings based on the transaction in prl market
            profit = float((price * amount) * 4)
            self.battery.charge_log.append(self.battery.get_soc())
            self.savings += profit
            # add the next four hours to the trade log. They should be equal to each other and just differ from the
            # step value
            for i in range(4):
                trade_info = (
                    self.prl.get_current_step() + i,
                    'reserve',
                    self.prl.get_current_price(),
                    price,
                    amount,
                    profit,
                    'prl accepted',
                    self.battery.get_soc(),
                    self.savings
                )
                self.trade_log.append(trade_info)

            self.set_boundaries(amount)
            self.prl_cooldown = 4

            return float(price * amount) * 4
        else:
            # return penalty if the offer was not accepted
            return self.penalty

    def handle_holding(self):
        # Logic for handling the holding scenario
        self.holding.append((self.day_ahead.get_current_step(), 'hold'))
        return 1

    def get_observation(self) -> np.array:
        """
        Get the current state of the environment.
        :return: np.array containing the current state of the environment
        """
        da_step = self.day_ahead.get_current_step()
        observation = np.concatenate((self.da_dataframe.iloc[da_step].to_numpy(dtype=float),
                                      self.prl_dataframe.iloc[self.prl.get_current_step()].to_numpy(dtype=float),
                                      price_features(self.da_dataframe, da_step)))

        # Append the agent's own state and the current SOC boundaries
        return np.append(observation, [self.battery.get_soc(), self.savings,
                                       self.prl_cooldown, self.lower_bound, self.upper_bound])

    def reset(self, seed=None, options=None) -> np.array:
        """
        Important: the observation must be a numpy array
        :return: np.array containing the initial state of the environment
        """
        # Reset the state of the environment to an initial state
        super().reset(seed=seed, options=options)
        self.savings = 50
        self.battery.reset()
        observation = self.get_observation().astype(np.float32)
        return observation, {}

    def log_step(self, reward):
        """
        Update the logs for the current step.

        :param reward: The reward obtained in this step.
        """
        self.rewards.append(reward)
        self.reward_log.append((self.reward_log[-1] + reward) if self.reward_log else reward)
        self.upper_bound_log.append(self.upper_bound)
        self.lower_bound_log.append(self.lower_bound)
        self.soc_log.append(self.battery.get_soc())

    def render(self, mode='human'):
        """
        Render the environment to the screen
        :param mode:
        :return:
        """
        plot_savings(self.trade_log, 'multi_compact')
        plot_savings_on_trade_steps(trade_log=self.trade_log, total_steps=self.da_dataframe.shape[0],
                                    model_name='multi_compact')
        plot_charge(self.battery, 'multi_compact')
        plot_trades_timeline(trade_source=self.trade_log, title='Trades', buy_color='green', sell_color='red',
                             model_name='multi_compact', data=self.da_dataframe, plot_name='trades')
        plot_trades_timeline(trade_source=self.invalid_trades, title='Invalid Trades', buy_color='black',
                             sell_color='brown', model_name='multi_compact', data=self.da_dataframe, plot_name='invalid_trades')
        plot_holding(self.holding, 'multi_compact', da_data=self.da_dataframe)
        plot_soc_and_boundaries(self.soc_log, self.upper_bound_log, self.lower_bound_log, 'multi_compact')
        kernel_density_estimation(self.trade_log, 'multi_compact', da_data=self.da_dataframe)

    def get_trades(self) -> list:
        """
        Returns the trade log
        :return: list of tuples
        """
        return self.trade_log

    def get_invalid_trades(self) -> list:
        """
        Returns the trade log
        :return: list of tuples
        """
        return self.invalid_trades

    def get_prl_trades(self) -> list:
        """
        Returns the trade log for the PRL market
        :return: list of tuples
        """
        return self.prl_trades

    def get_holdings(self):
        return self.holding

    def log_trades(self, valid: bool, type: str, offered_price: float, amount: float, reward: float,
                   case: str) -> None:
        """
        Log the trades
        :param type: the trade type
        :param valid: if the trade was accepted or not
        :param offered_price: the offered price
        :param amount: the trade amount
        :param reward: the reward based on the offer
        :param case: if not valid, why
        :return: None
        """
        if valid:
            self.trade_log.append(
                (self.day_ahead.get_current_step(), type, self.day_ahead.get_current_price(), offered_price, amount,
                 reward, case, self.battery.get_soc(), self.savings))
        else:
            self.invalid_trades.append(
                (self.day_ahead.get_current_step(), type, self.day_ahead.get_current_price(), offered_price, amount,
                 reward, case, self.battery.get_soc(), self.savings))
