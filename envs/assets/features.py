import numpy as np

# Default price lags, in hours. 1/2 h capture the very strong short-term autocorrelation
# (r = 0.99 / 0.96 on the training set), 24 h the daily cycle (r = 0.91) and 168 h the weekly
# one (r = 0.84). Six cyclical time columns explain only about 2 % of the price variance, so the
# price's own history is by far the strongest signal available.
DEFAULT_LAGS = (1, 2, 24, 168)
DEFAULT_WINDOW = 24


def price_features(dataframe, step: int, lags=DEFAULT_LAGS, window: int = DEFAULT_WINDOW,
                   price_column: str = 'price') -> np.array:
    """
    Build a compact set of price-history features for the given step.

    This is the cheap alternative to tiling the whole observation row over an 8-hour window as
    TrendEnv does: that spends 48 of its 72 dimensions on cyclical time columns that barely move
    within the window, and still misses the daily and weekly structure.

    Returned in order: one entry per lag, then the rolling mean, the rolling standard deviation,
    the ratio of the current price to the rolling mean, and an exponentially weighted average
    over the same window.

    :param dataframe: the market dataframe to read prices from.
    :param step: the current step, used as the right edge of every window.
    :param lags: the lags in hours to sample the price at.
    :param window: the size of the rolling window in hours.
    :param price_column: the name of the price column.
    :return: the feature vector as a numpy array of shape (len(lags) + 4,).
    """
    prices = dataframe[price_column].to_numpy(dtype=float)
    current = prices[step]

    # Clamp instead of zero-filling: early in the dataset a zero price is a value the market can
    # actually take, so it would be indistinguishable from real data.
    lagged = [prices[max(0, step - lag)] for lag in lags]

    start = max(0, step - window + 1)
    recent = prices[start:step + 1]
    rolling_mean = float(recent.mean())
    rolling_std = float(recent.std())

    # Guard the ratio: prices are in €/kWh and can legitimately be zero or negative on the
    # German day-ahead market.
    ratio = current / rolling_mean if abs(rolling_mean) > 1e-9 else 0.0

    weights = np.exp(np.linspace(-1, 0, len(recent)))
    weights /= weights.sum()
    ewma = float(np.dot(recent, weights))

    return np.array(lagged + [rolling_mean, rolling_std, ratio, ewma], dtype=float)


def price_feature_count(lags=DEFAULT_LAGS) -> int:
    """
    Number of entries price_features() returns, for sizing the observation space.

    :param lags: the lags the env is configured with.
    :return: the feature count.
    """
    return len(lags) + 4


def clip_to_battery(amount: float, battery, trade_type: str) -> float:
    """
    Clip a day-ahead trade amount so the battery can physically absorb it.

    The multi-market envs already do this against the PRL flexibility band via
    clip_trade_amount(); the day-ahead-only envs did not, which is why 92 % of BaseEnv's
    rejected trades were 'battery' rejections. Charge and discharge efficiency are applied
    because Battery.can_charge()/can_discharge() compare amount * rate against the headroom.

    :param amount: the requested amount, positive to buy/charge and negative to sell/discharge.
    :param battery: the Battery the trade would act on.
    :param trade_type: 'buy' or 'sell'.
    :return: the amount clipped into what the battery can take.
    """
    # Shave a hair off the limit: can_charge()/can_discharge() compare with a strict >, so an
    # amount clipped to exactly the headroom can still fail on floating-point rounding.
    margin = 1.0 - 1e-9
    if trade_type == 'buy':
        headroom = (battery.capacity - battery.get_soc()) / battery.charge_rate
        return float(min(amount, max(0.0, headroom) * margin))
    if trade_type == 'sell':
        available = battery.get_soc() / battery.discharge_rate
        return float(max(amount, -max(0.0, available) * margin))
    raise ValueError(f"Invalid trade type: {trade_type}")


def clip_to_budget(amount: float, price: float, savings: float) -> float:
    """
    Clip a buy amount to what the agent can actually pay for at the offered price.

    is_trade_valid() rejects a buy when price * amount exceeds savings, but labels that
    rejection 'battery' whenever savings is still positive, so the two causes are
    indistinguishable in the trade log. Clipping removes the rejection entirely.

    :param amount: the requested amount; returned unchanged when selling (negative).
    :param price: the offered price in euro per kWh.
    :param savings: the capital currently available.
    :return: the amount clipped to what the capital covers.
    """
    if amount <= 0:
        return float(amount)
    if price <= 0 or savings <= 0:
        return 0.0
    return float(min(amount, (savings / price) * (1.0 - 1e-9)))
