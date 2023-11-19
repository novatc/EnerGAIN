import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates


def add_noise_to_columns(df, columns, noise_level=0.01):
    """
    Adds small random noise to specified columns in a DataFrame.
    """
    for col in columns:
        if col in df.columns:
            noise = np.random.normal(loc=0.0, scale=noise_level, size=len(df[col]))
            df.loc[:, col] = df.loc[:, col] + df.loc[:, col] * noise
    return df


def extract_save_month_data_with_noise(env_data, month, year=2022, noise_level=0.01):
    """
    Extracts data for a specified month and year from env_data, processes it, adds noise, and saves to CSV.

    Args:
    env_data (pd.DataFrame): The environment data DataFrame.
    month (int): The month for which to extract data.
    year (int): The year for which to extract data.
    noise_level (float): The level of noise to add to specified columns.

    Returns:
    None
    """
    if not 1 <= month <= 12:
        raise ValueError("Month must be an integer between 1 and 12.")

    # Extract data for the specified month and year
    month_data = env_data[(env_data.index.month == month) & (env_data.index.year == year)].copy()

    # Drop the original 'day_of_week', 'month', and 'hour' columns
    month_data.drop(['day_of_week', 'month', 'hour'], axis=1, inplace=True)

    # Add noise to specified columns
    columns_to_add_noise = ['price', 'consumption', 'prediction']
    month_data_noisy = add_noise_to_columns(month_data, columns_to_add_noise, noise_level)

    # Save the noisy month data to a CSV file
    filename = f'data/in-use/month_{month}_data_da.csv'
    return month_data_noisy


energy_consumption = pd.read_csv('../data/clean/energy_consumption_01102018_01012023.csv', index_col=0)
energy_prediction = pd.read_csv('../data/clean/energy_prediction_01102018_01012023.csv', index_col=0)
energy_price = pd.read_csv('../data/clean/trading_prices_01102018_01012023.csv', index_col=0)
solar_power = pd.read_csv('../data/clean/solar_power_01102018_01012023.csv', sep='\t', index_col=0)

energy_prediction_column = "Gesamt (Netzlast) [MWh] Berechnete Auflösungen"
energy_price_column = "Deutschland/Luxemburg [€/MWh] Original resolutions"
energy_consumption_column = "Gesamt (Netzlast) [MWh] Berechnete Auflösungen"

energy_prediction_series = energy_prediction[energy_prediction_column]
energy_price_series = energy_price[energy_price_column]
energy_consumption_series = energy_consumption[energy_consumption_column]
date_column = solar_power.index

energy_prediction_series = energy_prediction_series.replace('-', np.nan)
energy_price_series = energy_price_series.replace('-', np.nan)
energy_consumption_series = energy_consumption_series.replace('-', np.nan)

energy_prediction_series = energy_prediction_series.str.replace('.', '').str.replace(',', '.').astype(float)
energy_price_series = energy_price_series.str.replace('.', '').str.replace(',', '.').astype(float)
energy_consumption_series = energy_consumption_series.str.replace('.', '').str.replace(',', '.').astype(float)

# Interpolate missing values
energy_prediction_series = energy_prediction_series.interpolate(
    method='linear', limit_direction='both', axis=0)

energy_price_series = energy_price_series.interpolate(
    method='linear', limit_direction='both', axis=0
)
energy_consumption_series = energy_consumption_series.interpolate(
    method='linear', limit_direction='both', axis=0
)

# Reset the index of each series
energy_prediction_series = energy_prediction_series.reset_index(drop=True)
energy_price_series = energy_price_series.reset_index(drop=True)
energy_consumption_series = energy_consumption_series.reset_index(drop=True)

# append the energy prediction series, the energy price series, and the energy consumption series to the solar power
# dataframe
solar_power['date'] = date_column
solar_power = solar_power.reset_index(drop=True)
solar_power['prediction'] = energy_prediction_series
solar_power['price'] = energy_price_series / 1000  # convert to €/kWh
solar_power['consumption'] = energy_consumption_series

# make the date column the index
solar_power = solar_power.set_index('date')

# make the index a datetime object
solar_power.index = pd.to_datetime(solar_power.index)

# scale values separately
price_values = solar_power['price'].values.reshape(-1, 1)  # reshape to 2D array
amount_values = solar_power['consumption'].values.reshape(-1, 1)  # reshape to 2D array
prediction_values = solar_power['prediction'].values.reshape(-1, 1)  # reshape to 2D array

dataset = pd.DataFrame(solar_power, columns=solar_power.columns, index=solar_power.index)

env_data = dataset[['price', 'consumption', 'prediction']].copy()

# save the number of the day of the week in a new column
env_data['day_of_week'] = env_data.index.dayofweek
# save the number of the month in a new column
env_data['month'] = env_data.index.month
# save the number of the hour in a new column
env_data['hour'] = env_data.index.hour

may_normal = extract_save_month_data_with_noise(env_data, 5, noise_level=0.00)  # For May data with 1% noise
may_augmented = extract_save_month_data_with_noise(env_data, 5, noise_level=0.1)  # For May data with 1% noise

# plot the data in comparison
plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
date_format = mdates.DateFormatter('%d.%m.%Y')
plt.gca().xaxis.set_major_formatter(date_format)
plt.plot(may_augmented['price'], label='mit Rauschsignal', color='red')
plt.plot(may_normal['price'], label='orginal', alpha=1)
plt.ylabel('Preis [€/kWh]')
plt.xlabel('Datum')
plt.legend()
plt.title('Preis mit Rauschsignal vs. Original')

plt.subplot(2, 1, 2)
date_format = mdates.DateFormatter('%d.%m.%Y')
plt.gca().xaxis.set_major_formatter(date_format)
plt.plot(may_augmented['consumption'], label='mit Rauschsignal', color='red')
plt.plot(may_normal['consumption'], label='orginal', alpha=1)
plt.ylabel('Verbrauch [kWh]')
plt.xlabel('Datum')
plt.legend()
plt.title('Verbrauch mit Rauschsignal vs. Original')

plt.tight_layout()
plt.savefig('../plots/augmentation_month.svg', format='svg', dpi=1200)
plt.show()




