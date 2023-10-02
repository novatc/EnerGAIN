import numpy as np
import pandas as pd


energy_consumption = pd.read_csv('data/clean/energy_consumption_01102018_01012023.csv', index_col=0)
energy_prediction = pd.read_csv('data/clean/energy_prediction_01102018_01012023.csv', index_col=0)
energy_price = pd.read_csv('data/clean/trading_prices_01102018_01012023.csv', index_col=0)
solar_power = pd.read_csv('data/clean/solar_power_01102018_01012023.csv', sep='\t', index_col=0)

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

# env_data = dataset[['price', 'consumption', 'prediction', 'Einstrahlung auf die Horizontale (kWh/m²)',
#                     'Diffusstrahlung auf die Horizontale (kWh/m²)']].copy()

env_data = dataset[['price', 'consumption', 'prediction']].copy()

# save the number of the day of the week in a new column
env_data['day_of_week'] = env_data.index.dayofweek
# save the number of the month in a new column
env_data['month'] = env_data.index.month
# save the number of the hour in a new column
env_data['hour'] = env_data.index.hour

# Apply cyclical encoding using sine and cosine transformations
env_data['hour_sin'] = np.sin(2 * np.pi * env_data['hour'] / 24)
env_data['hour_cos'] = np.cos(2 * np.pi * env_data['hour'] / 24)

env_data['day_of_week_sin'] = np.sin(2 * np.pi * env_data['day_of_week'] / 7)
env_data['day_of_week_cos'] = np.cos(2 * np.pi * env_data['day_of_week'] / 7)

env_data['month_sin'] = np.sin(2 * np.pi * env_data['month'] / 12)
env_data['month_cos'] = np.cos(2 * np.pi * env_data['month'] / 12)

# Drop the original 'day_of_week' column if no longer needed
env_data.drop('day_of_week', axis=1, inplace=True)

# Drop the original 'month' column if no longer needed
env_data.drop('month', axis=1, inplace=True)

# Drop the original 'hour' column if no longer needed
env_data.drop('hour', axis=1, inplace=True)

# set the price column as the index
env_data = env_data.set_index('price')

# cut the last 120 (one week) rows of the dataframe and save them as the test set
test_set = env_data.tail(24 * 30)
test_set.to_csv('data/in-use/eval_data.csv')

env_data.to_csv('data/in-use/train_data.csv')
# save the dataframe to a csv file
solar_power.to_csv('data/clean/dataset_01102018_01012023.csv')
# save some test data to a csv file
test_data = solar_power.tail(24 * 5)
test_data = test_data[['price', 'consumption', 'prediction']]
test_data.to_csv('data/in-use/test_data.csv')

time_data = env_data.iloc[:, -6:]

unscaled_data = solar_power[['price', 'consumption', 'prediction']]

# Reset indexes of the dataframes
unscaled_data = unscaled_data.reset_index(drop=True)
time_data = time_data.reset_index(drop=True)

# Concatenate unscaled_data and time_data
final_data = pd.concat([unscaled_data, time_data], axis=1)
final_data = final_data.set_index('price')

# cut off the last 30 days
train_data = final_data.head(-24 * 30)
train_data.to_csv('data/in-use/unscaled_train_data.csv')

test_data = final_data.tail(24 * 30)
test_data.to_csv('data/in-use/unscaled_eval_data.csv')
