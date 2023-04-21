import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from sklearn.preprocessing import MinMaxScaler

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
solar_power['prize'] = energy_price_series
solar_power['consumption'] = energy_consumption_series

# make the date column the index
solar_power = solar_power.set_index('date')

# make the index a datetime object
solar_power.index = pd.to_datetime(solar_power.index)

# scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(solar_power)

dataset = pd.DataFrame(scaled_data, columns=solar_power.columns, index=solar_power.index)

# save the dataframe to a csv file
dataset.to_csv('data/clean/dataset_01102018_01012023.csv')
