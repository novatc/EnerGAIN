import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

energy_consumption = pd.read_csv('data/clean/energy_consumption_01102018_01012023.csv', index_col=0)
energy_prediction = pd.read_csv('data/clean/energy_prediction_01102018_01012023.csv', index_col=0)
energy_price = pd.read_csv('data/clean/trading_prices_01102018_01012023.csv', index_col=0)
solar_value = pd.read_csv('data/clean/solar_values_01102018_01012023.csv', index_col=0)


energy_prediction_column = "Gesamt (Netzlast) [MWh] Berechnete Auflösungen"
energy_price_column = "Deutschland/Luxemburg [€/MWh] Original resolutions"
solar_value_column = "ZENIT"
date_column = "MESS_DATUM_WOZ"

energy_prediction_series = energy_prediction[energy_prediction_column]
energy_price_series = energy_price[energy_price_column]
solar_value_series = solar_value[solar_value_column]
date_series = solar_value[date_column]

# replace '-' with 0 and convert to float
energy_prediction_series = energy_prediction_series.replace('-', 0)
energy_price_series = energy_price_series.replace('-', 0)
solar_value_series = solar_value_series.replace('-', 0)

energy_prediction_series = energy_prediction_series.str.replace('.', '').str.replace(',', '.').astype(float)
energy_price_series = energy_price_series.str.replace('.', '').str.replace(',', '.').astype(float)



