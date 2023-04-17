from datetime import datetime

import pandas as pd

# load the table into a pandas DataFrame
df = pd.read_csv('../data/og/Gro_handelspreise_2018_2023_hour.csv', sep=';')

df['Date'] = df['Date'].apply(lambda x: datetime.strptime(x, '%d.%m.%Y'))

start_date = pd.to_datetime('01.10.2018')
end_date = pd.to_datetime('01.01.2023')
mask = (df['Date'] >= start_date) & (df['Date'] < end_date)
df = df.loc[mask]

df['Date'] = df['Date'].dt.strftime('%d.%m.%Y')

# save the DataFrame to a CSV file
df.to_csv('../data/clean/trading_prices_01102018_01012023.csv', index=False)