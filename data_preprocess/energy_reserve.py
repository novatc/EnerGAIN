from datetime import datetime

import pandas as pd

# load the table into a pandas DataFrame
df = pd.read_csv('../data/og/Prim_rregelreserve_201810010000_202303012359_Stunde.csv', sep=';')

df['Datum'] = df['Datum'].apply(lambda x: datetime.strptime(x, '%d.%m.%Y'))

start_date = pd.to_datetime('01.10.2018')
end_date = pd.to_datetime('01.01.2023')
mask = (df['Datum'] >= start_date) & (df['Datum'] < end_date)
df = df.loc[mask]

df['Datum'] = df['Datum'].dt.strftime('%d.%m.%Y')

# save the DataFrame to a CSV file
df.to_csv('../data/clean/energy_reserve_01102018_01012023.csv', index=False)