import pandas as pd

# load the table into a pandas DataFrame
df = pd.read_csv('../data/original/solar_values_produkt_st_stunde_20090101_20230228_00691.csv', sep=';')

# convert the "MESS_DATUM" column to a datetime object
df['MESS_DATUM_WOZ'] = pd.to_datetime(df['MESS_DATUM_WOZ'], format='%Y%m%d%H:%M')
# trim the DataFrame to include only dates between October 1, 2018, and January 1, 2023
start_date = pd.to_datetime('2018-10-01')
end_date = pd.to_datetime('2023-01-01')
mask = (df['MESS_DATUM_WOZ'] >= start_date) & (df['MESS_DATUM_WOZ'] < end_date)
df = df.loc[mask]
df['MESS_DATUM_WOZ'] = df['MESS_DATUM_WOZ'].dt.strftime('%d.%m.%Y %H:%M')

# save the DataFrame to a CSV file
df.to_csv('../data/clean/solar_values_01102018_01012023.csv', index=False)


