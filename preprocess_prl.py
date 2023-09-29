import numpy as np
import pandas as pd

prl = pd.read_csv('data/prm/prl.csv', index_col=0, sep=';')

# Print missing values
print(prl.isnull().sum())

# rename the columns to "date, start, end, amount, price"
prl.columns = ['start', 'end', 'amount', 'price']

# make the index a datetime object
prl.index = pd.to_datetime(prl.index, dayfirst=True)

# Convert the "price" column to a numeric type
prl['price'] = pd.to_numeric(prl['price'], errors='coerce')

# Interpolate missing values in the "price, amount" column
prl['price'] = prl['price'].interpolate(method='linear')
prl['price'] = prl['price'] / 1000  # convert to €/kWh
prl['amount'] = prl['amount'].interpolate(method='linear')

print(prl.isnull().sum())

# save the number of the day of the week in a new column
prl['day_of_week'] = prl.index.dayofweek
# save the number of the month in a new column
prl['month'] = prl.index.month
# for prl['hour'] use prl.start and convert to float
prl['hour'] = prl['start'].str.split(':').str[0]
prl['hour'] = pd.to_numeric(prl['hour'], errors='coerce')

# Apply cyclical encoding using sine and cosine transformations
prl['hour_sin'] = np.sin(2 * np.pi * prl['hour'] / 24)
prl['hour_cos'] = np.cos(2 * np.pi * prl['hour'] / 24)

prl['day_of_week_sin'] = np.sin(2 * np.pi * prl['day_of_week'] / 7)
prl['day_of_week_cos'] = np.cos(2 * np.pi * prl['day_of_week'] / 7)

prl['month_sin'] = np.sin(2 * np.pi * prl['month'] / 12)
prl['month_cos'] = np.cos(2 * np.pi * prl['month'] / 12)

# Drop the "day_of_week, month, hour" columns
prl = prl.drop(['day_of_week', 'month', 'hour', 'start', 'end',], axis=1)

# make price the index
prl = prl.set_index('price')


# Save the preprocessed data to a csv file
prl.to_csv('data/prm/preprocessed_prl.csv')
