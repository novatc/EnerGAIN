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

# Save the preprocessed data to a csv file
prl.to_csv('data/prm/preprocessed_prl.csv')
