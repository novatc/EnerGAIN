import numpy as np
import pandas as pd


def add_noise_to_columns(df, columns, noise_level=0.01):
    """
    Adds small random noise to specified columns in a DataFrame.
    """
    for col in columns:
        if col in df.columns:
            noise = np.random.normal(loc=0.0, scale=noise_level, size=len(df[col]))
            df.loc[:, col] = df.loc[:, col] + df.loc[:, col] * noise
    return df


def extract_save_month_data_with_noise(env_data, month, year=2022,  noise_level=0.01):
    """
    Extracts data for a specified month from env_data, processes it, adds noise, and saves to CSV.

    Args:
    env_data (pd.DataFrame): The environment data DataFrame.
    month (int): The month for which to extract data.
    noise_level (float): The level of noise to add to specified columns.

    Returns:
    None
    """
    if not 1 <= month <= 12:
        raise ValueError("Month must be an integer between 1 and 12.")

    # Extract data for the specified month and create a new DataFrame
    month_data = env_data[(env_data.index.month == month) & (env_data.index.year == year)].copy()
    # Drop the original 'day_of_week', 'month', and 'hour' columns
    month_data.drop(['day_of_week', 'month', 'hour', 'start', 'end'], axis=1, inplace=True)

    # Add noise to specified columns
    columns_to_add_noise = ['price', 'amount']
    month_data_noisy = add_noise_to_columns(month_data, columns_to_add_noise, noise_level)

    # Set the 'price' column as the index
    month_data_noisy.set_index('price', inplace=True)

    # Save the noisy month data to a CSV file
    filename = f'data/in-use/month_{month}_data_prl.csv'
    month_data_noisy.to_csv(filename)
    print(f"Noisy data for month {month} saved to {filename}")


prl = pd.read_csv('data/prm/prl.csv', index_col=0, sep=';')

# Print missing values
print(prl.isnull().sum())

# rename the columns to "date, start, end, amount, price"
prl.columns = ['start', 'end', 'amount', 'price']

# make the index a datetime object
prl.index = pd.to_datetime(prl.index, dayfirst=True)

# Convert the "price" column to a numeric type
prl['price'] = pd.to_numeric(prl['price'], errors='coerce')

# Interpolate missing values in the "price, amount" column using something else than linear interpolation
# for example, use the method='time'
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


prl.to_csv('data/prm/env_prl.csv')

# Drop the "day_of_week, month, hour" columns
prl = prl.drop(['day_of_week', 'month', 'hour', 'start', 'end', ], axis=1)

# make price the index
prl = prl.set_index('price')


# Save the preprocessed data to a csv file
prl.to_csv('data/prm/preprocessed_prl.csv')
