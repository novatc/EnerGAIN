import numpy as np
import pandas as pd


def extract_save_month_data(env_data, month):
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

    # Create a copy of the DataFrame and convert 'date' column to datetime
    data_copy = env_data.copy()
    # Extract data for the specified month and year
    month_data = data_copy[(data_copy.month == month)].copy()

    # Check for the existence of columns before dropping
    columns_to_drop = ['day', 'month', 'hour', 'day']
    month_data = month_data.drop(columns=[col for col in columns_to_drop if col in month_data.columns])

    # Set the 'price' column as the index
    month_data.set_index('price', inplace=True)

    # Save the noisy month data to a CSV file
    filename = f'data/in-use/eval_data/month_{month}_data_da.csv'
    month_data.to_csv(filename)
    print(f"data for month {month} saved to {filename}")


# Load the data from the CSV file
path = 'data/in-use/env_data.csv'
env_prl = pd.read_csv(path)

env_prl['date'] = pd.to_datetime(env_prl['date'])
filtered_df = env_prl[(env_prl['date'] >= '2019-01-01') & (env_prl['date'] <= '2022-12-31')]

# Add a 'day' column to the dataframe
filtered_df['day'] = filtered_df['date'].dt.day

# Group by month, day, and hour, and calculate the average amount and price
average_year_df_detailed = filtered_df.groupby(['month', 'day', 'hour']).agg(
    {'price': 'mean', 'consumption': 'mean', 'prediction': 'mean'}).reset_index()

# Apply cyclic encoding to the 'month' and 'day' hour columns
average_year_df_detailed['month_sin'] = np.sin(average_year_df_detailed['month'] * (2. * np.pi / 12))
average_year_df_detailed['month_cos'] = np.cos(average_year_df_detailed['month'] * (2. * np.pi / 12))
average_year_df_detailed['day_sin'] = np.sin(average_year_df_detailed['day'] * (2. * np.pi / 31))
average_year_df_detailed['day_cos'] = np.cos(average_year_df_detailed['day'] * (2. * np.pi / 31))
average_year_df_detailed['hour_sin'] = np.sin(average_year_df_detailed['hour'] * (2. * np.pi / 24))
average_year_df_detailed['hour_cos'] = np.cos(average_year_df_detailed['hour'] * (2. * np.pi / 24))

# Sort the dataframe in chronological order (by month, day, and hour)
average_year_df_detailed = average_year_df_detailed.sort_values(by=['month', 'day', 'hour'])

for i in range(1, 13):
    extract_save_month_data(average_year_df_detailed, i)

# Drop the 'month', 'day', and 'hour' columns
average_year_df_detailed.drop(['month', 'day', 'hour'], axis=1, inplace=True)

average_year_df_detailed.to_csv('data/in-use/eval_data/average_da_year.csv', index=False)
