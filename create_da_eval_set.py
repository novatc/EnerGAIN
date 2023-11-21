import numpy as np
import pandas as pd
from datetime import datetime

# Load the data from the CSV file
file_path = 'data/in-use/env_data.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Converting the 'date' column to datetime format for filtering and extraction
data['date'] = pd.to_datetime(data['date'])

# Filtering the data between January 1, 2019, and December 31, 2022
filtered_data = data[(data['date'] >= '2019-01-01') & (data['date'] <= '2022-12-31')]

# Extracting month and day
filtered_data['month'] = filtered_data['date'].dt.month
filtered_data['day'] = filtered_data['date'].dt.day
filtered_data['hour'] = filtered_data['date'].dt.hour

# Grouping the data by month, day, and hour to calculate the average
average_by_hour = filtered_data.groupby(['month', 'day', 'hour']).agg({'price': 'mean', 'consumption': 'mean', 'prediction': 'mean'}).reset_index()

# Generating a DataFrame for the average year
year_dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
average_year_df = pd.DataFrame({'date': year_dates})

# Extracting month, day, and hour for the average year DataFrame
average_year_df['month'] = average_year_df['date'].dt.month
average_year_df['day'] = average_year_df['date'].dt.day
average_year_df['hour'] = average_year_df['date'].dt.hour

# Merging the average values with the average year DataFrame
average_year_complete = average_year_df.merge(average_by_hour, on=['month', 'day', 'hour'], how='left')

# Apply cyclical encoding using sine and cosine transformations
average_year_complete['hour_sin'] = np.sin(2 * np.pi * average_year_complete['hour'] / 24)
average_year_complete['hour_cos'] = np.cos(2 * np.pi * average_year_complete['hour'] / 24)

average_year_complete['day_of_week_sin'] = np.sin(2 * np.pi * average_year_complete['day'] / 7)
average_year_complete['day_of_week_cos'] = np.cos(2 * np.pi * average_year_complete['day'] / 7)

average_year_complete['month_sin'] = np.sin(2 * np.pi * average_year_complete['month'] / 12)
average_year_complete['month_cos'] = np.cos(2 * np.pi * average_year_complete['month'] / 12)

# Dropping the extra columns and setting 'date' as index
average_year_complete.drop(['month', 'day', 'hour'], axis=1, inplace=True)
average_year_complete.set_index('price', inplace=True)

# drop the date column
average_year_complete.drop('date', axis=1, inplace=True)

# Saving the final DataFrame to a CSV file
average_year_csv_path_complete = 'data/in-use/eval_data/average_year_da.csv'
average_year_complete.to_csv(average_year_csv_path_complete)

# Output the path to the saved CSV file
print(f"CSV file saved as: {average_year_csv_path_complete}")
