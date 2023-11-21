import pandas as pd
from datetime import datetime

# Load the data from the CSV file
new_file_path = 'data/prm/env_prl.csv'  # Replace with your new file path
new_data = pd.read_csv(new_file_path)

# Converting the 'Datum' column to datetime format for filtering and extraction
new_data['Datum'] = pd.to_datetime(new_data['Datum'])

# Filtering the data between January 1, 2019, and December 31, 2022
filtered_new_data = new_data[(new_data['Datum'] >= '2019-01-01') & (new_data['Datum'] <= '2022-12-31')].copy()

# Extracting month and day using .loc
filtered_new_data.loc[:, 'month'] = filtered_new_data['Datum'].dt.month
filtered_new_data.loc[:, 'day'] = filtered_new_data['Datum'].dt.day
filtered_new_data.loc[:, 'hour'] = filtered_new_data['Datum'].dt.hour

# Grouping the data by month, day, and hour to calculate the average
average_by_hour_new = filtered_new_data.groupby(['month', 'day', 'hour']).agg({'price': 'mean', 'amount': 'mean'}).reset_index()

# Generating a DataFrame for the average year
year_dates_new = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
average_year_new_df = pd.DataFrame({'date': year_dates_new})

# Extracting month, day, and hour for the average year DataFrame
average_year_new_df['month'] = average_year_new_df['date'].dt.month
average_year_new_df['day'] = average_year_new_df['date'].dt.day
average_year_new_df['hour'] = average_year_new_df['date'].dt.hour

# Merging the average values with the average year DataFrame
average_year_complete_new = average_year_new_df.merge(average_by_hour_new, on=['month', 'day', 'hour'], how='left')

# Dropping the extra columns and setting 'date' as index
average_year_complete_new.drop(['month', 'day', 'hour'], axis=1, inplace=True)
average_year_complete_new.set_index('date', inplace=True)

# Saving the final DataFrame to a CSV file
average_year_csv_path_complete_new = 'data/in-use/eval_data/average_year_prl.csv'
average_year_complete_new.to_csv(average_year_csv_path_complete_new)

# Output the path to the saved CSV file
print(f"CSV file saved as: {average_year_csv_path_complete_new}")
