import pandas as pd
from datetime import datetime


def extract_save_month_data(env_data, month, year=2023):
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
    data_copy['Datum'] = pd.to_datetime(data_copy['date'])
    data_copy.set_index('Datum', inplace=True)

    # Extract data for the specified month and year
    month_data = data_copy[(data_copy.index.month == month) & (data_copy.index.year == year)].copy()

    # Check for the existence of columns before dropping
    columns_to_drop = ['day', 'month', 'hour', 'day']
    month_data = month_data.drop(columns=[col for col in columns_to_drop if col in month_data.columns])

    # Set the 'price' column as the index
    month_data.set_index('price', inplace=True)

    # Save the noisy month data to a CSV file
    filename = f'data/in-use/eval_data/month_{month}_data_prl.csv'
    month_data.to_csv(filename)
    print(f"data for month {month}, year {year} saved to {filename}")


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
average_by_hour_new = filtered_new_data.groupby(['month', 'day', 'hour']).agg(
    {'price': 'mean', 'amount': 'mean'}).reset_index()

# Generating a DataFrame for the average year
year_dates_new = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
average_year_new_df = pd.DataFrame({'date': year_dates_new})

# Extracting month, day, and hour for the average year DataFrame
average_year_new_df['month'] = average_year_new_df['date'].dt.month
average_year_new_df['day'] = average_year_new_df['date'].dt.day
average_year_new_df['hour'] = average_year_new_df['date'].dt.hour

# Merging the average values with the average year DataFrame
average_year_complete_new = average_year_new_df.merge(average_by_hour_new, on=['month', 'day', 'hour'], how='left')

for i in range(1, 13):
    extract_save_month_data(average_year_complete_new, i)

# Dropping the extra columns and setting 'date' as index
average_year_complete_new.drop(['month', 'day', 'hour'], axis=1, inplace=True)
average_year_complete_new.set_index('date', inplace=True)

# Saving the final DataFrame to a CSV file
average_year_csv_path_complete_new = 'data/in-use/eval_data/average_year_prl.csv'
average_year_complete_new.to_csv(average_year_csv_path_complete_new)

# Output the path to the saved CSV file
print(f"CSV file saved as: {average_year_csv_path_complete_new}")
