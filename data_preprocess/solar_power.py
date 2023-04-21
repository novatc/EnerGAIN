import pandas as pd
from datetime import datetime, timedelta

# load the table into a pandas DataFrame
solar_power = pd.read_csv('../data/original/pv_ol_25deg_south.csv', sep=';')

start_date = datetime(2018, 10, 1)
end_date = datetime(2023, 1, 1)
current_date = start_date

result_df = pd.DataFrame(columns=solar_power.columns)

while current_date < end_date:
    # Create a copy of the original dataframe and update the dates
    temp_df = solar_power.copy()
    temp_df['Zeit'] = temp_df['Zeit'].apply(lambda x: (datetime.strptime(x, "%d.%m. %H:%M") - datetime.strptime(
        "01.01. 00:00", "%d.%m. %H:%M")) + current_date)

    # Append the updated dataframe to the result dataframe
    result_df = pd.concat([result_df, temp_df], ignore_index=True)

    # Increment the current date by one year
    current_date += timedelta(days=365)

# remove values  after the 2022-12-31
result_df = result_df[result_df['Zeit'] < datetime(2023, 1, 1)]

# Save the new dataframe to a csv file
result_df.to_csv('../data/clean/solar_power_01102018_01012023.csv', index=False, sep='\t')
result_df.to_excel('../data/clean/solar_power_01102018_01012023.xlsx', index=False)
