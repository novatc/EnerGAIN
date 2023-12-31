import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Load the data from the file
file_path = '../data/prm/preprocessed_prl.csv'
data = pd.read_csv(file_path)

# Starting date and time
start_date = datetime.strptime("01.10.2018 00:00", "%d.%m.%Y %H:%M")

# Create a date range with hourly frequency
date_range = pd.date_range(start=start_date, periods=len(data), freq='H')

# Add the date range as a new column to the dataframe
data['date'] = date_range

# Grouping the data by the hour of the day
hourly_avg_day = data.copy()
hourly_avg_day['hour'] = hourly_avg_day['date'].dt.hour
average_per_hour = hourly_avg_day.groupby('hour')['amount'].mean()

# Preparing x-axis labels in hh:mm format
hour_labels = [f'{hour:02d}:00' for hour in range(24)]

# Plotting
plt.figure(figsize=(14, 7))
average_per_hour.plot(kind='line', marker='o', color='blue')
plt.xlabel('Stunde', fontsize=12)
plt.ylabel('Menge [MWh]', fontsize=12)
plt.xticks(range(0, 24), hour_labels, rotation=45)
plt.tight_layout()
plt.savefig('../plots/avg_prl_amount.svg', format='svg', dpi=1200)
plt.show()

