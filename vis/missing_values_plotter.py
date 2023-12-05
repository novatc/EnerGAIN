import matplotlib.dates as mdates
from matplotlib import pyplot as plt
import pandas as pd

# Load the dataset
file_path = '../data/prm/prl.csv'
data = pd.read_csv(file_path, delimiter=';')

# Combine 'Datum' and 'Anfang' into a single datetime column
data['DateTime'] = pd.to_datetime(data['Datum'] + ' ' + data['Anfang'], format='%d.%m.%Y %H:%M')

# Set the new datetime column as the index
data.set_index('DateTime', inplace=True)

# Plotting the data with European date format on the x-axis
fig, ax1 = plt.subplots(figsize=(14, 7))

color = 'tab:blue'
ax1.set_xlabel('DateTime')
ax1.set_ylabel('Vorgehaltene Menge [MW]', color=color)
line1 = ax1.plot(data.index, data['Vorgehaltene Menge [MW]'], color=color, label='Vorgehaltene Menge [MW]')
ax1.tick_params(axis='y', labelcolor=color)

# Instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('Leistungspreis', color=color)
line2 = ax2.plot(data.index, data['Leistungspreis'], color=color, label='Leistungspreis')
ax2.tick_params(axis='y', labelcolor=color)

# Setting European date format (day-month-year) for the x-axis
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))

# Creating a combined legend for both axes
lns = line1 + line2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper left')

# Title and grid
plt.title('Vorgehaltene Menge [MW] und Leistungspreis')
fig.tight_layout()
plt.savefig('../plots/prl_missing_values.svg', format='svg', dpi=1200)
plt.show()

file_path = '../data/prm/preprocessed_prl.csv'

fixed_data = pd.read_csv(file_path, delimiter=',')

# Combine 'Datum' and 'Anfang' into a single datetime column
fixed_data['DateTime'] = data.index

# Set the new datetime column as the index
fixed_data.set_index('DateTime', inplace=True)
fig, ax1 = plt.subplots(figsize=(14, 7))

color = 'tab:blue'
ax1.set_xlabel('DateTime')
ax1.set_ylabel('Vorgehaltene Menge [MW]', color=color)
line1 = ax1.plot(fixed_data.index, fixed_data['amount'], color=color, label='Vorgehaltene Menge [MW]')
ax1.tick_params(axis='y', labelcolor=color)

# Instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('Leistungspreis', color=color)
line2 = ax2.plot(fixed_data.index, fixed_data['price'], color=color, label='Leistungspreis')
ax2.tick_params(axis='y', labelcolor=color)

# Setting European date format (day-month-year) for the x-axis
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))

# Creating a combined legend for both axes
lns = line1 + line2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper left')

# Title and grid
plt.title('Vorgehaltene Menge [MW] und Leistungspreis mit interpolierten Werten')
fig.tight_layout()
plt.savefig('../plots/prl_interpolated_values.svg', format='svg', dpi=1200)
plt.show()
