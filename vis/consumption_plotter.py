from datetime import datetime
import pandas as pd

from matplotlib import pyplot as plt

# read in the data under data/in-use/unscaled_train_data.csv

dataset = pd.read_csv('../data/clean/dataset_01102018_01012023.csv')

price_amount_relationship = dataset[['date', 'price', 'consumption']]
price_amount_relationship['date'] = pd.to_datetime(price_amount_relationship['date'], format='%Y-%m-%d %H:%M:%S')

grouped_by_hour = price_amount_relationship.groupby(price_amount_relationship['date'].dt.hour)
grouped_by_hour = grouped_by_hour.mean()
grouped_by_hour['time'] = grouped_by_hour.index

plt.figure(figsize=(14, 7))
x_axis_labels = [f"{hour:02d}:00" for hour in range(24)]
plt.plot(x_axis_labels, grouped_by_hour['consumption'] / 1000, label='Durchschnittlicher Verbrauch', marker='o')
plt.xlabel('Stunde', fontsize=12)
plt.ylabel('Verbrauch [Tsd. MWh]', fontsize=12)
plt.xticks(rotation=45)

plt.tight_layout()  # Adjust layout to prevent clipping of datetime labels
plt.savefig('../plots/avg_consumption.svg', format='svg', dpi=1200)
plt.show()
