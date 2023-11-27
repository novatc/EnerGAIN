import pandas as pd
import matplotlib.pyplot as plt

# Load the data
years = pd.read_csv("../data/in-use/env_data.csv", parse_dates=True)
years['date'] = pd.to_datetime(years['date'])
years['day_of_year'] = years['date'].dt.dayofyear
average_year = pd.read_csv("../data/in-use/average_da_year.csv")
average_year['date'] = pd.to_datetime(average_year['date'])
average_year['day_of_year'] = average_year['date'].dt.dayofyear

# Filtering data for each year
year_2019 = years['date'].dt.year == 2019
year_2020 = years['date'].dt.year == 2020
year_2021 = years['date'].dt.year == 2021
year_2022 = years['date'].dt.year == 2022

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(years[year_2019]['day_of_year'], years[year_2019]['price'], label='2019', linewidth=0.2)
plt.plot(years[year_2020]['day_of_year'], years[year_2020]['price'], label='2020', linewidth=0.2)
plt.plot(years[year_2021]['day_of_year'], years[year_2021]['price'], label='2021', linewidth=0.2)
plt.plot(years[year_2022]['day_of_year'], years[year_2022]['price'], label='2022', linewidth=0.2)
plt.plot(average_year['day_of_year'], average_year['price'], label='Durchschnitt', linewidth=2)


plt.title('Preisverlauf der einzelnen Jahre im Vergleich zum Durchschnitt')
plt.xlabel('Tag des Jahres')
plt.ylabel('Preis [€]')
plt.legend()
plt.tight_layout()
plt.savefig('../plots/price_year_average.svg', format='svg')
plt.show()

plt.figure(figsize=(14, 7))

plt.plot(years[year_2019]['date'], years[year_2019]['price'], label='2019', linewidth=0.2)
plt.plot(years[year_2020]['date'], years[year_2020]['price'], label='2020', linewidth=0.2)
plt.plot(years[year_2021]['date'], years[year_2021]['price'], label='2021', linewidth=0.2)
plt.plot(years[year_2022]['date'], years[year_2022]['price'], label='2022', linewidth=0.2)

plt.title('Preisverlauf')
plt.xlabel('Tag des Jahres')
plt.ylabel('Preis [€]')
plt.legend()
plt.tight_layout()
plt.savefig('../plots/price_timeline.svg', format='svg')
plt.show()
