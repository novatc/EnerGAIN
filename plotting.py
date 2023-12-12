from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns


def load_csv_files_from_folder(folder_path):
    """
    Lädt alle CSV-Dateien aus einem bestimmten Ordner in ein Dictionary von DataFrames.
    """
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dfs = {}
    for csv_file in csv_files:
        dfs[csv_file] = pd.read_csv(os.path.join(folder_path, csv_file))
    return dfs


def plot_trade_data(dfs, column, colors, trade_type=None):
    """
    Zeichnet Handelsdaten basierend auf einer bestimmten Spalte mit unterschiedlichen Farben für jeden Agenten.
    Optionaler Filter nach Handelstyp (z.B. 'buy' oder 'sell').
    """
    plt.figure(figsize=(20, 8))
    for i, (name, df) in enumerate(dfs.items()):
        if trade_type:
            df = df[df['type'] == trade_type]
        x_values = df.index.values
        y_values = df[column].values
        plt.scatter(x_values, y_values, label=name, alpha=0.5, s=50, color=colors[i % len(colors)])

    plt.title(f'Handels-{column.capitalize()} im Zeitverlauf (Alle Agenten)')
    plt.xlabel('Zeitschritt')
    plt.ylabel(column.capitalize())
    plt.legend()
    plt.show()


def plot_cumulative_reward(dfs, colors):
    """
    Zeichnet die kumulative Belohnung über die Zeit für jeden Agenten.
    Optionaler Filter nach Handelstyp (z.B. 'buy' oder 'sell').
    """
    plt.figure(figsize=(14, 7))
    for i, (name, df) in enumerate(dfs.items()):
        name = name.split('.')[0]
        plt.plot(df['reward'].cumsum(), label=name, color=colors[i % len(colors)])

    plt.title('Kapitalgewinne/-verluste im Zeitverlauf')
    plt.xlabel('Anzahl der Handel')
    plt.ylabel('Kapital')
    #plt.yscale('log')
    plt.legend(fontsize=24)
    plt.legend()
    plt.tight_layout()
    plt.savefig('img/money_over_time.svg', dpi=1200, format='svg')
    plt.show()



def plot_capital_over_time(dfs, colors):
    """
    Zeichnet die kumulative Kapitalgewinne oder -verluste über die Zeit für jeden Agenten.

    Parameter:
        dfs (dict): Ein Dictionary, bei dem die Schlüssel Dateinamen und die Werte DataFrames sind.
        colors (list): Eine Liste von unterschiedlichen Farben für jeden Agenten.
    """
    plt.figure(figsize=(14, 7))
    for i, (name, df) in enumerate(dfs.items()):
        # Berechnet die kumulative Summe der 'reward'-Spalte
        df['kumulative_belohnung'] = df['reward'].cumsum()
        plt.plot(df['kumulative_belohnung'], label=name.split('.')[0], color=colors[i % len(colors)])

    plt.title('Kumulative Kapitalgewinne/-verluste im Zeitverlauf')
    plt.xlabel('Anzahl der Handel')
    plt.ylabel('Kumulative Belohnung (Kapitalgewinn/-verlust)')
    plt.legend()
    plt.show()


def plot_trade_durations(dfs, colors):
    """
    Zeichnet die Verteilung der Dauern zwischen Handelsgeschäften für jede Handelsstrategie.
    Verwendet eine logarithmische x-Achse für eine bessere Sichtbarkeit der Daten über mehrere Skalen.

    Parameter:
        dfs (dict): Ein Dictionary, bei dem die Schlüssel Dateinamen und die Werte DataFrames sind.
        colors (list): Eine Liste von unterschiedlichen Farben für jeden Agenten.
    """
    # Initialisiert das Diagramm mit einer größeren Größe
    plt.figure(figsize=(14, 7))

    # Durchläuft jede Handelsstrategie und berechnet die Handelsdauern
    for i, (name, df) in enumerate(dfs.items()):
        name = name.split('.')[0]

        # Berechnet die Zeitschritte zwischen jedem Handel
        handelsdauern = df['step'].diff().dropna()

        # Zeichnet die Handelsdauern
        plt.hist(handelsdauern, bins=np.logspace(np.log10(min(handelsdauern)), np.log10(max(handelsdauern)), 50),
                 alpha=0.5, color=colors[i % len(colors)], label=name)

    # Fügt Titel und Beschriftungen hinzu
    plt.title('Verteilung der Handelsdauern (Alle Agenten)')
    plt.xlabel('Handelsdauer (Zeitschritte)')
    plt.ylabel('Häufigkeit')

    # Macht die x-Achse logarithmisch für eine bessere Sichtbarkeit
    plt.xscale('log')

    # Fügt eine Legende hinzu
    plt.legend()

    # Zeigt das Diagramm
    plt.show()


def plot_trade_sizes(dfs, colors):
    """
    Zeichnet die Verteilung der Handelsgrößen (Mengen) für jede Handelsstrategie,
    getrennt nach 'buy' und 'sell' Handel.

    Parameter:
        dfs (dict): Ein Dictionary, bei dem die Schlüssel Dateinamen und die Werte DataFrames sind.
        colors (list): Eine Liste von unterschiedlichen Farben für jeden Agenten.
    """
    # Initialisiert zwei Unterdiagramme: eins für 'buy'-Handel und ein weiteres für 'sell'-Handel
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    # Titel für die Unterdiagramme
    axs[0].set_title('Verteilung der Handelsgrößen (Kaufen)')
    axs[1].set_title('Verteilung der Handelsgrößen (Verkaufen)')

    # Durchläuft jede Handelsstrategie und berechnet die Handelsgrößen
    for i, (name, df) in enumerate(dfs.items()):
        name = name.split('.')[0]
        for j, handelstyp in enumerate(['buy', 'sell']):
            # Filtert nach Handelstyp ('buy' oder 'sell')
            gefilterter_df = df[df['type'] == handelstyp]

            # Extrahiert die 'amount'-Spalte
            handelsgrößen = gefilterter_df['amount']

            # Zeichnet das Histogramm der Handelsgrößen
            axs[j].hist(handelsgrößen, bins=50, alpha=0.5, color=colors[i % len(colors)], label=name)

    # Beschriftungen für die Unterdiagramme
    axs[0].set_xlabel('Handelsgröße (Menge)')
    axs[0].set_ylabel('Häufigkeit')
    axs[1].set_xlabel('Handelsgröße (Menge)')
    axs[1].set_ylabel('Häufigkeit')

    # Fügt eine Legende zu jedem Unterdiagramm hinzu
    axs[0].legend()
    axs[1].legend()
    plt.savefig('img/trade_amount_dist.svg', dpi=1200, format='svg')

    # Zeigt die Diagramme
    plt.show()


def plot_correlation_matrix(dfs):
    """
    Berechnet und zeichnet die Korrelationsmatrix der Handelsstrategien basierend auf kumulativen Belohnungen.

    Parameter:
        dfs (dict): Ein Dictionary, bei dem die Schlüssel Dateinamen und die Werte DataFrames sind.
    """
    # Initialisiert ein leeres DataFrame, um die kumulativen Belohnungen für jede Strategie zu speichern
    kumulative_belohnungen_df = pd.DataFrame()

    # Durchläuft jede Handelsstrategie und berechnet kumulative Belohnungen
    for name, df in dfs.items():
        name = name.split('.')[0]
        kumulative_belohnungen = df['reward'].cumsum()
        kumulative_belohnungen_df[name] = kumulative_belohnungen

    # Berechnet die Korrelationsmatrix zwischen den Strategien
    korrelationsmatrix = kumulative_belohnungen_df.corr()

    # Zeichnet die Korrelationsmatrix mit einem Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(korrelationsmatrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Korrelationsmatrix der Handelsstrategien')
    plt.show()


# Example usage:
# Define a list of distinct colors
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# Call the function to plot trade sizes

# Example usage
folder_path = 'trade_logs'  # Update this path
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

dfs = load_csv_files_from_folder(folder_path)
# plot_trade_data(dfs, 'amount', colors, trade_type='buy')
# plot_trade_data(dfs, 'amount', colors, trade_type='sell')
# plot_trade_data(dfs, 'offered_price', colors, trade_type='buy')
# plot_trade_data(dfs, 'offered_price', colors, trade_type='sell')
# plot_trade_durations(dfs, colors)
plot_trade_sizes(dfs, colors)
plot_cumulative_reward(dfs, colors)
