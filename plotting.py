import pandas as pd
import matplotlib.pyplot as plt
import os


def load_csv_files_from_folder(folder_path):
    """
    Load all CSV files from a specified folder into a dictionary of DataFrames.
    """
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dfs = {}
    for csv_file in csv_files:
        dfs[csv_file] = pd.read_csv(os.path.join(folder_path, csv_file))
    return dfs


def plot_trade_data(dfs, column, colors, trade_type=None):
    """
    Plot trade data based on a specified column using distinct colors for each agent.
    Optionally filter by trade type (e.g., 'buy' or 'sell').
    """
    plt.figure(figsize=(20, 8))
    for i, (name, df) in enumerate(dfs.items()):
        if trade_type:
            df = df[df['trade_type'] == trade_type]
        x_values = df.index.values
        y_values = df[column].values
        plt.scatter(x_values, y_values, label=name, alpha=0.5, s=50, color=colors[i % len(colors)])

    plt.title(f'Trade {column.capitalize()} Over Time (All Agents)')
    plt.xlabel('Time Step')
    plt.ylabel(column.capitalize())
    plt.legend()
    plt.show()


def plot_cumulative_reward(dfs, colors, trade_type=None):
    """
    Plot the cumulative reward over time for each agent.
    Optionally filter by trade type (e.g., 'buy' or 'sell').
    """
    plt.figure(figsize=(15, 10))
    for i, (name, df) in enumerate(dfs.items()):
        if trade_type:
            df = df[df['trade_type'] == trade_type]
        plt.plot(df['reward'].cumsum(), label=name, color=colors[i % len(colors)])

    plt.title(f'Cumulative Reward Over Time {trade_type.capitalize() if trade_type else ""})')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.show()


# Example usage
folder_path = 'trade_logs'  # Update this path
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

dfs = load_csv_files_from_folder(folder_path)
plot_trade_data(dfs, 'amount', colors, trade_type='buy')
plot_trade_data(dfs, 'amount', colors, trade_type='sell')
plot_trade_data(dfs, 'price', colors, trade_type='buy')
plot_trade_data(dfs, 'price', colors, trade_type='sell')
plot_cumulative_reward(dfs, colors, trade_type='sell')
