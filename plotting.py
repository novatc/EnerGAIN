from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns


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


def plot_cumulative_reward(dfs, colors):
    """
    Plot the cumulative reward over time for each agent.
    Optionally filter by trade type (e.g., 'buy' or 'sell').
    """
    plt.figure(figsize=(15, 10))
    for i, (name, df) in enumerate(dfs.items()):
        name = name.split('.')[0]
        plt.plot(df['reward'].cumsum(), label=name, color=colors[i % len(colors)])

    plt.title(f'Cumulative Reward Over Time')
    plt.xlabel('Number of Trades')
    plt.ylabel('Cumulative Reward')
    plt.yscale('log')
    plt.legend(fontsize=24)
    plt.legend()
    plt.tight_layout()
    plt.savefig('img/cumulative_reward.png', dpi=400)
    plt.show()


def plot_trade_durations(dfs, colors):
    """
    Plot the distribution of durations between trades for each trading strategy.
    Uses a logarithmic x-axis for better visibility of data across multiple scales.

    Parameters:
        dfs (dict): A dictionary where keys are file names and values are DataFrames.
        colors (list): A list of distinct colors to use for each agent.
    """
    # Initialize the plot with a larger size
    plt.figure(figsize=(20, 8))

    # Loop through each trading strategy and calculate trade durations
    for i, (name, df) in enumerate(dfs.items()):
        name = name.split('.')[0]

        # Calculate the time steps between each trade
        trade_durations = df['step'].diff().dropna()

        # Plot the trade durations
        plt.hist(trade_durations, bins=np.logspace(np.log10(min(trade_durations)), np.log10(max(trade_durations)), 50),
                 alpha=0.5, color=colors[i % len(colors)], label=name)

    # Add title and labels
    plt.title('Distribution of Trade Durations (All Agents)')
    plt.xlabel('Trade Duration (Time Steps)')
    plt.ylabel('Frequency')

    # Make x-axis logarithmic for better visibility
    plt.xscale('log')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()


def plot_trade_sizes(dfs, colors):
    """
    Plot the distribution of trade sizes (amounts) for each trading strategy,
    separated by 'buy' and 'sell' trades.

    Parameters:
        dfs (dict): A dictionary where keys are file names and values are DataFrames.
        colors (list): A list of distinct colors to use for each agent.
    """
    # Initialize two subplots: one for 'buy' trades and another for 'sell' trades
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))

    # Titles for the subplots
    axs[0].set_title('Distribution of Trade Sizes (Buy Trades)')
    axs[1].set_title('Distribution of Trade Sizes (Sell Trades)')

    # Loop through each trading strategy and calculate trade sizes
    for i, (name, df) in enumerate(dfs.items()):
        name = name.split('.')[0]
        for j, trade_type in enumerate(['buy', 'sell']):
            # Filter by trade type ('buy' or 'sell')
            filtered_df = df[df['trade_type'] == trade_type]

            # Extract the 'amount' column
            trade_sizes = filtered_df['amount']

            # Plot the histogram of trade sizes
            axs[j].hist(trade_sizes, bins=50, alpha=0.5, color=colors[i % len(colors)], label=name)

    # Labels for the subplots
    axs[0].set_xlabel('Trade Size (Amount)')
    axs[0].set_ylabel('Frequency')
    axs[1].set_xlabel('Trade Size (Amount)')
    axs[1].set_ylabel('Frequency')

    # Add a legend to each subplot
    axs[0].legend()
    axs[1].legend()

    # Show the plots
    plt.show()


def plot_correlation_matrix(dfs):
    """
    Calculate and plot the correlation matrix of trading strategies based on cumulative rewards.

    Parameters:
        dfs (dict): A dictionary where keys are file names and values are DataFrames.
    """
    # Initialize an empty DataFrame to store the cumulative rewards for each strategy
    cumulative_rewards_df = pd.DataFrame()

    # Loop through each trading strategy and calculate cumulative rewards
    for name, df in dfs.items():
        name = name.split('.')[0]
        cumulative_rewards = df['reward'].cumsum()
        cumulative_rewards_df[name] = cumulative_rewards

    # Calculate the correlation matrix between the strategies
    correlation_matrix = cumulative_rewards_df.corr()

    # Plot the correlation matrix using a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Trading Strategies')
    plt.show()


def plot_avg_trade_data(trade_data, eval_data='data/in-use/month_5_data_da.csv'):
    """
    Generate plots for an average day

    Parameters:
    - trade_data : Path to the second dataset containing trade data
    - eval_data : Path to the first dataset containing hourly data

    """
    start_time = datetime.strptime('2023-05-01 00:00', '%Y-%m-%d %H:%M')

    # Read the first dataset and calculate 'time' and 'abs_price'
    first_dataset_df = pd.read_csv(eval_data)
    first_dataset_df['time'] = [start_time + timedelta(hours=i) for i in range(first_dataset_df.shape[0])]
    first_dataset_df['hour_of_day'] = first_dataset_df['time'].dt.hour
    avg_abs_price_by_hour = first_dataset_df.groupby('hour_of_day')['price'].mean().reset_index()

    model_name = trade_data.split('/')[-1].split('.')[0]

    # Load the second dataset
    df2 = pd.read_csv(trade_data)

    # Calculate time based on the 'step' column
    df2['time'] = df2['step'].apply(lambda x: start_time + timedelta(hours=x))

    # Filter 'buy' and 'sell' trades
    df2_buy = df2[df2['trade_type'] == 'buy']
    df2_sell = df2[df2['trade_type'] == 'sell']

    # Plot for 'buy' trades
    plt.figure(figsize=(14, 8))
    plt.plot(avg_abs_price_by_hour['hour_of_day'], avg_abs_price_by_hour['price'],
             label='Average Price')
    plt.scatter(df2_buy['step'] % 24, df2_buy['price'].abs(), color='g', marker='x', label='Buy Price')
    plt.xticks(range(0, 24), [str(i).zfill(2) + ':00' for i in range(0, 24)], rotation=90)
    plt.xlim(-1, 24)
    plt.xlabel('Hour of the Day')
    plt.ylabel('Price for Buy')
    plt.title(f'Price Data for Buy Trades - {model_name}')
    plt.legend()
    plt.show()

    # Plot for 'sell' trades
    plt.figure(figsize=(14, 8))
    plt.plot(avg_abs_price_by_hour['hour_of_day'], avg_abs_price_by_hour['price'],
             label='Average Price')
    plt.scatter(df2_sell['step'] % 24, df2_sell['price'], color='r', marker='x',
                label='Sell Price')
    plt.xticks(range(0, 24), [str(i).zfill(2) + ':00' for i in range(0, 24)], rotation=90)
    plt.xlim(-1, 24)
    plt.xlabel('Hour of the Day')
    plt.ylabel('Price for Sell')
    plt.title(f'Price Data for Sell Trades - {model_name}')
    plt.legend()
    plt.show()


# Example usage:
# Define a list of distinct colors
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# Call the function to plot trade sizes

# Example usage
folder_path = 'trade_logs'  # Update this path
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

dfs = load_csv_files_from_folder(folder_path)
plot_trade_data(dfs, 'amount', colors, trade_type='buy')
plot_trade_data(dfs, 'amount', colors, trade_type='sell')
plot_trade_data(dfs, 'price', colors, trade_type='buy')
plot_trade_data(dfs, 'price', colors, trade_type='sell')
plot_trade_durations(dfs, colors)
plot_trade_sizes(dfs, colors)
plot_cumulative_reward(dfs, colors)

# iterate over all files in the folder 'trade_logs' that end with '.csv'
# for file in os.listdir('trade_logs'):
#     plot_avg_trade_data(os.path.join('trade_logs', file))
