import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from envs.assets.env_utilities import moving_average


def plot_reward(reward_log: list, window_size: int, model_name: str):
    """
    Plot the reward over time.
    :param window_size: the size of the window for the moving average.
    :param model_name: name of the model
    :param reward_log: the reward log.
    :return:
    """
    os.makedirs(f'img/{model_name}', exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(reward_log, label='Original', alpha=0.5)
    smoothed_data = moving_average(reward_log, window_size)
    smoothed_steps = np.arange(window_size - 1, len(reward_log))
    plt.plot(smoothed_steps, smoothed_data, label=f'Smoothed (window size = {window_size})')
    plt.title('Reward Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'img/{model_name}/{model_name}_reward.png', dpi=400)
    plt.show()


def plot_savings(savings_log: list, window_size: int, model_name: str):
    """
    Plot the savings over time.
    :param model_name: name of the model
    :param window_size: size of the window for the moving average.
    :param savings_log: the savings log.
    :return:
    """
    plt.figure(figsize=(10, 6))
    plt.plot(savings_log, label='Original', alpha=0.5)
    smoothed_data = moving_average(savings_log, window_size)
    smoothed_steps = np.arange(window_size - 1, len(savings_log))
    plt.plot(smoothed_steps, smoothed_data, label=f'Smoothed (window size = {window_size})')
    plt.title('Savings Over Time')
    plt.xlabel('Number of trades')
    plt.ylabel('Savings (€)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'img/{model_name}/{model_name}_savings.png', dpi=400)
    plt.show()


def plot_charge(window_size: int, battery, model_name: str):
    """
    Plot the charge over time.
    :param window_size: size of the window for the moving average.
    :param battery: the battery object.
    :param model_name: name of the model
    :return:
    """
    plt.figure(figsize=(10, 6))
    charge_log = battery.get_charge_log()

    # Original data
    plt.plot(charge_log, label='Original', alpha=0.5)

    # Smoothed data
    smoothed_data = moving_average(charge_log, window_size)
    smoothed_steps = np.arange(window_size - 1,
                               len(charge_log))  # Adjust the x-axis for the smoothed data

    plt.plot(smoothed_steps, smoothed_data, label=f'Smoothed (window size = {window_size})')

    plt.title('Charge Over Time')
    plt.xlabel('Number of trades')
    plt.ylabel('Charge (kWh)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'img/{model_name}/{model_name}_charge.png', dpi=400)
    plt.show()


def plot_trades_timeline(trade_source: list, title: str, buy_color: str, sell_color: str, model_name: str):
    """
    Plot the given trades over time.
    :param trade_source: list of trades, could be valid or invalid trades
    :param title: name for the plot
    :param buy_color:  color for buy trades
    :param sell_color: plot color for sale trades
    :param model_name: name of the model
    :return:
    """
    plt.figure(figsize=(10, 6))
    trade_log = trade_source
    eval_data_df = pd.read_csv('data/in-use/unscaled_eval_data.csv')
    total_trades = len(trade_log)

    # Get the buy and sell trades from the trade log
    buys = [trade for trade in trade_log if trade[3] == 'buy']
    sells = [trade for trade in trade_log if trade[3] == 'sell']
    reserve = [trade for trade in trade_log if trade[3] == 'reserve']

    # Check if there are any buy or sell trades to plot
    if not buys and not sells:
        print("No trades to plot.")
        return

    # Plot real market prices from evaluation dataset
    plt.plot(eval_data_df.index, eval_data_df['price'], color='blue', label='Real Market Price', alpha=0.6)

    # Plot trade data if available
    if buys:
        buy_steps, buy_prices, _, _, _ = zip(*buys)
        plt.scatter(buy_steps, buy_prices, c=buy_color, marker='o', label='Buy', alpha=0.6, s=10)
    if sells:
        sell_steps, sell_prices, _, _, _ = zip(*sells)
        plt.scatter(sell_steps, sell_prices, c=sell_color, marker='x', label='Sell', alpha=0.6, s=10)
    if reserve:
        reserve_steps, reserve_price, _, _, _ = zip(*reserve)
        plt.scatter(reserve_steps, reserve_price, c='darkgoldenrod', marker='s', label='Reserve', alpha=0.6, s=10)

    plt.title(title + f' ({total_trades} trades)')
    plt.ylabel('Price (€/kWh)')
    plt.xlabel('Step')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'img/{model_name}/{model_name}_trades_timeline.png', dpi=400)
    plt.show()


def plot_holding(holding_logs: list, model_name: str):
    """
    Plot the holding over time.
    :param model_name: name of the model
    :param holding_logs: list with timestamps when the agent decided to hold
    :return:
    """
    if not holding_logs:
        print("No trades to plot.")
        return
    plt.figure(figsize=(10, 6))
    eval_data_timeline = pd.read_csv('data/in-use/unscaled_eval_data.csv')
    plt.plot(eval_data_timeline.index, eval_data_timeline['price'], color='blue', label='Real Market Price', alpha=0.6)
    steps, _, _, _, _ = zip(*holding_logs)
    plt.scatter(steps, [eval_data_timeline['price'][step] for step in steps], c='black', marker='o', label='Hold',
                alpha=0.6, s=10)
    plt.title('Hold')
    plt.ylabel('Price (€/kWh)')
    plt.xlabel('Step')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'img/{model_name}/{model_name}_hold.png', dpi=400)
    plt.show()


def kernel_density_estimation(trade_list: list):
    generated_prices = [trade[1] for trade in trade_list]
    historic_prices = pd.read_csv('data/in-use/unscaled_eval_data.csv')

    plt.figure(figsize=(10, 6))
    sns.kdeplot(generated_prices, label='Generated Prices', fill=True)
    sns.kdeplot(historic_prices['price'], label='Historic Prices', fill=True)

    plt.title('Kernel Density Estimate of Generated and Historic Prices')
    plt.xlabel('Price')
    plt.ylabel('Density')
    plt.legend()

    # Display the plot
    plt.show()



