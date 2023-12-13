import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

from envs.assets.env_utilities import moving_average


def plot_reward(reward_log: list, window_size: int, model_name: str):
    """
    Plot the reward over time.
    :param window_size: the size of the window for the moving average.
    :param model_name: name of the model
    :param reward_log: the reward log.
    :return:
    """
    os.makedirs(f'agent_data/{model_name}', exist_ok=True)
    plt.figure(figsize=(14, 7))
    plt.plot(reward_log, label='Original', alpha=0.5)
    smoothed_data = moving_average(reward_log, window_size)
    smoothed_steps = np.arange(window_size - 1, len(reward_log))
    plt.plot(smoothed_steps, smoothed_data, label=f'Durchschnitt (n = {window_size})')
    plt.title('Reward über die Zeit')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'agent_data/{model_name}/{model_name}_reward.svg', dpi=1200, format='svg')
    plt.show()


def plot_savings(savings_log: list, window_size: int, model_name: str):
    """
    Plot the savings over time.
    :param model_name: name of the model
    :param window_size: size of the window for the moving average.
    :param savings_log: the savings log.
    :return:
    """
    plt.figure(figsize=(14, 7))
    plt.plot(savings_log, label='Kapital')
    smoothed_data = moving_average(savings_log, window_size)
    smoothed_steps = np.arange(window_size - 1, len(savings_log))
    plt.title('Kapital über die Zeit')
    plt.xlabel('Anzahl der Handelssignale')
    plt.ylabel('Kapital (€)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'agent_data/{model_name}/{model_name}_savings.svg', dpi=1200, format='svg')
    plt.show()


def plot_charge(window_size: int, battery, model_name: str):
    """
    Plot the charge over time.
    :param window_size: size of the window for the moving average.
    :param battery: the battery object.
    :param model_name: name of the model
    :return:
    """
    plt.figure(figsize=(14, 7))
    charge_log = battery.get_charge_log()

    # Original data
    plt.plot(charge_log, label='SOC', alpha=0.5)

    plt.title('SOC über die Zeit')
    plt.xlabel('Anzahl der Handelssignale')
    plt.ylabel('SOC')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'agent_data/{model_name}/{model_name}_charge.svg', dpi=1200, format='svg')
    plt.show()


def plot_trades_timeline(trade_source: list, title: str, buy_color: str, sell_color: str,
                         model_name: str, data: pd.DataFrame, plot_name: str):
    """
    Plot the given trades over time.
    :param plot_name:
    :param data:
    :param trade_source: list of trades, could be valid or invalid trades
    :param title: name for the plot
    :param buy_color:  color for buy trades
    :param sell_color: plot color for sale trades
    :param model_name: name of the model
    :return:
    """
    plt.figure(figsize=(14, 7))
    trade_log = trade_source
    eval_data_df = data
    total_trades = len(trade_log)

    # Get the buy and sell trades from the trade log
    buys = [trade for trade in trade_log if trade[1] == 'buy']
    sells = [trade for trade in trade_log if trade[1] == 'sell']
    reserve = [trade for trade in trade_log if trade[1] == 'reserve']

    # Check if there are any buy or sell trades to plot
    if not buys and not sells:
        print("No trades to plot.")
        return

    # Plot real market prices from evaluation dataset
    plt.plot(eval_data_df.index, eval_data_df['price'], color='blue', label='Marktpreis', alpha=0.6)

    # Plot trade data if available
    if buys:
        buy_steps, _, _, buy_prices, _, _, _ = zip(*buys)
        plt.scatter(buy_steps, buy_prices, c=buy_color, marker='o', label='Kaufen', alpha=0.6, s=10)
    if sells:
        sell_steps, _, _, sell_prices, _, _, _ = zip(*sells)
        plt.scatter(sell_steps, sell_prices, c=sell_color, marker='x', label='Verkaufen', alpha=0.6, s=10)
    if reserve:
        reserve_steps, _, _, reserve_price, _, _, _ = zip(*reserve)
        plt.scatter(reserve_steps, reserve_price, c='darkgoldenrod', marker='s', label='Reserve', alpha=0.6, s=10)

    plt.title(title + f' ({total_trades} trades)')
    plt.ylabel('Preis (€/kWh)')
    plt.xlabel('Schritt')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'agent_data/{model_name}/{model_name}_{plot_name}_timeline.svg', dpi=1200, format='svg')
    plt.show()


def plot_holding(holding_logs: list, model_name: str, da_data: pd.DataFrame):
    """
    Plot the holding over time.
    :param da_data_path: the path to the used da data
    :param model_name: name of the model
    :param holding_logs: list with timestamps when the agent decided to hold
    :return:
    """
    if not holding_logs:
        print("No trades to plot.")
        return
    plt.figure(figsize=(14, 7))
    eval_data_timeline = da_data
    plt.plot(eval_data_timeline.index, eval_data_timeline['price'], color='blue', label='Real Market Price', alpha=0.6)
    steps, _ = zip(*holding_logs)
    plt.scatter(steps, [eval_data_timeline['price'][step] for step in steps], c='red', marker='o', label='Hold', s=15)
    plt.title('Halten über die Zeit')
    plt.ylabel('Preis (€/kWh)')
    plt.xlabel('Schritt')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'agent_data/{model_name}/{model_name}_hold.svg', dpi=1200, format='svg')
    plt.show()


def kernel_density_estimation(trade_list: list, model_name: str, da_data: pd.DataFrame):
    generated_prices = [trade[3] for trade in trade_list]
    historic_prices = da_data['price'].values

    # Generate density functions
    generated_density = gaussian_kde(generated_prices)
    historic_density = gaussian_kde(historic_prices)

    # Generate points for plotting
    x_vals = np.linspace(min(generated_prices + list(historic_prices)),
                         max(generated_prices + list(historic_prices)), 1000)

    # Create figure and plot data
    plt.figure(figsize=(14, 7))
    plt.plot(x_vals, generated_density(x_vals), label='Generated Prices')
    plt.plot(x_vals, historic_density(x_vals), label='Historic Prices')
    plt.title('Kerndichteschätzung von generierten und historischen Preisen')
    plt.xlabel('Preis (€/kWh)')
    plt.ylabel('Verteilung')
    plt.legend()
    plt.savefig(f'agent_data/{model_name}/{model_name}_KDE.svg', dpi=1200, format='svg')
    plt.show()


def plot_soc_and_boundaries(soc_log, upper_bound_log, lower_bound_log, model_name: str):
    plt.figure(figsize=(14, 7))
    plt.plot(soc_log, label='SOC', color='blue')
    plt.plot(upper_bound_log, label='obere Grenze', color='red', linestyle='--')
    plt.plot(lower_bound_log, label='untere Grenze', color='green', linestyle='--')
    plt.fill_between(range(len(soc_log)), lower_bound_log, upper_bound_log, color='gray', alpha=0.1)
    plt.title('SOC und Flexibilitätsgrenzen über die Zeit')
    plt.xlabel('Schritt')
    plt.ylabel('SOC')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'agent_data/{model_name}/{model_name}_soc_and_boundaries.svg', dpi=1200, format='svg')
    plt.show()
