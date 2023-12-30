import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from envs.assets.env_utilities import moving_average


def plot_reward(trade_log: list, model_name: str):
    """
    Plot the reward over time.
    :param trade_log: the list that holds all the trade infos
    :param model_name: name of the model
    :return:
    """
    os.makedirs(f'agent_data/{model_name}', exist_ok=True)
    reward_log = [trade[4] for trade in trade_log]
    cumulative_reward_log = np.cumsum(reward_log)

    plt.figure(figsize=(14, 7))
    plt.plot(cumulative_reward_log, label='Reward', alpha=0.5)
    plt.xlabel('Schritte', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'agent_data/{model_name}/{model_name}_reward.svg', dpi=1200, format='svg')
    plt.show()


def plot_savings_on_trade_steps(trade_log: list, total_steps: int, model_name: str):
    """
    Plot the savings over time, only updating when a trade occurs.
    :param trade_log: the trade log, a list of tuples.
    :param total_steps: total number of steps.
    :param model_name: name of the model.
    :return:
    """
    # Extract the step and savings from each trade log entry
    steps = [trade[0] for trade in trade_log]  # assuming step is at the 1st position
    savings = [trade[8] for trade in trade_log]  # assuming savings is at the 9th position

    plt.figure(figsize=(14, 7))
    plt.plot(steps, savings, drawstyle='steps-post', label='Kapital')
    plt.xlim(0, total_steps)
    plt.xlabel('Schritte', fontsize=12)
    plt.ylabel('Kapital (€)', fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'agent_data/{model_name}/{model_name}_savings_on_trade_steps.svg', dpi=1200, format='svg')
    plt.show()


def plot_savings(trade_log: list, model_name: str):
    """
    Plot the savings over time based on a trade log.
    :param model_name: name of the model.
    :param trade_log: the trade log, a list of tuples.
    :return:
    """
    # Extract the 'savings' from each trade log entry
    savings_log = [trade[8] for trade in trade_log]

    plt.figure(figsize=(14, 7))
    plt.plot(savings_log, label='Kapital')
    plt.xlabel('Anzahl der Handelssignale', fontsize=12)
    plt.ylabel('Kapital (€)', fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'agent_data/{model_name}/{model_name}_savings.svg', dpi=1200, format='svg')
    plt.show()


def plot_charge(battery, model_name: str):
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
    plt.plot(charge_log, label='SOC')

    plt.xlabel('Anzahl der Handelssignale', fontsize=12)
    plt.ylabel('SOC', fontsize=12)
    plt.legend(fontsize=12)
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
        buy_steps, _, _, buy_prices, _, _, _, _, _ = zip(*buys)
        plt.scatter(buy_steps, buy_prices, c=buy_color, marker='o', label='Kaufen', alpha=0.6, s=10)
    if sells:
        sell_steps, _, _, sell_prices, _, _, _, _, _ = zip(*sells)
        plt.scatter(sell_steps, sell_prices, c=sell_color, marker='x', label='Verkaufen', alpha=0.6, s=10)
    if reserve:
        reserve_steps, _, _, reserve_price, _, _, _, _, _ = zip(*reserve)
        plt.scatter(reserve_steps, reserve_price, c='darkgoldenrod', marker='s', label='Reserve', alpha=0.6, s=10)

    plt.ylabel('Preis (€/kWh)', fontsize=12)
    plt.xlabel('Schritt', fontsize=12)

    plt.legend(fontsize=12)
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
    plt.plot(eval_data_timeline.index, eval_data_timeline['price'], color='blue', label='Marktpreis', alpha=0.6)
    steps, _ = zip(*holding_logs)
    plt.scatter(steps, [eval_data_timeline['price'][step] for step in steps], c='red', marker='o', label='Hold', s=15)
    plt.ylabel('Preis (€/kWh)', fontsize=12)
    plt.xlabel('Schritt', fontsize=12)
    plt.legend(fontsize=12)
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
    plt.plot(x_vals, generated_density(x_vals), label='Erzeugte Preise', color='blue')
    plt.plot(x_vals, historic_density(x_vals), label='Historische Preise', color='orange')

    # Fill color under the curves
    plt.fill_between(x_vals, generated_density(x_vals), color='blue', alpha=0.3)
    plt.fill_between(x_vals, historic_density(x_vals), color='orange', alpha=0.3)

    plt.xlabel('Preis (€/kWh)', fontsize=12)
    plt.ylabel('Verteilung', fontsize=12)
    plt.legend(fontsize=12)
    plt.savefig(f'agent_data/{model_name}/{model_name}_KDE.svg', dpi=1200, format='svg')
    plt.show()


def plot_soc_and_boundaries(soc_log, upper_bound_log, lower_bound_log, model_name: str):
    plt.figure(figsize=(14, 7))
    plt.plot(soc_log, label='SOC', color='blue')
    plt.plot(upper_bound_log, label='obere Grenze', color='red', linestyle='--')
    plt.plot(lower_bound_log, label='untere Grenze', color='green', linestyle='--')
    plt.fill_between(range(len(soc_log)), lower_bound_log, upper_bound_log, color='gray', alpha=0.1)
    plt.xlabel('Schritt', fontsize=12)
    plt.ylabel('SOC', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'agent_data/{model_name}/{model_name}_soc_and_boundaries.svg', dpi=1200, format='svg')
    plt.show()
