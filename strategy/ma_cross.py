from datetime import timedelta

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import matplotlib.patches as patches

window = 20  # Define the window for local maxima and minima detection
volume_threshold = None


def calculate_smma(data, window):
    return data.ewm(alpha=1 / window, adjust=False).mean()


def calculate_supply_demand_zones(data, window, volume_threshold):
    # Create subsets where volume is greater than volume_threshold
    high_vol_data = data.loc[data['Volume'] > volume_threshold]

    # Find local peaks (supply zones)
    high_vol_data['supply'] = high_vol_data['High'].iloc[
        argrelextrema(high_vol_data['High'].values, np.greater_equal, order=window)[0]]

    # Find local troughs (demand zones)
    high_vol_data['demand'] = high_vol_data['Low'].iloc[
        argrelextrema(high_vol_data['Low'].values, np.less_equal, order=window)[0]]

    # Copy supply and demand zones to original data
    data['supply'] = high_vol_data['supply']
    data['demand'] = high_vol_data['demand']

    # Remove zones that are revisited in the future
    for i in range(len(data['High'])):
        if not np.isnan(data['demand'].iloc[i]):
            if data['Low'].iloc[i + 1:].min() <= data['demand'].iloc[i]:
                data['demand'].iloc[i] = np.nan
        if not np.isnan(data['supply'].iloc[i]):
            if data['High'].iloc[i + 1:].max() >= data['supply'].iloc[i]:
                data['supply'].iloc[i] = np.nan

    return data


def ma_crossover_strategy(data, short_window, long_window):
    short_moving_avg = calculate_smma(data, short_window)
    long_moving_avg = calculate_smma(data, long_window)

    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0.0
    signals['signal'][short_window:] = np.where(short_moving_avg[short_window:] > long_moving_avg[short_window:], 1.0,
                                                0.0)
    signals['positions'] = signals['signal'].diff()

    return signals


def backtest(data, signals, initial_investment=100000.0):
    positions = pd.DataFrame(index=signals.index).fillna(0.0)
    positions['Stock'] = 100 * signals['signal']
    portfolio = positions.multiply(data['Close'], axis=0)
    pos_diff = positions.diff()
    portfolio['holdings'] = (positions.multiply(data['Close'], axis=0)).sum(axis=1)
    portfolio['cash'] = initial_investment - (pos_diff.multiply(data['Close'], axis=0)).sum(axis=1).cumsum()
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    portfolio['returns'] = portfolio['total'].pct_change()

    return portfolio


def calculate_metrics(portfolio):
    total_return = (portfolio['total'][-1] / portfolio['total'][0]) - 1
    annual_return = (1 + total_return) ** (1 / len(set(portfolio.index.year))) - 1
    excess_returns = portfolio['returns'] - 0
    sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())

    return total_return, annual_return, sharpe_ratio


def automate_backtesting(ticker, start_date, end_date, short_window=33, long_window=144):
    data = yf.download(ticker, start=start_date, end=end_date, interval='1d')

    # If volume_threshold is not provided, set it as the 70% of average volume
    volume_threshold = data['Volume'].mean() * 0.7

    # Calculate support and resistance
    window = 20  # Define the window for local maxima and minima detection
    data = calculate_supply_demand_zones(data, window, volume_threshold)

    signals = ma_crossover_strategy(data['Close'], short_window, long_window)
    results = backtest(data, signals)
    total_return, annual_return, sharpe_ratio = calculate_metrics(results)

    print(f"Initial investment: {results['total'][0]}")
    print(f"Final portfolio value: {results['total'][-1]}")
    print(f"Total return: {total_return * 100}%")
    print(f"Annual return: {annual_return * 100}%")
    print(f"Sharpe ratio: {sharpe_ratio}")

    # Generate signals using the strategy
    signals = ma_crossover_strategy(data['Close'], short_window, long_window)
    plot_signals(data, signals, short_window, long_window)

    return results


def plot_signals(data, signals, short_window, long_window):
    # Plot the closing price
    _, ax = plt.subplots(figsize=(16, 9))
    ax.plot(data.index, data['Close'], label='Close Price', color='blue')

    # Plot the short and long moving averages
    ax.plot(data.index, calculate_smma(data['Close'], short_window), label=f'Short SMMA ({short_window} days)',
            color='green')
    ax.plot(data.index, calculate_smma(data['Close'], long_window), label=f'Long SMMA ({long_window} days)',
            color='red')

    # Add the demand zones as red boxes
    for i in range(len(data['demand'])):
        if not np.isnan(data['demand'].iloc[i]):
            ax.add_patch(patches.Rectangle((data.index[i], data['Low'].iloc[i]), width=timedelta(days=window),
                                           height=data['High'].iloc[i] - data['Low'].iloc[i], color='r', alpha=0.2))

    # Add the supply zones as green boxes
    for i in range(len(data['supply'])):
        if not np.isnan(data['supply'].iloc[i]):
            ax.add_patch(patches.Rectangle((data.index[i], data['Low'].iloc[i]), width=timedelta(days=window),
                                           height=data['High'].iloc[i] - data['Low'].iloc[i], color='g', alpha=0.2))

    # Plot buy signals
    buys = signals.loc[signals.positions == 1.0]
    ax.plot(buys.index, data.loc[buys.index]['Close'], '^', markersize=10, color='m', label='Buy')

    # Plot sell signals
    sells = signals.loc[signals.positions == -1.0]
    ax.plot(sells.index, data.loc[sells.index]['Close'], 'v', markersize=10, color='k', label='Sell')

    # Show the plot with a legend
    ax.set_title('Moving Average Crossover Strategy with Supply and Demand Zones')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend(loc='best')
    ax.grid(True)
    plt.show()


results = automate_backtesting('PLTR', '2020-05-04', '2023-07-28')
