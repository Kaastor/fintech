from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cryptocompare
import requests


def smoothed_moving_average(values, window):
    """Calculate the Smoothed Moving Average."""
    return pd.Series(values).rolling(window=window, min_periods=1).mean().values


def triangular_moving_average(values, window):
    """Calculate the Triangular Moving Average."""
    simple_ma = pd.Series(values).rolling(window=window, min_periods=1).mean()
    return simple_ma.rolling(window=window, min_periods=1).mean().values


def calculate_ad(high, low, close, volume):
    """Calculate Accumulation Distribution (AD) indicator."""
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = np.where((high - low) == 0, 0, mfm)
    mfv = mfm * volume
    return np.cumsum(mfv)


def load_and_prepare_data(file_path, ticker, live):
    """Load and prepare data from CSV file."""
    if live is False:
        df = pd.read_csv(file_path, skiprows=0)
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        df = pd.DataFrame(cryptocompare.get_historical_price_hour(ticker, currency='USDT', exchange='Binance'))
        if True is df.empty:
            return None
        df['Date'] = pd.to_datetime(df['time'], unit='s')
        df['Close'] = df['close']
        df['High'] = df['high']
        df['Low'] = df['low']
        df['Open'] = df['open']
        df['Volume'] = df['volumefrom']

    return df


def add_indicators_to_df(df):
    """Add AD and TMA indicators to the DataFrame."""
    high, low, close, volume = df['High'].values, df['Low'].values, df['Close'].values, df['Volume'].values
    df['AD'] = calculate_ad(high, low, close, volume)
    df['TMA_AD'] = triangular_moving_average(df['AD'].values, 60)
    df['TMA_PRICE'] = triangular_moving_average(df['Close'].values, 60)
    df['SMA_PRICE'] = smoothed_moving_average(df['Close'], 14)
    df['SMA_AD'] = smoothed_moving_average(df['AD'], 22)
    df['SMA_33'] = smoothed_moving_average(df['High'], 33)
    df['SMA_144'] = smoothed_moving_average(df['High'], 144)


def plot_price_and_tma_with_sma_crossover(df, buy_signals, sell_signals, ticker):
    """Plot Close Price, TMA_PRICE, and SMA_PRICE with fill between based on SMA/TMA crossovers."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plotting the lines
    ax.plot(df['Date'], df['Close'], label='Close Price', color='black', alpha=0.75)
    ax.plot(df['Date'], df['TMA_PRICE'], label='TMA of Price', color='red', alpha=0.75)
    ax.plot(df['Date'], df['SMA_PRICE'], label='SMA of Price', color='green', alpha=0.75)

    # FIXME ala Larsson Line buy/sell signals only based on Price
    # Calculate differences for crossover detection
    # sma_tma_diff = df['SMA_PRICE'] - df['TMA_PRICE']
    # prev_sma_tma_diff = sma_tma_diff.shift(1)
    # # Define crossovers
    # crossover_up = (prev_sma_tma_diff < 0) & (sma_tma_diff >= 0)
    # crossover_down = (prev_sma_tma_diff > 0) & (sma_tma_diff <= 0)
    #
    # ax.scatter(df['Date'][crossover_up], df['SMA_PRICE'][crossover_up], color='orange', label='Crossover Up', zorder=5)
    # ax.scatter(df['Date'][crossover_down], df['SMA_PRICE'][crossover_down], color='darkblue', label='Crossover Down',
    #            zorder=5)

    # Plot Buy and Sell signals
    if buy_signals is not None and sell_signals is not None:
        for buy_signal in buy_signals:
            ax.axvspan(buy_signal - pd.Timedelta(days=0.5), buy_signal + pd.Timedelta(days=0.5), color='green',
                       alpha=0.7)

        for sell_signal in sell_signals:
            ax.axvspan(sell_signal - pd.Timedelta(days=0.5), sell_signal + pd.Timedelta(days=0.5), color='red',
                       alpha=0.7)
    else:
        signal_two(ax)

    # Fill based on crossovers
    ax.fill_between(df['Date'], df['SMA_PRICE'], df['TMA_PRICE'], where=df['SMA_PRICE'] >= df['TMA_PRICE'],
                    color='gold', alpha=0.5, label='SMA Crossover Up')
    ax.fill_between(df['Date'], df['SMA_PRICE'], df['TMA_PRICE'], where=df['SMA_PRICE'] < df['TMA_PRICE'], color='blue',
                    alpha=0.5, label='SMA Crossover Down')

    # Labeling and formatting
    ax.set_title(f'{ticker}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend(loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

'''Strategy'''

dates = []

'''Safest, most hits but rare'''


def signal_one(ax):
    # Initialize a variable to track the previous row's SMA_PRICE and TMA_PRICE
    previous_sma_price = None
    previous_tma_price = None

    # Iterate through the DataFrame
    for index, row in df.iterrows():
        # Check if this is the first row
        if previous_sma_price is None or previous_tma_price is None:
            previous_sma_price = row['SMA_PRICE']
            previous_tma_price = row['TMA_PRICE']
            continue

        # Check if SMA_PRICE crossed TMA_PRICE to the upside
        if previous_sma_price <= previous_tma_price and row['SMA_PRICE'] > row['TMA_PRICE']:
            # Also ensure SMA_AD > TMA_AD at the crossover point
            if row['SMA_AD'] > row['TMA_AD']:
                # if row['SMA_33'] > row['SMA_144']:
                ax.axvline(x=row['Date'], color='green', alpha=0.7)
                dates.append(row['Date'])

        # Update the previous values for the next iteration
        previous_sma_price = row['SMA_PRICE']
        previous_tma_price = row['TMA_PRICE']


# Using AD MA crossover
def signal_two(ax):
    sma_tma_diff = df['SMA_AD'] - df['TMA_AD']
    prev_sma_tma_diff = sma_tma_diff.shift(1)
    # Get indices of crossovers
    tolerance = 0.01  # Define a suitable tolerance level for your data
    crossover_up = (prev_sma_tma_diff < -tolerance) & (sma_tma_diff >= tolerance)
    crossover_down = (prev_sma_tma_diff > tolerance) & (sma_tma_diff <= -tolerance)
    crossover_up_indices = np.where(crossover_up)[0]
    crossover_down_indices = np.where(crossover_down)[0]

    # Loop through the crossover_up indices and draw vertical lines at each corresponding date
    for idx in crossover_up_indices:
        ax.axvline(x=df['Date'][idx], color='green', label='Crossover Up' if idx == crossover_up[0] else "", zorder=5)


def simulate_trading_strategy_with_ad_signals(df):
    df['Next_Open'] = df['Open'].shift(-1)  # Next day's opening price for trade execution

    buy_signals, sell_signals = [], []  # Lists to store buy and sell dates

    # Simulate the trading strategy
    initial_balance = 1.0  # Starting with $1
    position_open = False
    for i in range(1, len(df)):
        if df['SMA_PRICE'].iloc[i] > df['TMA_PRICE'].iloc[i] and not position_open:
            # Buy
            position_open = True
            buy_price = df['Close'].iloc[i]
            buy_signals.append(df.loc[i, 'Date'])  # Record buy date
        elif df['SMA_PRICE'].iloc[i] < df['TMA_PRICE'].iloc[i] and position_open:
            # Sell
            sell_signals.append(df.loc[i, 'Date'])  # Record sell date
            position_open = False
            sell_price = df['Close'].iloc[i]
            initial_balance *= sell_price / buy_price
    if position_open:
        final_price = df['Close'].iloc[-1]
        initial_balance *= final_price / buy_price

    return initial_balance, buy_signals, sell_signals


def plot_ad_tma_sma_with_signals(df, buy_signals, sell_signals):
    fig, ax = plt.subplots(figsize=(14, 7))
    # ax.plot(df['Date'], df['AD'], label='AD', color='black', alpha=0.75)
    ax.plot(df['Date'], df['TMA_AD'], label='TMA_AD', color='red', alpha=0.75)
    ax.plot(df['Date'], df['SMA_AD'], label='SMA_AD (12)', color='blue', alpha=0.75)  # SMA line

    # Fill areas based on AD and TMA_AD
    ax.fill_between(df['Date'], df['AD'], df['TMA_AD'], where=df['AD'] >= df['TMA_AD'], color='gold', alpha=0.5)
    ax.fill_between(df['Date'], df['AD'], df['TMA_AD'], where=df['AD'] < df['TMA_AD'], color='blue', alpha=0.5)

    # Plot Buy and Sell signals
    if buy_signals is not None and sell_signals is not None:
        for buy_signal in buy_signals:
            ax.axvspan(buy_signal - pd.Timedelta(days=0.5), buy_signal + pd.Timedelta(days=0.5), color='green',
                       alpha=0.5)

        for sell_signal in sell_signals:
            ax.axvspan(sell_signal - pd.Timedelta(days=0.5), sell_signal + pd.Timedelta(days=0.5), color='red',
                       alpha=0.5)
    else:
        signal_two(ax)

    ax.fill_between(df['Date'], ax.get_ylim()[0], ax.get_ylim()[1], where=df['TMA_PRICE'] <= df['SMA_PRICE'],
                    color='yellow', alpha=0.2, label='TMA > Close (Yellow)')
    ax.fill_between(df['Date'], ax.get_ylim()[0], ax.get_ylim()[1], where=df['TMA_PRICE'] >= df['SMA_PRICE'],
                    color='blue', alpha=0.2, label='TMA <= Close (Blue)')

    ax.set_title('AD, TMA_AD, and SMA_AD with Buy/Sell Signals')
    ax.set_xlabel('Date')
    ax.set_ylabel('AD Value')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Main script
file_path = '../indicators/data/Bitcoin Price (Oct2015-2022)_daily.csv'
df = load_and_prepare_data(file_path, 'btc', False)
add_indicators_to_df(df)

# Run the simulation with the new AD-based buy condition
final_balance, buy_signals, sell_signals = simulate_trading_strategy_with_ad_signals(df)
print(f"Final USD Balance: ${final_balance:.2f}")
# Plot the AD, TMA_AD, and buy/sell signals
plot_price_and_tma_with_sma_crossover(df, buy_signals, sell_signals, 'btc')
plot_ad_tma_sma_with_signals(df, buy_signals, sell_signals)

'''Scan'''


def remove_usdt_from_symbols(symbols):
    # Remove 'USDT' from each symbol
    return [symbol.replace('USDT', '') for symbol in symbols]


def list_binance_perpetuals():
    # Binance API endpoint for futures exchange information
    url = 'https://fapi.binance.com/fapi/v1/exchangeInfo'

    try:
        response = requests.get(url)
        response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
        data = response.json()

        # Extract the symbols that are perpetual futures
        perpetuals = [symbol['symbol'] for symbol in data['symbols'] if symbol['contractType'] == 'PERPETUAL']
        return perpetuals

    except requests.RequestException as e:
        print(f"Error fetching data from Binance: {e}")
        return []


def test_tickers():
    return ['btc', 'theta', 'grt', 'icp']

# for ticker in remove_usdt_from_symbols(list_binance_perpetuals()):
# for ticker in test_tickers():
#     df = load_and_prepare_data(file_path, ticker, True)
#     if df is None:
#         continue
#     add_indicators_to_df(df)
#     plot_price_and_tma_with_sma_crossover(df, None, None, ticker)
    # print(f'{ticker} done.')

print(sorted(dates))