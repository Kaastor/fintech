from datetime import datetime, timedelta

import cryptocompare
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from strategy.PS.util import calculate_ad, triangular_moving_average, smoothed_moving_average


# TODO: analiza prędkości (siły trendu) jako 2 pochodna TMA_AD i wchodzić tylko jak ROŚNIE (a nie spada)!
# FIXME: na btc nie pokazało na 1h wejścia poprawnego gold P&V
def load_and_prepare_data_daily(file_path, ticker, live):
    """Load and prepare data from CSV file."""
    if live is False:
        df = pd.read_csv(file_path, skiprows=0).iloc[::-1].reset_index(drop=True)
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        df = pd.DataFrame(cryptocompare.get_historical_price_day(ticker, currency='USDT', exchange='Binance'))
        if True is df.empty:
            return None
        df['Date'] = pd.to_datetime(df['time'], unit='s')
        df['Close'] = df['close']
        df['High'] = df['high']
        df['Low'] = df['low']
        df['Open'] = df['open']
        df['Volume'] = df['volumefrom']

    return df


def get_binance_price(symbol):
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    response = requests.get(url)
    data = response.json()
    return data


def load_and_prepare_data_hour(ticker):
    data = get_binance_price(ticker)
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
    df['SMA_AD'] = smoothed_moving_average(df['AD'], 14)
    df['TMA_AD'] = triangular_moving_average(df['AD'].values, 59)

    df['TMA_PRICE'] = triangular_moving_average(df['Close'].values, 60)
    df['SMA_PRICE'] = smoothed_moving_average(df['Close'], 14)

    df['SMA_33'] = smoothed_moving_average(df['Close'], 33)
    df['SMA_144'] = smoothed_moving_average(df['Close'], 144)

    df['TMA_AD_1DERIVATIVE'] = df['TMA_AD'].diff()
    df['TMA_AD_2DERIVATIVE'] = df['TMA_AD_1DERIVATIVE'].diff()
    df['TMA_AD_TREND'] = 'Flat'
    df.loc[df['TMA_AD_1DERIVATIVE'] > 0, 'TMA_AD_TREND'] = 'Ascending'
    df.loc[df['TMA_AD_1DERIVATIVE'] < 0, 'TMA_AD_TREND'] = 'Descending'
    df['TMA_AD_CROSSES'] = np.where(df['AD'] >= df['TMA_AD'], 'Ascending', 'Descending')


def plot_price(df, ticker, tf):
    """Plot Close Price, TMA_PRICE, and SMA_PRICE with fill between based on SMA/TMA crossovers."""
    fig, ax = plt.subplots(figsize=(14, 7))

    signal_one(df, ax, tf)

    # Plotting the lines
    ax.plot(df['Date'], df['TMA_PRICE'], label='TMA of Price', color='red', alpha=0.75)
    ax.plot(df['Date'], df['SMA_PRICE'], label='SMA of Price', color='blue', alpha=0.75)

    ax.fill_between(df['Date'], df['SMA_PRICE'], df['TMA_PRICE'], where=df['SMA_PRICE'] >= df['TMA_PRICE'],
                    color='gold', alpha=0.5, label='SMA Crossover Up')
    ax.fill_between(df['Date'], df['SMA_PRICE'], df['TMA_PRICE'], where=df['SMA_PRICE'] < df['TMA_PRICE'], color='blue',
                    alpha=0.5, label='SMA Crossover Down')

    ax.fill_between(df['Date'], ax.get_ylim()[0], ax.get_ylim()[1], where=df['TMA_AD'] <= df['SMA_AD'], color='yellow',
                    alpha=0.2, label='TMA > Close (Yellow)')
    ax.fill_between(df['Date'], ax.get_ylim()[0], ax.get_ylim()[1], where=df['TMA_AD'] >= df['SMA_AD'], color='blue',
                    alpha=0.2, label='TMA <= Close (Blue)')

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


def plot_volume(df, ticker, tf):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df['Date'], df['TMA_AD'], label='TMA_AD', color='red', alpha=0.75)
    ax.plot(df['Date'], df['SMA_AD'], label='SMA_AD (12)', color='blue', alpha=0.75)  # SMA line

    signal_one(df, ax, tf)

    ax.fill_between(df['Date'], df['AD'], df['TMA_AD'], where=df['AD'] >= df['TMA_AD'], color='gold', alpha=0.5)
    ax.fill_between(df['Date'], df['AD'], df['TMA_AD'], where=df['AD'] < df['TMA_AD'], color='blue', alpha=0.5)

    ax.set_title('AD, TMA_AD, and SMA_AD with Buy/Sell Signals')
    ax.set_xlabel('Date')
    ax.set_ylabel('AD Value')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


'''Strategy'''

'''
Price gold,
AD gold
'''


def signal_one(df, ax=None, tf='day'):
    # Initialize a variable to track the previous row's SMA_PRICE and TMA_PRICE
    previous_sma_price = None
    previous_tma_price = None
    dates = []

    # Iterate through the DataFrame
    for index, row in df.iterrows():
        # Check if this is the first row
        if previous_sma_price is None or previous_tma_price is None:
            previous_sma_price = row['SMA_PRICE']
            previous_tma_price = row['TMA_PRICE']
            continue

        if previous_sma_price <= previous_tma_price and row['SMA_PRICE'] > row['TMA_PRICE']:
            if row['SMA_AD'] > row['TMA_AD']:
                if tf == 'day':
                    if ax is not None:
                        ax.axvline(x=row['Date'], color='green', alpha=0.7)
                    dates.append(row['Date'])
                elif tf == 'hour':
                    if row['SMA_33'] > row['SMA_144']:
                        if ax is not None:
                            ax.axvline(x=row['Date'], color='green', alpha=0.7)
                        dates.append(row['Date'])

        # Update the previous values for the next iteration
        previous_sma_price = row['SMA_PRICE']
        previous_tma_price = row['TMA_PRICE']
    # print(f'{tf}, {dates}')
    return dates


# Main script
ticker = 'btc'
file_path = '../indicators/data/Binance_BTCUSDT_d.csv'
df = load_and_prepare_data_daily(file_path, ticker, True)
add_indicators_to_df(df)
plot_price(df, ticker, 'day')
plot_volume(df, ticker, 'day')

df = load_and_prepare_data_hour(ticker)
add_indicators_to_df(df)
plot_price(df, ticker, 'hour')
plot_volume(df, ticker, 'hour')

print(signal_one(df, tf='hour'))


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


'''Scan'''
# for ticker in remove_usdt_from_symbols(list_binance_perpetuals()):
#     df = load_and_prepare_data_hour(ticker)
#     if df is None:
#         continue
#     add_indicators_to_df(df)
#     dates = signal_one(df, tf='hour')
#     filtered_dates = [ts for ts in dates if datetime.now() - ts < timedelta(hours=3)]
#     print(f'{ticker}-{filtered_dates}')
