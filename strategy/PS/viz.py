from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter1d

import cryptocompare
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import requests

from strategy.PS.util import *

'''
1. Napisz strategię (3x gold? 2x gold?)
2. Przenieś na TW
3. Zrób back-testing

Ważniejsze jest ryzyko niż zysk!
'''


def load_and_prepare_data_daily(file_path, ticker, live):
    """Load and prepare data from CSV file."""
    if live is False:
        df = pd.read_csv(file_path, skiprows=0).reset_index(drop=True)
        # df = pd.read_csv(file_path, skiprows=0).iloc[::-1].reset_index(drop=True)
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
    df['SMA_AD'] = exponential_moving_average(df['AD'], 12)
    df['TMA_AD'] = exponential_moving_average(df['AD'].values, 50)

    df['ADW'] = calculate_ad_williams(high, low, close, df['Open'].values, volume)
    df['SMA_ADW'] = exponential_moving_average(df['ADW'], 12)
    df['TMA_ADW'] = exponential_moving_average(df['ADW'].values, 60)
    df['TMA_ADW_1DERIVATIVE'] = df['TMA_ADW'].diff()
    df['GAUSS_ADW_1DERIVATIVE'] = gaussian_filter1d(df['TMA_ADW_1DERIVATIVE'], sigma=6)

    df['SMA_PRICE'] = exponential_moving_average(df['Close'].values, 12)
    df['TMA_PRICE'] = exponential_moving_average(df['Close'].values, 60)
    df['TMA_PRICE_1DERIVATIVE'] = df['TMA_PRICE'].diff()
    df['GAUSS_PRICE_1DERIVATIVE'] = gaussian_filter1d(df['TMA_PRICE_1DERIVATIVE'], sigma=6)

    df['TMA_AD_1DERIVATIVE'] = df['TMA_AD'].diff()
    df['GAUSS_AD_1DERIVATIVE'] = gaussian_filter1d(df['TMA_AD_1DERIVATIVE'], sigma=6)


def plot_price_volume_and_acceleration(df, ticker, tf):
    fig, ax = plt.subplots(3, 1, figsize=(25, 15), dpi=300)
                           #gridspec_kw={'height_ratios': [3, 3, 3, 1]})  # 2 Rows, 1 Column

    ax[0].plot(df['Date'], df['TMA_PRICE'], label='TMA of Price', color='red', alpha=0.75)
    ax[0].plot(df['Date'], df['SMA_PRICE'], label='SMA of Price', color='blue', alpha=0.75)

    ax[0].fill_between(df['Date'], df['SMA_PRICE'], df['TMA_PRICE'], where=df['SMA_PRICE'] >= df['TMA_PRICE'],
                       color='gold', alpha=0.5, label='SMA Crossover Up')
    ax[0].fill_between(df['Date'], df['SMA_PRICE'], df['TMA_PRICE'], where=df['SMA_PRICE'] < df['TMA_PRICE'],
                       color='blue',
                       alpha=0.5, label='SMA Crossover Down')
    # signal_one(df, ax[0], tf)
    # Labeling and formatting
    ax[0].set_title(f'PRICE-{ticker}')
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax[0].grid(True)

    # Plot 2: AD values, not needed besides 1 derivative
    ax[1].plot(df['Date'], df['TMA_AD'], label='TMA_AD', color='red', alpha=0.75)
    ax[1].plot(df['Date'], df['SMA_AD'], label='SMA_AD (12)', color='blue', alpha=0.75)  # SMA line
    # signal_one(df, ax, tf)
    ax[1].fill_between(df['Date'], df['SMA_AD'], df['TMA_AD'], where=df['SMA_AD'] >= df['TMA_AD'], color='gold', alpha=0.5)
    ax[1].fill_between(df['Date'], df['SMA_AD'], df['TMA_AD'], where=df['SMA_AD'] < df['TMA_AD'], color='blue', alpha=0.5)
    # signal_one(df, ax[1], tf)
    ax[1].set_title(f'AD-{ticker}')
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax[1].set_yticklabels([])
    ax[1].grid(True)

    ax[2].plot(df['Date'], df['TMA_ADW'], label='TMA_ADW', color='red', alpha=0.75)
    ax[2].plot(df['Date'], df['SMA_ADW'], label='SMA_ADW (12)', color='blue', alpha=0.75)  # SMA line
    # signal_one(df, ax, tf)
    ax[2].fill_between(df['Date'], df['SMA_ADW'], df['TMA_ADW'], where=df['SMA_ADW'] >= df['TMA_ADW'], color='gold',
                       alpha=0.5)
    ax[2].fill_between(df['Date'], df['SMA_ADW'], df['TMA_ADW'], where=df['SMA_ADW'] < df['TMA_ADW'], color='blue',
                       alpha=0.5)
    # signal_one(df, ax[2], tf)
    ax[2].set_title(f'ADW-{ticker}')
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax[2].set_yticklabels([])
    ax[2].grid(True)
    # Displaying the corrected plots
    plt.tight_layout()
    plt.show()


'''Strategy'''


def is_ad_gold(last_rows):
    return all(row["GAUSS_AD_1DERIVATIVE"] > 0 for row in last_rows)


def is_ADW_gold(last_rows):
    return all(row["GAUSS_ADW_1DERIVATIVE"] > 0 for row in last_rows)


def is_price_gold(last_rows):
    return all(row["GAUSS_PRICE_1DERIVATIVE"] > 0 for row in last_rows)


def is_price_gold_crossover(last_rows):
    return last_rows[0]['SMA_PRICE'] < last_rows[0]['TMA_PRICE'] and last_rows[1]['SMA_PRICE'] > last_rows[1][
        'TMA_PRICE']


def is_ad_gold_crossover(last_rows):
    return last_rows[0]['SMA_AD'] < last_rows[0]['TMA_AD'] and last_rows[1]['SMA_AD'] > last_rows[1]['TMA_AD']


def is_ADW_gold_crossover(last_rows):
    return last_rows[0]['SMA_ADW'] < last_rows[0]['TMA_ADW'] and last_rows[1]['SMA_ADW'] > last_rows[1]['TMA_ADW']


def is_ad_gold_accelerating(rows):
    # Calculate the first differences between consecutive elements
    diffs = [rows[i]["GAUSS_AD_1DERIVATIVE"] - rows[i - 1]["GAUSS_AD_1DERIVATIVE"] for i in range(1, len(rows))]
    # Check if each subsequent difference is greater than the previous one
    for i in range(1, len(diffs)):
        if not (diffs[i] >= diffs[i - 1] and rows[i + 1]["GAUSS_AD_1DERIVATIVE"] >= rows[i]["GAUSS_AD_1DERIVATIVE"]):
            return False

    return True


def only_ad(last_rows):
    return (is_price_gold_crossover(last_rows) or is_ad_gold_crossover(last_rows)) and \
           is_ad_gold(last_rows) and is_price_gold(last_rows) and \
           is_ad_gold_accelerating(last_rows)


def ad_ADW(last_rows):
    return is_ad_gold(last_rows) and \
           ((is_price_gold_crossover(last_rows) and is_ADW_gold(last_rows)) or is_ADW_gold_crossover(last_rows))


def signal_one(df, ax=None, tf='day'):
    dates = []
    last_rows_len = 3
    last_rows = []
    # Iterate through the DataFrame
    for index, row in df.iterrows():
        if index < last_rows_len:
            last_rows.append(row)
        else:
            if ad_ADW(last_rows):
                ax.axvline(x=row['Date'], color='green', alpha=0.7)
                dates.append(row['Date'])
            last_rows = last_rows[1:]
            last_rows.append(row)
    # print(f'{tf}, {dates}')
    return dates


# Main script
ticker = 'xrp'
file_path = '../indicators/data/Bitcoin Price (2014-2023)_daily.csv'
df = load_and_prepare_data_daily(file_path, ticker, True)
add_indicators_to_df(df)
plot_price_volume_and_acceleration(df, ticker, 'day')


# df = load_and_prepare_data_hour(ticker)
# add_indicators_to_df(df)
# plot_price_volume_and_acceleration(df, ticker, 'hour')


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
