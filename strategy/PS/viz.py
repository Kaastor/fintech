import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cryptocompare
import requests

from strategy.PS.util import exponential_moving_average, calculate_ad, triangular_moving_average, \
    smoothed_moving_average


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


def highlight_ad_trend(df):
    # Initialize the start variables for both Ascending and Descending trends
    start_asc = None
    start_desc = None
    # Loop through the DataFrame to find Ascending and Descending segments
    for i in range(1, len(df)):
        # Check for Ascending segments
        if df['TMA_AD_TREND'].iloc[i] == 'Ascending' and start_asc is None:
            start_asc = i  # Mark the start of an Ascending segment
        elif df['TMA_AD_TREND'].iloc[i] != 'Ascending' and start_asc is not None:
            # Highlight the Ascending segment found
            plt.fill_betweenx(y=[df['TMA_AD'].min(), df['TMA_AD'].max()], x1=df['Date'].iloc[start_asc], x2=df['Date'].iloc[i], color='gold', alpha=0.1)
            start_asc = None  # Reset start for the next Ascending segment
        # Check for Descending segments
        if df['TMA_AD_TREND'].iloc[i] == 'Descending' and start_desc is None:
            start_desc = i  # Mark the start of a Descending segment
        elif df['TMA_AD_TREND'].iloc[i] != 'Descending' and start_desc is not None:
            # Highlight the Descending segment found
            plt.fill_betweenx(y=[df['TMA_AD'].min(), df['TMA_AD'].max()], x1=df['Date'].iloc[start_desc], x2=df['Date'].iloc[i], color='blue', alpha=0.1)
            start_desc = None  # Reset start for the next Descending segment

    # Highlight the last segments if they are Ascending or Descending
    if start_asc is not None:
        plt.fill_betweenx(y=[df['TMA_AD'].min(), df['TMA_AD'].max()], x1=df['Date'].iloc[start_asc], x2=df['Date'].iloc[-1], color='gold', alpha=0.1)
    if start_desc is not None:
        plt.fill_betweenx(y=[df['TMA_AD'].min(), df['TMA_AD'].max()], x1=df['Date'].iloc[start_desc], x2=df['Date'].iloc[-1], color='blue', alpha=0.1)



def add_indicators_to_df(df):
    """Add AD and TMA indicators to the DataFrame."""
    high, low, close, volume = df['High'].values, df['Low'].values, df['Close'].values, df['Volume'].values
    df['AD'] = calculate_ad(high, low, close, volume)
    df['SMA_AD'] = smoothed_moving_average(df['AD'], 14)
    df['TMA_AD'] = triangular_moving_average(df['AD'].values, 59)

    df['TMA_PRICE'] = exponential_moving_average(df['Close'].values, 60)
    df['SMA_PRICE'] = smoothed_moving_average(df['Close'], 14)

    df['TMA_AD_1DERIVATIVE'] = df['TMA_AD'].diff()
    df['TMA_AD_TREND'] = 'Flat'
    df.loc[df['TMA_AD_1DERIVATIVE'] > 0, 'TMA_AD_TREND'] = 'Ascending'
    df.loc[df['TMA_AD_1DERIVATIVE'] < 0, 'TMA_AD_TREND'] = 'Descending'


def plot_price_and_tma_with_sma_crossover(df, ticker):
    """Plot Close Price, TMA_PRICE, and SMA_PRICE with fill between based on SMA/TMA crossovers."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plotting the lines
    ax.plot(df['Date'], df['TMA_PRICE'], label='TMA of Price', color='red', alpha=0.75)
    ax.plot(df['Date'], df['SMA_PRICE'], label='SMA of Price', color='green', alpha=0.75)

    # signal_one(ax)
    highlight_ad_trend(df)

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

'''Safest, most hits but rare
Price gold,
MAs up,
AD gold
'''


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
                ax.axvline(x=row['Date'], color='green', alpha=0.7)
                dates.append(row['Date'])

        # Update the previous values for the next iteration
        previous_sma_price = row['SMA_PRICE']
        previous_tma_price = row['TMA_PRICE']


def plot_ad_tma_sma_with_signals(df):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df['Date'], df['TMA_AD'], label='TMA_AD', color='red', alpha=0.75)
    ax.plot(df['Date'], df['SMA_AD'], label='SMA_AD (12)', color='blue', alpha=0.75)  # SMA line

    ax.fill_between(df['Date'], df['AD'], df['TMA_AD'], where=df['AD'] >= df['TMA_AD'], color='gold', alpha=0.5)
    ax.fill_between(df['Date'], df['AD'], df['TMA_AD'], where=df['AD'] < df['TMA_AD'], color='blue', alpha=0.5)

    # signal_one(ax)
    highlight_ad_trend(df)

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
plot_price_and_tma_with_sma_crossover(df, 'btc')
plot_ad_tma_sma_with_signals(df)
