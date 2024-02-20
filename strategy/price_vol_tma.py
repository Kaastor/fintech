import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def smoothed_moving_average(values, window):
    """Calculate the Smoothed Moving Average."""
    return pd.Series(values).rolling(window=window, min_periods=1).mean().values


def triangular_moving_average(values, window):
    """Calculate the Triangular Moving Average."""
    weights = np.arange(1, window + 1)
    tma = np.convolve(values, weights / weights.sum(), 'valid')
    return np.concatenate((np.full(window - 1, np.nan), tma))


def calculate_ad(high, low, close, volume):
    """Calculate Accumulation Distribution (AD) indicator."""
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = np.where((high - low) == 0, 0, mfm)
    mfv = mfm * volume
    return np.cumsum(mfv)


def load_and_prepare_data(file_path):
    """Load and prepare data from CSV file."""
    df = pd.read_csv(file_path, skiprows=0)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def add_indicators_to_df(df):
    """Add AD and TMA indicators to the DataFrame."""
    high, low, close, volume = df['High'].values, df['Low'].values, df['Close'].values, df['Volume'].values
    df['AD'] = calculate_ad(high, low, close, volume)
    df['TMA_AD'] = triangular_moving_average(df['AD'].values, 60)
    df['TMA_PRICE'] = triangular_moving_average(df['Close'].values, 60)
    df['SMA_PRICE'] = smoothed_moving_average(df['Close'], 14)
    df['SMA_AD'] = smoothed_moving_average(df['AD'], 14)
    df['SMA_33'] = smoothed_moving_average(df['Close'], 33)
    df['SMA_144'] = smoothed_moving_average(df['Close'], 144)


def plot_price_and_tma_with_sma_crossover(df, buy_signals, sell_signals):
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
        signal_one(ax)

    # Fill based on crossovers
    ax.fill_between(df['Date'], df['SMA_PRICE'], df['TMA_PRICE'], where=df['SMA_PRICE'] >= df['TMA_PRICE'],
                    color='gold', alpha=0.5, label='SMA Crossover Up')
    ax.fill_between(df['Date'], df['SMA_PRICE'], df['TMA_PRICE'], where=df['SMA_PRICE'] < df['TMA_PRICE'], color='blue',
                    alpha=0.5, label='SMA Crossover Down')

    # Labeling and formatting
    ax.set_title('Close Price, TMA, and SMA with Crossovers')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend(loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Main script
file_path = '../indicators/data/Bitcoin Price (Oct2015-2022)_daily.csv'
df = load_and_prepare_data(file_path)
add_indicators_to_df(df)
'''Strategy'''


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
                if row['SMA_33'] > row['SMA_144']:
                    ax.axvline(x=row['Date'], color='green', alpha=0.7)
                    print(row['Date'])

        # Update the previous values for the next iteration
        previous_sma_price = row['SMA_PRICE']
        previous_tma_price = row['TMA_PRICE']


def signal_two(ax):
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

        # Update the previous values for the next iteration
        previous_sma_price = row['SMA_PRICE']
        previous_tma_price = row['TMA_PRICE']


def simulate_trading_strategy_with_ad_signals(df):
    df['Next_Open'] = df['Open'].shift(-1)  # Next day's opening price for trade execution

    usd_balance = 1.0
    asset_holding = 0.0
    buy_signals, sell_signals = [], []  # Lists to store buy and sell dates

    for i in range(len(df) - 1):  # Exclude the last row as it has no Next_Open
        ad = df.loc[i, 'AD']
        tma_ad = df.loc[i, 'TMA_AD']
        tma_price = df.loc[i, 'TMA_PRICE']
        sma_ad = df.loc[i, 'SMA_AD']
        sma_price = df.loc[i, 'SMA_PRICE']
        next_open_price = df.loc[i, 'Next_Open']

        if sma_ad > tma_ad and usd_balance > 0:  # Buy condition
            asset_holding = usd_balance / next_open_price
            usd_balance = 0
            buy_signals.append(df.loc[i, 'Date'])  # Record buy date

        # Sell condition - sell when first blue appears, then wait for buy
        elif sma_ad < tma_ad and asset_holding > 0:
            usd_balance = asset_holding * next_open_price
            asset_holding = 0
            sell_signals.append(df.loc[i, 'Date'])  # Record sell date

    final_usd_balance = usd_balance + asset_holding * df.iloc[-1]['Open']

    return final_usd_balance, buy_signals, sell_signals


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
        signal_one(ax)

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


# Run the simulation with the new AD-based buy condition
final_balance, buy_signals, sell_signals = simulate_trading_strategy_with_ad_signals(df)
print(f"Final USD Balance: ${final_balance:.2f}")
# Plot the AD, TMA_AD, and buy/sell signals
plot_price_and_tma_with_sma_crossover(df, None, None)
plot_ad_tma_sma_with_signals(df, None, None)
