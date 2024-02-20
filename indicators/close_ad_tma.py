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
    #df = pd.read_csv(file_path, skiprows=0).iloc[::-1].reset_index(drop=True) reverse order
    df = pd.read_csv(file_path, skiprows=0)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def add_indicators_to_df(df):
    """Add AD and TMA indicators to the DataFrame."""
    high, low, close, volume = df['High'].values, df['Low'].values, df['Close'].values, df['Volume'].values
    df['AD'] = calculate_ad(high, low, close, volume)
    df['TMA_AD'] = triangular_moving_average(df['AD'].values, 66)
    df['TMA_PRICE'] = triangular_moving_average(df['Close'].values, 66)
    df['SMA_PRICE'] = smoothed_moving_average(df['Close'], 14)
    df['SMA_AD'] = smoothed_moving_average(df['AD'], 14)


def plot_price_and_tma(df):
    """Plot Close Price and TMA_PRICE with fill between."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df['Date'], df['Close'], label='Close Price', color='black', alpha=0.75)
    ax.plot(df['Date'], df['TMA_PRICE'], label='TMA of Price', color='red', alpha=0.75)
    ax.fill_between(df['Date'], df['Close'], df['TMA_PRICE'], where=df['Close'] >= df['TMA_PRICE'], color='gold',
                    alpha=0.5, label='Above TMA')
    ax.fill_between(df['Date'], df['Close'], df['TMA_PRICE'], where=df['Close'] < df['TMA_PRICE'], color='blue',
                    alpha=0.5, label='Below TMA')
    ax.set_title('Close Price and TMA with Fill')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_price_and_tma_with_sma_crossover(df):
    """Plot Close Price, TMA_PRICE, and SMA_PRICE with fill between based on SMA/TMA crossovers."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plotting the lines
    ax.plot(df['Date'], df['Close'], label='Close Price', color='black', alpha=0.75)
    ax.plot(df['Date'], df['TMA_PRICE'], label='TMA of Price', color='red', alpha=0.75)
    ax.plot(df['Date'], df['SMA_PRICE'], label='SMA of Price', color='green', alpha=0.75)

    # Calculate differences for crossover detection
    sma_tma_diff = df['SMA_PRICE'] - df['TMA_PRICE']
    prev_sma_tma_diff = sma_tma_diff.shift(1)

    # Define crossovers
    crossover_up = (prev_sma_tma_diff < 0) & (sma_tma_diff >= 0)
    crossover_down = (prev_sma_tma_diff > 0) & (sma_tma_diff <= 0)

    ax.scatter(df['Date'][crossover_up], df['SMA_PRICE'][crossover_up], color='orange', label='Crossover Up', zorder=5)
    ax.scatter(df['Date'][crossover_down], df['SMA_PRICE'][crossover_down], color='darkblue', label='Crossover Down', zorder=5)

    # Fill based on crossovers
    ax.fill_between(df['Date'], df['SMA_PRICE'], df['TMA_PRICE'], where=df['SMA_PRICE'] >= df['TMA_PRICE'], color='gold', alpha=0.5, label='SMA Crossover Up')
    ax.fill_between(df['Date'], df['SMA_PRICE'], df['TMA_PRICE'], where=df['SMA_PRICE'] < df['TMA_PRICE'], color='blue', alpha=0.5, label='SMA Crossover Down')

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


def plot_ad_and_tma_ad(df):
    """Plot AD and TMA_AD with fill between."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df['Date'], df['AD'], label='AD', color='black', alpha=0.75)
    ax.plot(df['Date'], df['TMA_AD'], label='TMA of AD', color='red', alpha=0.75)
    ax.fill_between(df['Date'], df['AD'], df['TMA_AD'], where=df['AD'] >= df['TMA_AD'], color='gold', alpha=0.5,
                    label='AD above TMA')
    ax.fill_between(df['Date'], df['AD'], df['TMA_AD'], where=df['AD'] < df['TMA_AD'], color='blue', alpha=0.5,
                    label='AD below TMA')
    ax.set_title('AD and TMA of AD with Fill')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_ad_with_tma_price_colors(df):
    """
    Plot the AD chart with background colors based on TMA_PRICE colors
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot AD
    ax.plot(df['Date'], df['AD'], label='AD', color='black', alpha=0.75)

    ax.fill_between(df['Date'], ax.get_ylim()[0], ax.get_ylim()[1], where=df['TMA_PRICE'] <= df['SMA_PRICE'], color='yellow', alpha=0.2, label='TMA > Close (Yellow)')
    ax.fill_between(df['Date'], ax.get_ylim()[0], ax.get_ylim()[1], where=df['TMA_PRICE'] >= df['SMA_PRICE'], color='blue', alpha=0.2, label='TMA <= Close (Blue)')

    # Labeling and formatting
    ax.set_title('AD with TMA_PRICE Colors')
    ax.set_xlabel('Date')
    ax.set_ylabel('AD Value')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    # Show plot
    plt.show()


# Main script
file_path = './data/Bitcoin Price (Oct2015-2022)_daily.csv'
df = load_and_prepare_data(file_path)
add_indicators_to_df(df)
plot_price_and_tma_with_sma_crossover(df)
plot_ad_with_tma_price_colors(df)

'''Strategy'''


def simulate_trading_strategy_with_ad_signals(df):
    df['Next_Open'] = df['Open'].shift(-1)  # Next day's opening price for trade execution

    usd_balance = 1.0
    asset_holding = 0.0
    buy_signals, sell_signals = [], []  # Lists to store buy and sell dates

    for i in range(len(df) - 1):  # Exclude the last row as it has no Next_Open
        ad = df.loc[i, 'AD']
        tma_ad = df.loc[i, 'TMA_AD']
        sma_ad = df.loc[i, 'SMA_AD']
        next_open_price = df.loc[i, 'Next_Open']

        if sma_ad > tma_ad and usd_balance > 0:  # Buy condition based on AD
            asset_holding = usd_balance / next_open_price
            usd_balance = 0
            buy_signals.append(df.loc[i, 'Date'])  # Record buy date

        elif sma_ad < tma_ad and asset_holding > 0:  # Sell condition based on AD falling below TMA_AD
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
    ax.scatter(buy_signals, df[df['Date'].isin(buy_signals)]['AD'], label='Buy Signal', marker='^', color='green',
               alpha=1.0)
    ax.scatter(sell_signals, df[df['Date'].isin(sell_signals)]['AD'], label='Sell Signal', marker='v', color='red',
               alpha=1.0)

    ax.fill_between(df['Date'], ax.get_ylim()[0], ax.get_ylim()[1], where=df['TMA_PRICE'] <= df['SMA_PRICE'], color='yellow', alpha=0.2, label='TMA > Close (Yellow)')
    ax.fill_between(df['Date'], ax.get_ylim()[0], ax.get_ylim()[1], where=df['TMA_PRICE'] >= df['SMA_PRICE'], color='blue', alpha=0.2, label='TMA <= Close (Blue)')

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
plot_ad_tma_sma_with_signals(df, buy_signals, sell_signals)
