import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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
    df = pd.read_csv(file_path, skiprows=1).iloc[::-1].reset_index(drop=True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def add_indicators_to_df(df):
    """Add AD and TMA indicators to the DataFrame."""
    high, low, close, volume = df['High'].values, df['Low'].values, df['Close'].values, df['Volume USDT'].values
    df['AD'] = calculate_ad(high, low, close, volume)
    df['TMA_AD'] = triangular_moving_average(df['AD'].values, 66)
    df['TMA_PRICE'] = triangular_moving_average(df['Close'].values, 66)


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


# Main script
file_path = './data/Binance_IOTAUSDT_d.csv'
df = load_and_prepare_data(file_path)
add_indicators_to_df(df)
plot_price_and_tma(df)
plot_ad_and_tma_ad(df)
