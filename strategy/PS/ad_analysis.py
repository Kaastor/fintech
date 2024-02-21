import cryptocompare
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from strategy.PS.util import *

'''
IDEA: MA AD do sprzedaży musi być szybsze, żeby znaleźć TOP? 
'''

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
    df['SMA_AD'] = smoothed_moving_average(df['AD'], 14)
    df['SMA_33'] = smoothed_moving_average(df['High'], 33)
    df['SMA_144'] = smoothed_moving_average(df['High'], 144)


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


def plot_price_and_tma_with_sma_crossover(df, buy_signals, sell_signals, ticker):
    """Plot Close Price, TMA_PRICE, and SMA_PRICE with fill between based on SMA/TMA crossovers."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plotting the lines
    ax.plot(df['Date'], df['Close'], label='Close Price', color='black', alpha=0.75)
    ax.plot(df['Date'], df['TMA_PRICE'], label='TMA of Price', color='red', alpha=0.75)
    ax.plot(df['Date'], df['SMA_PRICE'], label='SMA of Price', color='green', alpha=0.75)

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

    ax.set_title('AD, TMA_AD, and SMA_AD with Buy/Sell Signals')
    ax.set_xlabel('Date')
    ax.set_ylabel('AD Value')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


import pandas as pd


# Define your MA functions here
def simple_moving_average(values, window):
    return pd.Series(values).rolling(window=window, min_periods=1).mean().values


# Add other MA functions here (EMA, WMA, etc.)

# Optimization function
def optimize_ma_strategy(df):
    ma_types = {
        'SMA': simple_moving_average,
        'EMA': exponential_moving_average,
        # 'WMA': weighted_moving_average,
        # 'HMA': hull_moving_average,
        # 'TEMA': triple_exponential_moving_average,
        'TMA': triangular_moving_average,
    }
    results = []

    for ma_type_f, ma_function_f in ma_types.items():
        for period in range(3, 20):
            print(f'{ma_type_f}, {period}')
            key = ma_type_f + '_fast'
            df[key] = ma_function_f(df['AD'], period)
            for ma_type, ma_function in ma_types.items():
                for period in range(10, 101):  # Range from 10 to 100
                    df[ma_type] = ma_function(df['AD'], period)

                    # Simulate the trading strategy
                    initial_balance = 1.0  # Starting with $1
                    position_open = False
                    for i in range(1, len(df)):
                        if df['SMA_AD'].iloc[i] > df[ma_type].iloc[i] and not position_open:
                            # Buy
                            position_open = True
                            buy_price = df['Close'].iloc[i]
                        elif df['SMA_AD'].iloc[i] < df[ma_type].iloc[i] and position_open:
                            # Sell
                            position_open = False
                            sell_price = df['Close'].iloc[i]
                            initial_balance *= sell_price / buy_price

                    # Calculate final profit if position remains open at the end
                    if position_open:
                        final_price = df['Close'].iloc[-1]
                        initial_balance *= final_price / buy_price

                    results.append((ma_type, period, initial_balance))

    # Sort results by profit in descending order
    results.sort(key=lambda x: x[2], reverse=True)

    # Format the results
    formatted_results = [f"{ma_type}, {period}, {profit:.2f}" for ma_type, period, profit in results]

    return formatted_results


# Main script
file_path = '../../indicators/data/Bitcoin Price (Oct2015-2022)_daily.csv'
df = load_and_prepare_data(file_path, 'btc', False)
add_indicators_to_df(df)
# plot_price_and_tma_with_sma_crossover(df, None, None, 'btc')
# plot_ad_tma_sma_with_signals(df, None, None)

# Use the function with your DataFrame (make sure it has a 'Close_MA' column)
results = optimize_ma_strategy(df)
print(results)
