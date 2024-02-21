import cryptocompare
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def smoothed_moving_average(values, window):
    """Calculate the Smoothed Moving Average."""
    return pd.Series(values).rolling(window=window, min_periods=1).mean().values


def exponential_moving_average(values, span):
    """Calculate the Exponential Moving Average."""
    return pd.Series(values).ewm(span=span, adjust=False).mean().values


def weighted_moving_average(values, window):
    """Calculate the Weighted Moving Average."""
    weights = np.arange(1, window + 1)
    return pd.Series(values).rolling(window=window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True).values


def hull_moving_average(values, window):
    """Calculate the Hull Moving Average."""
    half_window = int(window / 2)
    sqrt_window = int(np.sqrt(window))
    wma_half = weighted_moving_average(values, half_window)
    wma_full = weighted_moving_average(values, window)
    raw_hma = 2 * wma_half - wma_full
    return pd.Series(raw_hma).rolling(window=sqrt_window).apply(
        lambda x: np.dot(x, np.arange(1, sqrt_window + 1)) / np.arange(1, sqrt_window + 1).sum(), raw=True).values


def triple_exponential_moving_average(values, span):
    """Calculate the Triple Exponential Moving Average."""
    ema1 = pd.Series(values).ewm(span=span, adjust=False).mean()
    ema2 = ema1.ewm(span=span, adjust=False).mean()
    ema3 = ema2.ewm(span=span, adjust=False).mean()
    return (3 * ema1 - 3 * ema2 + ema3).values


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
        'WMA': weighted_moving_average,
        'HMA': hull_moving_average,
        'TEMA': triple_exponential_moving_average,
        'TMA': triangular_moving_average,
    }
    results = []

    for ma_type, ma_function in ma_types.items():
        for period in range(10, 101):  # Range from 10 to 100
            df[ma_type] = ma_function(df['Close'], period)

            # Simulate the trading strategy
            initial_balance = 1.0  # Starting with $1
            position_open = False
            for i in range(1, len(df)):
                if df['SMA_PRICE'].iloc[i] > df[ma_type].iloc[i] and not position_open:
                    # Buy
                    position_open = True
                    buy_price = df['Close'].iloc[i]
                elif df['SMA_PRICE'].iloc[i] < df[ma_type].iloc[i] and position_open:
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

# ['WMA, 10, 100518.90', 'WMA, 11, 91378.55', 'WMA, 12, 83406.84', 'SMA, 10, 73948.81', 'WMA, 13, 72477.96', 'SMA,
# 11, 63911.13', 'WMA, 14, 63608.60', 'WMA, 15, 58428.16', 'SMA, 12, 54298.30', 'WMA, 16, 51546.46', 'EMA, 10,
# 49468.29', 'SMA, 13, 45660.27', 'WMA, 17, 45299.15', 'EMA, 11, 42114.73', 'WMA, 18, 40265.52', 'SMA, 14, 38954.03',
# 'HMA, 26, 36720.43', 'WMA, 19, 36544.71', 'EMA, 12, 35439.99', 'HMA, 18, 34538.30', 'HMA, 16, 33954.20', 'WMA, 20,
# 33538.26', 'SMA, 15, 33482.14', 'HMA, 22, 32387.02', 'HMA, 20, 31952.68', 'EMA, 13, 31863.92', 'HMA, 24, 31160.93',
# 'HMA, 28, 31122.72', 'WMA, 21, 30607.01', 'SMA, 16, 29563.73', 'EMA, 14, 29317.43', 'HMA, 27, 28926.68', 'HMA, 25,
# 28652.90', 'WMA, 22, 27862.46', 'HMA, 30, 27534.99', 'HMA, 32, 26460.28', 'TMA, 10, 26399.93', 'WMA, 23, 24983.18',
# 'SMA, 17, 24891.87', 'HMA, 29, 24371.15', 'EMA, 15, 24052.18', 'HMA, 23, 23823.15', 'HMA, 34, 23788.19', 'HMA, 36,
# 23121.68', 'EMA, 16, 22500.61', 'HMA, 31, 22333.30', 'SMA, 18, 21932.43', 'HMA, 33, 21819.64', 'WMA, 24, 21654.81',
# 'HMA, 10, 21521.73', 'HMA, 38, 21086.73', 'TMA, 11, 20315.92', 'HMA, 37, 20197.76', 'HMA, 21, 20136.47', 'HMA, 35,
# 20127.09', 'EMA, 17, 19995.32', 'SMA, 19, 19952.79', 'WMA, 25, 19775.39', 'HMA, 40, 19607.92', 'HMA, 17, 19585.93',
# 'HMA, 42, 19298.58', 'HMA, 12, 19282.95', 'HMA, 14, 18721.25', 'WMA, 26, 18676.69', 'HMA, 39, 18551.82', 'HMA, 44,
# 18551.28', 'HMA, 19, 18470.62', 'HMA, 41, 18122.39', 'HMA, 43, 17615.94', 'SMA, 20, 17351.43', 'HMA, 46, 17183.92',
# 'EMA, 18, 17145.66', 'WMA, 27, 16883.77', 'HMA, 45, 16690.34', 'TMA, 12, 16089.41', 'EMA, 19, 15662.24', 'HMA, 49,
# 15377.85', 'HMA, 48, 15326.88', 'SMA, 21, 15309.38', 'WMA, 28, 15045.29', 'HMA, 47, 14710.61', 'EMA, 20, 14473.57',
# 'HMA, 50, 14284.54', 'HMA, 51, 14017.90', 'SMA, 22, 13828.52', 'WMA, 29, 13783.13', 'TEMA, 33, 13478.38', 'TEMA,
# 35, 13300.08', 'EMA, 21, 13274.18', 'TMA, 13, 13240.08', 'TEMA, 34, 12931.86', 'TEMA, 37, 12849.03', 'HMA, 52,
# 12845.60', 'TEMA, 36, 12806.02', 'TEMA, 38, 12617.52', 'TEMA, 39, 12522.54', 'WMA, 30, 12411.91', 'SMA, 23,
# 12368.93', 'TEMA, 40, 12326.44', 'TEMA, 32, 12188.42', 'HMA, 53, 12067.52', 'EMA, 22, 11962.90', 'TEMA, 41,
# 11884.32', 'WMA, 31, 11749.15', 'TEMA, 30, 11735.70', 'TEMA, 42, 11733.51', 'HMA, 54, 11554.10', 'HMA, 56,
# 11307.06', 'TEMA, 27, 11278.46', 'TEMA, 28, 11271.83', 'HMA, 55, 11257.12', 'SMA, 24, 11235.88', 'EMA, 23,
# 11227.95', 'TEMA, 29, 11215.86', 'TEMA, 25, 11084.40', 'TEMA, 31, 11082.66', 'TMA, 14, 11079.83', 'HMA, 57,
# 11067.02', 'HMA, 58, 11046.45', 'WMA, 32, 11033.98', 'TEMA, 43, 10974.21', 'TEMA, 26, 10853.55', 'TEMA, 44,
# 10488.27', 'HMA, 59, 10465.14', 'HMA, 60, 10440.66', 'TEMA, 24, 10368.80', 'EMA, 24, 10056.98', 'WMA, 33,
# 10028.17', 'SMA, 25, 9963.23', 'TEMA, 45, 9858.59', 'TEMA, 46, 9811.30', 'TEMA, 47, 9736.71', 'TEMA, 23, 9416.34',
# 'HMA, 61, 9356.92', 'EMA, 25, 9242.02', 'TMA, 15, 9162.50', 'HMA, 62, 9119.72', 'SMA, 26, 9102.65', 'TEMA, 48,
# 8962.08', 'WMA, 34, 8929.34', 'EMA, 26, 8872.15', 'SMA, 27, 8620.38', 'EMA, 27, 8564.05', 'HMA, 63, 8541.06', 'WMA,
# 35, 8137.93', 'TEMA, 22, 8130.90', 'TEMA, 49, 7987.12', 'EMA, 28, 7967.33', 'HMA, 64, 7966.49', 'TMA, 16, 7891.46',
# 'TEMA, 50, 7867.37', 'SMA, 28, 7634.14', 'EMA, 29, 7519.94', 'WMA, 36, 7489.60', 'TEMA, 51, 7418.62', 'HMA, 65,
# 7233.09', 'EMA, 30, 7071.24', 'SMA, 29, 7034.64', 'TEMA, 52, 7028.14', 'WMA, 37, 7005.50', 'TEMA, 54, 6890.32',
# 'TEMA, 21, 6843.74', 'TEMA, 53, 6732.95', 'EMA, 31, 6713.54', 'SMA, 30, 6690.91', 'HMA, 66, 6683.62', 'TEMA, 55,
# 6676.40', 'TMA, 17, 6561.14', 'HMA, 67, 6414.26', 'WMA, 38, 6331.43', 'TEMA, 56, 6325.37', 'HMA, 68, 6221.95',
# 'SMA, 31, 6115.26', 'TEMA, 60, 6112.39', 'EMA, 32, 6097.62', 'TEMA, 64, 6083.44', 'TEMA, 57, 6026.76', 'TEMA, 62,
# 5986.90', 'HMA, 69, 5976.26', 'EMA, 33, 5971.55', 'TEMA, 61, 5887.83', 'TEMA, 58, 5886.09', 'HMA, 70, 5873.68',
# 'TEMA, 59, 5842.09', 'WMA, 39, 5811.45', 'TEMA, 65, 5783.26', 'TEMA, 66, 5759.37', 'TEMA, 63, 5745.14', 'TEMA, 67,
# 5709.92', 'SMA, 32, 5688.33', 'TEMA, 68, 5623.33', 'EMA, 34, 5589.46', 'TEMA, 20, 5589.21', 'TMA, 18, 5568.09',
# 'TEMA, 69, 5502.50', 'TEMA, 71, 5492.78', 'TEMA, 19, 5472.90', 'SMA, 33, 5415.33', 'EMA, 35, 5360.56', 'HMA, 71,
# 5359.19', 'TEMA, 72, 5333.19', 'TEMA, 75, 5261.32', 'HMA, 72, 5256.16', 'WMA, 40, 5248.13', 'SMA, 34, 5192.24',
# 'TEMA, 73, 5152.55', 'TEMA, 70, 5152.31', 'WMA, 41, 5087.63', 'TEMA, 76, 5060.93', 'HMA, 15, 5059.44', 'TEMA, 74,
# 5024.83', 'HMA, 73, 4976.07', 'EMA, 36, 4949.94', 'TEMA, 77, 4924.02', 'TEMA, 18, 4897.83', 'WMA, 42, 4895.64',
# 'SMA, 35, 4786.04', 'TEMA, 79, 4783.26', 'TEMA, 78, 4750.47', 'TMA, 19, 4725.49', 'HMA, 74, 4719.94', 'EMA, 37,
# 4694.35', 'HMA, 76, 4690.39', 'TEMA, 81, 4681.16', 'TEMA, 82, 4675.53', 'TEMA, 80, 4563.46', 'EMA, 38, 4506.46',
# 'HMA, 75, 4497.25', 'SMA, 36, 4488.26', 'HMA, 77, 4488.08', 'TMA, 20, 4466.98', 'TEMA, 83, 4437.77', 'WMA, 43,
# 4435.52', 'TEMA, 84, 4430.32', 'WMA, 44, 4408.93', 'HMA, 78, 4389.39', 'WMA, 45, 4270.88', 'EMA, 39, 4249.90',
# 'EMA, 40, 4243.54', 'TEMA, 85, 4231.91', 'HMA, 79, 4222.30', 'SMA, 37, 4143.17', 'TEMA, 17, 4125.16', 'TEMA, 86,
# 4106.59', 'WMA, 46, 4034.79', 'HMA, 80, 3979.61', 'SMA, 38, 3951.32', 'WMA, 47, 3863.77', 'TMA, 21, 3834.82', 'WMA,
# 48, 3823.92', 'EMA, 41, 3802.72', 'TEMA, 87, 3787.13', 'WMA, 49, 3762.24', 'HMA, 81, 3742.66', 'TEMA, 16, 3712.30',
# 'HMA, 83, 3699.90', 'TEMA, 88, 3693.70', 'EMA, 42, 3691.58', 'HMA, 82, 3681.10', 'SMA, 39, 3672.36', 'TEMA, 89,
# 3642.44', 'WMA, 50, 3628.40', 'HMA, 85, 3584.41', 'WMA, 51, 3574.40', 'EMA, 43, 3518.04', 'TEMA, 90, 3473.75',
# 'HMA, 86, 3470.33', 'SMA, 40, 3467.48', 'TEMA, 91, 3435.36', 'TMA, 22, 3434.68', 'HMA, 84, 3432.89', 'WMA, 52,
# 3421.56', 'EMA, 44, 3344.91', 'SMA, 41, 3334.76', 'WMA, 53, 3322.36', 'HMA, 87, 3302.95', 'WMA, 54, 3293.52', 'EMA,
# 46, 3256.59', 'TEMA, 92, 3207.43', 'WMA, 55, 3194.66', 'EMA, 45, 3187.56', 'EMA, 47, 3185.63', 'HMA, 88, 3172.35',
# 'TEMA, 94, 3158.32', 'TEMA, 97, 3151.22', 'TEMA, 95, 3141.28', 'HMA, 90, 3118.58', 'EMA, 48, 3104.40', 'TEMA, 93,
# 3091.02', 'HMA, 89, 3075.81', 'SMA, 42, 3071.37', 'TEMA, 98, 3063.35', 'TMA, 23, 3056.33', 'TEMA, 99, 3054.46',
# 'TEMA, 96, 3038.43', 'HMA, 92, 2984.53', 'WMA, 56, 2978.02', 'TEMA, 100, 2976.06', 'HMA, 91, 2959.64', 'HMA, 93,
# 2924.11', 'EMA, 49, 2897.81', 'SMA, 43, 2886.55', 'WMA, 57, 2883.11', 'TMA, 24, 2870.06', 'TEMA, 15, 2865.17',
# 'HMA, 95, 2826.42', 'WMA, 58, 2809.38', 'EMA, 51, 2785.86', 'HMA, 94, 2774.96', 'HMA, 13, 2770.85', 'SMA, 44,
# 2763.69', 'EMA, 52, 2701.78', 'HMA, 97, 2697.47', 'HMA, 96, 2688.10', 'EMA, 50, 2687.88', 'WMA, 59, 2659.74', 'EMA,
# 53, 2637.04', 'WMA, 60, 2604.20', 'SMA, 45, 2595.69', 'HMA, 98, 2589.05', 'EMA, 54, 2577.76', 'TMA, 25, 2564.38',
# 'HMA, 99, 2506.90', 'EMA, 55, 2493.86', 'SMA, 46, 2479.54', 'WMA, 62, 2459.92', 'WMA, 61, 2433.99', 'TMA, 26,
# 2417.18', 'SMA, 47, 2412.24', 'EMA, 56, 2410.10', 'EMA, 57, 2392.20', 'HMA, 100, 2371.09', 'WMA, 63, 2368.01',
# 'SMA, 48, 2335.90', 'EMA, 59, 2322.45', 'EMA, 58, 2311.29', 'WMA, 64, 2281.51', 'SMA, 49, 2272.21', 'TEMA, 14,
# 2270.98', 'EMA, 60, 2253.51', 'EMA, 61, 2234.85', 'WMA, 65, 2225.77', 'SMA, 50, 2214.52', 'EMA, 62, 2175.66', 'SMA,
# 51, 2150.66', 'WMA, 66, 2070.43', 'TMA, 27, 2065.58', 'WMA, 67, 2063.32', 'EMA, 63, 2059.97', 'SMA, 52, 2057.58',
# 'EMA, 64, 2037.04', 'SMA, 54, 2026.19', 'WMA, 68, 2019.25', 'SMA, 55, 2004.37', 'EMA, 66, 1994.13', 'SMA, 53,
# 1992.27', 'EMA, 65, 1982.20', 'EMA, 67, 1907.66', 'TEMA, 13, 1903.07', 'TMA, 28, 1891.66', 'SMA, 56, 1881.77',
# 'WMA, 69, 1878.21', 'TMA, 29, 1864.45', 'EMA, 68, 1835.68', 'WMA, 70, 1808.80', 'EMA, 69, 1800.34', 'SMA, 58,
# 1785.82', 'SMA, 57, 1775.33', 'HMA, 11, 1761.94', 'SMA, 59, 1758.28', 'TMA, 30, 1750.89', 'EMA, 70, 1748.23', 'EMA,
# 71, 1737.49', 'SMA, 60, 1737.36', 'WMA, 71, 1717.55', 'TMA, 31, 1689.22', 'EMA, 72, 1685.26', 'EMA, 73, 1621.02',
# 'SMA, 61, 1620.35', 'SMA, 62, 1606.74', 'TMA, 32, 1595.30', 'SMA, 63, 1586.89', 'EMA, 74, 1575.20', 'EMA, 78,
# 1566.09', 'WMA, 72, 1565.81', 'EMA, 79, 1559.79', 'EMA, 77, 1556.93', 'EMA, 75, 1555.86', 'EMA, 76, 1548.17', 'EMA,
# 80, 1540.41', 'TMA, 33, 1493.89', 'WMA, 73, 1493.89', 'EMA, 81, 1493.59', 'SMA, 64, 1467.33', 'EMA, 82, 1459.00',
# 'WMA, 75, 1454.09', 'WMA, 74, 1453.50', 'TMA, 34, 1436.15', 'TMA, 35, 1408.14', 'SMA, 66, 1406.33', 'EMA, 83,
# 1401.50', 'SMA, 65, 1392.50', 'WMA, 76, 1390.91', 'WMA, 77, 1383.17', 'TEMA, 12, 1381.15', 'EMA, 84, 1380.14',
# 'EMA, 85, 1375.58', 'EMA, 86, 1371.65', 'EMA, 87, 1357.86', 'SMA, 68, 1354.28', 'SMA, 69, 1350.24', 'TMA, 36,
# 1340.02', 'SMA, 67, 1324.61', 'EMA, 88, 1320.87', 'SMA, 70, 1315.49', 'SMA, 74, 1306.96', 'SMA, 72, 1305.63', 'EMA,
# 92, 1295.18', 'SMA, 73, 1291.97', 'TMA, 37, 1291.72', 'EMA, 91, 1283.62', 'SMA, 76, 1281.34', 'EMA, 90, 1280.69',
# 'EMA, 89, 1273.03', 'SMA, 81, 1272.19', 'SMA, 77, 1269.52', 'SMA, 71, 1268.66', 'SMA, 78, 1268.01', 'WMA, 78,
# 1267.51', 'SMA, 82, 1265.58', 'SMA, 75, 1264.77', 'EMA, 93, 1258.52', 'SMA, 79, 1243.95', 'SMA, 83, 1239.89', 'EMA,
# 95, 1228.02', 'EMA, 94, 1227.89', 'WMA, 79, 1218.52', 'EMA, 96, 1213.96', 'SMA, 80, 1210.85', 'WMA, 80, 1205.15',
# 'TMA, 39, 1167.90', 'WMA, 81, 1163.93', 'SMA, 84, 1161.18', 'SMA, 85, 1153.30', 'EMA, 98, 1152.78', 'WMA, 82,
# 1151.33', 'TMA, 38, 1139.92', 'EMA, 97, 1138.83', 'WMA, 83, 1137.82', 'EMA, 99, 1133.86', 'TMA, 41, 1126.58', 'SMA,
# 86, 1122.36', 'SMA, 87, 1116.05', 'TMA, 43, 1113.39', 'SMA, 88, 1111.56', 'WMA, 84, 1108.56', 'TMA, 42, 1103.31',
# 'EMA, 100, 1094.07', 'SMA, 89, 1093.37', 'TMA, 40, 1087.62', 'TMA, 46, 1081.90', 'TMA, 44, 1072.82', 'TMA, 45,
# 1067.45', 'WMA, 85, 1049.07', 'SMA, 90, 1041.17', 'SMA, 91, 1032.50', 'SMA, 93, 1028.62', 'SMA, 94, 1022.48', 'WMA,
# 89, 1018.86', 'WMA, 87, 1008.49', 'TMA, 48, 997.04', 'SMA, 92, 994.71', 'WMA, 86, 992.78', 'WMA, 88, 990.31', 'TMA,
# 49, 989.33', 'WMA, 90, 986.10', 'WMA, 91, 982.40', 'WMA, 95, 979.52', 'WMA, 96, 978.73', 'WMA, 92, 977.31', 'WMA,
# 97, 974.77', 'TMA, 47, 970.45', 'WMA, 93, 970.13', 'TMA, 50, 964.68', 'WMA, 94, 962.37', 'TMA, 51, 959.10', 'SMA,
# 95, 958.66', 'SMA, 97, 957.94', 'SMA, 96, 943.71', 'WMA, 99, 940.56', 'SMA, 98, 940.24', 'WMA, 98, 924.82', 'SMA,
# 99, 922.17', 'WMA, 100, 919.48', 'SMA, 100, 907.50', 'TMA, 52, 898.50', 'TMA, 53, 882.40', 'TMA, 54, 879.12', 'TMA,
# 55, 801.79', 'TMA, 57, 792.40', 'TMA, 56, 786.01', 'TMA, 60, 774.92', 'TMA, 61, 758.02', 'TMA, 59, 756.18', 'TMA,
# 58, 749.46', 'TMA, 62, 726.76', 'TMA, 64, 725.51', 'TMA, 65, 720.95', 'TEMA, 11, 711.77', 'TMA, 63, 709.56', 'TMA,
# 66, 694.22', 'TMA, 67, 689.63', 'TMA, 68, 664.42', 'TMA, 69, 630.30', 'TMA, 70, 625.38', 'TMA, 71, 600.90', 'TMA,
# 72, 597.71', 'TMA, 73, 577.46', 'TMA, 74, 573.59', 'TMA, 75, 564.59', 'TMA, 76, 561.98', 'TMA, 77, 555.65', 'TMA,
# 78, 532.13', 'TMA, 79, 523.75', 'TMA, 81, 507.09', 'TMA, 80, 506.93', 'TMA, 82, 503.96', 'TMA, 86, 501.15', 'TMA,
# 83, 494.71', 'TMA, 89, 488.03', 'TMA, 85, 484.51', 'TMA, 88, 476.60', 'TMA, 87, 476.01', 'TMA, 84, 466.41', 'TMA,
# 90, 464.12', 'TMA, 93, 459.43', 'TMA, 94, 453.28', 'TMA, 92, 452.90', 'TMA, 95, 440.98', 'TMA, 91, 440.64', 'TMA,
# 96, 439.82', 'TEMA, 10, 427.07', 'TMA, 97, 422.36', 'TMA, 98, 415.41', 'TMA, 99, 406.14', 'TMA, 100, 403.46']