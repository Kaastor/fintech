import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Kalman Filter calculation
def calc_kalman_filter_v2(source, high, low, length=12):
    value1 = np.zeros_like(source)
    value2 = np.zeros_like(source)
    klmf = np.zeros_like(source)
    klmf[0] = source[0]  # Initialize the first value of klmf with the first source value

    for i in range(1, len(source)):
        value1[i] = 0.2 * (source[i] - source[i - 1]) + 0.8 * (value1[i - 1] if i > 1 else 0)
        value2[i] = 0.1 * (high[i] - low[i]) + 0.8 * (value2[i - 1] if i > 1 else 0)
        lambda_val = np.abs(value1[i] / value2[i]) if value2[i] != 0 else 0
        alpha = (-lambda_val ** 2 + np.sqrt(lambda_val ** 4 + 16 * lambda_val ** 2)) / 8 if lambda_val != 0 else 0
        klmf[i] = alpha * source[i] + (1 - alpha) * (klmf[i - 1] if i > 1 else source[0])

    return klmf


# Triangular Moving Average calculation
def triangular_moving_average(values, window):
    weights = np.arange(1, window + 1)
    tma = np.convolve(values, weights / weights.sum(), 'valid')
    return np.concatenate((np.full(window - 1, np.nan), tma))  # Padding the start with NaNs to maintain array length


# Load and prepare data
file_path = './data/Binance_BTCUSDT_d.csv'  # Adjust the file path as necessary
df = pd.read_csv(file_path, skiprows=1)
df_reverted = df.iloc[::-1].reset_index(drop=True)

# AD
def calculate_ad(high, low, close, volume):
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = np.where((high - low) == 0, 0, mfm)  # Avoid division by zero
    mfv = mfm * volume
    ad = np.cumsum(mfv)  # Accumulate the Money Flow Volume to get AD
    return ad

# Calculate Accumulation Distribution (AD) indicator
high = df_reverted['High'].values
low = df_reverted['Low'].values
close = df_reverted['Close'].values
volume = df_reverted['Volume USDT'].values  # Using 'Volume BTC' as the volume

ad = calculate_ad(high, low, close, volume)

# Calculate Kalman Filter for AD
klmf_ad = calc_kalman_filter_v2(ad, high, low)

# Calculate TMA of Kalman-filtered AD
tma_klmf_ad = triangular_moving_average(klmf_ad, 66)

# Data preparation for plotting
df_reverted['AD'] = ad
df_reverted['KLMF_AD'] = klmf_ad
df_reverted['TMA_KLMF_AD'] = tma_klmf_ad

df_reverted[['AD', 'KLMF_AD', 'TMA_KLMF_AD']].head()


# Using Close price as 'source', and High and Low prices for the calculations
source = df_reverted['Close'].values
high = df_reverted['High'].values
low = df_reverted['Low'].values

# Calculate Kalman Filter
klmf = calc_kalman_filter_v2(source, high, low)

# Calculate Triangular Moving Average (TMA) of Kalman Filter with length 66
tma_klmf = triangular_moving_average(klmf, 66)

# Data preparation for plotting
df_reverted['KLMF'] = klmf
df_reverted['TMA_KLMF'] = tma_klmf
df_reverted['Date'] = pd.to_datetime(df_reverted['Date'])

fig, ax1 = plt.subplots(figsize=(14, 7), dpi=300)

# First set of plots using ax1
ax1.plot(df_reverted['Date'], df_reverted['Close'], label='Close Price', color='black', alpha=0.75)
ax1.plot(df_reverted['Date'], df_reverted['KLMF'], label='KLMF', color='green', alpha=0.75)
ax1.plot(df_reverted['Date'], df_reverted['TMA_KLMF'], label='TMA of KLMF', color='red', alpha=0.75)
ax1.fill_between(df_reverted['Date'], df_reverted['Close'], df_reverted['TMA_KLMF'], where=df_reverted['Close'] >= df_reverted['TMA_KLMF'], color='gold', alpha=0.5, label='Above TMA')
ax1.fill_between(df_reverted['Date'], df_reverted['Close'], df_reverted['TMA_KLMF'], where=df_reverted['Close'] < df_reverted['TMA_KLMF'], color='blue', alpha=0.5, label='Below TMA')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price', color='black')
ax1.tick_params(axis='y', labelcolor='black')
# ax1.legend(loc='upper left')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# Second Y-axis for the AD plots
ax2 = ax1.twinx()
ax2.plot(df_reverted['Date'], df_reverted['AD'], label='AD', color='purple', alpha=0.75, linestyle='dashed')
ax2.plot(df_reverted['Date'], df_reverted['KLMF_AD'], label='KLMF of AD', color='orange', alpha=0.75, linestyle='dashed')
ax2.plot(df_reverted['Date'], df_reverted['TMA_KLMF_AD'], label='TMA of KLMF_AD', color='cyan', alpha=0.75, linestyle='dashed')
ax2.fill_between(df_reverted['Date'], df_reverted['AD'], df_reverted['TMA_KLMF_AD'], where=df_reverted['AD'] >= df_reverted['TMA_KLMF_AD'], color='gold', alpha=0.3, label='AD above TMA', linestyle='dashed')
ax2.fill_between(df_reverted['Date'], df_reverted['AD'], df_reverted['TMA_KLMF_AD'], where=df_reverted['AD'] < df_reverted['TMA_KLMF_AD'], color='blue', alpha=0.3, label='AD below TMA', linestyle='dashed')
ax2.set_ylabel('Value', color='black')
ax2.tick_params(axis='y', labelcolor='black')
# ax2.legend(loc='upper right')

# Final formatting
fig.suptitle('Combined Plot with Dual Y-Axes')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()