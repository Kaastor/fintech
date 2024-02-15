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

# Plotting
fig, ax = plt.subplots(figsize=(14, 7))

# Plot Close, KLMF, and TMA_KLMF
ax.plot(df_reverted['Date'], df_reverted['Close'], label='Close Price', color='black', alpha=0.75)
ax.plot(df_reverted['Date'], df_reverted['KLMF'], label='KLMF', color='green', alpha=0.75)
ax.plot(df_reverted['Date'], df_reverted['TMA_KLMF'], label='TMA of KLMF', color='red', alpha=0.75)

# Fill
ax.fill_between(df_reverted['Date'], df_reverted['Close'], df_reverted['TMA_KLMF'],
                where=df_reverted['Close'] >= df_reverted['TMA_KLMF'], color='gold', alpha=0.5, label='Above TMA')
ax.fill_between(df_reverted['Date'], df_reverted['Close'], df_reverted['TMA_KLMF'],
                where=df_reverted['Close'] < df_reverted['TMA_KLMF'], color='blue', alpha=0.5, label='Below TMA')

# Formatting
ax.set_title('Close Price, KLMF, and TMA of KLMF with Fill')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
