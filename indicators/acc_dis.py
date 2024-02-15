import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file, skipping the first non-important row and setting the second as headers
file_path = './data/Binance_BTCUSDT_d.csv'
data = pd.read_csv(file_path, skiprows=1)

# Convert 'Date' to datetime and reverse the DataFrame to start from the earliest date
data['Date'] = pd.to_datetime(data['Date'])
data_reversed = data.iloc[::-1].reset_index(drop=True)

# Recalculate the AD Indicator using the reversed DataFrame and 'Volume BTC'
data_reversed['AD_Pine_BTCVol_Corrected'] = data_reversed.apply(
    lambda row: 0 if (row['Close'] == row['High'] and row['Close'] == row['Low']) or row['High'] == row['Low']
    else ((2 * row['Close'] - row['Low'] - row['High']) / (row['High'] - row['Low']) * row['Volume BTC']), axis=1).cumsum()

# Create a figure with two subplots
fig, ax1 = plt.subplots(2, 1, figsize=(14, 14))

# Plotting the closing price chart on the first subplot
ax1[0].plot(data_reversed['Date'], data_reversed['Close'], label='Close Price', color='blue')
ax1[0].set_title('Bitcoin Closing Price Chart')
ax1[0].set_xlabel('Date')
ax1[0].set_ylabel('Closing Price (USDT)')
ax1[0].legend()
ax1[0].grid(True)

# Plotting the corrected Accumulation Distribution Indicator on the second subplot
ax1[1].plot(data_reversed['Date'], data_reversed['AD_Pine_BTCVol_Corrected'], label='AD (Corrected Order, BTC Volume)', color='red')
ax1[1].set_title('Accumulation Distribution Indicator (Corrected Order, BTC Volume)')
ax1[1].set_xlabel('Date')
ax1[1].set_ylabel('AD Indicator Value')
ax1[1].legend()
ax1[1].grid(True)

plt.tight_layout()
plt.show()
