import pandas as pd
import matplotlib.pyplot as plt

# Load and prepare the data
file_path = './data/Binance_IOTAUSDT_d.csv'
data = pd.read_csv(file_path, skiprows=1)
data['Date'] = pd.to_datetime(data['Date'])
data_reversed = data.iloc[::-1].reset_index(drop=True)
data_reversed['AD_Pine_BTCVol_Corrected'] = data_reversed.apply(
    lambda row: 0 if (row['Close'] == row['High'] and row['Close'] == row['Low']) or row['High'] == row['Low']
    else ((2 * row['Close'] - row['Low'] - row['High']) / (row['High'] - row['Low']) * row['Volume USDT']), axis=1).cumsum()

# Calculate Rate of Change (RoC) for Price and AD Indicator over a 50-day window
window = 30
data_reversed['Price_RoC'] = data_reversed['Close'].pct_change(periods=window)
data_reversed['AD_RoC'] = data_reversed['AD_Pine_BTCVol_Corrected'].pct_change(periods=window)

# Define the RoC threshold
roc_threshold = 0.1  # 15%

# Update the divergence identification logic with the RoC threshold condition
data_reversed['Significant_Positive_Divergence'] = (data_reversed['Price_RoC'] < -roc_threshold) & (data_reversed['AD_RoC'] > 0)
data_reversed['Significant_Negative_Divergence'] = (data_reversed['Price_RoC'] > roc_threshold) & (data_reversed['AD_RoC'] < 0)

# Plotting the Price Chart with Significant Divergence Highlights
fig, ax_sig_div = plt.subplots(figsize=(14, 7))

# Plot the closing price
ax_sig_div.plot(data_reversed['Date'], data_reversed['Close'], label='Close Price', color='blue')

# Highlight significant positive divergences in green
ax_sig_div.fill_between(data_reversed['Date'], data_reversed['Close'].min(), data_reversed['Close'].max(),
                        where=data_reversed['Significant_Positive_Divergence'], color='green', alpha=0.3, label='Significant Positive Divergence')

# Highlight significant negative divergences in red
ax_sig_div.fill_between(data_reversed['Date'], data_reversed['Close'].min(), data_reversed['Close'].max(),
                        where=data_reversed['Significant_Negative_Divergence'], color='red', alpha=0.3, label='Significant Negative Divergence')

ax_sig_div.set_title('Bitcoin Closing Price with Significant Divergences Highlighted')
ax_sig_div.set_xlabel('Date')
ax_sig_div.set_ylabel('Closing Price (USDT)')
ax_sig_div.legend()
ax_sig_div.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
