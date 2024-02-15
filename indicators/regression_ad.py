import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Define a function to calculate the angle between two lines
def calculate_angle(m1, m2):
    if (1 + m1 * m2) == 0:
        return np.pi / 2  # Handle the undefined case
    return np.abs(np.arctan((m2 - m1) / (1 + m1 * m2)))


# Load the CSV file
file_path = './data/Binance_BTCUSDT_d.csv'
data = pd.read_csv(file_path, skiprows=1)
data['Date'] = pd.to_datetime(data['Date'])
data_reversed = data.iloc[::-1].reset_index(drop=True)

data_reversed['AD_Indicator'] = data_reversed.apply(
    lambda row: 0 if (row['Close'] == row['High'] and row['Close'] == row['Low']) or row['High'] == row['Low']
    else ((2 * row['Close'] - row['Low'] - row['High']) / (row['High'] - row['Low']) * row['Volume BTC']),
    axis=1).cumsum()

# Calculate the slopes of the regression lines and the angles between them
window_size = 30
overlap = 5
alpha_threshold_degrees = 15  # The angle threshold

# Initialize columns for the regression slopes and angles
data_reversed['Price_Slope'] = np.nan
data_reversed['AD_Slope'] = np.nan
data_reversed['Angle_Between_Lines_Degrees'] = np.nan

for i in range(window_size, len(data_reversed), overlap):
    if i + window_size > len(data_reversed):  # Ensure the window does not exceed the data range
        window = data_reversed.iloc[i:]
    else:
        window = data_reversed.iloc[i:i+window_size]
    X = np.array(range(len(window))).reshape(-1, 1)  # Time as independent variable
    y_price = window['Close'].values.reshape(-1, 1)  # Price as dependent variable
    y_ad = window['AD_Indicator'].values.reshape(-1, 1)  # AD Indicator as dependent variable
    reg_price = LinearRegression().fit(X, y_price)
    reg_ad = LinearRegression().fit(X, y_ad)
    data_reversed.at[window.index[-1], 'Price_Slope'] = reg_price.coef_[0][0]
    data_reversed.at[window.index[-1], 'AD_Slope'] = reg_ad.coef_[0][0]
    angle = calculate_angle(reg_price.coef_[0][0], reg_ad.coef_[0][0])
    data_reversed.at[window.index[-1], 'Angle_Between_Lines_Degrees'] = np.degrees(angle)

# Identify the bullish and bearish divergences
data_reversed['Bullish_Divergence'] = (data_reversed['Price_Slope'] < 0) & (data_reversed['AD_Slope'] > 0) & (data_reversed['Angle_Between_Lines_Degrees'] >= alpha_threshold_degrees)
data_reversed['Bearish_Divergence'] = (data_reversed['Price_Slope'] > 0) & (data_reversed['AD_Slope'] < 0) & (data_reversed['Angle_Between_Lines_Degrees'] >= alpha_threshold_degrees)

# Plotting the Price Chart with Bullish and Bearish Divergences Highlighted
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(data_reversed['Date'], data_reversed['Close'], label='Close Price', color='blue')

# Highlight bullish divergences in green
ax.fill_between(data_reversed['Date'], data_reversed['Close'].min(), data_reversed['Close'].max(),
                where=data_reversed['Bullish_Divergence'], color='green', alpha=0.3, label='Bullish Divergence')

# Highlight bearish divergences in red
ax.fill_between(data_reversed['Date'], data_reversed['Close'].max(), data_reversed['Close'].min(),
                where=data_reversed['Bearish_Divergence'], color='red', alpha=0.3, label='Bearish Divergence')

ax.set_title('Bitcoin Closing Price with Bullish and Bearish Divergences')
ax.set_xlabel('Date')
ax.set_ylabel('Closing Price (USDT)')
ax.legend()
ax.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()