import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# Function to calculate Money Flow Multiplier and Money Flow Volume
def money_flow_multiplier(row):
    high, low, close = row['High'], row['Low'], row['Close']
    return ((close - low) - (high - close)) / (high - low) if (high - low) != 0 else 0


def money_flow_volume(row):
    return row['money_flow_multiplier'] * row['Volume USDT']


# Function to find peaks and troughs
def find_peaks_troughs(series, distance=5):
    peaks, _ = find_peaks(series, distance=distance)
    troughs, _ = find_peaks(-series, distance=distance)
    return peaks, troughs


# Function to calculate the slope between two points
def calculate_slope(x1, y1, x2, y2):
    if x1 == x2:  # Avoid division by zero
        return np.inf
    return (y2 - y1) / (x2 - x1)


# Load and preprocess the data
data = pd.read_csv('./data/Binance_IOTAUSDT_d.csv', skiprows=1, index_col='Date', parse_dates=['Date'])
data.sort_index(inplace=True)
data['money_flow_multiplier'] = data.apply(money_flow_multiplier, axis=1)
data['money_flow_volume'] = data.apply(money_flow_volume, axis=1)
data['AD'] = data['money_flow_volume'].cumsum()

# Initialize dictionaries for counting divergences
bullish_divergence_counts = {}
bearish_divergence_counts = {}

# Define window size and overlap
window_size = 25
overlap = 5
step = window_size - overlap

# Loop through the dataset in windows
for start in range(0, len(data) - window_size + 1, step):
    end = start + window_size
    windowed_data = data.iloc[start:end]
    window_key = f'{data.index[start].date()} to {data.index[end - 1].date()}'
    bullish_divergence_counts[window_key] = 0
    bearish_divergence_counts[window_key] = 0

    # Find peaks and troughs within the window
    window_price_peaks, window_price_troughs = find_peaks_troughs(windowed_data['Close'])
    window_ad_peaks, window_ad_troughs = find_peaks_troughs(windowed_data['AD'])

    # Analyze troughs for Bullish Divergence
    for i in range(1, min(len(window_price_troughs), len(window_ad_troughs))):
        price_slope = calculate_slope(window_price_troughs[i - 1],
                                      windowed_data['Close'].iloc[window_price_troughs[i - 1]],
                                      window_price_troughs[i], windowed_data['Close'].iloc[window_price_troughs[i]])
        ad_slope = calculate_slope(window_ad_troughs[i - 1], windowed_data['AD'].iloc[window_ad_troughs[i - 1]],
                                   window_ad_troughs[i], windowed_data['AD'].iloc[window_ad_troughs[i]])
        if price_slope < 0 and ad_slope > 0:
            bullish_divergence_counts[window_key] += 1

    # Analyze peaks for Bearish Divergence
    for i in range(1, min(len(window_price_peaks), len(window_ad_peaks))):
        price_slope = calculate_slope(window_price_peaks[i - 1], windowed_data['Close'].iloc[window_price_peaks[i - 1]],
                                      window_price_peaks[i], windowed_data['Close'].iloc[window_price_peaks[i]])
        ad_slope = calculate_slope(window_ad_peaks[i - 1], windowed_data['AD'].iloc[window_ad_peaks[i - 1]],
                                   window_ad_peaks[i], windowed_data['AD'].iloc[window_ad_peaks[i]])
        if price_slope > 0 and ad_slope < 0:
            bearish_divergence_counts[window_key] += 1

# Filter for windows with 2 or more divergences
filtered_bullish_divergences = [key for key, count in bullish_divergence_counts.items() if count >= 2]
filtered_bearish_divergences = [key for key, count in bearish_divergence_counts.items() if count >= 2]

# Plotting
plt.figure(figsize=(14, 10))
plt.plot(data.index, data['Close'], label='Close Price', color='blue', linewidth=0.5)

# Plot shaded areas for filtered divergences
for window_key in filtered_bullish_divergences:
    start_date, end_date = window_key.split(' to ')
    plt.axvspan(pd.Timestamp(start_date), pd.Timestamp(end_date), color='gold', alpha=0.3)

for window_key in filtered_bearish_divergences:
    start_date, end_date = window_key.split(' to ')
    plt.axvspan(pd.Timestamp(start_date), pd.Timestamp(end_date), color='darkblue', alpha=0.3)

plt.title('BTC Close Price with Filtered Divergences')
plt.ylabel('Price in USDT')
plt.xlabel('Date')
plt.legend()
plt.tight_layout()
plt.show()
