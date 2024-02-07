import cryptocompare
import pandas as pd
import matplotlib.pyplot as plt
import requests

cryptocompare.cryptocompare._set_api_key_parameter('50356a0892a6e43eccacfec39e02a8929787e1d02cf83a0fca9b80f53b3dcdd3')


def fetch_data(ticker):
    return pd.DataFrame(cryptocompare.get_historical_price_hour(ticker, currency='USDT', exchange='Binance'))


def plot(data):
    # Histogram for visualizing the distribution
    plt.figure(figsize=(10, 6))
    plt.hist(data['Body_to_Wick_Ratio'], bins=20, color='blue', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Body-to-Wick Ratio')
    plt.xlabel('Body-to-Wick Ratio')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)


def analyze_body_wick_ratio(ticker):
    data = fetch_data(ticker)
    if isinstance(data, pd.DataFrame) and data.empty:
        return None
    # Analyzing the distribution of the Body-to-Wick Ratio
    # Calculating basic statistics
    # Calculate the body size (absolute difference between open and close)
    # Filter for green candles and create a deep copy to avoid SettingWithCopyWarning
    green_candles = data[data['close'] > data['open']].copy()

    # Calculate Body Size, Total Candle Size, and Body-to-Wick Ratio
    green_candles['Body_Size'] = green_candles['close'] - green_candles['open']
    green_candles['Total_Candle_Size'] = green_candles['high'] - green_candles['low']
    green_candles['Body_to_Wick_Ratio'] = green_candles['Body_Size'] / green_candles['Total_Candle_Size']

    # Replace infinities with zero in case of division by zero
    green_candles['Body_to_Wick_Ratio'] = green_candles['Body_to_Wick_Ratio'].replace([float('inf'), -float('inf')], 0)

    # Calculate mean and standard deviation
    mean_ratio = green_candles['Body_to_Wick_Ratio'].mean()
    std_dev_ratio = green_candles['Body_to_Wick_Ratio'].std()

    return ticker, mean_ratio, std_dev_ratio


def remove_usdt_from_symbols(symbols):
    # Remove 'USDT' from each symbol
    return [symbol.replace('USDT', '') for symbol in symbols]


def list_binance_perpetuals():
    # Binance API endpoint for futures exchange information
    url = 'https://fapi.binance.com/fapi/v1/exchangeInfo'

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Extract the symbols that are perpetual futures
        perpetuals = [symbol['symbol'] for symbol in data['symbols'] if symbol['contractType'] == 'PERPETUAL']
        return perpetuals

    except requests.RequestException as e:
        print(f"Error fetching data from Binance: {e}")
        return []


results = []
# Call the function and print the results
for ticker in remove_usdt_from_symbols(list_binance_perpetuals()):
    results.append(analyze_body_wick_ratio(ticker))

results = [i for i in results if i is not None and i[1] > 0.0]
sorted = sorted(results, key=lambda x: x[1])
for t in sorted:
    print(t)
