import matplotlib.pyplot as plt
import seaborn as sns
import cryptocompare
import pandas as pd
import requests

cryptocompare.cryptocompare._set_api_key_parameter('50356a0892a6e43eccacfec39e02a8929787e1d02cf83a0fca9b80f53b3dcdd3')


def _minute_data(ticker):
    return pd.DataFrame(cryptocompare.get_historical_price_minute(ticker, currency='USDT', exchange='Binance'))


def _calculate_wick_len(data):
    # Calculating the wick lengths for each candlestick
    data['Upper Wick'] = data.apply(lambda row: row['high'] - max(row['open'], row['close']), axis=1)
    data['Lower Wick'] = data.apply(lambda row: min(row['open'], row['close']) - row['low'], axis=1)


def _remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def _normalize(data):
    # Calculating the reference price (average of Open and Close prices)
    data['Reference Price'] = (data['open'] + data['close']) / 2
    # Converting wick lengths to percentages of the reference price
    data['Upper Wick'].round(3)
    data['Upper Wick'].round(3)
    data['Upper Wick %'] = (data['Upper Wick'] / data['Reference Price']) * 100
    data['Lower Wick %'] = (data['Lower Wick'] / data['Reference Price']) * 100


def calculate_stats(ticker):
    data = _minute_data(ticker)
    if isinstance(data, pd.DataFrame) and data.empty:
        return None
    _calculate_wick_len(data)
    data = _remove_outliers(data, 'Upper Wick')
    data = _remove_outliers(data, 'Lower Wick')
    _normalize(data)
    # Calculating the standard deviation of upper and lower wick lengths
    upper_wick_deviation = data['Upper Wick %'].std().round(4)
    lower_wick_deviation = data['Lower Wick %'].std().round(4)
    # _plot(data)
    return ticker, upper_wick_deviation, lower_wick_deviation


def _plot(data):
    # Plotting the distribution of wick lengths
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(data['Upper Wick %'], kde=True)
    plt.title("Distribution of Upper Wick Lengths")
    plt.xlabel("Wick Length")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    sns.histplot(data['Lower Wick %'], kde=True)
    plt.title("Distribution of Lower Wick Lengths")
    plt.xlabel("Wick Length")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


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
    results.append(calculate_stats(ticker))

results = [i for i in results if i is not None and i[2] > 0.0]
sorted = sorted(results, key=lambda x: x[2])
for t in sorted:
    print(t)
