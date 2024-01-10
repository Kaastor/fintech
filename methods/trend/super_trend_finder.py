import cryptocompare
import pandas as pd
import matplotlib.pyplot as plt
import requests


def remove_usdt_from_symbols(symbols):
    # Remove 'USDT' from each symbol
    return [symbol.replace('USDT', '') for symbol in symbols]


def list_binance_perpetuals():
    # Binance API endpoint for futures exchange information
    url = 'https://fapi.binance.com/fapi/v1/exchangeInfo'

    try:
        response = requests.get(url)
        response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
        data = response.json()

        # Extract the symbols that are perpetual futures
        perpetuals = [symbol['symbol'] for symbol in data['symbols'] if symbol['contractType'] == 'PERPETUAL']
        return perpetuals

    except requests.RequestException as e:
        print(f"Error fetching data from Binance: {e}")
        return []


def plot_chart(data):
    # Plotting
    plt.plot(data['time'], data['close'], label='Close Price', color='blue')
    plt.plot(data['time'], data['MA33'], label='33-Day MA', color='green')
    plt.plot(data['time'], data['MA144'], label='144-Day MA', color='red')

    # Labeling the plot
    plt.title('BTC Close Price with 33 and 144 Day Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()


def is_in_up_trend(data):
    """
    Function to determine if the green moving average (MA33) is above the red moving average (MA144).

    :param data: DataFrame containing the 'MA33' and 'MA144' columns.
    :return: A boolean value indicating if MA33 is above MA144.
    """
    # Check if data is complete
    if 'close' not in data.columns:
        return False
    # Check if the latest value of MA33 is greater than the latest value of MA144
    data['MA33'] = data['close'].rolling(window=33).mean()
    data['MA144'] = data['close'].rolling(window=144).mean()
    # plot_chart(data)
    return data['MA33'].iloc[-1] > data['MA144'].iloc[-1]


def daily_data(ticker):
    return pd.DataFrame(cryptocompare.get_historical_price_day(ticker, currency='USDT', exchange='Binance'))


def hour_data(ticker):
    return pd.DataFrame(cryptocompare.get_historical_price_hour(ticker, currency='USDT', exchange='Binance'))


def minute_data(ticker):
    return pd.DataFrame(cryptocompare.get_historical_price_minute(ticker, currency='USDT', exchange='Binance'))


def bull():
    # Call the function and print the results
    for ticker in remove_usdt_from_symbols(list_binance_perpetuals()):
        if is_in_up_trend(daily_data(ticker)) and \
                is_in_up_trend(hour_data(ticker)) and \
                is_in_up_trend(minute_data(ticker)):
            print(f'{ticker}')

    print('Done bull.')


def bear():
    # Call the function and print the results
    for ticker in remove_usdt_from_symbols(list_binance_perpetuals()):
        if not is_in_up_trend(daily_data(ticker)) and \
                not is_in_up_trend(hour_data(ticker)) and \
                not is_in_up_trend(minute_data(ticker)):
            print(f'{ticker}')

    print('Done bear.')


bull()
