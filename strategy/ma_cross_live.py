import time
from findatapy.market import Market, MarketDataRequest
from findatapy.util import DataConstants
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

data_constants = DataConstants()

# create a time series of market data
market = Market(market_data_generator=data_constants.default_market_data_generator)


#FIXME: should work + not get so much data
def fetch_data():
    md_request = MarketDataRequest(
        start_date="01 Jan 2020",
        finish_date=datetime.now().strftime("%d %b %Y"),
        category="crypto",
        fields=["close"],
        freq="tick",
        data_source="bloomberg",
        tickers=["BTC"],
        cache_algo="internet_load_return"
    )

    # Fetch the data
    data = market.fetch_market(md_request)

    # The fetched data is in a multi-index DataFrame. We can flatten it for easier use:
    data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns.values]
    data.rename(columns={'BTC.close': 'Close'}, inplace=True)

    return data


def calculate_sma(data, window):
    return data.rolling(window=window).mean()


def ma_crossover_strategy(data, short_window=33, long_window=144):
    short_moving_avg = calculate_sma(data, short_window)
    long_moving_avg = calculate_sma(data, long_window)

    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0.0
    signals['signal'][short_window:] = np.where(short_moving_avg[short_window:] > long_moving_avg[short_window:], 1.0,
                                                0.0)
    signals['positions'] = signals['signal'].diff()

    return signals


def backtest(data, signals, initial_investment=100000.0):
    positions = pd.DataFrame(index=signals.index).fillna(0.0)
    positions['Stock'] = 100 * signals['signal']
    portfolio = positions.multiply(data['Close'], axis=0)
    pos_diff = positions.diff()
    portfolio['holdings'] = (positions.multiply(data['Close'], axis=0)).sum(axis=1)
    portfolio['cash'] = initial_investment - (pos_diff.multiply(data['Close'], axis=0)).sum(axis=1).cumsum()
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    portfolio['returns'] = portfolio['total'].pct_change()

    return portfolio


def calculate_metrics(portfolio):
    total_return = (portfolio['total'][-1] / portfolio['total'][0]) - 1
    annual_return = (1 + total_return) ** (1 / len(set(portfolio.index.year))) - 1
    excess_returns = portfolio['returns'] - 0
    sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())

    return total_return, annual_return, sharpe_ratio


def plot_signals(data, signals, short_window=33, long_window=144):
    # Plot the closing price
    plt.figure(figsize=(16,9))
    plt.plot(data.index, data['Close'], label='Close Price', color='blue')

    # Plot the short and long moving averages
    plt.plot(data.index, calculate_sma(data['Close'], short_window), label=f'Short SMA ({short_window} days)', color='red')
    plt.plot(data.index, calculate_sma(data['Close'], long_window), label=f'Long SMA ({long_window} days)', color='green')

    # Plot buy signals
    buys = signals.loc[signals.positions == 1.0]
    plt.plot(buys.index, data.loc[buys.index]['Close'], '^', markersize=10, color='m', label='Buy')

    # Plot sell signals
    sells = signals.loc[signals.positions == -1.0]
    plt.plot(sells.index, data.loc[sells.index]['Close'], 'v', markersize=10, color='k', label='Sell')

    # Show the plot with a legend
    plt.title('Moving Average Crossover Strategy')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


while True:
    # Fetch the data
    data = fetch_data()

    # Resample the data to 1-minute intervals
    data = data.resample('1Min').mean()

    # Generate signals using the strategy
    signals = ma_crossover_strategy(data['Close'])

    # Backtest the strategy
    results = backtest(data, signals)

    # Calculate and print the metrics
    total_return, annual_return, sharpe_ratio = calculate_metrics(results)
    print(f"\nTime: {datetime.now()}")
    print(f"Initial investment: {results['total'][0]}")
    print(f"Final portfolio value: {results['total'][-1]}")
    print(f"Total return: {total_return * 100}%")
    print(f"Annual return: {annual_return * 100}%")
    print(f"Sharpe ratio: {sharpe_ratio}")

    # Plot the signals
    plot_signals(data, signals)

    # Pause for 60 seconds
    time.sleep(60)
