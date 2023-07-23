import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from findatapy.market import Market, MarketDataRequest, MarketDataGenerator


#FIXME not working with this data source
def fetch_data():
    market = Market(market_data_generator=MarketDataGenerator())

    md_request = MarketDataRequest(start_date="11 Nov 2017",
                                   finish_date="02 Feb 2018",
                                   cut="LOC",
                                   freq="tick",
                                   data_source="bitcoincharts",
                                   category="crypto",
                                   fields=["close"],
                                   tickers=["XBTUSD_itbit"])

    # Fetch the data
    data = market.fetch_market(md_request)

    # The fetched data is in a multi-index DataFrame. We can flatten it for easier use:
    data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns.values]
    data.rename(columns={'XBTUSD_itbit.close': 'Close'}, inplace=True)
    return data


def calculate_sma(data, window):
    return data.rolling(window=window).mean()


def ma_crossover_strategy(data, short_window, long_window):
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


def automate_backtesting(ticker, start_date, end_date, short_window=33, long_window=144):
    data = yf.download(ticker, start=start_date, end=end_date)
    # data = fetch_data()
    signals = ma_crossover_strategy(data['Close'], short_window, long_window)
    results = backtest(data, signals)
    total_return, annual_return, sharpe_ratio = calculate_metrics(results)

    print(f"Initial investment: {results['total'][0]}")
    print(f"Final portfolio value: {results['total'][-1]}")
    print(f"Total return: {total_return * 100}%")
    print(f"Annual return: {annual_return * 100}%")
    print(f"Sharpe ratio: {sharpe_ratio}")

    # Generate signals using the strategy
    signals = ma_crossover_strategy(data['Close'], short_window, long_window)
    plot_signals(data, signals, short_window, long_window)

    return results


def plot_signals(data, signals, short_window, long_window):
    # Plot the closing price
    plt.figure(figsize=(16, 9))
    plt.plot(data.index, data['Close'], label='Close Price', color='blue')

    # Plot the short and long moving averages
    plt.plot(data.index, calculate_sma(data['Close'], short_window), label=f'Short SMA ({short_window} days)',
             color='red')
    plt.plot(data.index, calculate_sma(data['Close'], long_window), label=f'Long SMA ({long_window} days)',
             color='green')

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


results = automate_backtesting('AAPL', '2020-01-01', '2023-12-31')
