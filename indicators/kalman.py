import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pykalman import KalmanFilter
from scipy.optimize import brute

ohlcv_data = pd.read_csv('../data/btc_data.csv')
# ohlcv_data = pd.read_csv('../data/Binance_SOLUSDT_1h_2023_10.csv', index_col=0)

# Convert 'Date' column to datetime and set it as index
ohlcv_data['Date'] = pd.to_datetime(ohlcv_data['Date'])
ohlcv_data.set_index('Date', inplace=True)

# Filter the relevant columns for the candlestick chart
ohlc_data = ohlcv_data.loc[:, ['Open', 'High', 'Low', 'Close']]
ohlc_data = ohlc_data.iloc[::-1]
closing_prices = ohlc_data['Close'].values

# Kalman
kf = KalmanFilter(transition_matrices=[1],
                  observation_matrices=[1],
                  initial_state_mean=0,
                  initial_state_covariance=1,
                  observation_covariance=2,
                  transition_covariance=.005)

# Use the observed values of the price to get a rolling mean
state_means, state_covariances = kf.filter(closing_prices)

state_means = pd.Series(state_means.flatten(), index=ohlc_data.index)
closing_prices = pd.Series(closing_prices, index=ohlc_data.index)


def trading_strategy(params, closing_prices):
    # Trading strategy implementation
    initial_capital = 1000.0
    capital = initial_capital
    holding = False
    trades = []

    observation_covariance, transition_covariance = params
    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      initial_state_mean=0,
                      initial_state_covariance=1,
                      observation_covariance=observation_covariance,
                      transition_covariance=transition_covariance)

    state_means, _ = kf.filter(closing_prices)
    # Strategy
    for date, (ma, close) in enumerate(zip(state_means, closing_prices)):
        if close > ma and not holding:
            # Buy
            holding = True
            buy_price = close
            trades.append(('buy', date, close))
        elif close < ma and holding:
            # Sell
            holding = False
            sell_price = close
            capital += sell_price - buy_price
            trades.append(('sell', date, close))

    # Final PnL calculation
    if holding:
        capital += closing_prices.iloc[-1] - buy_price

    pnl = capital - initial_capital
    print(f'PnL:{pnl}, params: {params}')
    return pnl


# Define the parameter ranges for optimization
ranges = [(0.001, 1), (0.001, 1)]  # Example ranges for observation_covariance and transition_covariance

# Run the optimization
result = brute(trading_strategy, ranges, args=(ohlcv_data['Close'],), finish=None)

print(f"Optimal observation_covariance: {result[0]}, Optimal transition_covariance: {result[1]}")

# Plot original data and estimated mean
plt.figure(figsize=(50, 25))
plt.plot(state_means)
plt.plot(closing_prices)
plt.title('Kalman filter estimate of average')
plt.legend(['Kalman Estimate', 'Price'])
plt.xlabel('Hour')
plt.ylabel('Price')
plt.grid(True)
plt.savefig('./plot/kalman.png')
