import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cryptocompare


def data(ticker):
    return pd.DataFrame(cryptocompare.get_historical_price_hour(ticker, currency='USDT', exchange='Binance'))


btc_data = data("RIF")
close_prices_btc = btc_data['close'].values


# Implementing Kalman Filter
class KalmanFilter:
    def __init__(self, transition_matrices, observation_matrices, initial_state_mean):
        self.transition_matrices = transition_matrices
        self.observation_matrices = observation_matrices
        self.current_state_mean = initial_state_mean
        self.current_state_covariance = 1.0  # initial covariance

    def filter(self, observations):
        predicted_states = []
        for observation in observations:
            # prediction step
            predicted_state_mean = self.transition_matrices * self.current_state_mean
            predicted_state_covariance = (
                    self.transition_matrices * self.current_state_covariance * self.transition_matrices + 1.0
            )

            # observation step
            innovation = observation - self.observation_matrices * predicted_state_mean
            innovation_covariance = (
                    self.observation_matrices * predicted_state_covariance * self.observation_matrices + 1.0
            )

            # update step
            kalman_gain = predicted_state_covariance * self.observation_matrices / innovation_covariance
            self.current_state_mean = predicted_state_mean + kalman_gain * innovation
            self.current_state_covariance = (
                    predicted_state_covariance - kalman_gain * self.observation_matrices * predicted_state_covariance
            )

            predicted_states.append(self.current_state_mean)

        return np.array(predicted_states), None


# Initializing and applying Kalman Filter to the 'Close' prices of the new dataset
kf_btc = KalmanFilter(
    transition_matrices=1,
    observation_matrices=1,
    initial_state_mean=close_prices_btc[0]
)
state_means_btc, _ = kf_btc.filter(close_prices_btc)


# Function to calculate Moving Average
def moving_average(data, period):
    return np.convolve(data, np.ones(period) / period, mode='valid')


# Function to calculate Triangular Moving Average
def triangular_moving_average(data, period):
    simple_ma = np.convolve(data, np.ones(period) / period, mode='valid')
    return np.convolve(simple_ma, np.ones(period) / period, mode='valid')


# Calculating Smoothed Moving Averages (SMA) for periods of 33 and 144
sma_33_btc = moving_average(state_means_btc.flatten(), 33)
sma_144_btc = moving_average(state_means_btc.flatten(), 144)

# Calculating Triangular Moving Average (TMA) for a period of 33
triangular_ma_33_btc = triangular_moving_average(state_means_btc.flatten(), 44)

# Implementing the trading strategy
initial_budget = 100.0
portfolio_value = initial_budget
position = 0  # 0 indicates no position, 1 indicates holding a position
portfolio_values = []

for i in range(len(state_means_btc)):
    current_price = close_prices_btc[i]
    current_kalman_estimate = state_means_btc[i]
    current_ma_33 = sma_33_btc[i - len(state_means_btc) + len(sma_33_btc)] if i >= len(state_means_btc) - len(
        sma_33_btc) else 0
    current_ma_144 = sma_144_btc[i - len(state_means_btc) + len(sma_144_btc)] if i >= len(state_means_btc) - len(
        sma_144_btc) else 0

    # Check for buy condition
    if current_ma_33 > current_ma_144 and current_price > current_kalman_estimate and position == 0:
        position = 1  # Open a position
        buy_price = current_price
        portfolio_value -= buy_price  # Deduct the buy price from the portfolio value

    # Check for sell condition
    elif current_price < current_kalman_estimate and position == 1:
        position = 0  # Close the position
        sell_price = current_price
        portfolio_value += sell_price  # Add the sell price to the portfolio value

    # Update the portfolio value if a position is open
    if position == 1:
        portfolio_values.append(portfolio_value + current_price)
    else:
        portfolio_values.append(portfolio_value)

# Final portfolio value after completing the trading
final_portfolio_value = portfolio_values[-1]

# Plotting the portfolio value over time
plt.figure(figsize=(15, 7))
plt.plot(btc_data['time'], portfolio_values, label="Portfolio Value", color='purple')
plt.title("Portfolio Value Over Time Using Trading Strategy")
plt.show()

# Plotting for the Bitcoin dataset with SMA periods of 33 and 144, along with TMA period 33
plt.figure(figsize=(15, 7))
plt.plot(btc_data['time'][len(btc_data) - len(triangular_ma_33_btc):], triangular_ma_33_btc, label="Triangular 33-Period Moving Average", color='green')
plt.plot(btc_data['time'][len(btc_data) - len(sma_33_btc):], sma_33_btc, label="SMA 33-Period", color='blue')
plt.plot(btc_data['time'][len(btc_data) - len(sma_144_btc):], sma_144_btc, label="SMA 144-Period", color='orange')
plt.plot(btc_data['time'], state_means_btc, label="Kalman Filter Estimate", color='red', alpha=0.5)
plt.plot(btc_data['time'], close_prices_btc, label="Close", color='black', alpha=0.5)
plt.title("Kalman Filter Estimates and Moving Averages for Bitcoin Close Prices")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()

print(f"final_portfolio_value: {final_portfolio_value}")
