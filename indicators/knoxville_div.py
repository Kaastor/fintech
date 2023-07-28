"""
Knoxville Divergence

Definition The Knoxville Divergence indicator suggests that price will reverse in direction, making it a
counter-trend indicator. For example, if price is in an uptrend with the Knoxville Divergence line set above the
price on the chart, the trader will understand that the indicator is suggesting price will potentially reverse and
shift to a downtrend shortly. This indicator is very useful for counter-trend traders and general trend-traders
alike, as it adds perspective for those seeking to enter a trade, while following a long-term trend.

History This indicator is used and admired by Rob Booker, an experienced entrepreneur and currency trader who has
helped bring popularity to the indicator.

Takeaways
When analyzing Knoxville Divergence, it should be known that the indicator is comprised of two main elements.

The Momentum indicator. If price is rising, but the momentum indicator is falling, the Knoxville Divergence indicator
might appear on the chart to signal to the trader that a price reversal may be closing in. This indicator must be
paired with another, however, because it does not tell the trader everything they need to know. This is why it pairs
nicely with our second element.

Relative Strength Index. If there is a simultaneous discrepancy between price and the momentum indicator,
the Relative Strength Index (RSI) will signal as overbought or oversold, and will trigger the Knoxville Divergence
line to appear on the chart.

Script:
First, we need to compute the Momentum indicator. The Momentum of a price is simply the difference in price over

the specified period. In this case, the period is 18.

Next, we will calculate the Relative Strength Index (RSI). The RSI is a bit more complex. It measures the speed and
change of price movements and is often used to identify overbought or oversold conditions in a market. When the RSI
for a security gets above 70, it could indicate that the asset is overbought, and when it gets below 30,
it could indicate the asset is oversold. """

import matplotlib.pyplot as plt
import ta
import yfinance as yf


def get_knoxville_divergence(data, period=21):
    # Calculate Momentum and RSI
    data['momentum'] = ta.momentum.AwesomeOscillatorIndicator(high=data['High'], low=data['Low'], window1=5, window2=period).awesome_oscillator()
    data['rsi'] = ta.momentum.RSIIndicator(close=data['Close'], window=period).rsi()

    # Initialize Knoxville Divergence series with 0
    data['knoxville_divergence'] = 0

    # Identify Bullish/Bearish Divergence
    for i in range(period, len(data)):
        # Bullish Divergence
        if data['Close'].iloc[i] > data['Close'].iloc[i-period] and data['momentum'].iloc[i] < data['momentum'].iloc[i-period] and data['rsi'].iloc[i] < 30:
            data['knoxville_divergence'].iloc[i] = -1
        # Bearish Divergence
        elif data['Close'].iloc[i] < data['Close'].iloc[i-period] and data['momentum'].iloc[i] > data['momentum'].iloc[i-period] and data['rsi'].iloc[i] > 70:
            data['knoxville_divergence'].iloc[i] = 1

    return data


# Fetch data from Yahoo Finance
data = yf.download('PLTR', start='2022-01-01', end='2023-01-01', interval='1d')

# Calculate Knoxville Divergence
data = get_knoxville_divergence(data)

# Plot the Close price and the Knoxville Divergence
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('Close', color=color)
ax1.plot(data.index, data['Close'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'
ax2.set_ylabel('Knoxville Divergence', color=color)
ax2.plot(data.index, data['knoxville_divergence'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('PLTR Close Price and Knoxville Divergence')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
