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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
np.random.seed(42)

# Generate synthetic financial data
date = pd.date_range(start='1/1/2020', periods=500)
price = np.cumsum(np.random.randn(500)) + 100

# Create a DataFrame
df = pd.DataFrame(price, columns=['close'], index=date)

# Calculate high, low, open prices
df['high'] = df['close'] + np.random.rand(500) * 2
df['low'] = df['close'] - np.random.rand(500) * 2
df['open'] = df['low'] + (df['high'] - df['low']) * np.random.rand(500)

# Calculate Momentum
df['momentum'] = df['close'] - df['close'].shift(18)

# Calculate RSI
delta = df['close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

average_gain = gain.rolling(18).mean()
average_loss = loss.rolling(18).mean()

rs = average_gain / average_loss
df['rsi'] = 100 - (100 / (1 + rs))

df.tail()

# Identify Knoxville Divergence points
df['price_increasing'] = df['close'].diff() > 0
df['momentum_decreasing'] = df['momentum'].diff() < 0
df['overbought'] = df['rsi'] > 70
df['oversold'] = df['rsi'] < 30

df['kd_bearish'] = df['price_increasing'] & df['momentum_decreasing'] & df['overbought']
df['kd_bullish'] = ~df['price_increasing'] & ~df['momentum_decreasing'] & df['oversold']

# Plot price data and overlay Knoxville Divergence points
plt.figure(figsize=(14, 7))
plt.plot(df['close'], label='Price', color='blue')

plt.plot(df[df['kd_bearish']]['close'], marker='v', markersize=7, linestyle='None', color='r', label='Bearish KD')
plt.plot(df[df['kd_bullish']]['close'], marker='^', markersize=7, linestyle='None', color='g', label='Bullish KD')

plt.title('Price with Knoxville Divergence')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()
