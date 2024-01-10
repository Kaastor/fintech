import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


def calculate_momentum(df, period):
    return df["Close"].diff(period)


def find_momentum_divergences(df, momentum_period=21, smoothing_period=9):
    momentum = calculate_momentum(df, momentum_period)
    smoothed_momentum = momentum.ewm(span=smoothing_period).mean()

    price_trend = df["Close"].diff()
    momentum_trend = smoothed_momentum.diff()

    bullish_divergences = df.loc[(price_trend > 0) & (momentum_trend < 0)]
    bearish_divergences = df.loc[(price_trend < 0) & (momentum_trend > 0)]

    return bullish_divergences, bearish_divergences


def plot_price_with_divergences(df, bullish_divergences, bearish_divergences):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["Close"], label="Price", color="black")
    plt.scatter(bullish_divergences.index, bullish_divergences["Close"], label="Bullish Divergence", color="green",
                marker="^", s=100)
    plt.scatter(bearish_divergences.index, bearish_divergences["Close"], label="Bearish Divergence", color="red",
                marker="v", s=100)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Price with Momentum Divergences")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_momentum_with_divergences(momentum, bullish_divergences, bearish_divergences):
    plt.figure(figsize=(12, 6))
    plt.plot(momentum.index, momentum, label="Momentum", color="blue")
    plt.scatter(bullish_divergences.index, bullish_divergences["Momentum"], label="Bullish Divergence", color="green",
                marker="^", s=100)
    plt.scatter(bearish_divergences.index, bearish_divergences["Momentum"], label="Bearish Divergence", color="red",
                marker="v", s=100)
    plt.xlabel("Date")
    plt.ylabel("Momentum")
    plt.title("Momentum with Divergences")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Replace 'PLTR' with any stock symbol you want to analyze
    stock_symbol = "PLTR"

    # Fetch data from Yahoo Finance using yfinance library
    data = yf.download(stock_symbol, start="2022-01-01", end="2023-01-01")

    # Calculate momentum divergences
    bullish_divergences, bearish_divergences = find_momentum_divergences(data)

    # Plot price data with marked divergences
    plot_price_with_divergences(data, bullish_divergences, bearish_divergences)

    # Plot momentum data with marked divergences
    momentum = calculate_momentum(data, period=21)
    plot_momentum_with_divergences(momentum, bullish_divergences, bearish_divergences)
