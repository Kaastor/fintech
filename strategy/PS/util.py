import numpy as np
import pandas as pd


def calculate_ad(high, low, close, volume):
    """Calculate Accumulation Distribution (AD) indicator."""
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = np.where((high - low) == 0, 0, mfm)
    mfv = mfm * volume
    return np.cumsum(mfv)


def smoothed_moving_average(values, window):
    """Calculate the Smoothed Moving Average."""
    return pd.Series(values).rolling(window=window, min_periods=1).mean().values


def exponential_moving_average(values, span):
    """Calculate the Exponential Moving Average."""
    return pd.Series(values).ewm(span=span, adjust=False).mean().values


def weighted_moving_average(values, window):
    """Calculate the Weighted Moving Average."""
    weights = np.arange(1, window + 1)
    return pd.Series(values).rolling(window=window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True).values


def hull_moving_average(values, window):
    """Calculate the Hull Moving Average."""
    half_window = int(window / 2)
    sqrt_window = int(np.sqrt(window))
    wma_half = weighted_moving_average(values, half_window)
    wma_full = weighted_moving_average(values, window)
    raw_hma = 2 * wma_half - wma_full
    return pd.Series(raw_hma).rolling(window=sqrt_window).apply(
        lambda x: np.dot(x, np.arange(1, sqrt_window + 1)) / np.arange(1, sqrt_window + 1).sum(), raw=True).values


def triple_exponential_moving_average(values, span):
    """Calculate the Triple Exponential Moving Average."""
    ema1 = pd.Series(values).ewm(span=span, adjust=False).mean()
    ema2 = ema1.ewm(span=span, adjust=False).mean()
    ema3 = ema2.ewm(span=span, adjust=False).mean()
    return (3 * ema1 - 3 * ema2 + ema3).values


def triangular_moving_average(values, window):
    """Calculate the Triangular Moving Average."""
    simple_ma = pd.Series(values).rolling(window=window, min_periods=1).mean()
    return simple_ma.rolling(window=window, min_periods=1).mean().values


def calculate_ad(high, low, close, volume):
    """Calculate Accumulation Distribution (AD) indicator."""
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = np.where((high - low) == 0, 0, mfm)
    mfv = mfm * volume
    return np.cumsum(mfv)