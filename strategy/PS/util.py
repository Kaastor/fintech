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

# ad = ta.cum(close==high and close==low or high==low ? 0 : ((2 * close - low - high) / (high - low)) * volume)
def calculate_ad(high, low, close, volume):
    """Calculate Accumulation Distribution (AD) indicator."""
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = np.where((high - low) == 0, 0, mfm)
    mfv = mfm * volume
    return np.cumsum(mfv)


def calculate_ad_williams(high, low, close, open_, volume):
    """Calculate Accumulation Distribution (AD) indicator."""
    mfm = ((close - open_) / (high - low))
    mfv = mfm * volume
    return np.cumsum(mfv)

def calculate_obv(closing_prices, volumes):
    """
    Calculate the On Balance Volume (OBV).

    Parameters:
    closing_prices: List of closing prices of the security.
    volumes: List of trading volumes of the security.

    Returns:
    List of OBV values.
    """
    obv = [0]  # Start the OBV from 0 for the first data point

    for i in range(1, len(closing_prices)):
        if closing_prices[i] > closing_prices[i - 1]:
            obv.append(obv[-1] + volumes[i])  # Price increased, add volume to OBV
        elif closing_prices[i] < closing_prices[i - 1]:
            obv.append(obv[-1] - volumes[i])  # Price decreased, subtract volume from OBV
        else:
            obv.append(obv[-1])  # Price unchanged, OBV remains the same

    return obv


def calculate_pvt(closing_prices, volumes):
    """
    Calculate the Price and Volume Trend (PVT).

    Parameters:
    closing_prices: List of closing prices of the security.
    volumes: List of trading volumes of the security.

    Returns:
    List of PVT values.
    """
    pvt = [0]  # Initialize PVT with first value as 0

    for i in range(1, len(closing_prices)):
        price_change_ratio = (closing_prices[i] - closing_prices[i - 1]) / closing_prices[i - 1]
        pvt_value = volumes[i] * price_change_ratio
        pvt.append(pvt[-1] + pvt_value)  # Add the current day's value to the cumulative total

    return pvt


def calculate_wad_ndarray(high, low, close):
    """
    Calculate Williams' Accumulation/Distribution using NumPy ndarrays for high, low, and close prices.

    :param high: NumPy ndarray of high prices.
    :param low: NumPy ndarray of low prices.
    :param close: NumPy ndarray of close prices.
    :return: NumPy ndarray of WAD values.
    """
    wad = np.zeros(close.shape)

    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            wad[i] = close[i] - np.minimum(low[i], close[i - 1])
        elif close[i] < close[i - 1]:
            wad[i] = close[i] - np.maximum(high[i], close[i - 1])
        else:
            wad[i] = 0.0

    return np.cumsum(wad)




