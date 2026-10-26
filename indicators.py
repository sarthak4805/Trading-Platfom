"""Candle checks, percent change, and VWAP.

Every function uses only the current row and earlier rows. None of them read a
future close to describe the past.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


def validate_ohlcv(df: pd.DataFrame) -> None:
    """Raise a helpful error when a candle frame is not safe to analyze."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Candle data must be a pandas DataFrame.")
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "Candle data is missing required columns: "
            + ", ".join(missing)
            + ". Expected timestamp, open, high, low, close, and volume."
        )
    if df.empty:
        raise ValueError(
            "Candle data is empty. Load a session before analyzing."
        )
    prices = df[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    if prices.isnull().any().any() or not np.isfinite(prices.to_numpy()).all():
        raise ValueError("Open, high, low, and close must all be finite numbers.")
    if (prices <= 0).any().any():
        raise ValueError("Open, high, low, and close must all be positive.")
    volume = pd.to_numeric(df["volume"], errors="coerce")
    if volume.isnull().any() or (volume < 0).any():
        raise ValueError("Volume must be a number greater than or equal to zero.")
    high_ok = prices["high"] + 1e-8 >= prices[["open", "close", "low"]].max(axis=1)
    low_ok = prices["low"] - 1e-8 <= prices[["open", "close", "high"]].min(axis=1)
    if not bool(high_ok.all() and low_ok.all()):
        raise ValueError(
            "Invalid candles: each high must be at least the open, close, and low, "
            "and each low must be at most the open, close, and high."
        )
    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    if timestamps.isnull().any():
        raise ValueError("Every candle needs a valid timestamp.")
    if not timestamps.is_monotonic_increasing:
        raise ValueError(
            "Candle timestamps must be sorted from oldest to newest. "
            "Signals cannot be trusted on an unsorted feed."
        )
    if timestamps.duplicated().any():
        raise ValueError("Candle timestamps must be unique within a session.")


def percentage_change(current: float, reference: float) -> float:
    """Percent change from ``reference`` to ``current``.

    A result of 1.0 means +1 percent, not +100 percent.
    """
    current_value = float(current)
    reference_value = float(reference)
    if not math.isfinite(current_value) or not math.isfinite(reference_value):
        raise ValueError("Percentage change requires finite prices.")
    if reference_value == 0:
        raise ValueError(
            "Cannot compute percentage change because the reference price is zero."
        )
    return (current_value - reference_value) / reference_value * 100.0


def vwap(df: pd.DataFrame) -> pd.Series:
    """Session VWAP using the typical price and cumulative volume.

    Typical price is (high + low + close) / 3. The series resets only when the
    caller passes a new frame, so pass one session at a time.
    """
    validate_ohlcv(df)
    typical = (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3.0
    volume = df["volume"].astype(float)
    cumulative_volume = volume.cumsum()
    cumulative_value = (typical * volume).cumsum()
    result = cumulative_value / cumulative_volume.replace(0, np.nan)
    result.name = "vwap"
    return result
