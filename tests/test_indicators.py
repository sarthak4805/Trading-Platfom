"""Tests for candle checks, percent change, and VWAP."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from indicators import percentage_change, validate_ohlcv, vwap
from market_data import IST


def _frame(rows: list[tuple[float, float, float, float, float]]) -> pd.DataFrame:
    start = datetime(2099, 1, 1, 9, 15, tzinfo=IST)
    return pd.DataFrame(
        [
            {"timestamp": start + timedelta(minutes=index), "open": o, "high": h, "low": l, "close": c, "volume": v}
            for index, (o, h, l, c, v) in enumerate(rows)
        ]
    )


def test_vwap_matches_hand_calculation() -> None:
    result = vwap(_frame([(9, 10, 8, 9, 100), (10, 12, 10, 11, 100)]))
    # Typical prices are 9 and 11. Equal volume makes VWAP 9 then 10.
    assert result.iloc[0] == pytest.approx(9)
    assert result.iloc[1] == pytest.approx(10)


def test_vwap_at_the_first_bar_ignores_later_bars() -> None:
    frame = _frame([(10, 11, 9, 10, 50), (10, 20, 10, 18, 500)])
    full = vwap(frame)
    assert full.iloc[0] == pytest.approx(vwap(frame.iloc[:1]).iloc[0])
    assert full.iloc[1] != pytest.approx(full.iloc[0])


def test_vwap_rejects_a_missing_column() -> None:
    frame = _frame([(10, 11, 9, 10, 100)]).drop(columns=["volume"])
    with pytest.raises(ValueError, match="volume"):
        vwap(frame)


def test_percentage_change_uses_percentage_points() -> None:
    assert percentage_change(110, 100) == pytest.approx(10)
    with pytest.raises(ValueError, match="zero"):
        percentage_change(1, 0)


def test_validate_rejects_unsorted_candles() -> None:
    frame = _frame([(10, 11, 9, 10, 100), (10, 11, 9, 10, 100)])
    frame.loc[0, "timestamp"] = frame.loc[1, "timestamp"] + pd.Timedelta(minutes=5)
    with pytest.raises(ValueError, match="sorted"):
        validate_ohlcv(frame)


def test_validate_rejects_a_high_below_the_close() -> None:
    with pytest.raises(ValueError, match="high"):
        validate_ohlcv(_frame([(10, 10.5, 9, 11, 100)]))
