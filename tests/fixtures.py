"""Test-only candle builder. Not used by the app.

Builds one-minute sessions for a few made-up stocks so the strategy, board, and
backtest can be tested without Zerodha. Some stocks gap and then trade back
through the PDF entry price so that orders and trades actually occur.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from live_data import HistoryDay
from market_data import IST
from universe import Stock

SECTORS = ("Alpha", "Beta", "Gamma", "Delta", "Epsilon")
BARS = 375


def make_stocks(count: int = 25) -> list[Stock]:
    return [Stock(f"S{i:02d}", f"Stock {i}", SECTORS[i % len(SECTORS)]) for i in range(count)]


def session_stamps(session_date: date, bars: int = BARS) -> list[datetime]:
    start = datetime.combine(session_date, time(9, 15), tzinfo=IST)
    return [start + timedelta(minutes=i) for i in range(bars)]


def _path(rng: np.random.Generator, kind: str, bars: int) -> np.ndarray:
    if kind == "flat":
        return np.exp(np.cumsum(rng.normal(0.0, 0.0007, bars)))
    dip = 0.012 + 0.006 * rng.random()
    recovered = 15 + int(rng.integers(10, 40))
    path = np.ones(bars)
    path[0:15] = np.linspace(1.0 - dip * 0.25, 1.0 - dip, 15)
    path[15:recovered] = np.linspace(1.0 - dip, 0.990, recovered - 15)
    path[recovered:] = path[recovered - 1] * np.exp(np.cumsum(rng.normal(0.0, 0.0008, bars - recovered)))
    return path if kind == "dip" else 2.0 - path


def session_arrays(stocks: list[Stock], seed: int, bars: int = BARS) -> dict[str, dict]:
    rng = np.random.default_rng(seed)
    sector_gap = {sector: rng.normal(0.0, 0.012) for sector in sorted({stock.sector for stock in stocks})}
    out: dict[str, dict] = {}
    for index, stock in enumerate(stocks):
        gap = sector_gap[stock.sector] + rng.normal(0.0, 0.01)
        kind = "flat"
        if abs(gap) > 0.02 and rng.random() < 0.7:
            kind = "dip" if gap > 0 else "pop"
        anchor = 100.0 + 25.0 * index
        closes = anchor * _path(rng, kind, bars)
        opens = np.concatenate([[anchor], closes[:-1]])
        wick = rng.uniform(0.0002, 0.0009, bars)
        highs = np.round(np.maximum(opens, closes) * (1 + wick), 2)
        lows = np.round(np.minimum(opens, closes) * (1 - wick), 2)
        opens, closes = np.round(opens, 2), np.round(closes, 2)
        highs = np.maximum(highs, np.maximum(opens, closes))
        lows = np.minimum(lows, np.minimum(opens, closes))
        volume = rng.lognormal(np.log(60_000), 0.3, bars)
        if kind != "flat":
            volume[:15] *= 2.5
        prev_close = round(anchor / (1 + gap), 2)
        out[stock.symbol] = {
            "open": opens, "high": highs, "low": lows, "close": closes,
            "volume": np.round(volume).astype(np.int64),
            "prev_close": prev_close,
            "close_3d_ago": round(prev_close / (1 + rng.normal(0, 0.02)), 2),
        }
    return out


def session_frames(stocks: list[Stock], session_date: date, seed: int, bars: int = BARS) -> dict[str, pd.DataFrame]:
    stamps = session_stamps(session_date, bars)
    frames = {}
    for symbol, arrays in session_arrays(stocks, seed, bars).items():
        frame = pd.DataFrame({"timestamp": stamps, **{k: arrays[k] for k in ("open", "high", "low", "close", "volume")}})
        frame["prev_close"] = arrays["prev_close"]
        frame["close_3d_ago"] = arrays["close_3d_ago"]
        frames[symbol] = frame
    return frames


def history_days(stocks: list[Stock], dates: list[date], seed: int = 1) -> list[HistoryDay]:
    return [
        HistoryDay(day, session_stamps(day), list(stocks), session_arrays(stocks, seed * 1000 + i))
        for i, day in enumerate(dates)
    ]


class FakeEngine:
    """The two things the board needs from the market: closed candles and a clock."""

    def __init__(self, frames: dict[str, pd.DataFrame], stocks: list[Stock], clock: datetime) -> None:
        self.frames, self.stocks, self.clock = frames, stocks, clock

    def visible(self, symbol: str) -> pd.DataFrame:
        frame = self.frames[symbol]
        return frame[frame["timestamp"] + pd.Timedelta(minutes=1) <= pd.Timestamp(self.clock)].reset_index(drop=True)
