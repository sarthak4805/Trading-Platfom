"""Bullish / bearish intraday strategy from the strategy PDF.

Bullish (buy)
    1. Rank sectors by average change from the previous close. Use the strongest.
    2. In each of those sectors, take the top gainers.
    3. Note each stock's 09:15 opening price.
    4. Change from the previous close must be more than +1.5%.
    5. Volume so far today must be more than 10 lakh shares.
    6. At 09:30, mark the high and low of 09:15 to 09:30 (the 15-minute range).
    7. Suggested buy price = opening price - 0.5%. The order fills only when a
       later candle trades at that price.
    8. Stop-loss = the 15-minute low.
    9. Target = buy price + 1.5%.
    10. Buy only when the price and the buy price are above VWAP.
Bearish (short) is the mirror: weakest sectors, top losers, change below -1.5%,
sell at open + 0.5%, stop at the 15-minute high, target 1.5% lower, below VWAP.
These are the PDF defaults. ``StrategyConfig`` can change the window length and
replace the window stop with a fixed percentage from the buy or sell price.

Every decision uses closed candles only. A step evaluated on candle t places an
order for candle t + 1. On the fill candle only the stop can trigger, because the
order of prices inside that candle is unknown. After that, a stop and target in
the same candle count as a stop. One trade per stock per day, a daily trade
limit across all stocks, no new entries after the cutoff, and every position is
closed at the session end. No real order is placed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from backtest import CostAssumptions, Trade, resolve_exit, trade_pnl
from market_data import IST
from risk_manager import RiskConfig, can_open_new_trade, position_size
from universe import Stock

BUY = "BUY"
SHORT = "SHORT-SELL"
NO_TRADE = "NO TRADE"

WAITING = "Waiting"
TRIGGERED = "Triggered"
STOPPED = "Stopped"
TARGET_HIT = "Target hit"
SESSION_CLOSE = "Session close"

EXIT_STATUS = {
    "stop_loss": STOPPED,
    "stop_loss_gap": STOPPED,
    "target": TARGET_HIT,
    "target_gap": TARGET_HIT,
    "session_close": SESSION_CLOSE,
}


@dataclass(frozen=True)
class StrategyConfig:
    """Editable PDF settings. Percent fields are percentage points (1.5 means 1.5%)."""

    sectors_each_side: int = 3
    stocks_per_sector: int = 5
    min_change_pct: float = 1.5
    min_volume: int = 1_000_000
    entry_offset_pct: float = 0.5
    target_pct: float = 1.5
    stop_pct: float | None = None  # None: the opening-window low (buy) or high (short), as in the PDF
    observation_start: time = time(9, 15)
    signal_start: time = time(9, 30)
    no_new_entries_after: time = time(15, 0)
    session_end: time = time(15, 30)

    def __post_init__(self) -> None:
        if not (self.observation_start < self.signal_start < self.session_end):
            raise ValueError(
                "The opening window must start before signals, and signals must start "
                "before the session end. Defaults are 09:15, 09:30, and 15:30."
            )
        if not (self.signal_start < self.no_new_entries_after <= self.session_end):
            raise ValueError(
                "The no-new-entries time must be after signals start and no later than "
                "the session end. The default is 15:00."
            )
        if self.sectors_each_side < 1 or self.stocks_per_sector < 1:
            raise ValueError("Use at least one sector and one stock per sector.")
        if self.min_change_pct < 0 or self.min_volume < 0:
            raise ValueError("Minimum change and minimum volume cannot be negative.")
        if self.entry_offset_pct < 0:
            raise ValueError("The entry offset from the open cannot be negative.")
        if self.target_pct <= 0:
            raise ValueError("The target must be greater than zero.")
        if self.stop_pct is not None and self.stop_pct <= 0:
            raise ValueError("A percentage stop-loss must be greater than zero.")

    @property
    def window_minutes(self) -> int:
        """Length of the opening window (09:15 to 09:30 is 15)."""
        start = self.observation_start.hour * 60 + self.observation_start.minute
        return self.signal_start.hour * 60 + self.signal_start.minute - start

    def stop_text(self, buy: bool = True) -> str:
        """Where the stop-loss sits, in words."""
        if self.stop_pct is None:
            return f"the {self.window_minutes}-minute {'low' if buy else 'high'}"
        return f"{self.stop_pct:g}% {'below the buy' if buy else 'above the sell'} price"


@dataclass(frozen=True)
class RuleResult:
    rule_id: str
    passed: bool
    detail: str


@dataclass
class Day:
    """One session for many stocks, as arrays of shape (stocks, candles)."""

    symbols: list[str]
    names: list[str]
    sectors: list[str]
    sector_names: list[str]
    sector_ids: np.ndarray
    stamps: list[datetime]
    times: list[time]
    bar_minutes: int
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    prev_close: np.ndarray
    close_3d_ago: np.ndarray
    vwap: np.ndarray
    cum_volume: np.ndarray
    change_pct: np.ndarray
    session_open: np.ndarray
    window_high: np.ndarray
    window_low: np.ndarray

    def index(self, symbol: str) -> int:
        return self.symbols.index(symbol)

    def __len__(self) -> int:
        return len(self.times)


@dataclass
class Screen:
    """Steps that do not depend on the entry offset or target."""

    in_window: np.ndarray
    sector_avg: np.ndarray
    sector_rank: np.ndarray
    rank_up: np.ndarray
    rank_down: np.ndarray
    sector_size: np.ndarray
    bull_sector: np.ndarray
    bear_sector: np.ndarray
    bull_top: np.ndarray
    bear_top: np.ndarray
    bull_change: np.ndarray
    bear_change: np.ndarray
    volume_ok: np.ndarray


@dataclass
class Entries:
    """Order prices, levels, and the full step result for every stock and candle."""

    buy_price: np.ndarray
    buy_stop: np.ndarray
    buy_target: np.ndarray
    short_price: np.ndarray
    short_stop: np.ndarray
    short_target: np.ndarray
    buy_stop_ok: np.ndarray
    short_stop_ok: np.ndarray
    buy_vwap: np.ndarray
    short_vwap: np.ndarray
    buy_ok: np.ndarray
    short_ok: np.ndarray


@dataclass(frozen=True)
class OpenPosition:
    symbol: str
    side: str
    signal_index: int
    entry_index: int
    entry_time: datetime
    entry: float
    stop: float
    target: float
    quantity: int
    mark: float
    net_if_closed: float


@dataclass(frozen=True)
class PendingOrder:
    symbol: str
    side: str
    signal_time: datetime
    price: float
    stop: float
    target: float


@dataclass
class DayReplay:
    trades: list[Trade] = field(default_factory=list)
    open_positions: list[OpenPosition] = field(default_factory=list)
    pending: list[PendingOrder] = field(default_factory=list)
    skipped_by_limit: int = 0


@dataclass
class StockView:
    """Everything the screens show for one stock at the latest closed candle."""

    symbol: str
    name: str
    sector: str
    side: str
    action: str
    status: str | None
    price: float
    prev_close: float
    day_high: float
    day_low: float
    window_high: float
    window_low: float
    change_pct: float
    three_day_pct: float
    volume: int
    vwap: float
    order_price: float
    stop: float
    target: float
    sector_rank: int
    sector_count: int
    rank_in_sector: int
    sector_size: int
    in_candidate_list: bool
    passed: list[RuleResult]
    failed: list[RuleResult]
    trade: Trade | None
    position: OpenPosition | None
    pending: PendingOrder | None
    summary: str


def _ist(stamps: pd.Series) -> pd.Series:
    stamps = pd.to_datetime(stamps)
    if stamps.dt.tz is not None:
        return stamps.dt.tz_convert(IST)
    return stamps


def build_day(
    frames: Mapping[str, pd.DataFrame],
    stocks: Sequence[Stock],
    config: StrategyConfig,
    bar_minutes: int = 1,
) -> Day:
    """Stack closed one-minute candles for every stock, optionally grouped into larger candles.

    Every frame must hold the same timestamps plus ``prev_close`` and ``close_3d_ago``.
    Candles outside the observation start to the session end are dropped.
    """
    if not stocks:
        raise ValueError("A trading day needs at least one stock.")
    first = frames[stocks[0].symbol]
    reference = pd.DatetimeIndex(first["timestamp"]).asi8
    arrays: dict[str, dict[str, np.ndarray | float]] = {}
    for stock in stocks:
        frame = frames[stock.symbol]
        if len(frame) != len(first) or not np.array_equal(pd.DatetimeIndex(frame["timestamp"]).asi8, reference):
            raise ValueError(f"{stock.symbol} does not share the same candle timestamps as the other stocks.")
        if "prev_close" not in frame.columns or "close_3d_ago" not in frame.columns:
            raise ValueError(f"{stock.symbol} needs prev_close and close_3d_ago columns.")
        arrays[stock.symbol] = {name: frame[name].to_numpy(dtype=float) for name in ("open", "high", "low", "close", "volume")}
        arrays[stock.symbol]["prev_close"] = float(frame["prev_close"].iloc[0])
        arrays[stock.symbol]["close_3d_ago"] = float(frame["close_3d_ago"].iloc[0])
    stamps = [stamp.to_pydatetime() for stamp in _ist(first["timestamp"])]
    return day_from_arrays(stamps, arrays, stocks, config, bar_minutes)


def day_from_arrays(
    stamps: Sequence[datetime],
    arrays: Mapping[str, Mapping[str, np.ndarray | float]],
    stocks: Sequence[Stock],
    config: StrategyConfig,
    bar_minutes: int = 1,
) -> Day:
    """``build_day`` for plain arrays: one-minute open/high/low/close/volume plus the two closes."""
    if bar_minutes < 1:
        raise ValueError("Candle size must be at least one minute.")
    if not stocks:
        raise ValueError("A trading day needs at least one stock.")
    keep = np.array([config.observation_start <= stamp.time() < config.session_end for stamp in stamps])
    if not keep.any():
        raise ValueError("No closed candles fall inside the session window yet.")
    open_, high, low, close, volume = (
        np.vstack([np.asarray(arrays[stock.symbol][name], dtype=float)[keep] for stock in stocks])
        for name in ("open", "high", "low", "close", "volume")
    )
    prev_close = [float(arrays[stock.symbol]["prev_close"]) for stock in stocks]
    close_3d = [float(arrays[stock.symbol]["close_3d_ago"]) for stock in stocks]
    kept_stamps = [stamp for stamp, flag in zip(stamps, keep) if flag]
    if bar_minutes > 1:
        count = open_.shape[1]
        starts = np.arange(0, count, bar_minutes)
        ends = np.minimum(starts + bar_minutes, count) - 1
        open_ = open_[:, starts]
        high = np.maximum.reduceat(high, starts, axis=1)
        low = np.minimum.reduceat(low, starts, axis=1)
        close = close[:, ends]
        volume = np.add.reduceat(volume, starts, axis=1)
        kept_stamps = [kept_stamps[i] for i in starts]
    times = [stamp.time() for stamp in kept_stamps]
    prev = np.array(prev_close)
    typical = (high + low + close) / 3.0
    cum_volume = np.cumsum(volume, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = np.cumsum(typical * volume, axis=1) / np.where(cum_volume > 0, cum_volume, np.nan)
    window = np.array([config.observation_start <= t < config.signal_start for t in times])
    if window.any():
        window_high = high[:, window].max(axis=1)
        window_low = low[:, window].min(axis=1)
    else:
        window_high = np.full(len(stocks), np.nan)
        window_low = np.full(len(stocks), np.nan)
    sectors = [stock.sector for stock in stocks]
    sector_names = sorted(set(sectors))
    return Day(
        symbols=[stock.symbol for stock in stocks],
        names=[stock.name for stock in stocks],
        sectors=sectors,
        sector_names=sector_names,
        sector_ids=np.array([sector_names.index(sector) for sector in sectors]),
        stamps=kept_stamps,
        times=times,
        bar_minutes=bar_minutes,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        prev_close=prev,
        close_3d_ago=np.array(close_3d),
        vwap=vwap,
        cum_volume=cum_volume,
        change_pct=(close / prev[:, None] - 1.0) * 100.0,
        session_open=open_[:, 0].copy(),
        window_high=window_high,
        window_low=window_low,
    )


def _ranks(values: np.ndarray, descending: bool) -> np.ndarray:
    """Rank along axis 0 for every column. 0 is the best. Ties keep the input order."""
    order = np.argsort(-values if descending else values, axis=0, kind="stable")
    ranks = np.empty_like(order)
    ranks[order, np.arange(values.shape[1])] = np.arange(values.shape[0])[:, None]
    return ranks


def screen_day(day: Day, config: StrategyConfig) -> Screen:
    """Steps 1, 2, 4, 5, and the time window for every stock and candle."""
    cutoff = min(config.no_new_entries_after, config.session_end)
    in_window = np.array([config.signal_start <= t < cutoff for t in day.times])
    sector_count = len(day.sector_names)
    sector_avg = np.vstack(
        [day.change_pct[day.sector_ids == sector].mean(axis=0) for sector in range(sector_count)]
    )
    sector_rank = _ranks(sector_avg, descending=True)
    each_side = min(config.sectors_each_side, sector_count)
    rank_up = np.zeros(day.change_pct.shape, dtype=int)
    rank_down = np.zeros(day.change_pct.shape, dtype=int)
    sector_size = np.zeros(len(day.symbols), dtype=int)
    for sector in range(sector_count):
        rows = np.nonzero(day.sector_ids == sector)[0]
        rank_up[rows] = _ranks(day.change_pct[rows], descending=True)
        rank_down[rows] = _ranks(day.change_pct[rows], descending=False)
        sector_size[rows] = len(rows)
    stock_sector_rank = sector_rank[day.sector_ids]
    return Screen(
        in_window=in_window,
        sector_avg=sector_avg,
        sector_rank=sector_rank,
        rank_up=rank_up,
        rank_down=rank_down,
        sector_size=sector_size,
        bull_sector=stock_sector_rank < each_side,
        bear_sector=stock_sector_rank >= sector_count - each_side,
        bull_top=rank_up < config.stocks_per_sector,
        bear_top=rank_down < config.stocks_per_sector,
        bull_change=day.change_pct > config.min_change_pct,
        bear_change=day.change_pct < -config.min_change_pct,
        volume_ok=day.cum_volume > config.min_volume,
    )


def entry_levels(day: Day, config: StrategyConfig, screen: Screen) -> Entries:
    """Steps 6 to 10 on top of the screen."""
    offset = config.entry_offset_pct / 100.0
    target = config.target_pct / 100.0
    buy_price = np.round(day.session_open * (1.0 - offset), 2)
    short_price = np.round(day.session_open * (1.0 + offset), 2)
    if config.stop_pct is None:
        buy_stop, short_stop = day.window_low, day.window_high
    else:
        stop = config.stop_pct / 100.0
        buy_stop = np.round(buy_price * (1.0 - stop), 2)
        short_stop = np.round(short_price * (1.0 + stop), 2)
    with np.errstate(invalid="ignore"):
        buy_stop_ok = np.isfinite(buy_stop) & (buy_stop < buy_price)
        short_stop_ok = np.isfinite(short_stop) & (short_stop > short_price)
        buy_vwap = (day.close > day.vwap) & (buy_price[:, None] > day.vwap)
        short_vwap = (day.close < day.vwap) & (short_price[:, None] < day.vwap)
    common = screen.in_window[None, :] & screen.volume_ok
    return Entries(
        buy_price=buy_price,
        buy_stop=buy_stop,
        buy_target=np.round(buy_price * (1.0 + target), 2),
        short_price=short_price,
        short_stop=short_stop,
        short_target=np.round(short_price * (1.0 - target), 2),
        buy_stop_ok=buy_stop_ok,
        short_stop_ok=short_stop_ok,
        buy_vwap=buy_vwap,
        short_vwap=short_vwap,
        buy_ok=common & screen.bull_sector & screen.bull_top & screen.bull_change & buy_stop_ok[:, None] & buy_vwap,
        short_ok=common & screen.bear_sector & screen.bear_top & screen.bear_change & short_stop_ok[:, None] & short_vwap,
    )


def replay_day(
    day: Day,
    config: StrategyConfig,
    risk: RiskConfig,
    costs: CostAssumptions,
    *,
    session_over: bool,
    screen: Screen | None = None,
) -> DayReplay:
    """Walk the candles in order and return trades, open positions, and orders still waiting.

    ``session_over`` says the last candle is the end of the session, so any open
    position is closed at its close. Otherwise positions stay open.
    """
    screen = screen or screen_day(day, config)
    entries = entry_levels(day, config, screen)
    count = len(day)
    fills: list[tuple[int, str, int, str, int]] = []
    for row, symbol in enumerate(day.symbols):
        best = None
        for side, ok, price in (
            (BUY, entries.buy_ok[row], entries.buy_price[row]),
            (SHORT, entries.short_ok[row], entries.short_price[row]),
        ):
            if count < 2:
                continue
            touched = (day.low[row, 1:] <= price) & (day.high[row, 1:] >= price)
            hits = np.nonzero(ok[:-1] & touched)[0]
            if hits.size and (best is None or hits[0] + 1 < best[0]):
                best = (int(hits[0]) + 1, symbol, row, side, int(hits[0]))
        if best is not None:
            fills.append(best)
    fills.sort(key=lambda item: (item[0], item[1]))

    result = DayReplay()
    traded: set[int] = set()
    opened = 0
    for fill_index, symbol, row, side, signal_index in fills:
        allowed, _reason = can_open_new_trade(0, opened, risk)
        if not allowed:
            result.skipped_by_limit += 1
            continue
        opened += 1
        traded.add(row)
        _play_trade(day, entries, row, side, signal_index, fill_index, risk, costs, session_over, result)

    if count and not session_over:
        last = count - 1
        limit_left = can_open_new_trade(0, opened, risk)[0]
        for row, symbol in enumerate(day.symbols):
            if row in traded or not limit_left:
                continue
            if entries.buy_ok[row, last]:
                result.pending.append(
                    PendingOrder(symbol, BUY, day.stamps[last], float(entries.buy_price[row]),
                                 float(entries.buy_stop[row]), float(entries.buy_target[row]))
                )
            elif entries.short_ok[row, last]:
                result.pending.append(
                    PendingOrder(symbol, SHORT, day.stamps[last], float(entries.short_price[row]),
                                 float(entries.short_stop[row]), float(entries.short_target[row]))
                )
    return result


def _play_trade(
    day: Day,
    entries: Entries,
    row: int,
    side: str,
    signal_index: int,
    fill_index: int,
    risk: RiskConfig,
    costs: CostAssumptions,
    session_over: bool,
    result: DayReplay,
) -> None:
    if side == BUY:
        entry, stop, target = float(entries.buy_price[row]), float(entries.buy_stop[row]), float(entries.buy_target[row])
    else:
        entry, stop, target = float(entries.short_price[row]), float(entries.short_stop[row]), float(entries.short_target[row])
    quantity = position_size(entry, risk.notional_inr)
    exit_index, reason, exit_raw = None, None, None
    stop_on_fill_bar = day.low[row, fill_index] <= stop if side == BUY else day.high[row, fill_index] >= stop
    if stop_on_fill_bar:
        exit_index, reason, exit_raw = fill_index, "stop_loss", stop
    else:
        for index in range(fill_index + 1, len(day)):
            bar = {"open": day.open[row, index], "high": day.high[row, index], "low": day.low[row, index]}
            reason, exit_raw = resolve_exit(side, stop, target, bar)
            if reason is not None:
                exit_index = index
                break
    if exit_index is None and session_over:
        exit_index, reason, exit_raw = len(day) - 1, "session_close", float(day.close[row, -1])
    symbol = day.symbols[row]
    if exit_index is None:
        mark = float(day.close[row, -1])
        result.open_positions.append(
            OpenPosition(
                symbol=symbol, side=side, signal_index=signal_index, entry_index=fill_index,
                entry_time=day.stamps[fill_index], entry=entry, stop=stop, target=target,
                quantity=quantity, mark=mark, net_if_closed=trade_pnl(side, entry, mark, quantity, costs)["net_pnl"],
            )
        )
        return
    money = trade_pnl(side, entry, float(exit_raw), quantity, costs)
    result.trades.append(
        Trade(
            symbol=symbol, side=side, signal_index=signal_index, entry_index=fill_index, exit_index=exit_index,
            signal_time=day.stamps[signal_index], entry_time=day.stamps[fill_index], exit_time=day.stamps[exit_index],
            entry_raw=entry, exit_raw=float(exit_raw), entry_fill=money["entry_fill"], exit_fill=money["exit_fill"],
            stop_loss=stop, target=target, quantity=quantity, exit_reason=str(reason),
            gross_pnl=money["gross_pnl"], slippage_cost=money["slippage_cost"],
            transaction_cost=money["transaction_cost"], net_pnl=money["net_pnl"],
        )
    )


# ---------------------------------------------------------------- plain-language steps

def _lakh(value: float) -> str:
    return f"{value / 100_000:,.1f} lakh"


def step_results(day: Day, config: StrategyConfig, screen: Screen, entries: Entries, row: int, index: int, side: str) -> list[RuleResult]:
    """The PDF steps for one stock at one closed candle, written for a beginner."""
    buy = side == BUY
    when = day.times[index].strftime("%H:%M")
    cutoff = min(config.no_new_entries_after, config.session_end)
    results = [
        RuleResult(
            "Time",
            bool(screen.in_window[index]),
            f"This candle opened at {when}. New orders are allowed from "
            f"{config.signal_start:%H:%M} until {cutoff:%H:%M}.",
        )
    ]
    sector = day.sectors[row]
    rank = int(screen.sector_rank[day.sector_ids[row], index])
    sector_count = len(day.sector_names)
    avg = float(screen.sector_avg[day.sector_ids[row], index])
    each_side = min(config.sectors_each_side, sector_count)
    if buy:
        results.append(RuleResult(
            "Step 1", bool(screen.bull_sector[row, index]),
            f"{sector} is #{rank + 1} of {sector_count} sectors by average change from yesterday ({avg:+.2f}%). "
            f"Buys come from the top {each_side}.",
        ))
        place = int(screen.rank_up[row, index])
        results.append(RuleResult(
            "Step 2", bool(screen.bull_top[row, index]),
            f"{day.symbols[row]} is #{place + 1} of {int(screen.sector_size[row])} gainers in {sector}. "
            f"Buys use the top {config.stocks_per_sector}.",
        ))
    else:
        results.append(RuleResult(
            "Step 1", bool(screen.bear_sector[row, index]),
            f"{sector} is #{sector_count - rank} of {sector_count} sectors from the weakest "
            f"({avg:+.2f}%). Shorts come from the weakest {each_side}.",
        ))
        place = int(screen.rank_down[row, index])
        results.append(RuleResult(
            "Step 2", bool(screen.bear_top[row, index]),
            f"{day.symbols[row]} is #{place + 1} of {int(screen.sector_size[row])} losers in {sector}. "
            f"Shorts use the top {config.stocks_per_sector}.",
        ))
    change = float(day.change_pct[row, index])
    prev = float(day.prev_close[row])
    if buy:
        results.append(RuleResult(
            "Step 4", bool(screen.bull_change[row, index]),
            f"Change from yesterday's close ({prev:.2f}) is {change:+.2f}%. A buy needs more than +{config.min_change_pct:.2f}%.",
        ))
    else:
        results.append(RuleResult(
            "Step 4", bool(screen.bear_change[row, index]),
            f"Change from yesterday's close ({prev:.2f}) is {change:+.2f}%. A short needs a fall of more than {config.min_change_pct:.2f}%.",
        ))
    results.append(RuleResult(
        "Step 5", bool(screen.volume_ok[row, index]),
        f"Volume so far today is {_lakh(day.cum_volume[row, index])}. It must be more than {_lakh(config.min_volume)}.",
    ))
    opening = float(day.session_open[row])
    if buy:
        price, stop, ok = float(entries.buy_price[row]), float(entries.buy_stop[row]), bool(entries.buy_stop_ok[row])
        level = "low"
        relation = "below" if ok else "not below"
    else:
        price, stop, ok = float(entries.short_price[row]), float(entries.short_stop[row]), bool(entries.short_stop_ok[row])
        level = "high"
        relation = "above" if ok else "not above"
    if not np.isfinite(stop) or day.times[index] < config.signal_start:
        results.append(RuleResult(
            "Steps 6-8", False,
            f"The {config.window_minutes}-minute range ({config.observation_start:%H:%M} to "
            f"{config.signal_start:%H:%M}) is still forming.",
        ))
    else:
        word = "buy" if buy else "sell"
        intro = (
            f"Open {opening:.2f}, so the suggested {word} price is {price:.2f} "
            f"({'-' if buy else '+'}{config.entry_offset_pct:g}%). "
        )
        if config.stop_pct is None:
            detail = f"The {config.window_minutes}-minute {level} {stop:.2f} is {relation} it" + (
                ", so it can be the stop-loss." if ok else ", so a stop-loss there would be on the wrong side."
            )
        else:
            detail = f"The stop-loss is {config.stop_text(buy)}, at {stop:.2f}."
        results.append(RuleResult("Steps 6-8", ok, intro + detail))
    vwap = float(day.vwap[row, index])
    close = float(day.close[row, index])
    if buy:
        results.append(RuleResult(
            "Step 10", bool(entries.buy_vwap[row, index]),
            f"Price {close:.2f} and buy price {price:.2f} versus VWAP {vwap:.2f}. A buy needs both above VWAP.",
        ))
    else:
        results.append(RuleResult(
            "Step 10", bool(entries.short_vwap[row, index]),
            f"Price {close:.2f} and sell price {price:.2f} versus VWAP {vwap:.2f}. A short needs both below VWAP.",
        ))
    return results


def stock_views(
    day: Day,
    config: StrategyConfig,
    replay: DayReplay,
    screen: Screen | None = None,
) -> list[StockView]:
    """One view per stock at the latest closed candle."""
    screen = screen or screen_day(day, config)
    entries = entry_levels(day, config, screen)
    last = len(day) - 1
    trades = {trade.symbol: trade for trade in replay.trades}
    positions = {position.symbol: position for position in replay.open_positions}
    pending = {order.symbol: order for order in replay.pending}
    views = []
    for row, symbol in enumerate(day.symbols):
        trade, position, order = trades.get(symbol), positions.get(symbol), pending.get(symbol)
        if trade is not None:
            side, status = trade.side, EXIT_STATUS.get(trade.exit_reason, trade.exit_reason)
        elif position is not None:
            side, status = position.side, TRIGGERED
        elif order is not None:
            side, status = order.side, WAITING
        else:
            side = BUY if day.change_pct[row, last] >= 0 else SHORT
            status = None
        rules = step_results(day, config, screen, entries, row, last, side)
        passed = [rule for rule in rules if rule.passed]
        failed = [rule for rule in rules if not rule.passed]
        ok = bool((entries.buy_ok if side == BUY else entries.short_ok)[row, last])
        action = side if ok else NO_TRADE
        buy = side == BUY
        order_price = float(entries.buy_price[row] if buy else entries.short_price[row])
        stop = float(entries.buy_stop[row] if buy else entries.short_stop[row])
        target = float(entries.buy_target[row] if buy else entries.short_target[row])
        candidate = bool((screen.bull_sector & screen.bull_top)[row, last] if buy else (screen.bear_sector & screen.bear_top)[row, last])
        word = "buy" if buy else "sell"
        if trade is not None:
            summary = f"{status}. {side} at {trade.entry_raw:.2f} ({trade.entry_time:%H:%M}), out at {trade.exit_raw:.2f} ({trade.exit_time:%H:%M})."
        elif position is not None:
            summary = f"Triggered. {side} at {position.entry:.2f} ({position.entry_time:%H:%M}). Stop {position.stop:.2f}, target {position.target:.2f}."
        elif order is not None:
            summary = f"Waiting. Every step passed. Suggested {word} price {order.price:.2f}; the order fills when the price trades there."
        elif failed:
            summary = f"NO TRADE. {failed[0].detail}"
        else:
            summary = "NO TRADE. The daily trade limit has been reached."
        rank = int(screen.sector_rank[day.sector_ids[row], last])
        views.append(
            StockView(
                symbol=symbol,
                name=day.names[row],
                sector=day.sectors[row],
                side=side,
                action=action,
                status=status,
                price=float(day.close[row, last]),
                prev_close=float(day.prev_close[row]),
                day_high=float(day.high[row, : last + 1].max()),
                day_low=float(day.low[row, : last + 1].min()),
                window_high=float(day.window_high[row]),
                window_low=float(day.window_low[row]),
                change_pct=float(day.change_pct[row, last]),
                three_day_pct=(float(day.close[row, last]) / float(day.close_3d_ago[row]) - 1.0) * 100.0,
                volume=int(day.cum_volume[row, last]),
                vwap=float(day.vwap[row, last]),
                order_price=order_price,
                stop=stop,
                target=target,
                sector_rank=rank + 1,
                sector_count=len(day.sector_names),
                rank_in_sector=int((screen.rank_up if buy else screen.rank_down)[row, last]) + 1,
                sector_size=int(screen.sector_size[row]),
                in_candidate_list=candidate,
                passed=passed,
                failed=failed,
                trade=trade,
                position=position,
                pending=order,
                summary=summary,
            )
        )
    return views
