"""Opening-range board: one-screen summary of the current session.

Every number here comes from Zerodha candles and the PDF strategy. The index tiles
use real index candles when Zerodha returns them; otherwise an equal-weight average
of the tracked names stands in and is labelled as such. Streamlit
strips inline <svg>, so every chart, sparkline, and icon is sent as an <img>.
"""

from __future__ import annotations

import base64
import html
from dataclasses import dataclass
from datetime import datetime, time

import numpy as np
import pandas as pd
import streamlit as st

from market_data import IST
from risk_manager import RiskConfig, position_size
from strategy import (
    BUY,
    SESSION_CLOSE,
    STOPPED,
    TARGET_HIT,
    TRIGGERED,
    WAITING,
    Day,
    DayReplay,
    Screen,
    StockView,
    StrategyConfig,
)

EMA_FAST = 9
EMA_SLOW = 21
INDEX_BASE = 10_000.0
TABLE_ROWS = 15
SECTOR_CARDS = 10

SHORT_SECTOR = {
    "Automobile": "Auto",
    "Capital Goods": "Cap Goods",
    "Consumer Durables": "Durables",
    "Consumer Services": "Cons Svcs",
    "Financial Services": "Finance",
    "Information Technology": "IT",
    "Infrastructure": "Infra",
    "Metals & Mining": "Metals",
    "Telecommunication": "Telecom",
}

STATUS_ORDER = {TRIGGERED: 0, WAITING: 1, TARGET_HIT: 2, SESSION_CLOSE: 3, STOPPED: 4}

GREEN = "#22c55e"
RED = "#ef4444"
BLUE = "#3b82f6"
ORANGE = "#f59e0b"
MUTED = "#8b95a5"
TEXT = "#e5e9f0"


@dataclass(frozen=True)
class TickerItem:
    label: str
    value: str
    change_points: float | None
    change_pct: float | None
    series: list[float]
    note: str


@dataclass(frozen=True)
class SectorCard:
    sector: str
    change_pct: float
    series: list[float]


@dataclass(frozen=True)
class Mover:
    symbol: str
    change_pct: float
    buyer_share: float
    session_volume: int


@dataclass(frozen=True)
class OpportunityRow:
    view: StockView
    setup: str
    risk_reward: float | None
    steps_passed: int
    steps_total: int
    signal: str


@dataclass
class ChartData:
    symbol: str
    name: str
    price: float
    change_pct: float
    bar_minutes: int
    candles: pd.DataFrame
    ema_fast: pd.Series
    ema_slow: pd.Series
    vwap: pd.Series
    or_high: float | None
    or_low: float | None
    window_end: time
    view: StockView


@dataclass
class BoardData:
    clock: datetime
    phase: str
    ticker: list[TickerItem]
    advancing: int
    declining: int
    best: SectorCard
    worst: SectorCard
    waiting: int
    in_trade: int
    trades_used: int
    max_trades: int
    day_pnl: float
    capital: float
    pnl_series: list[float]
    risk_used: float
    rows: list[OpportunityRow]
    sector_cards: list[SectorCard]
    performance: list[SectorCard]
    focus_sector: str
    movers: list[Mover]
    min_volume: int
    entry_offset_pct: float
    target_pct: float
    notional: float


def index_ticker(frames: dict[str, pd.DataFrame]) -> list[TickerItem]:
    """Ticker items from real index candles (live mode). Change is against the previous close."""
    items = []
    for label, frame in frames.items():
        if frame is None or frame.empty:
            continue
        prev = float(frame["prev_close"].iloc[0])
        closes = frame["close"].astype(float).tolist()
        last = closes[-1]
        items.append(
            TickerItem(
                label,
                f"{last:,.2f}",
                last - prev,
                (last / prev - 1.0) * 100.0 if prev else 0.0,
                [prev] + closes,
                f"{label} from Zerodha, last closed minute. Change is against the previous close.",
            )
        )
    return items


def short_sector(sector: str) -> str:
    return SHORT_SECTOR.get(sector, sector)


def session_phase(clock: time, strategy: StrategyConfig) -> str:
    if clock <= strategy.signal_start:
        return "Observation window"
    if clock >= strategy.session_end:
        return "Session closed"
    if clock > strategy.no_new_entries_after:
        return "No new entries"
    return "Trading window"


def resample_bars(frame: pd.DataFrame, minutes: int = 5) -> pd.DataFrame:
    """Group contiguous one-minute candles into fixed buckets from the first bar.

    The last bucket may be partial. It only holds minutes that already closed.
    Per-session constants such as ``prev_close`` are carried over.
    """
    base = ["timestamp", "open", "high", "low", "close", "volume"]
    constants = [name for name in ("prev_close", "close_3d_ago") if name in frame.columns]
    if frame.empty:
        return frame.iloc[0:0][base + constants]
    groups = np.arange(len(frame)) // minutes
    grouped = frame.reset_index(drop=True).groupby(groups)
    out = pd.DataFrame(
        {
            "timestamp": grouped["timestamp"].first(),
            "open": grouped["open"].first(),
            "high": grouped["high"].max(),
            "low": grouped["low"].min(),
            "close": grouped["close"].last(),
            "volume": grouped["volume"].sum(),
        }
    ).reset_index(drop=True)
    for name in constants:
        out[name] = frame[name].iloc[0]
    return out


def buyer_share(frame: pd.DataFrame) -> float:
    """Share of volume on up-candles. Flat candles count half to each side."""
    volume = frame["volume"].astype(float)
    total = float(volume.sum())
    if total <= 0:
        return 0.5
    up = float(volume[frame["close"] > frame["open"]].sum())
    flat = float(volume[frame["close"] == frame["open"]].sum())
    return (up + flat / 2.0) / total


def default_focus(views: list[StockView], ranked_sectors: list[str], min_names: int = 3) -> str:
    """Strongest sector with enough names to fill the movers card."""
    counts = pd.Series([view.sector for view in views]).value_counts()
    for sector in ranked_sectors:
        if counts.get(sector, 0) >= min_names:
            return sector
    return ranked_sectors[0]


def risk_reward(view: StockView) -> float | None:
    if view.side == BUY:
        risk, reward = view.order_price - view.stop, view.target - view.order_price
    else:
        risk, reward = view.stop - view.order_price, view.order_price - view.target
    if not np.isfinite(risk) or risk <= 0:
        return None
    return reward / risk


def signal_label(view: StockView) -> str:
    if view.status == TRIGGERED:
        return "BUY" if view.side == BUY else "SHORT"
    if view.status == WAITING:
        return "WAIT"
    if view.status == TARGET_HIT:
        return "TARGET"
    if view.status == STOPPED:
        return "STOPPED"
    if view.status == SESSION_CLOSE:
        return "CLOSED"
    return "NO TRADE"


def opportunity_rows(views: list[StockView], choice: str = "All", limit: int = TABLE_ROWS) -> list[OpportunityRow]:
    """Candidates from the sector screen plus anything that traded today, best first."""
    pool = [view for view in views if view.in_candidate_list or view.status is not None]
    if choice == "Long":
        pool = [view for view in pool if view.side == BUY]
    elif choice == "Short":
        pool = [view for view in pool if view.side != BUY]
    elif choice == "Waiting":
        pool = [view for view in pool if view.status == WAITING]
    pool.sort(key=lambda v: (STATUS_ORDER.get(v.status, 9), -len(v.passed), -abs(v.change_pct), v.symbol))
    rows = []
    for view in pool[:limit]:
        rows.append(
            OpportunityRow(
                view=view,
                setup="Gap-up dip buy" if view.side == BUY else "Gap-down pop sell",
                risk_reward=risk_reward(view),
                steps_passed=len(view.passed),
                steps_total=len(view.passed) + len(view.failed),
                signal=signal_label(view),
            )
        )
    return rows


def _pnl_series(day: Day, replay: DayReplay) -> list[float]:
    """Paper P&L through the day: open trades at the candle close, closed trades at their net."""
    total = np.zeros(len(day))
    for trade in replay.trades:
        row = day.index(trade.symbol)
        direction = 1.0 if trade.side == BUY else -1.0
        span = slice(trade.entry_index, trade.exit_index)
        total[span] += direction * (day.close[row, span] - trade.entry_raw) * trade.quantity
        total[trade.exit_index:] += trade.net_pnl
    for position in replay.open_positions:
        row = day.index(position.symbol)
        direction = 1.0 if position.side == BUY else -1.0
        total[position.entry_index:-1] += direction * (day.close[row, position.entry_index:-1] - position.entry) * position.quantity
        total[-1] += position.net_if_closed
    return [0.0] + total.tolist()


def build_board(
    engine,
    day: Day,
    screen: Screen,
    views: list[StockView],
    replay: DayReplay,
    strategy: StrategyConfig,
    risk: RiskConfig,
    choice: str = "All",
    focus_sector: str | None = None,
    mover_limit: int = 5,
    ticker_items: list[TickerItem] | None = None,
) -> BoardData:
    last = len(day) - 1
    index_series = day.change_pct.mean(axis=0)
    finance = np.array([sector == "Financial Services" for sector in day.sectors])
    finance_series = day.change_pct[finance].mean(axis=0) if finance.any() else index_series
    returns = np.diff(np.log(1.0 + day.change_pct.mean(axis=0) / 100.0))
    sessions_per_year = 252 * (375 // day.bar_minutes)
    vol_series = [
        float(np.std(returns[: i + 1]) * np.sqrt(sessions_per_year) * 100.0) if i >= 4 else 0.0
        for i in range(len(returns))
    ]
    ticker = [
        _index_item("Tracked · eq-wt", index_series, "Equal-weight average of the tracked names, base 10,000 at the previous close."),
        _index_item("Finance · eq-wt", finance_series, "Equal-weight average of the tracked Financial Services names, base 10,000."),
        TickerItem(
            "Realized vol · sim",
            f"{vol_series[-1]:.1f}" if vol_series else "—",
            None,
            None,
            vol_series or [0.0],
            "Annualized volatility of the equal-weight average so far today. Not India VIX.",
        ),
    ]
    sector_series = {name: screen.sector_avg[i].tolist() for i, name in enumerate(day.sector_names)}
    ranked = sorted(day.sector_names, key=lambda name: sector_series[name][last], reverse=True)
    all_cards = [SectorCard(name, sector_series[name][last], sector_series[name]) for name in ranked]
    cards = all_cards
    if len(cards) > SECTOR_CARDS:
        cards = cards[: SECTOR_CARDS // 2] + cards[-SECTOR_CARDS // 2 :]
    each_side = strategy.sectors_each_side
    performance = all_cards if len(all_cards) <= each_side * 2 else all_cards[:each_side] + all_cards[-each_side:]
    focus = focus_sector if focus_sector in day.sector_names else default_focus(views, ranked)
    liquid = sorted(
        (view for view in views if view.sector == focus and view.volume > strategy.min_volume),
        key=lambda view: view.change_pct,
        reverse=True,
    )
    movers = [
        Mover(view.symbol, view.change_pct, buyer_share(engine.visible(view.symbol)), view.volume)
        for view in liquid[:mover_limit]
    ]
    changes = np.array([view.change_pct for view in views])
    pnl = _pnl_series(day, replay)
    capital = risk.notional_inr * risk.max_daily_trades
    open_risk = sum(abs(p.entry - p.stop) * p.quantity for p in replay.open_positions)
    return BoardData(
        clock=engine.clock,
        phase=session_phase(engine.clock.time(), strategy),
        ticker=ticker_items if ticker_items else ticker,
        advancing=int((changes > 0).sum()),
        declining=int((changes < 0).sum()),
        best=SectorCard(ranked[0], sector_series[ranked[0]][last], sector_series[ranked[0]]),
        worst=SectorCard(ranked[-1], sector_series[ranked[-1]][last], sector_series[ranked[-1]]),
        waiting=sum(1 for view in views if view.status == WAITING),
        in_trade=sum(1 for view in views if view.status == TRIGGERED),
        trades_used=len(replay.trades) + len(replay.open_positions),
        max_trades=risk.max_daily_trades,
        day_pnl=pnl[-1],
        capital=capital,
        pnl_series=pnl,
        risk_used=open_risk / capital if capital else 0.0,
        rows=opportunity_rows(views, choice),
        sector_cards=cards,
        performance=performance,
        focus_sector=focus,
        movers=movers,
        min_volume=strategy.min_volume,
        entry_offset_pct=strategy.entry_offset_pct,
        target_pct=strategy.target_pct,
        notional=risk.notional_inr,
    )


def _index_item(label: str, change_series: np.ndarray, note: str) -> TickerItem:
    levels = INDEX_BASE * (1.0 + change_series / 100.0)
    return TickerItem(
        label,
        f"{levels[-1]:,.2f}",
        float(levels[-1] - INDEX_BASE),
        float(change_series[-1]),
        [INDEX_BASE] + levels.tolist(),
        note,
    )


def build_chart(frame: pd.DataFrame, view: StockView, strategy: StrategyConfig, bar_minutes: int = 5) -> ChartData:
    candles = resample_bars(frame, bar_minutes) if bar_minutes > 1 else frame.reset_index(drop=True)
    typical = (candles["high"] + candles["low"] + candles["close"]) / 3.0
    vwap = (typical * candles["volume"]).cumsum() / candles["volume"].cumsum()
    return ChartData(
        symbol=view.symbol,
        name=view.name,
        price=view.price,
        change_pct=view.change_pct,
        bar_minutes=bar_minutes,
        candles=candles,
        ema_fast=candles["close"].ewm(span=EMA_FAST, adjust=False).mean(),
        ema_slow=candles["close"].ewm(span=EMA_SLOW, adjust=False).mean(),
        vwap=vwap,
        or_high=view.window_high if np.isfinite(view.window_high) else None,
        or_low=view.window_low if np.isfinite(view.window_low) else None,
        window_end=strategy.signal_start,
        view=view,
    )


# ---------------------------------------------------------------- rendering

BOARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
.orb, .st-key-orb_opps, .st-key-orb_chart, .st-key-orb_plan { font-family: Inter, 'Segoe UI', system-ui, sans-serif; color: #e5e9f0; }
.orb .up { color: #22c55e; } .orb .down { color: #ef4444; } .orb .muted { color: #8b95a5; }
.orb .mono { font-family: 'JetBrains Mono', Consolas, monospace; }
.st-key-orb_opps, .st-key-orb_chart, .st-key-orb_plan { background: #111821; border: 1px solid #1f2833; border-radius: 10px; padding: 14px 16px 12px; }
.st-key-orb_plan { margin-top: 4px; }
[data-testid="stHorizontalBlock"]:has(.st-key-orb_opps) { align-items: stretch; }
[data-testid="stColumn"]:has(.st-key-orb_opps) { display: flex; flex-direction: column; }
[data-testid="stColumn"]:has(.st-key-orb_opps) > div { flex: 1 1 auto; display: flex; flex-direction: column; }
.st-key-orb_opps { flex: 1 1 auto; box-sizing: border-box; }
.orb-ticker { display: flex; flex-wrap: wrap; row-gap: 10px; border-bottom: 1px solid #1f2833; padding: 2px 0 12px; margin-bottom: 14px; }
.orb-tick { flex: 1 1 250px; min-width: 0; display: flex; align-items: center; gap: 10px; padding: 0 14px; border-right: 1px solid #1f2833; }
.orb-tick:first-child { padding-left: 2px; }
.orb-tick .txt { min-width: 0; flex: 1 1 auto; }
.orb-tick .label { display: block; font-weight: 600; font-size: 0.78rem; color: #a3adbb; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.orb-tick .line { display: flex; flex-wrap: wrap; align-items: baseline; column-gap: 8px; }
.orb-tick .value { font-family: 'JetBrains Mono', Consolas, monospace; font-size: 1.0rem; font-weight: 600; white-space: nowrap; }
.orb-tick .chg { font-family: 'JetBrains Mono', Consolas, monospace; font-size: 0.76rem; white-space: nowrap; }
.orb-tick img { flex: 0 0 auto; width: 78px; height: 26px; }
.orb-updated { flex: 0 0 auto; padding-left: 14px; text-align: right; font-size: 0.72rem; color: #8b95a5; white-space: nowrap; }
.orb-updated b { display: block; color: #e5e9f0; font-family: 'JetBrains Mono', Consolas, monospace; font-size: 0.9rem; }
.orb-kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin-bottom: 14px; }
.orb-kpi { min-width: 0; overflow: hidden; background: #111821; border: 1px solid #1f2833; border-radius: 10px; padding: 14px 12px; display: grid; grid-template-columns: 36px minmax(0, 1fr); gap: 10px; align-items: start; min-height: 96px; }
.orb-kpi > div { min-width: 0; }
.orb-kpi .icon { width: 36px; height: 36px; border-radius: 8px; display: block; }
.orb-kpi .label { font-size: 0.64rem; letter-spacing: 0.06em; text-transform: uppercase; color: #a3adbb; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.orb-kpi .value { font-size: clamp(1.0rem, 1.45vw, 1.35rem); font-weight: 700; margin-top: 2px; line-height: 1.25; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.orb-kpi .value small { font-size: 0.74rem; font-weight: 600; margin-left: 6px; }
.orb-kpi .sub { font-size: 0.7rem; color: #8b95a5; margin-top: 4px; }
.orb-kpi .spark { display: block; width: 100%; max-width: 100%; height: 26px; margin-top: 4px; }
.orb-bar { display: flex; height: 6px; border-radius: 3px; overflow: hidden; background: #243040; margin-top: 8px; }
.orb-segs { display: flex; gap: 4px; margin-top: 8px; }
.orb-segs span { flex: 1; height: 6px; border-radius: 3px; background: #243040; }
.orb-segs span.on { background: #22c55e; }
.orb-title { font-size: 0.95rem; font-weight: 700; letter-spacing: 0.02em; text-transform: uppercase; }
.orb-subtitle { font-size: 0.74rem; color: #8b95a5; margin-top: 2px; }
.orb-table-wrap { overflow-x: auto; container-type: inline-size; }
.orb-table tbody tr[data-symbol] { cursor: pointer; }
.orb-table tbody tr[data-symbol]:hover td { background: #16202b; }
.orb-table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 0.76rem; }
.orb-table th { color: #8b95a5; font-weight: 500; text-align: left; padding: 8px 6px; border-bottom: 1px solid #1f2833; white-space: normal; min-width: 52px; line-height: 1.25; vertical-align: bottom; font-size: 0.7rem; }
.orb-table .c-fix0, .orb-table .c-fix1 { position: sticky; z-index: 2; background: #111821; box-sizing: border-box; }
.orb-table .c-fix0 { left: 0; width: 26px; min-width: 26px; max-width: 26px; padding-left: 4px; padding-right: 4px; }
.orb-table .c-fix1 { left: 26px; box-shadow: 1px 0 0 #1f2833; }
.orb-table td { color: #e5e9f0; }
.orb-table td.muted { color: #8b95a5; }
.orb-table tr.sel td.c-fix0, .orb-table tr.sel td.c-fix1 { background: #13233b; }
.orb-table tbody tr[data-symbol]:hover td.c-fix0, .orb-table tbody tr[data-symbol]:hover td.c-fix1 { background: #16202b; }
.orb-table td { padding: 10px 8px; border-bottom: 1px solid #18202a; white-space: nowrap; }
.orb-table th:not(:last-child), .orb-table td:not(:last-child) { border-right: 1px solid #1f2833; }
.orb-table td.num { font-family: 'JetBrains Mono', Consolas, monospace; }
.orb-table tr.sel td { background: #13233b; border-top: 1px solid #2f6fed; border-bottom: 1px solid #2f6fed; }
.orb-table tr.sel td:first-child { border-left: 1px solid #2f6fed; border-radius: 6px 0 0 6px; }
.orb-table tr.sel td:last-child { border-right: 1px solid #2f6fed; border-radius: 0 6px 6px 0; }
.orb-table .sym { font-weight: 700; }
.orb-pill { display: inline-block; min-width: 52px; text-align: center; padding: 4px 8px; border-radius: 6px; font-size: 0.68rem; font-weight: 700; border: 1px solid; }
.orb-pill.none { font-size: 0.6rem; min-width: 52px; padding: 4px 4px; }
.orb-pill.buy { color: #22c55e; border-color: #1f7a42; background: #0f2a1b; }
.orb-pill.short { color: #f87171; border-color: #8a2b2b; background: #2a1215; }
.orb-pill.wait { color: #cbd5e1; border-color: #3a4656; background: #1a2230; }
.orb-pill.target { color: #22c55e; border-color: #1f7a42; background: transparent; }
.orb-pill.stop { color: #f87171; border-color: #8a2b2b; background: transparent; }
.orb-pill.none { color: #8b95a5; border-color: #2a3442; background: transparent; }
.orb-vwap { display: inline-block; margin-left: 6px; font-family: Inter, 'Segoe UI', system-ui, sans-serif; font-size: 0.66rem; font-weight: 600; }
.orb-steps { font-family: 'JetBrains Mono', Consolas, monospace; font-size: 0.7rem; margin-left: 6px; }
.orb-chart-head .sym { font-size: 1.05rem; font-weight: 700; }
.orb-chart-head .px { font-family: 'JetBrains Mono', Consolas, monospace; font-size: 1.0rem; margin-left: 10px; }
.orb-chart-head .name { font-size: 0.74rem; color: #8b95a5; margin-top: 2px; }
.orb-legend { display: flex; gap: 16px; font-size: 0.72rem; margin: 6px 0 4px; color: #a3adbb; flex-wrap: wrap; }
.orb-legend i { display: inline-block; width: 14px; height: 2px; vertical-align: middle; margin-right: 5px; }
.orb-legend i.box { height: 9px; width: 12px; background: #3b4a63; }
.orb img.chart { display: block; width: 100%; height: auto; }
.orb-plan-head { display: flex; align-items: center; flex-wrap: wrap; gap: 10px 12px; }
.orb-plan-head .orb-title { white-space: nowrap; }
.orb-plan-head .strength { margin-left: auto; font-size: 0.74rem; color: #a3adbb; text-align: right; min-width: 150px; }
.orb-plan-head .strength b { color: #22c55e; font-size: 0.9rem; margin-left: 8px; }
.orb-plan-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(92px, 1fr)); gap: 10px; margin-top: 12px; border-top: 1px solid #1f2833; padding-top: 10px; }
.orb-plan-grid .k { font-size: 0.7rem; color: #8b95a5; }
.orb-plan-grid .v { font-family: 'JetBrains Mono', Consolas, monospace; font-size: 0.86rem; font-weight: 600; margin-top: 3px; }
.orb-sectors-head { display: flex; align-items: center; gap: 10px; margin: 16px 0 10px; }
.orb-tf { margin-left: auto; display: flex; gap: 4px; }
.orb-tf span { padding: 3px 10px; border-radius: 6px; font-size: 0.72rem; font-weight: 600; color: #8b95a5; border: 1px solid #1f2833; }
.orb-tf span.on { background: #2f6fed; color: white; border-color: #2f6fed; }
.orb-sectors { display: grid; grid-template-columns: repeat(10, minmax(0, 1fr)); gap: 10px; }
@media (max-width: 1200px) { .orb-sectors { grid-template-columns: repeat(5, minmax(0, 1fr)); } }
.orb-sector { background: #111821; border: 1px solid #1f2833; border-radius: 8px; padding: 10px 10px 6px; }
.orb-sector .n { font-size: 0.78rem; font-weight: 600; }
.orb-sector .p { font-family: 'JetBrains Mono', Consolas, monospace; font-size: 0.86rem; font-weight: 600; margin-top: 2px; }
.orb-sector img { width: 100%; height: 26px; margin-top: 4px; }
.orb-foot { font-size: 0.7rem; color: #8b95a5; margin-top: 12px; }
.orb-bottom { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 14px; }
@media (max-width: 1100px) { .orb-bottom { grid-template-columns: 1fr; } }
.orb-panel { background: #111821; border: 1px solid #1f2833; border-radius: 10px; padding: 18px 22px 16px; }
.orb-panel h4 { font-size: 1rem; font-weight: 700; margin: 0 0 18px; color: #e5e9f0; }
.orb-perf { display: grid; grid-template-columns: 120px 1fr 64px; align-items: center; gap: 16px; margin-bottom: 16px; font-size: 0.9rem; }
.orb-perf .pct { font-weight: 700; text-align: right; }
.orb-div { position: relative; height: 13px; background: #202a36; border-radius: 6px; }
.orb-div .mid { position: absolute; left: 50%; top: -3px; bottom: -3px; width: 2px; margin-left: -1px; background: #8b95a5; }
.orb-div .fill { position: absolute; top: 0; bottom: 0; border-radius: 4px; }
.orb-mover { display: grid; grid-template-columns: 140px 70px 1fr 96px; align-items: center; gap: 16px; margin-bottom: 14px; font-size: 0.9rem; }
.orb-mover .sym { font-weight: 700; }
.orb-mover .track { position: relative; height: 13px; background: #202a36; border-radius: 6px; }
.orb-mover .track .fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 6px; background: #34c78e; }
.orb-mover .track .tick { position: absolute; left: 50%; top: -3px; bottom: -3px; width: 2px; margin-left: -1px; background: #8b95a5; }
.orb-bias { text-align: center; padding: 5px 0; border-radius: 999px; font-size: 0.78rem; font-weight: 600; }
.orb-bias.bull { background: #12392a; color: #34c78e; }
.orb-bias.bear { background: #3a1519; color: #f0646f; }
.orb-panel .note { font-size: 0.8rem; color: #8b95a5; margin-top: 6px; }
/* Narrow table: shrink text so every column fits beside the chart. Must stay after the base table rules. */
@container (max-width: 860px) {
  .orb-table { font-size: 0.7rem; }
  .orb-table th { font-size: 0.64rem; padding: 7px 4px; white-space: normal; }
  .orb-table td { padding: 8px 4px; }
  .orb-pill { min-width: 44px; padding: 3px 5px; font-size: 0.62rem; }
  .orb-pill.none { font-size: 0.54rem; min-width: 44px; padding: 3px 2px; }
  .orb-steps { font-size: 0.64rem; margin-left: 4px; }
}
@container (max-width: 660px) {
  .orb-table { font-size: 0.62rem; }
  .orb-table th { font-size: 0.58rem; padding: 6px 2px; }
  .orb-table td { padding: 7px 2px; }
  .orb-table td.num { letter-spacing: -0.02em; }
  .orb-pill, .orb-pill.none { min-width: 38px; font-size: 0.52rem; padding: 3px 2px; }
  .orb-steps { display: block; margin: 2px 0 0; }
}
</style>
"""


def _e(text: object) -> str:
    return html.escape(str(text), quote=True)


def _img(svg: str, css: str = "", alt: str = "") -> str:
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"<img class='{css}' alt='{_e(alt)}' src='data:image/svg+xml;base64,{encoded}'/>"


def _inr(value: float, decimals: int = 2) -> str:
    sign = "−" if value < 0 else ""
    return f"{sign}₹{abs(value):,.{decimals}f}"


def _num(value: float) -> str:
    return "—" if value != value else f"{value:,.2f}"


def _signed_pct(value: float, decimals: int = 2) -> str:
    css = "up" if value >= 0 else "down"
    arrow = "▲" if value >= 0 else "▼"
    return f"<span class='{css}'>{arrow} {abs(value):.{decimals}f}%</span>"


def sparkline_svg(values: list[float], color: str, width: int = 120, height: int = 32, fill: bool = True) -> str:
    data = [float(v) for v in values if v == v] or [0.0]
    if len(data) == 1:
        data = data * 2
    low, high = min(data), max(data)
    span = (high - low) or 1.0
    step = width / (len(data) - 1)
    points = [(i * step, height - 2 - (v - low) / span * (height - 4)) for i, v in enumerate(data)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' width='{width}' height='{height}' preserveAspectRatio='none'>"]
    if fill:
        parts.append(f"<polygon points='0,{height} {line} {width},{height}' fill='{color}' fill-opacity='0.12'/>")
    parts.append(f"<polyline points='{line}' fill='none' stroke='{color}' stroke-width='1.6' stroke-linejoin='round'/>")
    parts.append("</svg>")
    return "".join(parts)


def _icon(kind: str, color: str) -> str:
    bg = {"#22c55e": "#0f2a1b", "#ef4444": "#2a1215", "#3b82f6": "#10213d", "#f59e0b": "#2b2110"}.get(color, "#1a2230")
    shapes = {
        "bars": f"<rect x='11' y='22' width='4' height='8' fill='{color}'/><rect x='18' y='16' width='4' height='14' fill='{color}'/><rect x='25' y='10' width='4' height='20' fill='{color}'/>",
        "trophy": f"<path d='M13 10h14v6a7 7 0 0 1-14 0z' fill='{color}'/><rect x='18' y='22' width='4' height='5' fill='{color}'/><rect x='14' y='27' width='12' height='3' rx='1' fill='{color}'/>",
        "down": f"<path d='M20 30 L11 19 H17 V10 H23 V19 H29 Z' fill='{color}'/>",
        "target": f"<circle cx='20' cy='20' r='10' fill='none' stroke='{color}' stroke-width='2.5'/><circle cx='20' cy='20' r='5' fill='none' stroke='{color}' stroke-width='2.5'/><circle cx='20' cy='20' r='1.8' fill='{color}'/>",
        "rupee": f"<text x='20' y='27' text-anchor='middle' font-size='19' font-family='Segoe UI, Arial' font-weight='700' fill='{color}'>₹</text>",
    }
    return (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 40 40' width='40' height='40'>"
        f"<rect width='40' height='40' rx='8' fill='{bg}'/>{shapes[kind]}</svg>"
    )


def donut_svg(share: float, color: str = BLUE) -> str:
    share = min(max(share, 0.0), 1.0)
    radius, circumference = 14.0, 2 * np.pi * 14.0
    return (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 40 40' width='40' height='40'>"
        "<rect width='40' height='40' rx='8' fill='#10213d'/>"
        f"<circle cx='20' cy='20' r='{radius}' fill='none' stroke='#243040' stroke-width='6'/>"
        f"<circle cx='20' cy='20' r='{radius}' fill='none' stroke='{color}' stroke-width='6' "
        f"stroke-dasharray='{circumference * share:.2f} {circumference:.2f}' transform='rotate(-90 20 20)'/></svg>"
    )


def render_top(board: BoardData) -> str:
    """Ticker strip and the six summary cards."""
    ticks = []
    for item in board.ticker:
        color = GREEN if (item.change_pct or 0) >= 0 else RED
        if item.change_pct is None:
            color = ORANGE
            change = "<span class='chg muted'>annualized %</span>"
        else:
            css = "up" if item.change_pct >= 0 else "down"
            change = f"<span class='chg {css}'>{item.change_points:+,.2f} &nbsp;{item.change_pct:+.2f}%</span>"
        ticks.append(
            f"<div class='orb-tick' title='{_e(item.note)}'><div class='txt'><span class='label'>{_e(item.label)}</span>"
            f"<div class='line'><span class='value'>{item.value}</span>{change}</div></div>"
            f"{_img(sparkline_svg(item.series, color, 78, 26, False), alt='trend')}</div>"
        )
    ticks.append(
        f"<div class='orb-updated'>Market clock<b>{board.clock:%H:%M}</b>"
        f"{_e(board.phase)} · Zerodha, {board.clock:%d %b}</div>"
    )
    total = max(board.advancing + board.declining, 1)
    breadth = "Positive" if board.advancing >= board.declining else "Negative"
    pnl_pct = board.day_pnl / board.capital * 100.0 if board.capital else 0.0
    pnl_color = GREEN if board.day_pnl >= 0 else RED
    segments = "".join(
        f"<span class='{'on' if i < board.trades_used else ''}'></span>" for i in range(board.max_trades)
    )
    kpis = [
        _kpi(
            _icon("bars", GREEN), "Market breadth",
            f"{board.advancing} / {board.advancing + board.declining}",
            f"<div class='orb-bar'><span style='width:{board.advancing / total * 100:.1f}%;background:{GREEN}'></span>"
            f"<span style='width:{board.declining / total * 100:.1f}%;background:{RED}'></span></div>"
            f"<div class='sub'><span class='{'up' if breadth == 'Positive' else 'down'}'>{breadth}</span> · "
            f"{board.advancing} up · {board.declining} down</div>",
        ),
        _kpi(
            _icon("trophy", GREEN), "Best sector",
            f"<span title='{_e(board.best.sector)}'>{_e(short_sector(board.best.sector).upper())}</span>",
            f"<div class='sub'>{_signed_pct(board.best.change_pct)}</div>{_img(sparkline_svg(board.best.series, GREEN), 'spark')}",
        ),
        _kpi(
            _icon("down", RED), "Weakest sector",
            f"<span title='{_e(board.worst.sector)}'>{_e(short_sector(board.worst.sector).upper())}</span>",
            f"<div class='sub'>{_signed_pct(board.worst.change_pct)}</div>{_img(sparkline_svg(board.worst.series, RED), 'spark')}",
        ),
        _kpi(
            _icon("target", BLUE), "Active setups", str(board.waiting + board.in_trade),
            f"<div class='sub'>{board.waiting} waiting · {board.in_trade} in trade</div>"
            f"<div class='orb-segs' title='Trades used today: {board.trades_used} of {board.max_trades}'>{segments}</div>",
        ),
        _kpi(
            _icon("rupee", pnl_color), "Daily P&amp;L (paper)",
            f"<span class='{'up' if board.day_pnl >= 0 else 'down'}'>{'+' if board.day_pnl >= 0 else ''}{_inr(board.day_pnl, 0)}</span>",
            f"<div class='sub'><span class='{'up' if pnl_pct >= 0 else 'down'}'>{pnl_pct:+.2f}%</span> · after default costs</div>"
            f"{_img(sparkline_svg(board.pnl_series, pnl_color), 'spark')}",
        ),
        _kpi(
            _img(donut_svg(board.risk_used), "icon"), "Risk used",
            f"{board.risk_used * 100:.1f}%",
            f"<div class='sub'>of paper capital {_inr(board.capital, 0)}</div>"
            f"<div class='orb-bar'><span style='width:{min(board.risk_used, 1) * 100:.1f}%;background:{GREEN}'></span></div>",
            icon_is_img=True,
        ),
    ]
    return BOARD_CSS + "<div class='orb'><div class='orb-ticker'>" + "".join(ticks) + "</div><div class='orb-kpis'>" + "".join(kpis) + "</div></div>"


def _kpi(icon: str, label: str, value: str, extra: str, icon_is_img: bool = False) -> str:
    icon_html = icon if icon_is_img else _img(icon, "icon")
    return (
        f"<div class='orb-kpi'>{icon_html}<div><div class='label'>{label}</div>"
        f"<div class='value'>{value}</div>{extra}</div></div>"
    )


def render_opportunities_head() -> str:
    return (
        BOARD_CSS
        + "<div class='orb'><div class='orb-title'>Top opportunities</div>"
        "<div class='orb-subtitle'>Candidates from the PDF screen: strongest and weakest sectors, top movers, "
        "opening range, VWAP. Zerodha data.</div></div>"
    )


def _pill(signal: str) -> str:
    css = {"BUY": "buy", "SHORT": "short", "WAIT": "wait", "TARGET": "target", "STOPPED": "stop", "CLOSED": "wait"}.get(signal, "none")
    return f"<span class='orb-pill {css}'>{_e(signal)}</span>"


def _vwap_tag(price: float, vwap: float) -> str:
    if np.isnan(price) or np.isnan(vwap):
        return ""
    if price >= vwap:
        return "<span class='orb-vwap up' title='Price is above VWAP'>▲ Above</span>"
    return "<span class='orb-vwap down' title='Price is below VWAP'>▼ Below</span>"


def render_table(rows: list[OpportunityRow], selected: str | None) -> str:
    if not rows:
        return BOARD_CSS + "<div class='orb'><p class='muted' style='font-size:0.82rem;padding:18px 4px'>No candidates match this filter right now.</p></div>"
    head = (
        "<tr><th class='c-fix0'>#</th><th class='c-fix1'>Stock</th><th>Sector</th>"
        "<th>Price</th><th>Previous day close</th><th>Day high</th><th>Day low</th>"
        "<th>Window high</th><th>Window low</th><th>% change from previous day</th>"
        "<th>Volume</th><th>VWAP</th><th>3-day % change</th>"
        "<th>Suggested buy</th><th>Suggested sell</th><th>SL</th>"
        "<th>R:R</th><th>Signal</th></tr>"
    )
    body = []
    for number, row in enumerate(rows, start=1):
        view = row.view
        rr = "—" if row.risk_reward is None else f"{row.risk_reward:.1f}"
        if view.side == BUY:
            buy_price, sell_price = view.order_price, view.target
            buy_tip, sell_tip = "Buy here (open − offset)", "Target"
        else:
            buy_price, sell_price = view.target, view.order_price
            buy_tip, sell_tip = "Target (buy-back)", "Short here (open + offset)"
        body.append(
            f"<tr class='{'sel' if view.symbol == selected else ''}' data-symbol='{_e(view.symbol)}' title='Show {_e(view.symbol)} on the chart'>"
            f"<td class='muted c-fix0'>{number}</td>"
            f"<td class='sym c-fix1'>{_e(view.symbol)}</td>"
            f"<td>{_e(short_sector(view.sector))}</td>"
            f"<td class='num'>{_num(view.price)}</td>"
            f"<td class='num'>{_num(view.prev_close)}</td>"
            f"<td class='num'>{_num(view.day_high)}</td>"
            f"<td class='num'>{_num(view.day_low)}</td>"
            f"<td class='num'>{_num(view.window_high)}</td>"
            f"<td class='num'>{_num(view.window_low)}</td>"
            f"<td class='num'>{_signed_pct(view.change_pct, 2)}</td>"
            f"<td class='num'>{view.volume / 100_000:,.1f}L</td>"
            f"<td class='num'>{_num(view.vwap)}{_vwap_tag(view.price, view.vwap)}</td>"
            f"<td class='num'>{_signed_pct(view.three_day_pct, 2)}</td>"
            f"<td class='num' title='{buy_tip}'>{_num(buy_price)}</td>"
            f"<td class='num' title='{sell_tip}'>{_num(sell_price)}</td>"
            f"<td class='num'>{_num(view.stop)}</td>"
            f"<td class='num'>{rr}</td>"
            f"<td>{_pill(row.signal)}<span class='orb-steps {'up' if row.steps_passed == row.steps_total else 'muted'}' "
            f"title='PDF steps passed on the latest closed candle'>{row.steps_passed}/{row.steps_total}</span></td>"
            "</tr>"
        )
    return BOARD_CSS + f"<div class='orb orb-table-wrap'><table class='orb-table'><thead>{head}</thead><tbody>{''.join(body)}</tbody></table></div>"


# Renders the table HTML from render_table and reports the symbol of a tapped row.
# Registered once here because app.py re-runs on every interaction.
OPPORTUNITY_TABLE = st.components.v2.component(
    "orb_opportunity_table",
    js="""
export default function(component) {
    const { data, setTriggerValue, parentElement } = component;
    let root = parentElement.querySelector('.orb-root');
    if (!root) {
        root = document.createElement('div');
        root.className = 'orb-root';
        parentElement.appendChild(root);
    }
    root.innerHTML = data;
    root.querySelectorAll('tr[data-symbol]').forEach((row) => {
        row.onclick = () => {
            root.querySelectorAll('tr.sel').forEach((other) => other.classList.remove('sel'));
            row.classList.add('sel');
            setTriggerValue('clicked', row.getAttribute('data-symbol'));
        };
    });
}
""",
)


def spread_labels(positions: list[float], gap: float, low: float = 0.0, high: float = 1e9) -> list[float]:
    """Push label y positions apart so none sit closer than ``gap``. Order is kept."""
    if not positions:
        return []
    order = sorted(range(len(positions)), key=lambda i: positions[i])
    placed = [positions[i] for i in order]
    for k in range(1, len(placed)):
        placed[k] = max(placed[k], placed[k - 1] + gap)
    overflow = placed[-1] - high
    if overflow > 0:
        placed = [value - overflow for value in placed]
    for k in range(len(placed) - 2, -1, -1):
        placed[k] = min(placed[k], placed[k + 1] - gap)
    placed = [max(value, low) for value in placed]
    result = [0.0] * len(positions)
    for rank, index in enumerate(order):
        result[index] = placed[rank]
    return result


def chart_svg(chart: ChartData, max_bars: int = 80) -> str:
    width, height = 480.0, 340.0
    left, plot_right = 4.0, 404.0
    top, bottom = 10.0, 236.0
    vol_top, vol_bottom = 248.0, 310.0
    offset = max(0, len(chart.candles) - max_bars)
    candles = chart.candles.iloc[offset:].reset_index(drop=True)
    ema_fast = chart.ema_fast.iloc[offset:].reset_index(drop=True)
    ema_slow = chart.ema_slow.iloc[offset:].reset_index(drop=True)
    vwap = chart.vwap.iloc[offset:].reset_index(drop=True)
    count = len(candles)
    if count == 0:
        return f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width:.0f} {height:.0f}'></svg>"
    view = chart.view
    buy = view.side == BUY
    levels = [
        ("Target", view.target, GREEN),
        ("Entry", view.order_price, BLUE),
        ("Stop", view.stop, RED),
    ]
    prices = [float(candles["low"].min()), float(candles["high"].max())] + [p for _, p, _ in levels if p == p]
    low, high = min(prices), max(prices)
    pad = (high - low) * 0.06 or high * 0.002
    low, high = low - pad, high + pad

    def y(price: float) -> float:
        return top + (high - price) / (high - low) * (bottom - top)

    step = (plot_right - left) / max(count, 24)

    def x(index: int) -> float:
        return left + (index + 0.5) * step

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width:.0f} {height:.0f}' width='{width:.0f}' "
        f"height='{height:.0f}' font-family='Segoe UI, Arial, sans-serif'>"
    ]
    for i in range(5):
        price = low + (high - low) * (i + 0.5) / 5
        yy = y(price)
        parts.append(f"<line x1='{left}' x2='{plot_right}' y1='{yy:.1f}' y2='{yy:.1f}' stroke='#1c2530' stroke-width='0.8'/>")
        parts.append(f"<text x='{width - 4:.0f}' y='{yy + 3.5:.1f}' font-size='10.5' fill='{MUTED}' text-anchor='end'>{price:,.2f}</text>")

    stamps = pd.to_datetime(candles["timestamp"]).dt.tz_convert(IST)
    times = list(stamps.dt.time)
    if chart.or_high is not None and chart.or_low is not None:
        inside = [i for i, t in enumerate(times) if t < chart.window_end]
        if inside:
            x0 = left + inside[0] * step
            x1 = left + (inside[-1] + 1) * step
            parts.append(
                f"<rect x='{x0:.1f}' y='{y(chart.or_high):.1f}' width='{x1 - x0:.1f}' "
                f"height='{max(1.0, y(chart.or_low) - y(chart.or_high)):.1f}' fill='#3b4a63' fill-opacity='0.45'/>"
            )

    for index, bar in enumerate(candles.itertuples(index=False)):
        color = GREEN if bar.close >= bar.open else RED
        cx = x(index)
        body_top = y(max(bar.open, bar.close))
        body_h = max(1.0, y(min(bar.open, bar.close)) - body_top)
        body_w = max(1.5, step * 0.6)
        parts.append(f"<line x1='{cx:.1f}' x2='{cx:.1f}' y1='{y(bar.high):.1f}' y2='{y(bar.low):.1f}' stroke='{color}' stroke-width='1'/>")
        parts.append(f"<rect x='{cx - body_w / 2:.1f}' y='{body_top:.1f}' width='{body_w:.1f}' height='{body_h:.1f}' fill='{color}'/>")

    for series, color, dash in ((ema_slow, ORANGE, ""), (ema_fast, BLUE, ""), (vwap, "#cbd5e1", "4 3")):
        points = " ".join(f"{x(i):.1f},{y(float(v)):.1f}" for i, v in enumerate(series))
        dash_attr = f" stroke-dasharray='{dash}'" if dash else ""
        parts.append(f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='1.4'{dash_attr}/>")

    valid = [(name, price, color) for name, price, color in levels if price == price]
    boxes = spread_labels([y(price) for _, price, _ in valid], gap=29.0, low=top + 13, high=bottom - 13)
    for (name, price, color), box_y in zip(valid, boxes):
        yy = y(price)
        parts.append(
            f"<line x1='{left}' x2='{plot_right}' y1='{yy:.1f}' y2='{yy:.1f}' stroke='{color}' stroke-width='1' stroke-dasharray='5 4'/>"
        )
        parts.append(f"<rect x='{plot_right + 2:.1f}' y='{box_y - 13:.1f}' width='{width - plot_right - 4:.1f}' height='26' rx='3' fill='{color}'/>")
        parts.append(
            f"<text x='{(plot_right + width) / 2:.1f}' y='{box_y - 2:.1f}' font-size='9.5' fill='white' text-anchor='middle'>{name}</text>"
            f"<text x='{(plot_right + width) / 2:.1f}' y='{box_y + 10:.1f}' font-size='10.5' font-weight='700' fill='white' text-anchor='middle'>{price:,.2f}</text>"
        )

    entry_time = view.trade.entry_time if view.trade else view.position.entry_time if view.position else None
    if entry_time is not None:
        index = int((stamps <= pd.Timestamp(entry_time)).sum()) - 1
        if 0 <= index < count:
            cx, yy = x(index), y(view.order_price)
            if buy:
                parts.append(f"<path d='M{cx:.1f} {yy + 4:.1f} l-5 9 h10 z' fill='{GREEN}'/>")
            else:
                parts.append(f"<path d='M{cx:.1f} {yy - 4:.1f} l-5 -9 h10 z' fill='{RED}'/>")

    volumes = candles["volume"].astype(float)
    top_volume = float(volumes.max()) or 1.0
    for index, bar in enumerate(candles.itertuples(index=False)):
        color = GREEN if bar.close >= bar.open else RED
        h = (float(bar.volume) / top_volume) * (vol_bottom - vol_top - 12)
        parts.append(
            f"<rect x='{x(index) - max(1.5, step * 0.6) / 2:.1f}' y='{vol_bottom - h:.1f}' width='{max(1.5, step * 0.6):.1f}' "
            f"height='{h:.1f}' fill='{color}' fill-opacity='0.55'/>"
        )
    total = float(volumes.iloc[-1])
    label = f"{total / 1_000_000:.2f}M" if total >= 1_000_000 else f"{total / 1_000:.0f}K"
    parts.append(f"<text x='{left + 2}' y='{vol_top + 9:.0f}' font-size='10.5' fill='{MUTED}'>Volume  {label}</text>")

    label_every = 30 if count * chart.bar_minutes > 150 else 15
    for index, moment in enumerate(times):
        if moment.minute % label_every == 0:
            parts.append(f"<text x='{x(index):.1f}' y='{height - 12:.0f}' font-size='10.5' fill='{MUTED}' text-anchor='middle'>{moment:%H:%M}</text>")
    parts.append("</svg>")
    return "".join(parts)


def render_chart_head(chart: ChartData) -> str:
    return (
        BOARD_CSS
        + "<div class='orb orb-chart-head'>"
        f"<span class='sym'>{_e(chart.symbol)}</span><span class='px'>{_inr(chart.price)}</span>"
        f"<span class='mono' style='margin-left:8px;font-size:0.86rem'>{_signed_pct(chart.change_pct, 1)}</span>"
        f"<div class='name'>{_e(chart.name)} · NSE</div></div>"
    )


def render_chart(chart: ChartData) -> str:
    legend = (
        "<div class='orb-legend'>"
        f"<span><i style='background:{BLUE}'></i>EMA {EMA_FAST}</span>"
        f"<span><i style='background:{ORANGE}'></i>EMA {EMA_SLOW}</span>"
        "<span><i style='background:repeating-linear-gradient(90deg,#cbd5e1 0 4px,transparent 4px 7px)'></i>VWAP</span>"
        "<span><i class='box'></i>Opening range</span>"
        "</div>"
    )
    return BOARD_CSS + "<div class='orb'>" + legend + _img(chart_svg(chart), "chart", "Price chart") + "</div>"


def render_plan(view: StockView, notional: float) -> str:
    signal = signal_label(view)
    if signal == "NO TRADE":
        signal = "NO TRADE"
    passed, total = len(view.passed), len(view.passed) + len(view.failed)
    share = passed / total if total else 0.0
    rr = risk_reward(view)
    quantity = position_size(view.order_price, notional) if view.order_price > 0 else 0
    return (
        BOARD_CSS
        + "<div class='orb'><div class='orb-plan-head'>"
        f"<span class='orb-title'>Trade plan — {_e(view.symbol)}</span>{_pill(signal)}"
        f"<div class='strength'>Steps passed<b>{passed}/{total}</b>"
        f"<div class='orb-bar'><span style='width:{share * 100:.0f}%;background:{GREEN if passed == total else ORANGE}'></span></div></div>"
        "</div><div class='orb-plan-grid'>"
        f"<div><div class='k'>{'Buy' if view.side == BUY else 'Sell'} price</div><div class='v'>{_inr(view.order_price)}</div></div>"
        f"<div><div class='k'>Stop loss</div><div class='v down'>{_inr(view.stop)}</div></div>"
        f"<div><div class='k'>Target</div><div class='v up'>{_inr(view.target)}</div></div>"
        f"<div><div class='k'>Risk / Reward</div><div class='v'>{'—' if rr is None else f'1 : {rr:.1f}'}</div></div>"
        f"<div><div class='k'>Position size</div><div class='v'>{quantity} sh</div></div>"
        "</div></div>"
    )


def render_sectors(board: BoardData) -> str:
    cards = []
    for card in board.sector_cards:
        color = GREEN if card.change_pct >= 0 else RED
        cards.append(
            f"<div class='orb-sector'><div class='n'>{_e(short_sector(card.sector))}</div>"
            f"<div class='p {'up' if card.change_pct >= 0 else 'down'}'>{card.change_pct:+.2f}%</div>"
            f"{_img(sparkline_svg(card.series, color), alt='trend')}</div>"
        )
    return (
        BOARD_CSS
        + "<div class='orb'><div class='orb-sectors-head'><span class='orb-title'>Sector momentum</span>"
        "<span class='muted' style='font-size:0.74rem'>(tracked names, illustrative sectors)</span>"
        "<div class='orb-tf'><span class='on'>1D</span>"
        "<span title='Needs several days of real history'>1W</span><span title='Needs several days of real history'>1M</span></div></div>"
        f"<div class='orb-sectors'>{''.join(cards)}</div></div>"
    )


def render_bottom(board: BoardData) -> str:
    """Sector performance bars and the movers card, with the page disclaimer."""
    largest = max((abs(card.change_pct) for card in board.performance), default=0.0) or 1.0
    perf = []
    for card in board.performance:
        half = abs(card.change_pct) / largest * 50.0
        side = "left:50%" if card.change_pct >= 0 else "right:50%"
        color = "#34c78e" if card.change_pct >= 0 else "#f0646f"
        perf.append(
            "<div class='orb-perf'>"
            f"<span>{_e(short_sector(card.sector))}</span>"
            f"<div class='orb-div'><div class='fill' style='{side};width:{max(half, 1.0):.1f}%;background:{color}'></div><div class='mid'></div></div>"
            f"<span class='pct' style='color:{color}'>{card.change_pct:+.1f}%</span>"
            "</div>"
        )
    lakh = f"{board.min_volume / 100_000:g}L"
    movers = []
    for mover in board.movers:
        up = mover.change_pct >= 0
        movers.append(
            "<div class='orb-mover'>"
            f"<span class='sym'>{_e(mover.symbol)}</span>"
            f"<span style='color:{'#34c78e' if up else '#f0646f'}'>{'▲' if up else '▼'} {abs(mover.change_pct):.1f}%</span>"
            f"<div class='track' title='Buyer share {mover.buyer_share * 100:.0f}%'><div class='fill' style='width:{mover.buyer_share * 100:.1f}%'></div><div class='tick'></div></div>"
            f"<span class='orb-bias {'bull' if up else 'bear'}'>{'Bullish' if up else 'Bearish'}</span>"
            "</div>"
        )
    if not movers:
        movers.append(f"<p class='note'>No name in this sector has traded more than {lakh} shares yet.</p>")
    return (
        BOARD_CSS
        + "<div class='orb'><div class='orb-bottom'>"
        "<div class='orb-panel'><h4>Sector performance</h4>" + "".join(perf) + "</div>"
        f"<div class='orb-panel'><h4>{_e(short_sector(board.focus_sector))} movers · volume &gt; {lakh}</h4>"
        + "".join(movers)
        + "<p class='note'>Bar = share of today's volume on up-minutes. Tick marks 50%. "
        "It is an estimate, not the order book.</p></div>"
        "</div>"
        "<p class='orb-foot'>LIVE DATA from Zerodha Kite Connect, read-only. Candles are one-minute bars built from the "
        "exchange feed; sector labels are an illustrative mapping. "
        "WAIT means every PDF step passed and the order "
        "fills only if a later candle trades at the suggested price. The stop-loss is the 15-minute low (buy) or high (short). "
        "Steps x/7 counts the PDF steps passed on the latest closed candle; it is not a probability. "
        "Signals are paper signals, not advice. No order is sent.</p></div>"
    )
