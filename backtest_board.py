"""Backtest of the PDF strategy across all stocks on real Zerodha history.

Each past session is screened and replayed candle by candle like the live
board. History is downloaded from Kite and kept in memory only. Results are
paper results on past data and do not predict future profitability.
"""

from __future__ import annotations

import base64
import html
from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Callable, Sequence

from backtest import CostAssumptions, Trade, performance_stats
from board import BOARD_CSS, GREEN, MUTED, RED
from live_data import HistoryDay
from risk_manager import RiskConfig
from strategy import StrategyConfig, day_from_arrays, replay_day, screen_day

PERIOD_DAYS = (30, 60)
GRID_TARGETS = (1.0, 1.5, 2.0)
GRID_OFFSETS = (0.25, 0.5, 0.75)

ProgressFn = Callable[[int, int], None]


@dataclass(frozen=True)
class PeriodStats:
    days: int
    sessions: int
    trades: int
    win_rate: float | None
    gross_pnl: float
    net_pnl: float
    max_drawdown: float
    profit_factor: float | None


@dataclass
class MultiDayResult:
    bar_minutes: int
    headline_days: int
    headline: PeriodStats
    periods: list[PeriodStats]
    equity: list[tuple[date, float]]
    drawdown: list[float]
    grid: dict[tuple[float, float], float]
    grid_targets: tuple[float, ...]
    grid_offsets: tuple[float, ...]
    trades: list[tuple[date, Trade]]
    skipped_by_limit: int
    entry_offset_pct: float
    target_pct: float
    first_session: date | None
    last_session: date | None
    stop_text: str = "the 15-minute low or high"
    symbol: str | None = None
    qualified_sessions: int = 0


def window_start(days: int, last_session: date) -> date:
    """First calendar day of a ``days``-long window ending on ``last_session``."""
    return last_session - timedelta(days=days - 1)


def period_stats(days: int, sessions: int, trades: list[Trade]) -> PeriodStats:
    stats = performance_stats([t.gross_pnl for t in trades], [t.net_pnl for t in trades])
    return PeriodStats(
        days=days,
        sessions=sessions,
        trades=int(stats["trade_count"]),
        win_rate=stats["win_rate"],  # type: ignore[arg-type]
        gross_pnl=float(stats["gross_pnl"]),
        net_pnl=float(stats["net_pnl"]),
        max_drawdown=float(stats["max_drawdown"]),
        profit_factor=stats["profit_factor"],  # type: ignore[arg-type]
    )


def run_history_backtest(
    history: Sequence[HistoryDay],
    strategy: StrategyConfig,
    risk: RiskConfig,
    costs: CostAssumptions,
    *,
    bar_minutes: int = 5,
    headline_days: int = 60,
    period_days: tuple[int, ...] = PERIOD_DAYS,
    grid_targets: tuple[float, ...] = GRID_TARGETS,
    grid_offsets: tuple[float, ...] = GRID_OFFSETS,
    progress: ProgressFn | None = None,
    symbol: str | None = None,
) -> MultiDayResult:
    """Replay every downloaded session in the longest period across all stocks.

    Each session is screened and replayed with the sidebar settings, then once per
    target and entry-offset cell. Positions never carry over to the next session.
    With ``symbol`` every stock is still replayed, because the sector ranking and the
    daily trade limit depend on all of them, but only that stock's trades are counted.
    """
    if bar_minutes not in (1, 5):
        raise ValueError("Candle size must be 1 or 5 minutes.")
    sessions = sorted(history, key=lambda day: day.session_date)
    last = sessions[-1].session_date if sessions else None
    headline_start = window_start(headline_days, last) if last else None
    longest_start = window_start(max(max(period_days), headline_days), last) if last else None
    sessions = [day for day in sessions if last and day.session_date >= longest_start]
    grid_configs = {
        (target, offset): replace(strategy, target_pct=target, entry_offset_pct=offset)
        for target in grid_targets
        for offset in grid_offsets
    }
    trades: list[tuple[date, Trade]] = []
    daily_net: dict[date, float] = {}
    grid = {cell: 0.0 for cell in grid_configs}
    skipped = 0
    qualified = 0

    def counted(day_trades: list[Trade]) -> list[Trade]:
        return day_trades if symbol is None else [trade for trade in day_trades if trade.symbol == symbol]

    for step, session in enumerate(sessions, start=1):
        day = day_from_arrays(session.stamps, session.arrays, session.stocks, strategy, bar_minutes)
        screen = screen_day(day, strategy)
        replay = replay_day(day, strategy, risk, costs, session_over=True, screen=screen)
        kept = counted(replay.trades)
        trades.extend((session.session_date, trade) for trade in kept)
        daily_net[session.session_date] = sum(trade.net_pnl for trade in kept)
        if session.session_date >= headline_start:
            if symbol is None:
                skipped += replay.skipped_by_limit
            elif symbol in day.symbols:
                row = day.index(symbol)
                picked = (screen.bull_sector[row] & screen.bull_top[row]) | (screen.bear_sector[row] & screen.bear_top[row])
                qualified += bool((picked & screen.in_window).any())
            for cell, cell_config in grid_configs.items():
                grid[cell] += sum(
                    trade.net_pnl
                    for trade in counted(replay_day(day, cell_config, risk, costs, session_over=True, screen=screen).trades)
                )
        if progress is not None:
            progress(step, len(sessions))

    dates = [session.session_date for session in sessions]
    periods = []
    for days in period_days:
        start = window_start(days, last) if last else None
        periods.append(
            period_stats(
                days,
                sum(1 for d in dates if d >= start),
                [trade for d, trade in trades if d >= start],
            )
        )
    headline = period_stats(
        headline_days,
        sum(1 for d in dates if headline_start and d >= headline_start),
        [trade for d, trade in trades if headline_start and d >= headline_start],
    )
    equity: list[tuple[date, float]] = []
    drawdown: list[float] = []
    running = peak = 0.0
    for session_date in dates:
        if session_date < headline_start:
            continue
        running += daily_net[session_date]
        peak = max(peak, running)
        equity.append((session_date, running))
        drawdown.append(peak - running)
    return MultiDayResult(
        bar_minutes=bar_minutes,
        headline_days=headline_days,
        headline=headline,
        periods=periods,
        equity=equity,
        drawdown=drawdown,
        grid=grid,
        grid_targets=tuple(grid_targets),
        grid_offsets=tuple(grid_offsets),
        trades=[(d, t) for d, t in trades if headline_start and d >= headline_start],
        skipped_by_limit=skipped,
        entry_offset_pct=strategy.entry_offset_pct,
        target_pct=strategy.target_pct,
        first_session=dates[0] if dates else None,
        last_session=last,
        stop_text=_stop_words(strategy),
        symbol=symbol,
        qualified_sessions=qualified,
    )


def _stop_words(strategy: StrategyConfig) -> str:
    if strategy.stop_pct is None:
        return f"the {strategy.window_minutes}-minute low or high"
    return f"{strategy.stop_pct:g}% from the buy or sell price"


# ---------------------------------------------------------------- rendering

BT_CSS = """
<style>
.orb-bt-head { display: flex; align-items: baseline; gap: 16px; margin: 2px 0 16px; }
.orb-bt-head .orb-title { font-size: 1.3rem; font-weight: 700; text-transform: none; letter-spacing: -0.01em; }
.orb-card { background: #111821; border: 1px solid #1f2833; border-radius: 10px; padding: 16px 14px 14px; }
.orb-label { font-size: 0.7rem; letter-spacing: 0.07em; text-transform: uppercase; color: #a3adbb; font-weight: 600; margin-bottom: 16px; }
.orb-note { margin-left: auto; font-size: 0.74rem; color: #8b95a5; }
.orb img.orb-chart { display: block; width: 100%; height: auto; }
.orb-kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }
.orb-kpi .value { font-family: 'IBM Plex Mono', Consolas, monospace; font-size: 1.55rem; margin-top: -4px; }
.orb-kpi .sub { font-size: 0.72rem; color: #8b939e; margin-top: 6px; }
.orb-grid2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 14px; }
@media (max-width: 1100px) { .orb-kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); } .orb-grid2 { grid-template-columns: 1fr; } }
.orb-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
.orb-table th { color: #a3abb5; font-weight: 600; text-align: right; padding: 6px 6px 9px; border-bottom: 1px solid #262b33; }
.orb-table th:first-child, .orb-table td:first-child { text-align: left; font-family: inherit; }
.orb-table td { text-align: right; padding: 8px 6px; border-bottom: 1px solid #22262d; font-family: 'IBM Plex Mono', Consolas, monospace; }
.orb-heat { display: grid; gap: 8px; align-items: center; margin-top: -4px; }
.orb-heat .col { font-size: 0.7rem; color: #a3abb5; padding-left: 2px; }
.orb-heat .row { font-size: 0.72rem; color: #a3abb5; }
.orb-heat .cell { border-radius: 5px; padding: 12px 6px; text-align: center; font-family: 'IBM Plex Mono', Consolas, monospace; font-size: 0.78rem; color: #15181d; font-weight: 600; }
.orb-heat .cell.current { outline: 2px solid #e6e8eb; outline-offset: 1px; }
</style>
"""


def _e(text: object) -> str:
    return html.escape(str(text), quote=True)


def _inr(value: float, decimals: int = 0) -> str:
    sign = "−" if value < 0 else ""
    return f"{sign}₹{abs(value):,.{decimals}f}"


def _signed(value: float) -> str:
    css = "up" if value > 0 else "down" if value < 0 else ""
    arrow = "▲ " if value > 0 else "▼ " if value < 0 else ""
    return f"<span class='{css}'>{arrow}{_inr(abs(value))}</span>"


def equity_svg(equity: list[tuple[date, float]], drawdown: list[float]) -> str:
    width, height = 600.0, 190.0
    left, right = 6.0, 594.0
    top, mid = 10.0, 118.0
    dd_top, dd_bottom = 130.0, 184.0
    parts = [
        f"<svg viewBox='0 0 {width:.0f} {height:.0f}' width='{width:.0f}' height='{height:.0f}' "
        "xmlns='http://www.w3.org/2000/svg' font-family='Segoe UI, Arial, sans-serif'>"
    ]
    values = [0.0] + [value for _, value in equity]
    dd = [0.0] + list(drawdown)
    count = len(values)
    low, high = min(values), max(values)
    if high - low < 1e-9:
        low, high = low - 1.0, high + 1.0
    pad = (high - low) * 0.08
    low, high = low - pad, high + pad

    def x(index: int) -> float:
        return left + index * (right - left) / max(count - 1, 1)

    def y(value: float) -> float:
        return top + (high - value) / (high - low) * (mid - top)

    zero = y(0.0)
    parts.append(f"<line x1='{left}' x2='{right}' y1='{zero:.1f}' y2='{zero:.1f}' stroke='#3a404a' stroke-width='0.8'/>")
    line_color = GREEN if values[-1] >= 0 else RED
    points = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values))
    parts.append(f"<polyline points='{points}' fill='none' stroke='{line_color}' stroke-width='1.8' stroke-linejoin='round'/>")

    worst = max(dd) or 1.0
    area = [f"{x(i):.1f},{dd_bottom - (v / worst) * (dd_bottom - dd_top):.1f}" for i, v in enumerate(dd)]
    polygon = f"{left:.1f},{dd_bottom:.1f} " + " ".join(area) + f" {x(count - 1):.1f},{dd_bottom:.1f}"
    parts.append(f"<polygon points='{polygon}' fill='{RED}' fill-opacity='0.28' stroke='{RED}' stroke-opacity='0.5' stroke-width='0.6'/>")
    parts.append(f"<text x='{left + 2:.1f}' y='{dd_top + 10:.1f}' font-size='8' fill='{MUTED}'>Drawdown</text>")
    parts.append("</svg>")
    return "".join(parts)


def _svg_img(svg: str, alt: str) -> str:
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"<img class='orb-chart' alt='{_e(alt)}' src='data:image/svg+xml;base64,{encoded}'/>"


def _heat_color(value: float, scale: float) -> str:
    strength = min(1.0, abs(value) / scale) if scale > 0 else 0.0
    if value > 0:
        light, strong = (0xC9, 0xEF, 0xDF), (0x4C, 0xC9, 0x9F)
    elif value < 0:
        light, strong = (0xFF, 0xD9, 0xD9), (0xFF, 0x96, 0x96)
    else:
        return "#2a2f37"
    mixed = tuple(round(a + (b - a) * strength) for a, b in zip(light, strong))
    return "#%02x%02x%02x" % mixed


def _kpi(label: str, value: str, sub: str = "") -> str:
    sub_html = f"<div class='sub'>{sub}</div>" if sub else ""
    return f"<div class='orb-card orb-kpi'><div class='orb-label'>{label}</div><div class='value'>{value}</div>{sub_html}</div>"


def render_backtest(result: MultiDayResult) -> str:
    head = result.headline
    net_css = "up" if head.net_pnl > 0 else "down" if head.net_pnl < 0 else ""
    win = "—" if head.win_rate is None else f"{head.win_rate * 100:.1f}%"
    pf = "—" if head.profit_factor is None else f"{head.profit_factor:.2f}"
    pf_sub = "no losing trades" if head.trades and head.profit_factor is None else ""
    dd = f"<span class='down'>▼ {_inr(head.max_drawdown)}</span>" if head.max_drawdown > 0 else _inr(0)
    scope = f"Last {result.headline_days} days · {head.sessions} sessions · {head.trades} trades"

    kpis = (
        "<div class='orb-kpis'>"
        + _kpi("Net P&amp;L", f"<span class='{net_css}'>{_inr(head.net_pnl)}</span>", scope)
        + _kpi("Win rate", win, f"{head.trades} trades")
        + _kpi("Max drawdown", dd, "largest drop from a peak")
        + _kpi("Profit factor", pf, pf_sub)
        + "</div>"
    )

    equity = (
        "<div class='orb-card'><div class='orb-label'>Equity curve (₹, net of costs)</div>"
        + _svg_img(equity_svg(result.equity, result.drawdown), "Equity curve")
        + "</div>"
    )

    rows = "".join(
        "<tr>"
        f"<td>{p.days} days</td><td>{p.trades}</td>"
        f"<td>{'—' if p.win_rate is None else f'{p.win_rate * 100:.1f}'}</td>"
        f"<td>{_signed(p.net_pnl)}</td><td>{_inr(p.max_drawdown)}</td>"
        "</tr>"
        for p in result.periods
    )
    last_label = f"{result.last_session:%d %b %Y}" if result.last_session else "the latest session"
    periods = (
        "<div class='orb-card'><div class='orb-label'>Period comparison</div>"
        "<table class='orb-table'><thead><tr><th>Period</th><th>Trades</th><th>Win %</th>"
        "<th>Net ₹</th><th>Max DD ₹</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        + "<p class='muted' style='font-size:0.7rem;margin:12px 0 0'>Periods count the trading sessions Zerodha returned "
        + f"in each window, ending {last_label} "
        + f"({', '.join(f'{p.days} days = {p.sessions} sessions' for p in result.periods)}).</p></div>"
    )

    scale = max((abs(v) for v in result.grid.values()), default=0.0)
    columns = len(result.grid_offsets)
    cells = ["<span></span>"] + [f"<span class='col'>Entry {o:g}% from open</span>" for o in result.grid_offsets]
    for target in result.grid_targets:
        cells.append(f"<span class='row'>Tgt {target:g}%</span>")
        for offset in result.grid_offsets:
            value = result.grid[(target, offset)]
            arrow = "▲ " if value > 0 else "▼ " if value < 0 else ""
            current = " current" if (target, offset) == (result.target_pct, result.entry_offset_pct) else ""
            text_color = "" if value != 0 else "color:#8b939e;"
            cells.append(
                f"<div class='cell{current}' style='background:{_heat_color(value, scale)};{text_color}'>"
                f"{arrow}{_inr(abs(value))}</div>"
            )
    grid = (
        "<div class='orb-card' style='margin-top:14px'><div class='orb-label'>"
        f"Net P&amp;L by target (rows) × entry offset (columns) · last {result.headline_days} days</div>"
        f"<div class='orb-heat' style='grid-template-columns:64px repeat({columns}, minmax(0, 1fr))'>"
        + "".join(cells)
        + "</div>"
        "<p class='muted' style='font-size:0.7rem;margin:12px 0 0'>Each cell replays the same sessions with a different "
        f"buy/sell distance from the open and a different target. The stop stays at {_e(result.stop_text)}. "
        "An outlined cell matches the sidebar settings. Picking the best cell on this same data is overfitting.</p></div>"
    )

    header = (
        "<div class='orb-bt-head'>"
        f"<span class='orb-title'>Backtest · {result.bar_minutes}-minute data · after costs</span>"
        f"<span class='muted' style='font-size:0.8rem'>{_e(result.symbol) if result.symbol else 'All stocks'} · PDF strategy</span>"
        "<span class='orb-note'>Zerodha history · paper results</span>"
        "</div>"
    )
    if result.symbol:
        header += (
            "<p class='muted' style='font-size:0.78rem;margin:-4px 0 12px'>"
            f"Full PDF rules: {_e(result.symbol)} can only trade on days it is in the top sectors and a top mover of its "
            f"sector. That happened on {result.qualified_sessions} of {head.sessions} sessions in the last "
            f"{result.headline_days} days. Other stocks are still replayed, so the daily trade limit counts them too.</p>"
        )
    return (
        BOARD_CSS
        + BT_CSS
        + "<div class='orb'>"
        + header
        + kpis
        + f"<div class='orb-grid2'>{equity}{periods}</div>"
        + grid
        + "<p class='orb-foot'>Past sessions from Zerodha Kite Connect, replayed candle by candle with the PDF rules. "
        "Costs are paper assumptions, not a Zerodha tariff. Past results do not predict future profitability. "
        "No order is sent.</p>"
        + "</div>"
    )
