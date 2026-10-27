"""Tests for the PDF strategy: screen, steps 4-10, order fills, exits, and no look-ahead."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import pytest

from backtest import CostAssumptions
from board import resample_bars
from indicators import vwap
from fixtures import make_stocks, session_frames
from market_data import IST
from risk_manager import RiskConfig
from strategy import (
    BUY,
    NO_TRADE,
    SHORT,
    STOPPED,
    TARGET_HIT,
    TRIGGERED,
    WAITING,
    StrategyConfig,
    build_day,
    entry_levels,
    replay_day,
    screen_day,
    stock_views,
)
from universe import Stock

START = datetime(2099, 1, 1, 9, 15, tzinfo=IST)
STOCKS = [
    Stock("AAA", "Alpha", "X"),
    Stock("BBB", "Beta", "X"),
    Stock("CCC", "Gamma", "Y"),
    Stock("DDD", "Delta", "Y"),
]
CONFIG = StrategyConfig(sectors_each_side=1, stocks_per_sector=1, min_volume=1_000)
FREE = CostAssumptions(
    brokerage_inr_per_order=0, gst_pct=0, exchange_charge_pct=0, stt_sell_pct=0, stamp_duty_buy_pct=0, slippage_pct=0
)


def _frame(bars: list[tuple[float, float, float, float, float]], prev_close: float, count: int = 40) -> pd.DataFrame:
    rows = list(bars)
    last = rows[-1][3]
    while len(rows) < count:
        rows.append((last, last + 0.05, last - 0.05, last, 2_000))
    return pd.DataFrame(
        {
            "timestamp": [START + timedelta(minutes=i) for i in range(len(rows))],
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [r[4] for r in rows],
            "prev_close": prev_close,
            "close_3d_ago": prev_close,
        }
    )


def _long_bars(fill_bar=(99.3, 99.6, 99.2, 99.5, 10_000), after=(99.5, 101.1, 99.4, 101.0, 10_000)):
    """Gap up from 97: dip in 09:15-09:30, close above VWAP at 09:30, trade at 99.5 at 09:31."""
    bars = [(100.0, 100.2, 99.0, 99.2, 10_000)]
    bars += [(99.2, 99.4, 98.5, 99.0, 10_000)] * 14
    bars += [(99.0, 99.3, 98.9, 99.2, 10_000), fill_bar, after]
    return bars


def _short_bars():
    """Gap down from 103: pop in 09:15-09:30, close below VWAP at 09:30, sell at 100.5 at 09:31."""
    bars = [(100.0, 101.0, 99.9, 100.8, 10_000)]
    bars += [(100.8, 101.5, 100.6, 101.0, 10_000)] * 14
    bars += [(100.9, 100.95, 100.6, 100.7, 10_000), (100.6, 100.7, 100.4, 100.5, 10_000), (100.5, 100.6, 98.9, 99.0, 10_000)]
    return bars


def _flat(price: float) -> list[tuple[float, float, float, float, float]]:
    return [(price, price + 0.05, price - 0.05, price, 10_000)]


def _frames(**overrides: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames = {
        "AAA": _frame(_long_bars(), 97.0),
        "BBB": _frame(_flat(100.0), 99.0),
        "CCC": _frame(_short_bars(), 103.0),
        "DDD": _frame(_flat(100.0), 101.0),
    }
    frames.update(overrides)
    return frames


def _replay(frames, config=CONFIG, risk=None, session_over=True):
    day = build_day(frames, STOCKS, config)
    return day, replay_day(day, config, risk or RiskConfig(), FREE, session_over=session_over)


def test_long_and_short_setups_fill_at_the_offset_price_and_reach_target() -> None:
    _day, result = _replay(_frames())
    trades = {trade.symbol: trade for trade in result.trades}
    assert set(trades) == {"AAA", "CCC"}
    buy = trades["AAA"]
    assert buy.side == BUY
    assert buy.signal_index == 15 and buy.entry_index == 16
    assert buy.entry_raw == pytest.approx(99.5)
    assert buy.stop_loss == pytest.approx(98.5)
    assert buy.target == pytest.approx(100.99)
    assert buy.exit_reason == "target" and buy.exit_raw == pytest.approx(100.99)
    short = trades["CCC"]
    assert short.side == SHORT
    assert short.entry_raw == pytest.approx(100.5)
    assert short.stop_loss == pytest.approx(101.5)
    assert short.target == pytest.approx(98.99)
    assert short.exit_reason == "target"
    assert short.gross_pnl > 0


def test_no_fill_when_no_later_candle_reaches_the_buy_price() -> None:
    bars = _long_bars(fill_bar=(99.7, 99.9, 99.65, 99.8, 10_000), after=(99.8, 99.9, 99.7, 99.8, 10_000))
    _day, result = _replay(_frames(AAA=_frame(bars, 97.0)))
    assert "AAA" not in {trade.symbol for trade in result.trades}


def test_waiting_order_is_reported_when_the_session_is_still_open() -> None:
    frames = {symbol: frame.iloc[:16].reset_index(drop=True) for symbol, frame in _frames().items()}
    day, result = _replay(frames, session_over=False)
    assert {order.symbol for order in result.pending} == {"AAA", "CCC"}
    views = {view.symbol: view for view in stock_views(day, CONFIG, result)}
    assert views["AAA"].status == WAITING and views["AAA"].action == BUY
    assert views["AAA"].order_price == pytest.approx(99.5)
    assert views["BBB"].status is None and views["BBB"].action == NO_TRADE


def test_stop_wins_on_the_fill_candle() -> None:
    bars = _long_bars(fill_bar=(99.3, 101.5, 98.4, 99.0, 10_000))
    _day, result = _replay(_frames(AAA=_frame(bars, 97.0)))
    trade = next(t for t in result.trades if t.symbol == "AAA")
    assert trade.exit_index == trade.entry_index
    assert trade.exit_reason == "stop_loss" and trade.exit_raw == pytest.approx(98.5)


def test_target_is_not_taken_on_the_fill_candle() -> None:
    bars = _long_bars(fill_bar=(99.3, 101.5, 99.2, 99.6, 10_000), after=(99.6, 99.7, 99.55, 99.6, 10_000))
    _day, result = _replay(_frames(AAA=_frame(bars, 97.0)))
    trade = next(t for t in result.trades if t.symbol == "AAA")
    assert trade.exit_index > trade.entry_index
    assert trade.exit_reason == "session_close"


def test_open_position_stays_open_until_the_session_is_over() -> None:
    bars = _long_bars(after=(99.5, 99.7, 99.4, 99.6, 10_000))
    frames = {symbol: frame.iloc[:25].reset_index(drop=True) for symbol, frame in _frames(AAA=_frame(bars, 97.0)).items()}
    day, live = _replay(frames, session_over=False)
    assert [position.symbol for position in live.open_positions] == ["AAA"]
    assert live.open_positions[0].mark == pytest.approx(float(frames["AAA"]["close"].iloc[-1]))
    views = {view.symbol: view for view in stock_views(day, CONFIG, live)}
    assert views["AAA"].status == TRIGGERED
    _day, closed = _replay(frames, session_over=True)
    assert next(t for t in closed.trades if t.symbol == "AAA").exit_reason == "session_close"


def test_vwap_step_blocks_a_buy_below_vwap() -> None:
    bars = _long_bars()
    bars[15] = (99.0, 99.1, 98.6, 98.7, 10_000)
    bars[16] = (98.7, 98.9, 98.6, 98.8, 10_000)
    bars[17] = (98.8, 98.9, 98.7, 98.8, 10_000)
    day = build_day(_frames(AAA=_frame(bars, 97.0)), STOCKS, CONFIG)
    screen = screen_day(day, CONFIG)
    entries = entry_levels(day, CONFIG, screen)
    assert not entries.buy_vwap[0, 15]
    assert not entries.buy_ok[0, 15]


def test_window_low_above_the_buy_price_cannot_be_a_stop() -> None:
    bars = [(100.0, 100.2, 99.6, 99.8, 10_000)] + [(99.8, 100.0, 99.6, 99.8, 10_000)] * 14
    bars += [(99.8, 99.9, 99.6, 99.8, 10_000), (99.8, 99.9, 99.4, 99.5, 10_000)]
    frames = _frames(AAA=_frame(bars, 97.0))
    day, result = _replay(frames)
    assert day.window_low[0] == pytest.approx(99.6)
    assert "AAA" not in {trade.symbol for trade in result.trades}
    view = next(v for v in stock_views(day, CONFIG, result) if v.symbol == "AAA")
    assert "Steps 6-8" in {rule.rule_id for rule in view.failed}


def test_percentage_stop_replaces_the_window_stop() -> None:
    bars = [(100.0, 100.2, 99.6, 99.8, 10_000)] + [(99.8, 100.0, 99.6, 99.8, 10_000)] * 14
    bars += [(99.8, 99.9, 99.6, 99.8, 10_000), (99.8, 99.9, 99.4, 99.5, 10_000)]
    config = replace(CONFIG, stop_pct=1.0)
    day = build_day(_frames(AAA=_frame(bars, 97.0)), STOCKS, config)
    entries = entry_levels(day, config, screen_day(day, config))
    assert np.allclose(entries.buy_stop, np.round(entries.buy_price * 0.99, 2))
    assert np.allclose(entries.short_stop, np.round(entries.short_price * 1.01, 2))
    assert entries.buy_stop_ok.all() and entries.short_stop_ok.all()
    assert config.stop_text() == "1% below the buy price" and CONFIG.stop_text(False) == "the 15-minute high"
    with pytest.raises(ValueError):
        replace(CONFIG, stop_pct=0.0)


def test_window_length_follows_the_window_end() -> None:
    config = replace(CONFIG, signal_start=time(9, 45))
    assert config.window_minutes == 30
    day = build_day(_frames(), STOCKS, config)
    assert screen_day(day, config).in_window[day.times.index(time(9, 45))]
    assert not screen_day(day, config).in_window[day.times.index(time(9, 44))]


@pytest.mark.parametrize(
    "config",
    [
        replace(CONFIG, min_change_pct=5.0),
        replace(CONFIG, min_volume=10_000_000),
    ],
)
def test_change_and_volume_steps_block_the_trade(config) -> None:
    _day, result = _replay(_frames(), config=config)
    assert result.trades == []


def test_only_the_top_stocks_of_the_top_sector_are_bought() -> None:
    stronger = _frame(_flat(100.0), 95.0)
    _day, result = _replay(_frames(BBB=stronger))
    assert "AAA" not in {trade.symbol for trade in result.trades}
    _day, wider = _replay(_frames(BBB=stronger), config=replace(CONFIG, stocks_per_sector=2))
    assert "AAA" in {trade.symbol for trade in wider.trades}


def test_daily_limit_takes_fills_in_time_then_alphabetical_order() -> None:
    _day, result = _replay(_frames(), risk=RiskConfig(max_daily_trades=1))
    assert [trade.symbol for trade in result.trades] == ["AAA"]
    assert result.skipped_by_limit == 1


def test_signal_window_and_entry_cutoff() -> None:
    config = replace(CONFIG, no_new_entries_after=time(9, 40))
    day = build_day(_frames(), STOCKS, config)
    screen = screen_day(day, config)
    for index, moment in enumerate(day.times):
        assert screen.in_window[index] == (time(9, 30) <= moment < time(9, 40))


DAY = date(2026, 10, 23)
MANY = make_stocks(25)


def _fixture_day(seed: int, bars: int | None = None):
    frames = session_frames(MANY, DAY, seed)
    if bars is not None:
        frames = {symbol: frame.iloc[:bars].reset_index(drop=True) for symbol, frame in frames.items()}
    return frames


def test_full_days_trade_and_never_enter_after_the_cutoff() -> None:
    config = StrategyConfig()
    total = 0
    for seed in range(1, 9):
        day = build_day(_fixture_day(seed), MANY, config)
        result = replay_day(day, config, RiskConfig(), CostAssumptions(), session_over=True)
        total += len(result.trades)
        for trade in result.trades:
            assert time(9, 31) <= trade.entry_time.time() <= time(15, 0)
            row = day.index(trade.symbol)
            expected = day.session_open[row] * (0.995 if trade.side == BUY else 1.005)
            assert trade.entry_raw == pytest.approx(round(expected, 2))
        assert len(result.trades) <= RiskConfig().max_daily_trades
    assert total > 0


def test_prefix_replay_matches_the_full_day() -> None:
    config = StrategyConfig()
    frames = _fixture_day(3)
    full = replay_day(build_day(frames, MANY, config), config, RiskConfig(), CostAssumptions(), session_over=True)
    for cut in (30, 60, 120, 200):
        prefix = {symbol: frame.iloc[:cut].reset_index(drop=True) for symbol, frame in frames.items()}
        part = replay_day(build_day(prefix, MANY, config), config, RiskConfig(), CostAssumptions(), session_over=False)
        closed = [(t.symbol, t.entry_index, t.exit_index, t.exit_reason) for t in full.trades if t.exit_index < cut]
        assert [(t.symbol, t.entry_index, t.exit_index, t.exit_reason) for t in part.trades] == closed
        still_open = {t.symbol for t in full.trades if t.entry_index < cut <= t.exit_index}
        assert {p.symbol for p in part.open_positions} == still_open


def test_changing_later_candles_does_not_change_earlier_decisions() -> None:
    config = StrategyConfig()
    frames = _fixture_day(5)
    day = build_day(frames, MANY, config)
    original = entry_levels(day, config, screen_day(day, config))
    changed_frames = {}
    for symbol, frame in frames.items():
        changed = frame.copy()
        changed.loc[200:, ["open", "high", "low", "close"]] *= 1.2
        changed.loc[200:, "volume"] *= 3
        changed_frames[symbol] = changed
    changed_day = build_day(changed_frames, MANY, config)
    changed = entry_levels(changed_day, config, screen_day(changed_day, config))
    assert np.array_equal(original.buy_ok[:, :200], changed.buy_ok[:, :200])
    assert np.array_equal(original.short_ok[:, :200], changed.short_ok[:, :200])


def test_five_minute_day_matches_resampled_candles_and_vwap_matches_indicator() -> None:
    frames = _fixture_day(9)
    one = build_day(frames, MANY, StrategyConfig())
    five = build_day(frames, MANY, StrategyConfig(), bar_minutes=5)
    row = one.index("S03")
    assert np.allclose(one.vwap[row], vwap(frames["S03"]).to_numpy())
    grouped = resample_bars(frames["S03"])
    for name in ("open", "high", "low", "close", "volume"):
        assert np.allclose(getattr(five, name)[row], grouped[name].to_numpy())
    assert five.window_low[row] == pytest.approx(one.window_low[row])


def test_views_show_the_step_11_fields() -> None:
    day, result = _replay(_frames())
    view = next(v for v in stock_views(day, CONFIG, result) if v.symbol == "AAA")
    assert view.status == TARGET_HIT
    assert view.prev_close == pytest.approx(97.0)
    assert view.window_low == pytest.approx(98.5)
    assert view.window_high == pytest.approx(100.2)
    assert view.order_price == pytest.approx(99.5)
    assert view.stop == pytest.approx(98.5)
    assert view.target == pytest.approx(100.99)


def test_stop_status_is_reported() -> None:
    bars = _long_bars(after=(99.5, 99.6, 98.0, 98.2, 10_000))
    day, result = _replay(_frames(AAA=_frame(bars, 97.0)))
    view = next(v for v in stock_views(day, CONFIG, result) if v.symbol == "AAA")
    assert view.status == STOPPED


def test_invalid_settings_are_rejected() -> None:
    with pytest.raises(ValueError):
        StrategyConfig(signal_start=time(9, 10))
    with pytest.raises(ValueError):
        StrategyConfig(no_new_entries_after=time(9, 20))
    with pytest.raises(ValueError):
        StrategyConfig(target_pct=0)
    with pytest.raises(ValueError):
        StrategyConfig(entry_offset_pct=-0.1)
