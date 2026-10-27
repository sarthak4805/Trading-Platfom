from datetime import date, datetime, time, timedelta

import pytest

import backtest_board
import live_data
from backtest import CostAssumptions
from fixtures import history_days, make_stocks
from live_data import IST, LiveMarket
from risk_manager import RiskConfig
from strategy import StrategyConfig
from universe import Stock

STOCKS = make_stocks(25)
LAST = date(2026, 10, 23)
WEEKDAYS = [LAST - timedelta(days=i) for i in range(59, -1, -1) if (LAST - timedelta(days=i)).weekday() < 5]


def test_history_backtest_totals_are_consistent():
    config = StrategyConfig()
    result = backtest_board.run_history_backtest(
        history_days(STOCKS, WEEKDAYS), config, RiskConfig(), CostAssumptions(), bar_minutes=5, headline_days=30
    )
    head = result.headline
    assert result.last_session == LAST
    assert head.sessions == sum(1 for d in WEEKDAYS if d >= LAST - timedelta(days=29))
    assert result.periods[0].days == 30 and result.periods[0].trades == head.trades
    assert result.periods[1].sessions == len(WEEKDAYS)
    assert result.grid[(config.target_pct, config.entry_offset_pct)] == pytest.approx(head.net_pnl)
    if result.equity:
        assert result.equity[-1][1] == pytest.approx(head.net_pnl)
    assert len(result.trades) == head.trades
    assert all(value >= 0 for value in result.drawdown)
    for _session, trade in result.trades:
        assert trade.entry_time.time() <= time(15, 0)
    page = backtest_board.render_backtest(result)
    assert "data:image/svg+xml;base64," in page and "<svg" not in page
    assert "Zerodha history" in page and "ynthetic" not in page


def test_daily_limit_and_one_trade_per_stock_hold_every_session():
    result = backtest_board.run_history_backtest(
        history_days(STOCKS, WEEKDAYS[-20:], seed=7), StrategyConfig(), RiskConfig(max_daily_trades=2),
        CostAssumptions(), bar_minutes=1, headline_days=60,
    )
    per_day: dict[date, list[str]] = {}
    for session, trade in result.trades:
        per_day.setdefault(session, []).append(trade.symbol)
    assert result.trades
    for symbols in per_day.values():
        assert len(symbols) <= 2
        assert len(symbols) == len(set(symbols))


def test_single_stock_backtest_keeps_only_that_stocks_trades_from_the_full_replay():
    history = history_days(STOCKS, WEEKDAYS[-20:], seed=7)
    everyone = backtest_board.run_history_backtest(history, StrategyConfig(), RiskConfig(), CostAssumptions(), headline_days=60)
    symbol = everyone.trades[0][1].symbol
    single = backtest_board.run_history_backtest(
        history, StrategyConfig(), RiskConfig(), CostAssumptions(), headline_days=60, symbol=symbol
    )
    expected = [(d, t) for d, t in everyone.trades if t.symbol == symbol]
    assert [(d, t.entry_time) for d, t in single.trades] == [(d, t.entry_time) for d, t in expected]
    assert single.headline.net_pnl == pytest.approx(sum(t.net_pnl for _d, t in expected))
    assert single.grid[(1.5, 0.5)] == pytest.approx(single.headline.net_pnl)
    assert len({d for d, _t in expected}) <= single.qualified_sessions <= single.headline.sessions
    page = backtest_board.render_backtest(single)
    assert f"{symbol} · PDF strategy" in page and "Full PDF rules" in page


class HistoryKite:
    """Fake Kite with several weekdays of minute and daily candles."""

    days = [date(2026, 10, d) for d in (19, 20, 21, 22, 23)]

    def instruments(self, exchange):
        return [
            {"tradingsymbol": "AAA", "instrument_token": 1, "segment": "NSE", "instrument_type": "EQ"},
            {"tradingsymbol": "BBB", "instrument_token": 2, "segment": "NSE", "instrument_type": "EQ"},
        ]

    def historical_data(self, token, start, end, interval):
        if interval == "day":
            history = [date(2026, 10, d) for d in (13, 14, 15, 16)] + self.days
            return [
                {"date": datetime.combine(d, time(0), tzinfo=IST), "open": 100, "high": 100, "low": 100,
                 "close": 100 + i + token, "volume": 1}
                for i, d in enumerate(history) if start.date() <= d <= end.date()
            ]
        rows = []
        for day in self.days:
            if not (start.date() <= day <= end.date()):
                continue
            moment = datetime.combine(day, time(9, 15), tzinfo=IST)
            for index in range(375):
                if token == 2 and index == 7:
                    moment += timedelta(minutes=1)
                    continue
                price = 100 + token + index * 0.01
                rows.append({"date": moment, "open": price, "high": price + 0.05, "low": price - 0.05,
                             "close": price, "volume": 500})
                moment += timedelta(minutes=1)
        return rows


def test_load_history_builds_aligned_sessions_with_previous_closes():
    stocks = [Stock("AAA", "Alpha", "X"), Stock("BBB", "Beta", "X")]
    now = datetime(2026, 10, 25, 18, 0, tzinfo=IST)
    market = LiveMarket(HistoryKite(), stocks, now=lambda: now, pause=lambda _s: None)
    market.load()
    assert market.state == "ready", market.message
    seen = []
    days = live_data.load_history(market, 10, progress=lambda done, total, text: seen.append((done, total)))
    assert [d.session_date for d in days] == HistoryKite.days
    assert seen[-1] == (2, 2)
    first = days[0]
    assert len(first.stamps) == 375
    assert [s.symbol for s in first.stocks] == ["AAA", "BBB"]
    assert len(first.arrays["BBB"]["close"]) == 375
    assert first.arrays["BBB"]["volume"][7] == 0
    # Daily closes are 100 + position + token; the session before 19 Oct is 16 Oct (position 3).
    assert first.arrays["AAA"]["prev_close"] == pytest.approx(104.0)
    assert first.arrays["AAA"]["close_3d_ago"] == pytest.approx(102.0)
    result = backtest_board.run_history_backtest(days, StrategyConfig(), RiskConfig(), CostAssumptions(), headline_days=30)
    assert result.headline.sessions == 5


def test_background_prefetch_fills_the_history_store():
    import time as clock

    stocks = [Stock("AAA", "Alpha", "X"), Stock("BBB", "Beta", "X")]
    now = datetime(2026, 10, 25, 18, 0, tzinfo=IST)
    market = LiveMarket(HistoryKite(), stocks, now=lambda: now, pause=lambda _s: None)
    market.load()
    market.prefetch_history(10)
    market.prefetch_history(10)  # a second call while running does nothing
    deadline = clock.time() + 20
    while market.history_state == "running" and clock.time() < deadline:
        clock.sleep(0.05)
    assert market.history_state == "ready", market.history_message
    assert [d.session_date for d in market.history[10]] == HistoryKite.days
    assert market.history_done == market.history_total == 2


def test_history_window_is_limited_to_sixty_days():
    market = LiveMarket(HistoryKite(), [], now=lambda: datetime(2026, 10, 25, tzinfo=IST), pause=lambda _s: None)
    with pytest.raises(ValueError):
        live_data.load_history(market, 61)
