from datetime import date, datetime, time, timedelta

import pandas as pd
import pytest

import board
from backtest import CostAssumptions
from fixtures import FakeEngine, make_stocks, session_frames
from market_data import IST
from risk_manager import RiskConfig
from strategy import BUY, TRIGGERED, WAITING, StrategyConfig, build_day, replay_day, screen_day, stock_views

DAY = date(2026, 10, 23)
STOCKS = make_stocks(25)


def _bars(rows):
    start = datetime(2026, 10, 23, 9, 15, tzinfo=IST)
    return pd.DataFrame(
        {
            "timestamp": [start + timedelta(minutes=i) for i in range(len(rows))],
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [r[4] for r in rows],
            "prev_close": 9.0,
        }
    )


def _state(clock: time, seed: int = 4):
    engine = FakeEngine(session_frames(STOCKS, DAY, seed), STOCKS, datetime.combine(DAY, clock, tzinfo=IST))
    config, risk = StrategyConfig(), RiskConfig()
    day = build_day({s.symbol: engine.visible(s.symbol) for s in STOCKS}, STOCKS, config)
    screen = screen_day(day, config)
    replay = replay_day(day, config, risk, CostAssumptions(), session_over=clock >= config.session_end, screen=screen)
    views = stock_views(day, config, replay, screen)
    return engine, day, screen, views, replay, config, risk


def test_resample_groups_five_minutes_and_keeps_constants():
    rows = [(10 + i, 11 + i, 9 + i, 10.5 + i, 100) for i in range(7)]
    out = board.resample_bars(_bars(rows))
    assert len(out) == 2
    first = out.iloc[0]
    assert (first["open"], first["high"], first["low"], first["close"], first["volume"]) == (10, 15, 9, 14.5, 500)
    assert out.iloc[1]["volume"] == 200
    assert set(out["prev_close"]) == {9.0}


def test_session_phase_boundaries():
    config = StrategyConfig()
    assert board.session_phase(time(9, 20), config) == "Observation window"
    assert board.session_phase(time(9, 30), config) == "Observation window"
    assert board.session_phase(time(9, 31), config) == "Trading window"
    assert board.session_phase(time(15, 1), config) == "No new entries"
    assert board.session_phase(time(15, 30), config) == "Session closed"


def test_spread_labels_keeps_minimum_gap_and_order():
    placed = board.spread_labels([50.0, 51.0, 52.0, 100.0], gap=8.0, low=0.0, high=140.0)
    ordered = sorted(placed[:3])
    assert ordered[1] - ordered[0] >= 8.0 - 1e-9
    assert ordered[2] - ordered[1] >= 8.0 - 1e-9
    assert placed[0] < placed[1] < placed[2] < placed[3]


@pytest.mark.parametrize("clock", [time(9, 25), time(10, 50), time(15, 30)])
def test_board_builds_and_renders_at_any_clock(clock):
    engine, day, screen, views, replay, config, risk = _state(clock)
    data = board.build_board(engine, day, screen, views, replay, config, risk)
    assert data.advancing + data.declining <= len(STOCKS)
    assert data.best.change_pct >= data.worst.change_pct
    assert len(data.sector_cards) == min(board.SECTOR_CARDS, len(day.sector_names))
    assert data.waiting == sum(1 for v in views if v.status == WAITING)
    assert data.in_trade == sum(1 for v in views if v.status == TRIGGERED)
    assert data.trades_used <= risk.max_daily_trades
    assert data.pnl_series[-1] == pytest.approx(data.day_pnl)
    expected = sum(t.net_pnl for t in replay.trades) + sum(p.net_if_closed for p in replay.open_positions)
    assert data.day_pnl == pytest.approx(expected)
    assert [c.change_pct for c in data.performance] == sorted((c.change_pct for c in data.performance), reverse=True)
    assert all(m.session_volume > config.min_volume for m in data.movers)
    assert all(0.0 <= m.buyer_share <= 1.0 for m in data.movers)
    pages = [board.render_top(data), board.render_table(data.rows, None), board.render_sectors(data), board.render_bottom(data)]
    if data.rows:
        view = data.rows[0].view
        chart = board.build_chart(engine.visible(view.symbol), view, config)
        pages += [board.render_chart_head(chart), board.render_chart(chart), board.render_plan(view, risk.notional_inr)]
    for page in pages:
        assert "<svg" not in page
    assert "Market clock" in pages[0] and "SIMULATION" not in "".join(pages)
    if clock == time(9, 25):
        assert data.waiting == 0 and data.in_trade == 0


def test_real_index_tiles_replace_the_equal_weight_fallback():
    engine, day, screen, views, replay, config, risk = _state(time(10, 50))
    frame = engine.visible("S01").copy()
    items = board.index_ticker({"NIFTY 100": frame})
    data = board.build_board(engine, day, screen, views, replay, config, risk, ticker_items=items)
    assert [item.label for item in data.ticker] == ["NIFTY 100"]
    assert data.ticker[0].change_points == pytest.approx(float(frame["close"].iloc[-1]) - float(frame["prev_close"].iloc[0]))
    fallback = board.build_board(engine, day, screen, views, replay, config, risk)
    assert "eq-wt" in fallback.ticker[0].label


def test_movers_follow_the_chosen_sector():
    engine, day, screen, views, replay, config, risk = _state(time(10, 50))
    data = board.build_board(engine, day, screen, views, replay, config, risk, focus_sector="Beta")
    assert data.focus_sector == "Beta"
    assert {m.symbol for m in data.movers} <= {v.symbol for v in views if v.sector == "Beta"}
    changes = [m.change_pct for m in data.movers]
    assert changes == sorted(changes, reverse=True)
    assert "Beta movers" in board.render_bottom(data)


def test_buyer_share_counts_up_volume_and_splits_flat():
    rows = [(10, 11, 9, 10.5, 300), (10, 11, 9, 9.5, 100), (10, 11, 9, 10, 200)]
    assert board.buyer_share(_bars(rows)) == pytest.approx((300 + 100) / 600)


def test_rows_filter_and_sort():
    engine, day, screen, views, replay, config, risk = _state(time(10, 50))
    rows = board.opportunity_rows(views)
    assert len(rows) <= board.TABLE_ROWS
    statuses = [board.STATUS_ORDER.get(r.view.status, 9) for r in rows]
    assert statuses == sorted(statuses)
    assert all(r.view.side == BUY for r in board.opportunity_rows(views, "Long"))
    assert all(r.view.side != BUY for r in board.opportunity_rows(views, "Short"))
    assert all(r.view.status == WAITING for r in board.opportunity_rows(views, "Waiting"))
    for row in rows:
        assert row.steps_total == 7
        if row.risk_reward is not None:
            assert row.risk_reward > 0


def test_table_shows_the_step_11_columns_without_setup():
    engine, day, screen, views, replay, config, risk = _state(time(10, 50))
    rows = board.opportunity_rows(views)
    page = board.render_table(rows, None)
    for header in ("Price", "Previous day close", "Day high", "Day low", "Window high", "Window low",
                   "% change from previous day", "Volume", "VWAP", "3-day % change",
                   "Suggested buy", "Suggested sell", "SL"):
        assert f">{header}</th>" in page
    assert ">Setup</th>" not in page
    view = rows[0].view
    expected_buy = view.order_price if view.side == BUY else view.target
    assert f"{expected_buy:,.2f}" in page
    assert f"{view.prev_close:,.2f}" in page


def test_vwap_column_says_above_or_below():
    assert "▲ Above" in board._vwap_tag(101.0, 100.0)
    assert "▼ Below" in board._vwap_tag(99.0, 100.0)
    assert board._vwap_tag(float("nan"), 100.0) == ""
    engine, day, screen, views, replay, config, risk = _state(time(10, 50))
    page = board.render_table(board.opportunity_rows(views), None)
    assert "orb-vwap" in page


def test_escaping_and_signal_labels():
    engine, day, screen, views, replay, config, risk = _state(time(10, 50))
    view = views[0]
    view.symbol = "M&M"
    page = board.render_table([board.OpportunityRow(view, "Gap-up dip buy", None, 3, 7, "NO TRADE")], "M&M")
    assert "M&amp;M" in page and "M&M<" not in page
    assert board.signal_label(view) in {"BUY", "SHORT", "WAIT", "TARGET", "STOPPED", "CLOSED", "NO TRADE"}
