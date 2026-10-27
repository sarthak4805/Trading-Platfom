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
    assert [c.change_pct for c in data.performance] == sorted((c.change_pct for c in data.performance), reverse=True)
    assert all(m.session_volume > config.min_volume for m in data.movers)
    assert all(0.0 <= m.buyer_share <= 1.0 for m in data.movers)
    pages = [board.render_top(data), board.render_table(data.rows, None), board.render_sectors(data), board.render_bottom(data)]
    for removed in ("Active setups", "Risk used", "Daily P&amp;L", "Market clock"):
        assert removed not in pages[0]
    assert len(data.ticker) == 1 and ">Time<b>" in pages[0]
    if data.rows:
        view = data.rows[0].view
        frame = engine.visible(view.symbol)
        chart = board.build_chart(frame, view, config)
        pages += [board.render_chart_head(chart), board.render_chart(chart), board.render_plan(view, risk.notional_inr)]
        assert f"Open ₹{float(frame['open'].iloc[0]):,.2f}" in pages[-3]
        assert "Entry price" in pages[-1]
        assert ">Open<" in board.chart_svg(chart)
    for page in pages:
        assert "<svg" not in page
    assert "SIMULATION" not in "".join(pages)
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


def test_sector_cards_list_the_tapped_sectors_top_five():
    engine, day, screen, views, replay, config, risk = _state(time(10, 50))
    data = board.build_board(engine, day, screen, views, replay, config, risk)
    plain = board.render_sectors(data)
    assert "data-sector=" in plain and "<div class='orb-sector-stocks'>" not in plain
    sector = data.sector_cards[0].sector
    leaders = board.sector_leaders(views, sector)
    members = [v for v in views if v.sector == sector]
    assert len(leaders) == min(5, len(members))
    rising = sum(v.change_pct for v in members) >= 0
    changes = [v.change_pct for v in leaders]
    assert changes == sorted(changes, reverse=rising)
    page = board.render_sectors(data, sector, views)
    assert "<div class='orb-sector-stocks'>" in page and "orb-sector sel" in page
    assert all(f"data-symbol='{v.symbol}'" in page for v in leaders)


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


def test_table_shows_the_step_11_columns_without_setup():
    engine, day, screen, views, replay, config, risk = _state(time(10, 50))
    rows = board.opportunity_rows(views)
    page = board.render_table(rows, None)
    for header in ("Price", "Previous day close", "Day high", "Day low", "Time Frame High", "Time Frame Low",
                   "% change from previous day", "Volume", "VWAP", "3-day % change",
                   "Suggested buy", "Suggested sell", "SL"):
        assert f">{header}</th>" in page
    assert ">Setup</th>" not in page and ">R:R</th>" not in page
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
    page = board.render_table([board.OpportunityRow(view, "Gap-up dip buy", 3, 7, "NO TRADE")], "M&M")
    assert "M&amp;M" in page and "M&M<" not in page
    assert board.signal_label(view) in {"BUY", "SHORT", "WAIT", "TARGET", "STOPPED", "CLOSED", "NO TRADE"}
