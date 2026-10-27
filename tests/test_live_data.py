"""Live data plumbing against a fake Kite client. No network, no real orders."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

import pytest

import board
from backtest import CostAssumptions
from live_data import IST, LiveEngine, LiveMarket, is_login_error
from risk_manager import RiskConfig
from strategy import StrategyConfig, build_day, replay_day, screen_day, stock_views
from universe import Stock

STOCKS = [
    Stock("AAA", "Alpha", "X"),
    Stock("BBB", "Beta", "X"),
    Stock("CCC", "Gamma", "Y"),
    Stock("ZZZ", "Missing", "Y"),
]
FRIDAY = date(2026, 10, 23)


class FakeKite:
    """Mimics the KiteConnect calls LiveMarket uses. Friday has full minute data."""

    def __init__(self, minute_days=(FRIDAY,)):
        self.minute_days = set(minute_days)
        self.calls = []

    def instruments(self, exchange):
        rows = [
            {"tradingsymbol": "AAA", "instrument_token": 1, "segment": "NSE", "instrument_type": "EQ"},
            {"tradingsymbol": "BBB", "instrument_token": 2, "segment": "NSE", "instrument_type": "EQ"},
            {"tradingsymbol": "CCC", "instrument_token": 3, "segment": "NSE", "instrument_type": "EQ"},
            {"tradingsymbol": "CCC-BE", "instrument_token": 30, "segment": "NSE", "instrument_type": "EQ"},
            {"tradingsymbol": "NIFTY 100", "instrument_token": 900, "segment": "INDICES", "instrument_type": "EQ"},
        ]
        return rows

    def historical_data(self, token, start, end, interval):
        self.calls.append((token, start, end, interval))
        base = {1: 100.0, 2: 200.0, 3: 300.0, 900: 25000.0}[token]
        if interval == "day":
            days = [FRIDAY - timedelta(days=offset) for offset in (7, 6, 5, 4, 3, 0)]
            return [
                {"date": datetime.combine(day, time(0), tzinfo=IST), "open": base, "high": base, "low": base,
                 "close": base + i, "volume": 1}
                for i, day in enumerate(days)
                if start.date() <= day <= end.date()
            ]
        rows = []
        day = start.date()
        if day not in self.minute_days:
            return rows
        moment = datetime.combine(day, time(9, 15), tzinfo=IST)
        stop = end.replace(tzinfo=IST) if end.tzinfo is None else end
        index = 0
        while moment < stop and moment.time() < time(15, 30):
            if not (token == 2 and index == 3):  # BBB has one quiet minute
                price = base + index * 0.1
                rows.append({"date": moment, "open": price, "high": price + 0.2, "low": price - 0.2,
                             "close": price + 0.1, "volume": 1000})
            moment += timedelta(minutes=1)
            index += 1
        return rows


def _market(now: datetime, client=None) -> LiveMarket:
    market = LiveMarket(client or FakeKite(), STOCKS, now=lambda: now, pause=lambda _s: None)
    market.load()
    return market


def test_weekend_loads_the_last_trading_day_with_previous_closes():
    market = _market(datetime(2026, 10, 25, 17, 0, tzinfo=IST))
    assert market.state == "ready", market.message
    assert market.session_date == FRIDAY
    assert market.missing == ["ZZZ"]
    assert [stock.symbol for stock in market.available] == ["AAA", "BBB", "CCC"]
    # Daily closes before Friday are base+0..base+4; the last one is the previous close.
    assert market.prev_close["AAA"] == pytest.approx(104.0)
    assert market.close_3d_ago["AAA"] == pytest.approx(102.0)
    engine = LiveEngine(market)
    assert engine.clock == datetime.combine(FRIDAY, time(15, 30), tzinfo=IST)
    assert engine.session_complete
    frame = engine.visible("AAA")
    assert len(frame) == 375
    assert set(frame["prev_close"]) == {104.0}


def test_quiet_minutes_are_filled_so_every_stock_shares_timestamps():
    engine = LiveEngine(_market(datetime(2026, 10, 25, 17, 0, tzinfo=IST)))
    a, b = engine.visible("AAA"), engine.visible("BBB")
    assert list(a["timestamp"]) == list(b["timestamp"])
    quiet = b.iloc[3]
    assert quiet["volume"] == 0
    assert quiet["open"] == quiet["close"] == pytest.approx(b.iloc[2]["close"])


def test_during_market_hours_only_closed_minutes_are_visible_and_ticks_update_them():
    now = datetime(2026, 10, 23, 10, 0, 30, tzinfo=IST)
    clock = {"now": now}
    market = LiveMarket(FakeKite(), STOCKS, now=lambda: clock["now"], pause=lambda _s: None)
    market.load()
    engine = LiveEngine(market)
    assert market.session_date == FRIDAY
    frame = engine.visible("AAA")
    assert frame["timestamp"].iloc[-1] == datetime(2026, 10, 23, 9, 59, tzinfo=IST)
    loaded_volume = market.last_volume[1]
    market.apply_ticks([
        {"instrument_token": 1, "last_price": 150.0, "volume_traded": loaded_volume + 500,
         "exchange_timestamp": datetime(2026, 10, 23, 10, 0, 40, tzinfo=IST)},
        {"instrument_token": 1, "last_price": 149.0, "volume_traded": loaded_volume + 800,
         "exchange_timestamp": datetime(2026, 10, 23, 10, 0, 50, tzinfo=IST)},
    ])
    assert len(engine.visible("AAA")) == len(frame)
    clock["now"] = datetime(2026, 10, 23, 10, 1, 5, tzinfo=IST)
    after = engine.visible("AAA")
    last = after.iloc[-1]
    assert last["timestamp"] == datetime(2026, 10, 23, 10, 0, tzinfo=IST)
    assert last["high"] == pytest.approx(150.0)
    assert last["close"] == pytest.approx(149.0)
    assert last["volume"] >= 800


def test_ticks_outside_the_session_are_ignored():
    market = _market(datetime(2026, 10, 23, 10, 0, 30, tzinfo=IST))
    before = dict(market.candles["AAA"])
    market.apply_ticks([{"instrument_token": 1, "last_price": 999.0, "volume_traded": 10,
                         "exchange_timestamp": datetime(2026, 10, 23, 16, 0, tzinfo=IST)}])
    assert market.candles["AAA"] == before


def test_live_engine_feeds_the_strategy_and_board():
    engine = LiveEngine(_market(datetime(2026, 10, 25, 17, 0, tzinfo=IST)))
    config = StrategyConfig(sectors_each_side=1, stocks_per_sector=1, min_volume=0)
    day = build_day({s.symbol: engine.visible(s.symbol) for s in engine.stocks}, engine.stocks, config)
    screen = screen_day(day, config)
    replay = replay_day(day, config, RiskConfig(), CostAssumptions(), session_over=True, screen=screen)
    views = stock_views(day, config, replay, screen)
    assert {view.symbol for view in views} == {"AAA", "BBB", "CCC"}
    items = board.index_ticker({"NIFTY 100": engine.index_frame("NIFTY 100")})
    assert items and items[0].label == "NIFTY 100"
    assert items[0].change_points == pytest.approx(float(engine.index_frame("NIFTY 100")["close"].iloc[-1]) - 25004.0)
    data = board.build_board(engine, day, screen, views, replay, config, RiskConfig(), ticker_items=items)
    page = board.render_top(data) + board.render_bottom(data)
    assert ">Time<b>" in page and "LIVE DATA" in page and "SIMULATION MODE" not in page


def test_load_failure_is_reported_not_raised():
    class Broken(FakeKite):
        def instruments(self, exchange):
            raise RuntimeError("Incorrect `api_key` or `access_token`.")

    market = _market(datetime(2026, 10, 25, 17, 0, tzinfo=IST), client=Broken())
    assert market.state == "error"
    assert "access_token" in market.message
    assert not market.login_expired


def test_fo_list_comes_from_zerodha_stock_futures():
    class WithFutures(FakeKite):
        def instruments(self, exchange):
            if exchange == "NFO":
                return [
                    {"tradingsymbol": "AAA26NOVFUT", "name": "AAA", "segment": "NFO-FUT", "instrument_type": "FUT"},
                    {"tradingsymbol": "CCC26NOVFUT", "name": "CCC", "segment": "NFO-FUT", "instrument_type": "FUT"},
                    {"tradingsymbol": "NIFTY26NOVFUT", "name": "NIFTY", "segment": "NFO-FUT", "instrument_type": "FUT"},
                    {"tradingsymbol": "BBB26NOV100CE", "name": "BBB", "segment": "NFO-OPT", "instrument_type": "CE"},
                ]
            return super().instruments(exchange)

    market = LiveMarket(WithFutures(), [], now=lambda: datetime(2026, 10, 25, 17, 0, tzinfo=IST),
                        pause=lambda _s: None, fo_list="NIFTY F&O")
    market.load()
    assert market.state == "ready", market.message
    assert [stock.symbol for stock in market.available] == ["AAA", "CCC"]


def test_expired_login_is_flagged():
    class TokenException(Exception):
        pass

    class Expired(FakeKite):
        def instruments(self, exchange):
            raise TokenException("Token is invalid or has expired.")

    market = _market(datetime(2026, 10, 27, 9, 50, tzinfo=IST), client=Expired())
    assert market.state == "error" and market.login_expired
    assert is_login_error("WebSocket connection upgrade failed (403 - Forbidden)")
    assert not is_login_error("Too many requests")


def test_a_new_trading_day_asks_for_a_reload_once_its_first_candle_exists():
    clock = {"now": datetime(2026, 10, 26, 9, 0, tzinfo=IST)}
    market = LiveMarket(FakeKite(), STOCKS, now=lambda: clock["now"], pause=lambda _s: None)
    market.load()
    assert market.session_date == FRIDAY
    assert not market.needs_new_session()
    clock["now"] = datetime(2026, 10, 26, 9, 20, tzinfo=IST)
    assert market.needs_new_session()
    market.load()
    assert market.session_date == FRIDAY  # no Monday candles in the fake, like a holiday
    assert not market.needs_new_session()
    weekend = _market(datetime(2026, 10, 25, 17, 0, tzinfo=IST))
    assert not weekend.needs_new_session()
