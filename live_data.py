"""Read-only market data from Zerodha Kite Connect.

Loads the latest session's one-minute candles, the previous close, and the close
three sessions back for every stock, then keeps the current session updated
from the Kite WebSocket feed. ``load_history`` downloads past sessions for the
backtest and keeps them in memory only. The screens read everything through
LiveEngine. Nothing here places, modifies, or cancels an order.
"""

from __future__ import annotations

import threading
import time as clock_time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from market_data import IST, SESSION_CLOSE, SESSION_OPEN
from universe import Stock, with_fo

INDEX_SYMBOLS = ("NIFTY 100",)
HISTORY_INTERVAL_SECONDS = 0.36  # Kite allows about 3 historical requests per second.
HISTORY_WORKERS = 3  # Requests overlap their network wait; the spacing above still caps the rate.
DAILY_LOOKBACK_DAYS = 14
FIRST_CANDLE_READY = time(9, 16)  # Kite has today's first minute candle once 09:15-09:16 closes.

LIVE_BANNER = (
    "LIVE DATA from Zerodha Kite Connect (read-only). Candles are built from the exchange feed and "
    "can lag by a few seconds. Signals are paper signals for research, not advice. No order is sent."
)


def login_url(api_key: str) -> str:
    from kiteconnect import KiteConnect

    return KiteConnect(api_key=api_key).login_url()


def exchange_request_token(api_key: str, api_secret: str, request_token: str) -> str:
    """Trade the one-time request token from the login redirect for today's access token."""
    from kiteconnect import KiteConnect

    session = KiteConnect(api_key=api_key).generate_session(request_token, api_secret=api_secret)
    return str(session["access_token"])


def to_ist(moment: datetime) -> datetime:
    """Naive times from the WebSocket are in the computer's local zone."""
    return moment.astimezone(IST)


def floor_minute(moment: datetime) -> datetime:
    return moment.replace(second=0, microsecond=0)


def is_login_error(error: object) -> bool:
    """Kite rejects an expired or invalid access token with TokenException or HTTP 403."""
    return type(error).__name__ == "TokenException" or "403" in str(error)


@dataclass
class LiveStatus:
    mode: str
    clock: datetime
    updated_at: datetime
    tracked_stocks: int
    session_date: date
    banner: str
    last_closed_bar: datetime | None


class LiveMarket:
    """Holds one session of minute candles per symbol and keeps today's updated from ticks.

    ``client`` is a logged-in KiteConnect. ``ticker_factory`` builds a KiteTicker; pass
    None in tests to skip the WebSocket. ``now`` returns the current time. ``fo_list``
    ('All' or 'NIFTY F&O') replaces ``stocks`` with that list once Zerodha's F&O stocks are known.
    """

    def __init__(
        self,
        client: Any,
        stocks: Sequence[Stock],
        *,
        ticker_factory: Callable[[], Any] | None = None,
        now: Callable[[], datetime] | None = None,
        pause: Callable[[float], None] = clock_time.sleep,
        fo_list: str | None = None,
    ) -> None:
        self.client = client
        self.stocks = list(stocks)
        self.fo_list = fo_list
        self.ticker_factory = ticker_factory
        self.now = now or (lambda: datetime.now(IST))
        self.pause = pause
        self.lock = threading.RLock()
        self.state = "starting"
        self.message = "Connecting to Kite..."
        self.done = 0
        self.total = 0
        self.errors: list[str] = []
        self.missing: list[str] = []
        self.available: list[Stock] = []
        self.session_date: date | None = None
        self.candles: dict[str, dict[datetime, list[float]]] = {}
        self.prev_close: dict[str, float] = {}
        self.close_3d_ago: dict[str, float] = {}
        self.tokens: dict[str, int] = {}
        self.token_symbol: dict[int, str] = {}
        self.last_volume: dict[int, float] = {}
        self.last_tick: datetime | None = None
        self.loaded_at: datetime | None = None
        self.ticker: Any = None
        self.login_expired = False
        self.history: dict[int, list] = {}
        self.history_state = "idle"
        self.history_message = ""
        self.history_done = 0
        self.history_total = 0
        self._slot_lock = threading.Lock()
        self._next_slot = 0.0
        self._thread: threading.Thread | None = None

    # ---------------------------------------------------------------- loading

    def start(self) -> None:
        self._thread = threading.Thread(target=self.load, daemon=True)
        self._thread.start()

    def load(self) -> None:
        try:
            self._load()
        except Exception as exc:  # surfaced in the sidebar
            with self.lock:
                self.login_expired = self.login_expired or is_login_error(exc)
                self.state = "error"
                self.message = f"Loading Kite data failed: {exc}"
                self.errors.append(str(exc))

    def _history(self, token: int, start: datetime, end: datetime, interval: str) -> list[dict]:
        """One historical request. Safe to call from several threads; requests are spaced out."""
        with self._slot_lock:
            slot = max(clock_time.monotonic(), self._next_slot)
            self._next_slot = slot + HISTORY_INTERVAL_SECONDS
        wait = slot - clock_time.monotonic()
        if wait > 0:
            self.pause(wait)
        return self.client.historical_data(token, start.replace(tzinfo=None), end.replace(tzinfo=None), interval)

    def prefetch_history(self, calendar_days: int) -> None:
        """Start downloading backtest history in the background unless it is running or done."""
        with self.lock:
            if self.history_state == "running" or calendar_days in self.history:
                return
            self.history_state = "running"
            self.history_message = "Starting the history download..."

        def progress(done: int, total: int, text: str) -> None:
            self.history_done, self.history_total, self.history_message = done, total, text

        def run() -> None:
            try:
                days = load_history(self, calendar_days, progress=progress)
                with self.lock:
                    self.history[calendar_days] = days
                    self.history_state = "ready"
                    self.history_message = f"{len(days)} sessions downloaded."
            except Exception as exc:
                with self.lock:
                    self.history_state = "error"
                    self.history_message = f"History download failed: {exc}"

        threading.Thread(target=run, daemon=True).start()

    def _load(self) -> None:
        self.message = "Downloading the NSE instrument list..."
        instruments = self.client.instruments("NSE")
        equities = {
            row["tradingsymbol"]: int(row["instrument_token"])
            for row in instruments
            if row.get("segment") == "NSE" and row.get("instrument_type") == "EQ"
        }
        indices = {
            row["tradingsymbol"]: int(row["instrument_token"])
            for row in instruments
            if row.get("segment") == "INDICES" and row.get("tradingsymbol") in INDEX_SYMBOLS
        }
        if self.fo_list is not None:
            self.message = "Downloading Zerodha's F&O list..."
            names = {row["tradingsymbol"]: str(row.get("name") or "") for row in instruments}
            fo = {
                str(row.get("name"))
                for row in self.client.instruments("NFO")
                if row.get("segment") == "NFO-FUT" and row.get("name") in equities
            }
            if not fo:
                raise RuntimeError("Zerodha returned no stock futures, so the F&O list is empty.")
            self.stocks = with_fo(self.fo_list, {symbol: names.get(symbol, "") for symbol in fo})
        with self.lock:
            self.missing = [stock.symbol for stock in self.stocks if stock.symbol not in equities]
            self.available = [stock for stock in self.stocks if stock.symbol in equities]
            self.tokens = {stock.symbol: equities[stock.symbol] for stock in self.available}
            self.tokens.update(indices)
            self.token_symbol = {token: symbol for symbol, token in self.tokens.items()}
        if not self.available:
            raise RuntimeError("None of the chosen symbols were found in the NSE instrument list.")

        now = self.now()
        self.session_date = self._find_session(now)
        session_start = datetime.combine(self.session_date, SESSION_OPEN, tzinfo=IST)
        session_end = datetime.combine(self.session_date, SESSION_CLOSE, tzinfo=IST)
        minute_end = min(now, session_end)
        daily_start = datetime.combine(self.session_date - timedelta(days=DAILY_LOOKBACK_DAYS), time(0), tzinfo=IST)

        symbols = list(self.tokens)
        self.total = len(symbols)

        def fetch(symbol: str):
            token = self.tokens[symbol]
            try:
                daily = self._history(token, daily_start, session_end, "day")
                minutes = self._history(token, session_start, minute_end, "minute")
                return symbol, token, daily, minutes, None
            except Exception as exc:
                return symbol, token, None, None, exc

        with ThreadPoolExecutor(max_workers=HISTORY_WORKERS) as pool:
            for symbol, token, daily, minutes, error in pool.map(fetch, symbols):
                self._keep_session(symbol, token, daily, minutes, error)
                self.done += 1
                self.message = f"Loaded {symbol} ({self.done} of {self.total})..."

        with self.lock:
            dropped = [s.symbol for s in self.available if s.symbol not in self.prev_close or not self.candles.get(s.symbol)]
            for symbol in dropped:
                self.errors.append(f"{symbol}: no previous close or no candles for {self.session_date}.")
            self.available = [s for s in self.available if s.symbol not in dropped]
            self.loaded_at = self.now()
            self.state = "ready"
            self.message = f"Loaded {len(self.available)} stocks for {self.session_date:%d %b %Y}."
        if self.ticker_factory is not None and self.session_date == now.date() and now.time() < SESSION_CLOSE:
            self._start_ticker()

    def _keep_session(self, symbol: str, token: int, daily, minutes, error) -> None:
        with self.lock:
            if error is not None:
                self.errors.append(f"{symbol}: {error}")
                self.login_expired = self.login_expired or is_login_error(error)
                return
            before = [row for row in daily if pd.Timestamp(row["date"]).date() < self.session_date]
            if before:
                self.prev_close[symbol] = float(before[-1]["close"])
                self.close_3d_ago[symbol] = float(before[-3]["close"] if len(before) >= 3 else before[0]["close"])
            store = self.candles.setdefault(symbol, {})
            for row in minutes:
                stamp = floor_minute(_row_time(row["date"]))
                store[stamp] = [float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]), float(row.get("volume", 0))]
            self.last_volume[token] = sum(candle[4] for candle in store.values())

    def _find_session(self, now: datetime) -> date:
        """Today once it has minute candles, otherwise the most recent trading day."""
        probe = self.tokens[self.available[0].symbol]
        today_open = datetime.combine(now.date(), SESSION_OPEN, tzinfo=IST)
        if now >= today_open + timedelta(minutes=1):
            if self._history(probe, today_open, now, "minute"):
                return now.date()
        start = datetime.combine(now.date() - timedelta(days=DAILY_LOOKBACK_DAYS), time(0), tzinfo=IST)
        daily = self._history(probe, start, now, "day")
        days = sorted({pd.Timestamp(row["date"]).date() for row in daily if pd.Timestamp(row["date"]).date() < now.date()})
        if not days:
            raise RuntimeError("Kite returned no recent trading days.")
        return days[-1]

    # ---------------------------------------------------------------- ticks

    def _start_ticker(self) -> None:
        ticker = self.ticker_factory()
        tokens = list(self.token_symbol)

        def on_connect(ws, _response):
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_FULL, tokens)

        def on_ticks(_ws, ticks):
            self.apply_ticks(ticks)

        def on_error(ws, code, reason):
            with self.lock:
                self.errors.append(f"Live feed error {code}: {reason}")
                if is_login_error(reason):
                    self.login_expired = True
            if is_login_error(reason):
                try:
                    ws.stop_retry()
                except Exception:
                    pass

        ticker.on_connect = on_connect
        ticker.on_ticks = on_ticks
        ticker.on_error = on_error
        from twisted.internet import reactor

        # The Twisted reactor runs once per process; later tickers must join it from its own thread.
        if reactor.running:
            reactor.callFromThread(ticker.connect, threaded=True)
        else:
            ticker.connect(threaded=True)
        self.ticker = ticker

    def stop(self) -> None:
        if self.ticker is not None:
            try:
                self.ticker.close()
            except Exception:
                pass
            self.ticker = None

    def apply_ticks(self, ticks: list[dict]) -> None:
        """Fold ticks into the minute candle they fall in. Volume is the change in day volume."""
        with self.lock:
            for tick in ticks:
                token = int(tick.get("instrument_token", 0))
                symbol = self.token_symbol.get(token)
                price = tick.get("last_price")
                if symbol is None or price is None:
                    continue
                raw = tick.get("exchange_timestamp") or tick.get("last_trade_time")
                moment = to_ist(raw) if isinstance(raw, datetime) else self.now()
                if moment.date() != self.session_date or not (SESSION_OPEN <= moment.time() < SESSION_CLOSE):
                    continue
                stamp = floor_minute(moment)
                volume_now = float(tick.get("volume_traded") or 0.0)
                previous = self.last_volume.get(token)
                added = max(volume_now - previous, 0.0) if previous is not None and volume_now else 0.0
                if volume_now:
                    self.last_volume[token] = volume_now
                store = self.candles.setdefault(symbol, {})
                candle = store.get(stamp)
                price = float(price)
                if candle is None:
                    store[stamp] = [price, price, price, price, added]
                else:
                    candle[1] = max(candle[1], price)
                    candle[2] = min(candle[2], price)
                    candle[3] = price
                    candle[4] += added
                self.last_tick = self.now()

    # ---------------------------------------------------------------- reading

    def clock(self) -> datetime:
        """End of the latest closed minute this session can show."""
        if self.session_date is None:
            return self.now()
        end = datetime.combine(self.session_date, SESSION_CLOSE, tzinfo=IST)
        now = self.now()
        if now.date() != self.session_date or now >= end:
            return end
        return max(floor_minute(now), datetime.combine(self.session_date, SESSION_OPEN, tzinfo=IST) + timedelta(minutes=1))

    def frame(self, symbol: str) -> pd.DataFrame:
        """Closed one-minute candles from 09:15, one row per minute. Quiet minutes repeat the last close."""
        clock = self.clock()
        start = datetime.combine(self.session_date, SESSION_OPEN, tzinfo=IST)
        minutes = []
        moment = start
        while moment + timedelta(minutes=1) <= clock:
            minutes.append(moment)
            moment += timedelta(minutes=1)
        with self.lock:
            store = dict(self.candles.get(symbol, {}))
            prev = self.prev_close.get(symbol)
            three = self.close_3d_ago.get(symbol, prev)
        rows = []
        last = None
        first_open = next((store[m][0] for m in minutes if m in store), prev)
        for minute in minutes:
            candle = store.get(minute)
            if candle is None:
                level = last if last is not None else (first_open or prev)
                rows.append((minute, level, level, level, level, 0.0))
            else:
                rows.append((minute, *candle))
                last = candle[3]
        frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        frame["volume"] = frame["volume"].round().astype("int64")
        frame["data_source"] = "kite"
        frame["prev_close"] = prev
        frame["close_3d_ago"] = three
        return frame

    def needs_new_session(self) -> bool:
        """True once a new trading day has its first candle but this market still holds an older day."""
        now = self.now()
        if self.state != "ready" or self.session_date is None or self.session_date >= now.date():
            return False
        if now.weekday() >= 5 or now.time() < FIRST_CANDLE_READY:
            return False
        opened = datetime.combine(now.date(), FIRST_CANDLE_READY, tzinfo=IST)
        return self.loaded_at is None or self.loaded_at < opened

    def version(self) -> tuple:
        """Changes whenever the screen should refresh."""
        return (
            self.state, self.done, self.clock().isoformat(), len(self.errors),
            self.login_expired, self.needs_new_session(),
        )


@dataclass
class HistoryDay:
    """One past session for every stock that traded, aligned to the same 375 minutes."""

    session_date: date
    stamps: list[datetime]
    stocks: list[Stock]
    arrays: dict[str, dict[str, Any]]


MAX_HISTORY_DAYS = 60  # Kite returns at most 60 days of minute candles per request.


def session_minutes(session_date: date) -> list[datetime]:
    start = datetime.combine(session_date, SESSION_OPEN, tzinfo=IST)
    count = int((datetime.combine(session_date, SESSION_CLOSE) - datetime.combine(session_date, SESSION_OPEN)).total_seconds() // 60)
    return [start + timedelta(minutes=i) for i in range(count)]


def _row_time(value: Any) -> datetime:
    """Kite sends datetimes already; other inputs go through pandas, which is far slower per row."""
    moment = value if isinstance(value, datetime) else pd.Timestamp(value).to_pydatetime()
    return to_ist(moment)


def align_minutes(rows: list[dict], minutes: list[datetime], fallback: float) -> dict[str, list[float]]:
    """Candles on the minute grid. A minute without trades repeats the last close with zero volume."""
    by_minute = {floor_minute(_row_time(row["date"])): row for row in rows}
    first = next((float(by_minute[m]["open"]) for m in minutes if m in by_minute), fallback)
    out = {name: [] for name in ("open", "high", "low", "close", "volume")}
    last = None
    for minute in minutes:
        row = by_minute.get(minute)
        if row is None:
            level = last if last is not None else first
            values = (level, level, level, level, 0.0)
        else:
            values = (float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]), float(row.get("volume", 0)))
            last = values[3]
        for name, value in zip(out, values):
            out[name].append(value)
    return out


def load_history(
    market: LiveMarket,
    calendar_days: int,
    progress: Callable[[int, int, str], None] | None = None,
) -> list[HistoryDay]:
    """Past complete sessions in the last ``calendar_days`` days, oldest first. Kept in memory only.

    Uses the market's client and rate limit. Today's session is included only after 15:30.
    """
    if not 1 <= calendar_days <= MAX_HISTORY_DAYS:
        raise ValueError(f"History can cover 1 to {MAX_HISTORY_DAYS} calendar days.")
    now = market.now()
    last_day = now.date() if now.time() >= SESSION_CLOSE else now.date() - timedelta(days=1)
    first_day = last_day - timedelta(days=calendar_days - 1)
    minute_start = datetime.combine(first_day, SESSION_OPEN, tzinfo=IST)
    minute_end = datetime.combine(last_day, SESSION_CLOSE, tzinfo=IST)
    daily_start = datetime.combine(first_day - timedelta(days=DAILY_LOOKBACK_DAYS), time(0), tzinfo=IST)
    stocks = list(market.available)
    by_day: dict[date, dict[str, dict[str, Any]]] = {}

    def fetch(stock: Stock):
        token = market.tokens[stock.symbol]
        try:
            daily = market._history(token, daily_start, minute_end, "day")
            minutes = market._history(token, minute_start, minute_end, "minute")
            return stock, daily, minutes, None
        except Exception as exc:
            return stock, None, None, exc

    if progress is not None:
        progress(0, len(stocks), f"Downloading history for {len(stocks)} stocks...")
    with ThreadPoolExecutor(max_workers=HISTORY_WORKERS) as pool:
        results = pool.map(fetch, stocks)
        for done, (stock, daily, minutes, error) in enumerate(results, start=1):
            if progress is not None:
                progress(done, len(stocks), f"Downloaded {stock.symbol} ({done} of {len(stocks)})...")
            if error is not None:
                with market.lock:
                    market.errors.append(f"{stock.symbol} history: {error}")
                continue
            _keep_history(stock, daily, minutes, by_day, first_day, last_day)
    if progress is not None:
        progress(len(stocks), len(stocks), "Arranging sessions...")
    days: list[HistoryDay] = []
    for session_date in sorted(by_day):
        arrays = by_day[session_date]
        included = [stock for stock in stocks if stock.symbol in arrays]
        days.append(HistoryDay(session_date, session_minutes(session_date), included, arrays))
    return days


def _keep_history(
    stock: Stock,
    daily: list[dict],
    minutes: list[dict],
    by_day: dict[date, dict[str, dict[str, Any]]],
    first_day: date,
    last_day: date,
) -> None:
    """Turn one stock's raw candles into compact per-session arrays right away, so 500 stocks fit in memory."""
    closes = sorted((_row_time(r["date"]).date(), float(r["close"])) for r in daily)
    grouped: dict[date, list[dict]] = {}
    for row in minutes:
        grouped.setdefault(_row_time(row["date"]).date(), []).append(row)
    for session_date, rows in grouped.items():
        before = [close for day, close in closes if day < session_date]
        if not first_day <= session_date <= last_day or not before:
            continue
        aligned = align_minutes(rows, session_minutes(session_date), before[-1])
        arrays: dict[str, Any] = {name: np.asarray(values, dtype=np.float32) for name, values in aligned.items()}
        arrays["prev_close"] = before[-1]
        arrays["close_3d_ago"] = before[-3] if len(before) >= 3 else before[0]
        by_day.setdefault(session_date, {})[stock.symbol] = arrays


class LiveEngine:
    """The market interface the screens use, backed by a LiveMarket."""

    def __init__(self, market: LiveMarket) -> None:
        self.market = market

    @property
    def stocks(self) -> list[Stock]:
        return self.market.available

    @property
    def clock(self) -> datetime:
        return self.market.clock()

    @property
    def session_date(self) -> date:
        return self.market.session_date

    @property
    def session_complete(self) -> bool:
        return self.clock.time() >= SESSION_CLOSE

    def visible(self, symbol: str) -> pd.DataFrame:
        return self.market.frame(symbol)

    def index_frame(self, label: str) -> pd.DataFrame | None:
        if label not in self.market.tokens or label not in self.market.prev_close:
            return None
        return self.market.frame(label)

    def status(self) -> LiveStatus:
        clock = self.clock
        return LiveStatus(
            mode="LIVE",
            clock=clock,
            updated_at=self.market.last_tick or self.market.loaded_at or self.market.now(),
            tracked_stocks=len(self.stocks),
            session_date=self.session_date,
            banner=LIVE_BANNER,
            last_closed_bar=clock - timedelta(minutes=1),
        )
