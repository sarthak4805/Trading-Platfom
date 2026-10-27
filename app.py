"""NSE intraday research desk running the bullish / bearish PDF strategy.

Prices come from Zerodha Kite Connect (read-only). The app never places,
modifies, or cancels an order.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from plotly.subplots import make_subplots

import backtest_board
import board
import live_data
from backtest import CostAssumptions, exit_reason_text
from market_data import IST, format_wall_clock
from risk_manager import RISK_LIMITATIONS, RiskConfig, position_size
from strategy import (
    BUY,
    NO_TRADE,
    StockView,
    StrategyConfig,
    build_day,
    replay_day,
    screen_day,
    stock_views,
)
from universe import DEFAULT_LIST, FO_LISTS, LIST_CHOICES, UNIVERSE_DISCLAIMER, index_stocks

st.set_page_config(
    page_title="NSE Intraday Desk",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PAGE_CSS = """
<style>
[data-testid="stMainBlockContainer"] { max-width: 100%; padding: 2.75rem 1.5rem 1.5rem; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppDeployButton"] { display: none; }
</style>
"""
load_dotenv(override=False)

def _trigger(key: str, name: str):
    state = st.session_state.get(key)
    if state is None:
        return None
    return state.get(name) if hasattr(state, "get") else getattr(state, name, None)


def _on_table_click() -> None:
    clicked = _trigger("orb_table", "clicked")
    if clicked:
        st.session_state["board_chart"] = clicked


def _on_sector_stock_click() -> None:
    clicked = _trigger("orb_sectors", "clicked")
    if clicked:
        st.session_state["board_chart"] = clicked


def _on_sector_click() -> None:
    sector = _trigger("orb_sectors", "sector")
    if sector:
        same = st.session_state.get("board_sector_pick") == sector
        st.session_state["board_sector_pick"] = None if same else sector


def main() -> None:
    """Render the research desk."""
    try:
        _render()
    except Exception as exc:
        st.error(f"The dashboard hit an unexpected error: {exc}")
        st.info(
            "Press 'Reload Zerodha data' in the sidebar. If it keeps failing, run pytest "
            "from the project folder and read the traceback below."
        )
        st.exception(exc)


def _render() -> None:
    st.html(PAGE_CSS)
    engine = _live_engine()
    if engine is None:
        return
    strategy, risk = _strategy_controls()
    market = _market(engine, strategy, risk)
    views: list[StockView] = market["views"]

    board_tab, overview, detail, backtest = st.tabs(
        ["Opening-range board", "Market overview", "Stock detail", "Backtesting"]
    )
    with board_tab:
        _board(engine, market, strategy, risk)
    with overview:
        _banner()
        _overview(engine, views)
    with detail:
        _banner()
        _detail(engine, strategy, risk, views)
    with backtest:
        _backtest(engine, strategy, risk, views)


def _banner() -> None:
    st.caption("NSE intraday research and paper-trading desk. No broker orders are sent.")
    st.info(live_data.LIVE_BANNER)


def _today():
    return datetime.now(IST).date()


@st.cache_resource
def _token_store() -> dict:
    """Today's Kite access token and the market it feeds, kept in server memory only. Never written to disk or shown."""
    return {}


@st.cache_resource(show_spinner=False)
def _live_market(api_key: str, access_token: str, stock_list: str) -> live_data.LiveMarket:
    from kiteconnect import KiteConnect, KiteTicker

    client = KiteConnect(api_key=api_key)
    client.set_access_token(access_token)
    market = live_data.LiveMarket(
        client,
        index_stocks(stock_list),
        ticker_factory=lambda: KiteTicker(api_key, access_token),
        fo_list=stock_list if stock_list in FO_LISTS else None,
    )
    market.start()
    return market


def _stock_list() -> str:
    """Sidebar choice of which NSE list to scan. Changing it reloads that list from Zerodha."""
    return st.sidebar.selectbox(
        "Stock list",
        LIST_CHOICES,
        index=LIST_CHOICES.index(DEFAULT_LIST),
        key="stock_list",
        help="All is NIFTY 500 plus any F&O stock outside it. Bigger lists take longer to load: "
        "Zerodha allows about 3 history requests a second.",
    )


def _live_engine() -> live_data.LiveEngine | None:
    """Log in to Zerodha if needed, load the session, and return a live engine once ready."""
    api_key = os.getenv("KITE_API_KEY", "").strip()
    api_secret = os.getenv("KITE_API_SECRET", "").strip()
    st.sidebar.header("Zerodha")
    if not api_key or not api_secret or api_key.startswith("your_"):
        st.sidebar.error("Add KITE_API_KEY and KITE_API_SECRET to the .env file, then restart the app.")
        st.info("Live mode needs your Kite Connect API key and secret in the .env file.")
        return None
    store = _token_store()
    request_token = st.query_params.get("request_token")
    if request_token:
        if st.query_params.get("status") == "success":
            try:
                token = live_data.exchange_request_token(api_key, api_secret, request_token)
                if store.get("market") is not None:
                    store["market"].stop()
                _live_market.clear()
                store.clear()
                store[_today()] = token
            except Exception as exc:
                st.sidebar.error(f"Zerodha login failed: {exc}")
        st.query_params.clear()
    access_token = store.get(_today())
    if access_token is None:
        st.sidebar.link_button("Log in with Zerodha", live_data.login_url(api_key), type="primary", width="stretch")
        st.sidebar.caption("Zerodha asks you to log in once a day. The access token stays in this app's memory.")
        st.title("NSE Intraday Desk")
        if st.session_state.pop("login_expired_note", False):
            st.warning("Your Zerodha login has expired. Zerodha ends every login each morning, so log in again to load today's data.")
        st.link_button("Log in with Zerodha", live_data.login_url(api_key), type="primary")
        st.info(
            "Log in with Zerodha to load market data. After you log in, Zerodha sends you back here and the app "
            "loads the latest session's candles. Your Kite app's redirect URL must be the address of this page, "
            "for example `http://127.0.0.1:8501/`. Log in again each day."
        )
        return None
    stock_list = _stock_list()
    if store.get("stock_list") not in (None, stock_list):
        if store.get("market") is not None:
            store["market"].stop()
        _live_market.clear()
        st.session_state.pop("market_key", None)
    store["stock_list"] = stock_list
    market = _live_market(api_key, access_token, stock_list)
    store["market"] = market
    if getattr(market, "login_expired", False):
        market.stop()
        _live_market.clear()
        store.clear()
        st.session_state["login_expired_note"] = True
        st.rerun()
    if hasattr(market, "needs_new_session") and market.needs_new_session():
        market.stop()
        _live_market.clear()
        st.rerun()
    if st.sidebar.button("Reload Zerodha data", width="stretch"):
        market.stop()
        _live_market.clear()
        st.session_state.pop("market_key", None)
        st.rerun()
    if st.sidebar.button("Log out of Zerodha", width="stretch"):
        market.stop()
        _live_market.clear()
        store.clear()
        st.rerun()
    _live_refresh(market)
    if market.state == "error":
        st.error(market.message)
        st.info("Press **Log out of Zerodha** and log in again. The access token expires every day.")
        return None
    if market.state != "ready":
        st.title("Loading Zerodha data")
        share = market.done / market.total if market.total else 0.0
        st.progress(share, text=market.message)
        if market.total:
            seconds = market.total * 2 * live_data.HISTORY_INTERVAL_SECONDS
            st.caption(
                f"{stock_list}: {market.total} symbols. Zerodha allows about 3 history requests a second, "
                f"so this load takes about {max(1, round(seconds / 60))} minute(s)."
            )
        return None
    engine = live_data.LiveEngine(market)
    if hasattr(market, "prefetch_history"):
        market.prefetch_history(max(backtest_board.PERIOD_DAYS))
    st.sidebar.success(f"Connected. {stock_list}, session {engine.session_date:%d %b %Y}, {len(engine.stocks)} stocks.")
    if market.last_tick is not None:
        st.sidebar.caption(f"Last live tick {market.last_tick:%H:%M:%S}.")
    elif engine.session_complete:
        st.sidebar.caption("Market closed. Showing the latest complete session.")
    if market.missing:
        st.sidebar.warning("Not found on NSE, so left out: " + ", ".join(market.missing))
    if market.errors:
        with st.sidebar.expander(f"Data notes ({len(market.errors)})"):
            for note in market.errors[-20:]:
                st.write(note)
    return engine


def _live_refresh(market: live_data.LiveMarket, seconds: int = 5) -> None:
    """Rerun the page when a new minute closes or the load makes progress."""

    def _check() -> None:
        version = repr(market.version())
        if st.session_state.get("live_version") != version:
            first = "live_version" not in st.session_state
            st.session_state["live_version"] = version
            if not first:
                st.rerun()

    st.fragment(run_every=timedelta(seconds=seconds))(_check)()


def _market(engine: live_data.LiveEngine, strategy: StrategyConfig, risk: RiskConfig) -> dict:
    """Screen and replay the visible session once per clock and settings."""
    key = json.dumps(
        {
            "market": id(engine.market),
            "loaded": str(engine.market.loaded_at),
            "clock": engine.clock.isoformat(),
            "strategy": {name: str(value) for name, value in asdict(strategy).items()},
            "risk": {name: str(value) for name, value in asdict(risk).items()},
        },
        sort_keys=True,
    )
    if st.session_state.get("market_key") == key:
        return st.session_state["market"]
    frames = {stock.symbol: engine.visible(stock.symbol) for stock in engine.stocks}
    day = build_day(frames, engine.stocks, strategy)
    screen = screen_day(day, strategy)
    replay = replay_day(
        day,
        strategy,
        risk,
        CostAssumptions(),
        session_over=engine.clock.time() >= strategy.session_end,
        screen=screen,
    )
    market = {"day": day, "screen": screen, "replay": replay, "views": stock_views(day, strategy, replay, screen)}
    st.session_state["market_key"] = key
    st.session_state["market"] = market
    return market


def _sector_names(engine: live_data.LiveEngine) -> list[str]:
    return sorted({stock.sector for stock in engine.stocks})


def _board(engine: live_data.LiveEngine, market: dict, strategy: StrategyConfig, risk: RiskConfig) -> None:
    views: list[StockView] = market["views"]
    choice = st.session_state.get("opp_filter") or "All"
    st.sidebar.header("Board")
    focus = st.sidebar.selectbox("Movers sector", ["Strongest sector"] + _sector_names(engine), key="board_sector")
    indices = board.index_ticker({label: engine.index_frame(label) for label in live_data.INDEX_SYMBOLS})
    data = board.build_board(
        engine, market["day"], market["screen"], views, market["replay"], strategy, risk, choice,
        focus_sector=None if focus == "Strongest sector" else focus,
        ticker_items=indices,
    )
    listed = [row.view.symbol for row in data.rows]
    options = listed + sorted(view.symbol for view in views if view.symbol not in listed)
    if st.session_state.get("board_chart") not in options:
        st.session_state["board_chart"] = options[0]
    selected = st.session_state["board_chart"]
    view = next(v for v in views if v.symbol == selected)

    st.html(board.render_top(data))
    left, right = st.columns([2.5, 1], gap="medium")
    with left:
        with st.container(key="orb_opps"):
            head, controls = st.columns([1.5, 1.9], vertical_alignment="center")
            head.html(board.render_opportunities_head())
            controls.segmented_control(
                "Filter", ["All", "Long", "Short", "Waiting"], default="All", key="opp_filter", label_visibility="collapsed"
            )
            board.CLICKABLE_HTML(
                data=board.render_table(data.rows, selected), key="orb_table", on_clicked_change=_on_table_click
            )
            st.caption("Tap a row to show that stock's candles and trade plan on the right. Scroll the table sideways to see every column.")
    with right:
        with st.container(key="orb_chart"):
            head, interval = st.columns([2.2, 1.25], vertical_alignment="center")
            minutes = interval.selectbox("Interval", ["5 min", "1 min"], key="board_interval", label_visibility="collapsed")
            chart = board.build_chart(engine.visible(selected), view, strategy, 5 if minutes.startswith("5") else 1)
            head.html(board.render_chart_head(chart))
            st.selectbox("Chart stock", options, key="board_chart", label_visibility="collapsed")
            st.html(board.render_chart(chart))
        with st.container(key="orb_plan"):
            st.html(board.render_plan(view, risk.notional_inr))
            with st.popover("Review setup →", type="primary", width="stretch"):
                st.markdown(f"**{view.symbol} — {view.summary}**")
                _rule_lists(view)
    board.CLICKABLE_HTML(
        data=board.render_sectors(data, st.session_state.get("board_sector_pick"), views, selected),
        key="orb_sectors",
        on_sector_change=_on_sector_click,
        on_clicked_change=_on_sector_stock_click,
    )
    st.html(board.render_bottom(data))


PERCENT_STEPS = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0)
VOLUME_CHOICES = {"All": 0, "More than 1 lakh": 100_000, "More than 5 lakh": 500_000, "More than 10 lakh": 1_000_000}
WINDOW_MINUTES = (5, 15, 30, 60)


def _pct(value: float) -> str:
    return f"{value:g} %"


def _strategy_controls() -> tuple[StrategyConfig, RiskConfig]:
    st.sidebar.header("Strategy (PDF)")
    st.sidebar.caption("A buy or short needs every step. Otherwise the result is NO TRADE.")
    window = st.sidebar.selectbox(
        "Time frame (opening window)",
        WINDOW_MINUTES,
        index=WINDOW_MINUTES.index(15),
        format_func=lambda minutes: f"{minutes} minutes",
        help="The high and low of this window after the open set the PDF stop-loss. Orders are allowed once it ends. "
        "The PDF uses 15 minutes (09:15 to 09:30).",
    )
    sectors_each_side = int(st.sidebar.number_input("Sectors to use (top for buys, bottom for shorts)", min_value=1, max_value=9, value=3, step=1))
    stocks_per_sector = int(st.sidebar.number_input("Stocks per sector", min_value=1, max_value=10, value=5, step=1))
    min_change = st.sidebar.number_input("Minimum change from yesterday's close (%)", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
    volume_label = st.sidebar.selectbox("Volume today", list(VOLUME_CHOICES), index=3, help="The PDF uses more than 10 lakh shares.")
    entry_offset = st.sidebar.selectbox(
        "Buy at (% below the open; shorts above)", PERCENT_STEPS, index=0, format_func=_pct, help="The PDF uses 0.5 %."
    )
    stop_choice = st.sidebar.selectbox(
        "SL (stop-loss)",
        (None,) + PERCENT_STEPS,
        index=0,
        format_func=lambda value: "Time Frame Low / High (PDF)" if value is None else f"{value:g} % from the buy/sell price",
        help="The PDF puts the stop at the opening-window low for a buy and the high for a short.",
    )
    target = st.sidebar.selectbox(
        "Book profit (% from the buy/sell price)", PERCENT_STEPS, index=PERCENT_STEPS.index(1.5), format_func=_pct,
        help="The PDF uses 1.5 %.",
    )
    with st.sidebar.expander("Times"):
        observation = st.time_input("Opening window starts", value=datetime.strptime("09:15", "%H:%M").time())
        signal_start = (datetime.combine(datetime.now().date(), observation) + timedelta(minutes=window)).time()
        st.caption(f"The window ends and orders are allowed from {signal_start:%H:%M}.")
        no_entries_after = st.time_input(
            "No new entries after",
            value=datetime.strptime("15:00", "%H:%M").time(),
            help="Late entries have little time to reach the target and mostly pay costs. Open positions still run to the session end.",
        )
        session_end = st.time_input("Session ends", value=datetime.strptime("15:30", "%H:%M").time())
    try:
        strategy = StrategyConfig(
            sectors_each_side=sectors_each_side,
            stocks_per_sector=stocks_per_sector,
            min_change_pct=float(min_change),
            min_volume=VOLUME_CHOICES[volume_label],
            entry_offset_pct=float(entry_offset),
            target_pct=float(target),
            stop_pct=None if stop_choice is None else float(stop_choice),
            observation_start=observation,
            signal_start=signal_start,
            no_new_entries_after=no_entries_after,
            session_end=session_end,
        )
        st.session_state["strategy"] = strategy
    except ValueError as exc:
        st.sidebar.error(str(exc))
        strategy = st.session_state.get("strategy", StrategyConfig())

    st.sidebar.header("Risk")
    st.sidebar.caption(f"The stop-loss is {strategy.stop_text(True)} for a buy and {strategy.stop_text(False)} for a short.")
    max_trades = int(st.sidebar.number_input("Maximum hypothetical trades per day", min_value=1, max_value=20, value=5, step=1))
    notional = st.sidebar.number_input("Hypothetical notional per trade (INR)", min_value=10_000.0, max_value=10_000_000.0, value=100_000.0, step=10_000.0)
    try:
        risk = RiskConfig(max_daily_trades=max_trades, notional_inr=float(notional))
        st.session_state["risk"] = risk
    except ValueError as exc:
        st.sidebar.error(str(exc))
        risk = st.session_state.get("risk", RiskConfig())

    return strategy, risk


def _status_label(view: StockView) -> str:
    if view.status is not None:
        return view.status
    return "NO TRADE"


def _table(views: list[StockView]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Symbol": view.symbol,
                "Company": view.name,
                "Sector": view.sector,
                "Price": round(view.price, 2),
                "Prev close": round(view.prev_close, 2),
                "Change %": round(view.change_pct, 2),
                "Volume": view.volume,
                "VWAP": round(view.vwap, 2),
                "Signal": view.action,
                "Status": _status_label(view),
            }
            for view in views
        ]
    )


def _stat_row(items: list[tuple[str, str]]) -> None:
    """Label and value row that wraps instead of clipping long words."""
    columns = st.columns(len(items))
    for column, (label, value) in zip(columns, items):
        column.markdown(
            f"<p style='margin:0 0 0.15rem 0; font-size:0.85rem; opacity:0.75'>{label}</p>"
            f"<p style='margin:0 0 0.8rem 0; font-size:1.35rem; font-weight:650; "
            f"line-height:1.25; word-break:break-word'>{value}</p>",
            unsafe_allow_html=True,
        )


def _overview(engine: live_data.LiveEngine, views: list[StockView]) -> None:
    status = engine.status()
    st.subheader("Market overview")
    st.caption(
        "Change is measured from each stock's previous session close. Counts and sector ranks "
        "cover the tracked names only, not the whole exchange."
    )
    table = _table(views)
    advancing = int((table["Change %"] > 0).sum())
    declining = int((table["Change %"] < 0).sum())
    unchanged = int(len(table) - advancing - declining)
    sector_average = table.groupby("Sector")["Change %"].mean().sort_values(ascending=False)
    _stat_row(
        [
            ("Market status", "Closed" if engine.session_complete else "Open"),
            ("Tracked names", str(status.tracked_stocks)),
            ("Advancing", str(advancing)),
            ("Declining", str(declining)),
            ("Unchanged", str(unchanged)),
        ]
    )
    st.markdown(
        f"**Strongest sector:** {sector_average.index[0]} ({float(sector_average.iloc[0]):+.2f}%)  \n"
        f"**Weakest sector:** {sector_average.index[-1]} ({float(sector_average.iloc[-1]):+.2f}%)"
    )
    st.write(f"Market clock: **{status.clock:%Y-%m-%d %H:%M} IST** (latest closed minute)")
    st.write(f"Session date: **{status.session_date:%d %b %Y}**, from Zerodha Kite Connect.")
    st.write(f"Last data update: **{format_wall_clock(status.updated_at)}**")
    st.info(UNIVERSE_DISCLAIMER)


def _step_table(views: list[StockView], side: str) -> pd.DataFrame:
    """Step 11 of the PDF: the per-stock table."""
    buy = side == BUY
    rows = []
    for view in views:
        rows.append(
            {
                "Symbol": view.symbol,
                "Sector": view.sector,
                "Price": round(view.price, 2),
                "Previous day close": round(view.prev_close, 2),
                "Day high": round(view.day_high, 2),
                "Day low": round(view.day_low, 2),
                "Time Frame High": round(view.window_high, 2),
                "Time Frame Low": round(view.window_low, 2),
                "% change from previous day": round(view.change_pct, 2),
                "Volume": view.volume,
                "VWAP": round(view.vwap, 2),
                "3-day % change": round(view.three_day_pct, 2),
                ("Suggested buy" if buy else "Suggested sell (short)"): round(view.order_price, 2),
                ("Suggested sell" if buy else "Target (buy-back)"): round(view.target, 2),
                "SL": round(view.stop, 2),
                "Status": view.status or ("Every step passed" if view.action != NO_TRADE else f"No: {view.failed[0].rule_id}" if view.failed else "No"),
            }
        )
    return pd.DataFrame(rows)


def _rule_lists(view: StockView) -> None:
    if view.passed:
        st.markdown("**Steps that passed**")
        for rule in view.passed:
            st.write(f"✓ {rule.rule_id}: {rule.detail}")
    if view.failed:
        st.markdown("**Steps that did not pass**")
        for rule in view.failed:
            st.write(f"✗ {rule.rule_id}: {rule.detail}")


def _strategy_steps(strategy: StrategyConfig, risk: RiskConfig) -> None:
    with st.expander("The strategy steps (from the PDF)"):
        st.write(
            f"1. Rank sectors by average change from yesterday's close. Buys use the top {strategy.sectors_each_side}, "
            f"shorts the bottom {strategy.sectors_each_side}.  \n"
            f"2. In each, take the top {strategy.stocks_per_sector} gainers (buys) or losers (shorts).  \n"
            "3. Note the 09:15 opening price.  \n"
            f"4. Change from yesterday's close above +{strategy.min_change_pct:g}% (buy) or below -{strategy.min_change_pct:g}% (short).  \n"
            f"5. Volume today above {strategy.min_volume:,} shares.  \n"
            f"6. At {strategy.signal_start:%H:%M}, mark the high and low since {strategy.observation_start:%H:%M}.  \n"
            f"7. Buy at the open minus {strategy.entry_offset_pct:g}% (short: open plus {strategy.entry_offset_pct:g}%). "
            "The order fills only when a later candle trades at that price.  \n"
            f"8. Stop-loss at {strategy.stop_text(True)} (short: {strategy.stop_text(False)}).  \n"
            f"9. Target {strategy.target_pct:g}% above the buy price (short: below the sell price).  \n"
            "10. Buy only above VWAP; short only below VWAP.  \n"
            f"No new entries after {strategy.no_new_entries_after:%H:%M}. Everything closes at {strategy.session_end:%H:%M}. "
            f"At most {risk.max_daily_trades} trades a day, one per stock."
        )


def _detail(engine: live_data.LiveEngine, strategy: StrategyConfig, risk: RiskConfig, views: list[StockView]) -> None:
    st.subheader("Stock detail")
    st.caption(
        "One-minute candles from Zerodha. Orders are decided on closed candles and fill only when a later "
        "candle reaches the price."
    )
    symbols = [view.symbol for view in views]
    default = next((v.symbol for v in views if v.status is not None), "RELIANCE")
    selected = st.selectbox("Symbol", options=symbols, index=symbols.index(default) if default in symbols else 0)
    view = next(v for v in views if v.symbol == selected)
    frame = engine.visible(selected)
    st.plotly_chart(_chart(frame, strategy, view), width="stretch")
    _stat_row(
        [
            ("Price", f"{view.price:,.2f}"),
            ("Change from yesterday", f"{view.change_pct:+.2f}%"),
            ("Volume today", f"{view.volume:,}"),
            ("VWAP", f"{view.vwap:,.2f}"),
            ("Signal", view.status or view.action),
        ]
    )
    st.markdown("**Step 11 table**")
    st.dataframe(_step_table([view], view.side), width="stretch", hide_index=True)
    st.markdown(f"**{view.summary}**")
    _rule_lists(view)
    _strategy_steps(strategy, risk)
    word = "buy" if view.side == BUY else "short-sell"
    quantity = position_size(view.order_price, risk.notional_inr)
    st.info(
        f"Hypothetical {word} plan: order at {view.order_price:.2f}, stop-loss {view.stop:.2f} "
        f"({strategy.stop_text(view.side == BUY)}), target {view.target:.2f}. "
        f"Quantity {quantity} at a notional of {risk.notional_inr:,.0f} INR. This is not an order."
    )
    for note in RISK_LIMITATIONS:
        st.caption(note)
    recent = frame.tail(15).copy()
    recent["time"] = pd.to_datetime(recent["timestamp"]).dt.tz_convert(IST).dt.strftime("%H:%M")
    st.markdown("**Latest candles**")
    st.dataframe(recent[["time", "open", "high", "low", "close", "volume"]], width="stretch", hide_index=True)


def _chart(frame: pd.DataFrame, strategy: StrategyConfig, view: StockView) -> go.Figure:
    stamps = pd.to_datetime(frame["timestamp"]).dt.tz_convert(IST)
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    vwap = (typical * frame["volume"]).cumsum() / frame["volume"].cumsum()
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
    figure.add_trace(
        go.Candlestick(x=stamps, open=frame["open"], high=frame["high"], low=frame["low"], close=frame["close"], name="OHLC"),
        row=1, col=1,
    )
    figure.add_trace(go.Scatter(x=stamps, y=vwap, name="VWAP", line={"width": 1.6}), row=1, col=1)
    figure.add_trace(go.Bar(x=stamps, y=frame["volume"], name="Volume", marker_color="#7f8c9b"), row=2, col=1)
    window = stamps[(stamps.dt.time >= strategy.observation_start) & (stamps.dt.time < strategy.signal_start)]
    if len(window) and view.window_high == view.window_high:
        figure.add_shape(
            type="rect", x0=window.iloc[0], x1=window.iloc[-1] + pd.Timedelta(minutes=1),
            y0=view.window_low, y1=view.window_high, fillcolor="#7f93b5", opacity=0.18, line_width=0, row=1, col=1,
        )
    buy = view.side == BUY
    levels = [
        (float(frame["open"].iloc[0]), "#8b939e", "Open"),
        (view.order_price, "#3ecf8e", f"{'Buy' if buy else 'Sell'} {'-' if buy else '+'}{strategy.entry_offset_pct:g}%"),
        (view.target, "#3ecf8e", f"Target {'+' if buy else '-'}{strategy.target_pct:g}%"),
        (view.stop, "#f0716f", f"Stop {strategy.window_minutes}m {'low' if buy else 'high'}" if strategy.stop_pct is None
         else f"Stop {'-' if buy else '+'}{strategy.stop_pct:g}%"),
    ]
    for price, color, label in levels:
        if price == price:
            figure.add_hline(y=price, line_dash="dash", line_color=color, annotation_text=label, row=1, col=1)
    entry_time = view.trade.entry_time if view.trade else view.position.entry_time if view.position else None
    if entry_time is not None:
        figure.add_trace(
            go.Scatter(x=[entry_time], y=[view.order_price], mode="markers", name="Hypothetical fill",
                       marker={"symbol": "triangle-up" if buy else "triangle-down", "size": 12, "color": "#3ecf8e" if buy else "#f0716f"}),
            row=1, col=1,
        )
    figure.update_layout(
        template="plotly_dark",
        height=620,
        title=f"One-minute candles, VWAP, {strategy.window_minutes}-minute range, and order levels",
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h"},
        margin=dict(l=10, r=10, t=50, b=10),
    )
    figure.update_yaxes(title_text="Price", row=1, col=1)
    figure.update_yaxes(title_text="Volume", row=2, col=1)
    return figure


def _history_progress(market: live_data.LiveMarket) -> None:
    """Show the background history download and redraw the page once it finishes."""

    def _show() -> None:
        if getattr(market, "history_state", "") != "running":
            st.rerun()
        share = market.history_done / market.history_total if market.history_total else 0.0
        st.progress(share, text=market.history_message or "Downloading history...")
        st.caption(
            "Downloading the last 60 days of one-minute history from Zerodha in the background, three stocks "
            "at a time. It started when you logged in. You can use the other tabs meanwhile."
        )

    st.fragment(run_every=timedelta(seconds=2))(_show)()


def _backtest(engine: live_data.LiveEngine, strategy: StrategyConfig, risk: RiskConfig, views: list[StockView]) -> None:
    market = engine.market
    longest = max(backtest_board.PERIOD_DAYS)
    # A market loaded by an older version of this module may not have the history store yet.
    store = market.__dict__.setdefault("history", {})
    history = store.get(longest)
    downloading = history is None and getattr(market, "history_state", "") == "running"
    screener = [row.view.symbol for row in board.opportunity_rows(views, st.session_state.get("opp_filter") or "All")]
    if st.session_state.get("mbt_stock") not in ["All stocks"] + screener:
        st.session_state["mbt_stock"] = "All stocks"
    c0, c1, c2, c3 = st.columns([2, 1.6, 1.6, 1.3], vertical_alignment="bottom")
    stock = c0.selectbox(
        "Stock",
        ["All stocks"] + screener,
        key="mbt_stock",
        help="The stocks in the Screener table on the Opening-range board. A single stock uses the full PDF rules, "
        "so it only trades on days it was in the top sectors and a top mover of its sector.",
    )
    symbol = None if stock == "All stocks" else stock
    bar_label = c1.selectbox("Candle size", ["5-minute", "1-minute"], key="mbt_bars")
    headline_days = c2.selectbox(
        "Period for cards, curve, and grid",
        list(backtest_board.PERIOD_DAYS),
        index=len(backtest_board.PERIOD_DAYS) - 1,
        format_func=lambda days: f"Last {days} days",
        key="mbt_days",
    )
    run_clicked = c3.button("Run backtest", type="primary", width="stretch", disabled=downloading)
    with st.expander("Cost assumptions (paper accounting, not a Zerodha tariff)"):
        cost_cols = st.columns(3)
        brokerage = cost_cols[0].number_input("Brokerage INR per order", min_value=0.0, value=20.0, step=1.0)
        gst = cost_cols[1].number_input("GST % on brokerage and exchange assumption", min_value=0.0, value=18.0, step=1.0)
        exchange = cost_cols[2].number_input("Exchange charge %", min_value=0.0, value=0.003, step=0.001, format="%.4f")
        cost_cols_b = st.columns(3)
        stt = cost_cols_b[0].number_input("STT % on sell value", min_value=0.0, value=0.025, step=0.001, format="%.4f")
        stamp = cost_cols_b[1].number_input("Stamp duty % on buy value", min_value=0.0, value=0.003, step=0.001, format="%.4f")
        slippage = cost_cols_b[2].number_input("Slippage % per fill", min_value=0.0, value=0.05, step=0.01)
    try:
        costs = CostAssumptions(
            brokerage_inr_per_order=float(brokerage),
            gst_pct=float(gst),
            exchange_charge_pct=float(exchange),
            stt_sell_pct=float(stt),
            stamp_duty_buy_pct=float(stamp),
            slippage_pct=float(slippage),
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    bar_minutes = 5 if bar_label.startswith("5") else 1
    key = json.dumps(
        {
            "bars": bar_minutes,
            "days": headline_days,
            "symbol": symbol,
            "market": id(market),
            "strategy": {name: str(value) for name, value in asdict(strategy).items()},
            "risk": {name: str(value) for name, value in asdict(risk).items()},
            "costs": asdict(costs),
        },
        sort_keys=True,
    )
    if downloading:
        _history_progress(market)
    elif history is None and getattr(market, "history_state", "") == "error":
        st.warning(f"{market.history_message} Press Run backtest to try again.")
    if run_clicked:
        progress = st.progress(0.0, text="Preparing...")
        try:
            if history is None:
                history = live_data.load_history(
                    market,
                    longest,
                    progress=lambda done, total, text: progress.progress(done / max(total, 1), text=text),
                )
                store[longest] = history
            if not history:
                raise RuntimeError("Zerodha returned no past sessions for this period.")
            result = backtest_board.run_history_backtest(
                history,
                strategy,
                risk,
                costs,
                bar_minutes=bar_minutes,
                headline_days=int(headline_days),
                progress=lambda done, total: progress.progress(done / total, text=f"Replaying session {done} of {total}"),
                symbol=symbol,
            )
            st.session_state["mbt_result"] = (key, result)
            st.session_state.pop("mbt_error", None)
        except Exception as exc:
            st.session_state["mbt_error"] = str(exc)
        progress.empty()
    if st.session_state.get("mbt_error"):
        st.error(f"The backtest could not finish: {st.session_state['mbt_error']}")
    stored = st.session_state.get("mbt_result")
    if stored is None:
        if history:
            st.info(
                f"History is ready ({len(history)} sessions). Press Run backtest; replaying takes a few seconds. "
                "It replays every stock on each session with the PDF strategy, then repeats each session for 9 "
                "target and entry settings."
            )
        elif not downloading:
            st.info(
                f"Press Run backtest to download the last {longest} days of history from Zerodha and replay it. "
                "History is kept in memory until the app restarts."
            )
        return
    stored_key, result = stored
    if stored_key != key:
        st.caption("Settings changed since this result was made. Press Run backtest to refresh it.")
    st.html(backtest_board.render_backtest(result))
    with st.expander(f"Trades in the last {result.headline_days} days ({len(result.trades)})"):
        if result.trades:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Session": day.isoformat(),
                            "Symbol": trade.symbol,
                            "Side": trade.side,
                            "Entry time": trade.entry_time.strftime("%H:%M"),
                            "Exit time": trade.exit_time.strftime("%H:%M"),
                            "Entry": round(trade.entry_raw, 2),
                            "Stop-loss": round(trade.stop_loss, 2),
                            "Target": round(trade.target, 2),
                            "Exit": round(trade.exit_raw, 2),
                            "Exit reason": exit_reason_text(trade.exit_reason),
                            "Net P&L": round(trade.net_pnl, 2),
                        }
                        for day, trade in result.trades
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
        else:
            who = result.symbol or "Any stock"
            st.write(f"No hypothetical trades. {who} got NO TRADE on every session in this period.")
    if result.skipped_by_limit:
        st.caption(f"{result.skipped_by_limit} fills were skipped because the daily trade limit was already reached.")


if __name__ == "__main__":
    main()
else:
    main()
