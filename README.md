# NIFTY 100 Intraday Research Desk

A local Streamlit desk that runs the bullish / bearish strategy from the strategy PDF on real NSE data from Zerodha Kite Connect: sector screen, opening range, offset entries, VWAP filter, stop at the 15-minute low or high, and a backtest on past sessions.

It is **read-only**. It downloads the instrument list, historical candles, and the live WebSocket feed. It never places, modifies, or cancels an order. Every trade on screen is a paper trade.

## What you need

- Python 3.10 (verified on this PC). The `py` launcher's 3.11 and 3.14 are ARM64 builds that could not install pandas without a compiler.
- A **paid Kite Connect** app. The free Personal plan does not include market data.

## Setup (Windows PowerShell)

```powershell
cd "C:\Users\MRUNALI\OneDrive\Desktop\trade"
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest
streamlit run app.py
```

If activation is blocked, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, or skip activation and use `.\.venv\Scripts\python.exe -m streamlit run app.py`.

Copy `.env.example` to `.env` and put your Kite Connect `KITE_API_KEY` and `KITE_API_SECRET` in it. `.env` is gitignored and the app never shows either value.

## Logging in to Zerodha

1. In the Kite developer console, set the app's **Redirect URL** to the address you open the app at, for example `http://127.0.0.1:8501/`. Leave the Postback URL empty.
2. Open the app at that exact address and press **Log in with Zerodha**.
3. Zerodha sends you back with a one-time code. The app exchanges it for today's access token and keeps it in memory only. Log in again each day.
4. The app downloads the NSE instrument list, then for each stock two weeks of daily candles (previous close and the close three sessions back) and the session's one-minute candles. Kite allows about 3 history requests a second, so this takes about a minute.

During market hours the app subscribes to the Kite WebSocket feed and builds the current one-minute candle from ticks; the page refreshes when a minute closes, and only closed minutes are used. Before 09:16, on weekends, and on holidays it shows the most recent complete session. **Reload Zerodha data** downloads again (use it after 09:15 if you opened the app earlier). **Log out of Zerodha** forgets the token. Symbols Zerodha does not recognise are listed in the sidebar and left out.

## How the desk is organized

The sidebar holds the Zerodha status, the strategy settings from the PDF, the paper-risk limits, and the movers-sector picker. It starts closed; open it with the arrow at the top left.

### Opening-range board

- A ticker strip with NIFTY 100, NIFTY BANK, and INDIA VIX from Zerodha, and the market clock. If Zerodha returns no index candles, an equal-weight average of the tracked names stands in, labelled "eq-wt".
- Six cards: market breadth, best and weakest sector, active setups (waiting orders and open trades, with the daily trade limit), daily paper P&L after default costs, and risk used (open stop distance times quantity, as a share of notional × daily trade limit).
- **Top opportunities**: candidates from the sector screen plus anything that traded today, with entry, stop, target, reward-to-risk, a signal (BUY or SHORT while in a trade, WAIT for a waiting order, TARGET, STOPPED, CLOSED, or NO TRADE), and how many PDF steps passed. The filter switches between All, Long, Short, and Waiting. Tap a row to open that stock in the chart and trade plan on the right. The step count is not a probability.
- The chart for the chosen stock in 5-minute or 1-minute candles, with EMA 9 and 21, VWAP, the shaded 15-minute range, and the entry, stop, and target. The EMA lines are for reading the picture only. Below it, the trade plan and a **Review setup** button that lists every step.
- **Sector momentum** for the five strongest and five weakest sectors today, and **Sector performance** and **movers** cards at the bottom. The movers buyer-share bar is estimated from up-minute volume; it is not the order book.

### Market overview, Stock screener, Sector analysis

Breadth, strongest and weakest sector, a searchable and sortable table of every tracked stock, top gainers and losers, and the sector ranking used in step 1. Change is always measured from the previous session's close.

### Stock detail

One stock's one-minute candles with VWAP, volume, the 15-minute range, and the order, stop, and target lines; its step-11 table; every step that passed or failed; and the paper plan with its quantity.

### Signals

Waiting buy and short orders, open and closed trades, the daily limit, and the step-11 tables for bullish candidates (top gainers in the strongest sectors) and bearish candidates (top losers in the weakest sectors). Pick any stock to read its steps.

### Backtesting

Pick 5-minute or 1-minute candles and the last 30 or 60 days, then click **Run backtest**. Right after you log in, the app starts downloading every stock's one-minute and daily history for the last 60 days in the background, three stocks at a time within Kite's request limit (about a minute or two). The Backtesting tab shows the progress, and once it finishes a run only replays the data, which takes a few seconds. History stays in memory until the app restarts; nothing is written to disk. Each session is screened and replayed with the sidebar settings. The page shows net P&L after costs, win rate, maximum drawdown, profit factor, an equity curve, a 30 / 60-day comparison, and a target × entry-offset grid. Choosing the best grid cell on the same data is overfitting.

## Strategy (from the PDF)

Every step is checked on closed candles only. All must pass, otherwise the result is **NO TRADE**.

**Bullish (buy)**

| Step | Rule | Default |
| --- | --- | --- |
| 1 | The stock's sector is one of the strongest by average change from the previous close | top 3 sectors |
| 2 | The stock is one of the top gainers in its sector | top 5 |
| 3 | Note the opening price | 09:15 open |
| 4 | Change from the previous close is above the minimum | more than +1.5% |
| 5 | Volume so far today is above the minimum | more than 10 lakh |
| 6 | Mark the 15-minute high and low | 09:15 to 09:30 |
| 7 | Suggested buy price = open − offset. The order fills only when a later candle trades at that price | 0.5% below the open |
| 8 | Stop-loss = the 15-minute low. If that low is not below the buy price, there is no trade | |
| 9 | Target = buy price + target % | 1.5% |
| 10 | The price and the buy price are both above VWAP | |

**Bearish (short)** is the mirror: the weakest sectors, the top losers, a fall of more than 1.5%, sell at the open + 0.5%, stop at the 15-minute high, target 1.5% below the sell price, and price and sell price both below VWAP. Intraday shorts are always squared off the same day.

Timing:

- A step result on candle `t` places an order for candle `t + 1`. The order fills at exactly the suggested price, and only if that candle's range includes it.
- Orders are allowed from 09:30. No new entries after 15:00. Everything still open closes at 15:30.
- On the fill candle only the stop can trigger, because the order of prices inside that candle is unknown. After that, if a candle touches both the stop and the target, the stop is assumed first. A candle that opens beyond a level fills at that open.
- One trade per stock per day. At most 5 trades per day across all stocks, taken in the order they fill (alphabetical on a tie).

All thresholds and times are editable in the sidebar. VWAP uses the typical price `(high + low + close) / 3`.

## Risk and costs

- Hypothetical size is whole shares of the sidebar notional (default 100,000 INR), with at least one share.
- Positions are intraday only. Averaging down cannot be turned on. No real order is created.
- Gross P&L uses the order prices. Slippage moves each fill against the trade once. Charges are computed on the slipped fills and subtracted for net P&L.
- Maximum drawdown is the largest peak-to-trough drop of cumulative net P&L. Profit factor is winning net divided by losing net, blank when there are no losses. Win rate counts a trade as a win only when net P&L is positive.

The default charge inputs are round teaching assumptions: 20 INR brokerage per order, 18% GST on brokerage plus the exchange-charge assumption, 0.003% exchange charge, 0.025% STT on sell value, 0.003% stamp duty on buy value, and 0.05% slippage per fill. **They are not a Zerodha tariff and they are not tax advice.**

## Layout

```
app.py                  Streamlit desk (Zerodha login, tabs, sidebar)
live_data.py            Zerodha login, session candles, live ticks, backtest history (read-only)
board.py                Opening-range board (data and HTML)
backtest_board.py       Backtest on past Zerodha sessions and its page
strategy.py             PDF strategy: screen, steps, order fills, exits
market_data.py          Market clock constants (IST, 09:15 open, 15:30 close)
indicators.py           Candle checks, percent change, VWAP
risk_manager.py         Size, daily limit, intraday-only rules
backtest.py             Paper P&L, slippage, charges, stop-first exits
universe.py             The 100 tracked names and their sectors
tests/                  Strategy, board, backtest, live-data, risk, and indicator tests
```

Tests use a small test-only candle builder and a fake Kite client in `tests/`, so they run without a Zerodha login. Market data stays in memory; the app does not write CSV files or use a database.

## Limits

- The stock list and sector labels are a working sample, not a verified copy of the current NIFTY 100.
- One-minute candles do not reveal which level was hit first inside a minute, so the replay assumes the stop.
- Candles built from the live feed can differ slightly from Zerodha's own minute candles.
- Costs are assumptions until you verify them against your broker's current charges.
- Past results do not predict future results. Nothing here is a recommendation to buy or sell.
