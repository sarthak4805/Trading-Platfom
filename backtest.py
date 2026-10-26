"""Paper-trade accounting shared by the live board and the backtest.

Slippage moves every fill against the position once. Charges are computed on the
slipped fills. Gross P&L uses the raw order prices; net P&L uses the slipped
fills minus charges. When a candle touches both the stop and the target, the
stop is assumed first because the tick path is unknown. A candle that opens
beyond a level fills at that open.

Fee inputs are assumptions. They are not a current Zerodha tariff. Past results
do not predict future profitability.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

BUY = "BUY"
SHORT = "SHORT-SELL"

SYNTHETIC_DISCLAIMER = (
    "Synthetic-data test. These results do not predict real-world profitability."
)


@dataclass
class CostAssumptions:
    """Paper-trading cost inputs.

    These are user-editable assumptions for accounting practice. They are not the
    official Zerodha brokerage schedule and they are not a statutory tax ruling.
    Replace every figure from your broker's current tariff and the current tax
    rules before you draw a conclusion.

    Percentage fields are percentage points. ``slippage_pct=0.05`` means 0.05%
    adverse movement on each fill. ``gst_pct=18`` means 18% of brokerage plus
    the exchange-charge assumption.
    """

    brokerage_inr_per_order: float = 20.0
    gst_pct: float = 18.0
    exchange_charge_pct: float = 0.003
    stt_sell_pct: float = 0.025
    stamp_duty_buy_pct: float = 0.003
    slippage_pct: float = 0.05

    def __post_init__(self) -> None:
        values = {
            "brokerage": self.brokerage_inr_per_order,
            "GST": self.gst_pct,
            "exchange charge": self.exchange_charge_pct,
            "STT": self.stt_sell_pct,
            "stamp duty": self.stamp_duty_buy_pct,
            "slippage": self.slippage_pct,
        }
        for name, value in values.items():
            if value < 0:
                raise ValueError(f"Cost assumption '{name}' cannot be negative.")


@dataclass
class Trade:
    """One closed hypothetical trade."""

    symbol: str
    side: str
    signal_index: int
    entry_index: int
    exit_index: int
    signal_time: datetime
    entry_time: datetime
    exit_time: datetime
    entry_raw: float
    exit_raw: float
    entry_fill: float
    exit_fill: float
    stop_loss: float
    target: float
    quantity: int
    exit_reason: str
    gross_pnl: float
    slippage_cost: float
    transaction_cost: float
    net_pnl: float


def apply_slippage(price: float, side: str, *, is_entry: bool, slippage_pct: float) -> float:
    """Move a fill against the hypothetical position."""
    if side not in (BUY, SHORT):
        raise ValueError("Slippage is only defined for BUY or SHORT-SELL.")
    rate = slippage_pct / 100.0
    if side == BUY:
        multiplier = 1.0 + rate if is_entry else 1.0 - rate
    else:
        multiplier = 1.0 - rate if is_entry else 1.0 + rate
    return float(price) * multiplier


def transaction_charges(
    entry_fill: float,
    exit_fill: float,
    quantity: int,
    side: str,
    costs: CostAssumptions,
) -> float:
    """Brokerage, GST, and the configured exchange, STT, and stamp assumptions.

    Charges use the slipped fill prices, which is the value that would have traded
    in this paper model. The rates themselves remain assumptions.
    """
    if quantity < 1:
        raise ValueError("Quantity must be at least 1 share to compute charges.")
    if side == BUY:
        buy_value = entry_fill * quantity
        sell_value = exit_fill * quantity
    elif side == SHORT:
        buy_value = exit_fill * quantity
        sell_value = entry_fill * quantity
    else:
        raise ValueError("Charges are only defined for BUY or SHORT-SELL.")
    brokerage = costs.brokerage_inr_per_order * 2.0
    exchange = (buy_value + sell_value) * (costs.exchange_charge_pct / 100.0)
    stt = sell_value * (costs.stt_sell_pct / 100.0)
    stamp = buy_value * (costs.stamp_duty_buy_pct / 100.0)
    gst = (brokerage + exchange) * (costs.gst_pct / 100.0)
    return brokerage + exchange + stt + stamp + gst


def trade_pnl(
    side: str,
    entry_raw: float,
    exit_raw: float,
    quantity: int,
    costs: CostAssumptions,
) -> dict[str, float]:
    """Gross price P&L, slippage drag, charges, and net P&L.

    Gross P&L uses the raw order prices before slippage and before charges.
    Net P&L uses adverse fills and then subtracts the charge assumptions.
    """
    entry_fill = apply_slippage(entry_raw, side, is_entry=True, slippage_pct=costs.slippage_pct)
    exit_fill = apply_slippage(exit_raw, side, is_entry=False, slippage_pct=costs.slippage_pct)
    direction = 1.0 if side == BUY else -1.0
    gross = direction * (exit_raw - entry_raw) * quantity
    slipped = direction * (exit_fill - entry_fill) * quantity
    charges = transaction_charges(entry_fill, exit_fill, quantity, side, costs)
    return {
        "entry_fill": entry_fill,
        "exit_fill": exit_fill,
        "gross_pnl": gross,
        "slippage_cost": gross - slipped,
        "transaction_cost": charges,
        "net_pnl": slipped - charges,
    }


def resolve_exit(
    side: str,
    stop: float,
    target: float,
    bar: Mapping[str, float],
) -> tuple[str | None, float | None]:
    """Exit inside one candle. A gap through a level fills at the open. Both levels: stop first."""
    open_ = float(bar["open"])
    high = float(bar["high"])
    low = float(bar["low"])
    if side == BUY:
        if open_ <= stop:
            return "stop_loss_gap", open_
        if open_ >= target:
            return "target_gap", open_
        hit_stop = low <= stop
        hit_target = high >= target
    elif side == SHORT:
        if open_ >= stop:
            return "stop_loss_gap", open_
        if open_ <= target:
            return "target_gap", open_
        hit_stop = high >= stop
        hit_target = low <= target
    else:
        raise ValueError("Exit checks are only defined for BUY or SHORT-SELL.")
    if hit_stop:
        return "stop_loss", stop
    if hit_target:
        return "target", target
    return None, None


def max_drawdown(net_pnls: list[float]) -> float:
    """Largest peak-to-trough drop of the closed-trade equity curve, in currency."""
    equity = 0.0
    peak = 0.0
    worst = 0.0
    for pnl in net_pnls:
        equity += float(pnl)
        if equity > peak:
            peak = equity
        worst = max(worst, peak - equity)
    return worst


def performance_stats(gross: list[float], net: list[float]) -> dict[str, float | int | str | None]:
    """Win rate, profit factor, and drawdown from closed-trade P&L."""
    if len(gross) != len(net):
        raise ValueError("Gross and net P&L lists must describe the same trades.")
    if not net:
        return {
            "trade_count": 0,
            "win_rate": None,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "profit_factor": None,
            "profit_factor_note": "Not calculable because there were no closed trades.",
            "max_drawdown": 0.0,
        }
    wins = [value for value in net if value > 0]
    losses = [value for value in net if value < 0]
    if losses:
        profit_factor = sum(wins) / abs(sum(losses))
        note = "Profit factor uses net trade results (after slippage and assumed charges)."
    else:
        profit_factor = None
        note = "Not calculable because there were no losing trades."
    return {
        "trade_count": len(net),
        "win_rate": len(wins) / len(net),
        "gross_pnl": float(sum(gross)),
        "net_pnl": float(sum(net)),
        "profit_factor": profit_factor,
        "profit_factor_note": note,
        "max_drawdown": max_drawdown(net),
    }


def exit_reason_text(reason: str) -> str:
    """Beginner-facing name for an exit code."""
    labels = {
        "stop_loss": "Stopped out at the stop-loss",
        "stop_loss_gap": "Stopped out at the open because price gapped through the stop",
        "target": "Exited at the target",
        "target_gap": "Exited at the open because price gapped through the target",
        "session_close": "Closed at the end of the session",
    }
    return labels.get(reason, reason)
