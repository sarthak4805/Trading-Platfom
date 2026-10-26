"""Paper position limits: size, daily trade count, and intraday-only holds.

The stop-loss and target come from the strategy (the 15-minute low or high, and
a percent target from the order price). Nothing here places an order.
"""

from __future__ import annotations

from dataclasses import dataclass


RISK_LIMITATIONS: tuple[str, ...] = (
    "The stop-loss is the 15-minute low for a buy or the 15-minute high for a short. It is a paper-trading rule, not a guarantee.",
    "A gap can open through the stop-loss. When that happens, the replay fills at the open rather than at the stop.",
    "On the candle where an order fills, only the stop can trigger. After that, a candle that touches both the stop and the target counts as a stop.",
    "Slippage and transaction costs are configurable assumptions. They are not a verified Zerodha fee schedule.",
    "Hypothetical positions are intraday only and are closed at the end of the session.",
    "One trade per stock per day. Averaging down is not allowed.",
    "No real order is created or sent.",
)


@dataclass
class RiskConfig:
    """Paper position limits."""

    max_daily_trades: int = 5
    intraday_only: bool = True
    allow_averaging_down: bool = False
    notional_inr: float = 100_000.0

    def __post_init__(self) -> None:
        if self.max_daily_trades < 1:
            raise ValueError("Maximum daily trades must be at least 1.")
        if not self.intraday_only:
            raise ValueError(
                "This platform allows hypothetical intraday positions only. "
                "Overnight holds are disabled."
            )
        if self.allow_averaging_down:
            raise ValueError(
                "Averaging down is disabled and cannot be turned on in this build."
            )
        if self.notional_inr <= 0:
            raise ValueError("Hypothetical notional must be greater than zero.")


def position_size(entry: float, notional_inr: float) -> int:
    """Whole-share quantity for a hypothetical notional. At least one share."""
    entry_price = float(entry)
    notional = float(notional_inr)
    if entry_price <= 0:
        raise ValueError("Entry price must be positive to size a hypothetical position.")
    if notional <= 0:
        raise ValueError("Notional must be positive to size a hypothetical position.")
    return max(1, int(notional // entry_price))


def can_open_new_trade(open_positions: int, trades_opened_today: int, config: RiskConfig) -> tuple[bool, str]:
    """Return whether a new hypothetical position is allowed, and why."""
    if open_positions < 0 or trades_opened_today < 0:
        raise ValueError("Position and trade counts cannot be negative.")
    if open_positions > 0:
        return (
            False,
            "A hypothetical position is already open. Adding to it, including averaging down, is not allowed.",
        )
    if trades_opened_today >= config.max_daily_trades:
        return (
            False,
            f"The daily hypothetical trade limit of {config.max_daily_trades} has been reached.",
        )
    return True, "Within the configured daily trade limit, and no position is open."
