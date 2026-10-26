"""Tests for paper position limits."""

from __future__ import annotations

import pytest

from risk_manager import RISK_LIMITATIONS, RiskConfig, can_open_new_trade, position_size


def test_averaging_down_cannot_be_enabled() -> None:
    with pytest.raises(ValueError, match="Averaging down"):
        RiskConfig(allow_averaging_down=True)


def test_overnight_holds_are_rejected() -> None:
    with pytest.raises(ValueError, match="intraday"):
        RiskConfig(intraday_only=False)


def test_open_position_blocks_a_new_trade() -> None:
    allowed, message = can_open_new_trade(1, 0, RiskConfig())
    assert allowed is False
    assert "Averaging" in message or "already open" in message


def test_daily_limit_blocks_a_new_trade() -> None:
    allowed, message = can_open_new_trade(0, 5, RiskConfig(max_daily_trades=5))
    assert allowed is False
    assert "5" in message
    allowed_again, _message = can_open_new_trade(0, 4, RiskConfig(max_daily_trades=5))
    assert allowed_again is True


def test_position_size_is_whole_shares_and_at_least_one() -> None:
    assert position_size(250, 100_000) == 400
    assert position_size(150_000, 100_000) == 1


def test_limitations_mention_the_stop_gaps_and_no_orders() -> None:
    text = " ".join(RISK_LIMITATIONS).lower()
    assert "15-minute low" in text
    assert "gap" in text
    assert "not a verified" in text
    assert "no real order" in text
