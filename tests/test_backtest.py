"""Tests for stop-first exits and paper P&L accounting."""

from __future__ import annotations

import pytest

from backtest import (
    CostAssumptions,
    apply_slippage,
    exit_reason_text,
    max_drawdown,
    performance_stats,
    resolve_exit,
    trade_pnl,
    transaction_charges,
)


def test_drawdown_from_a_known_equity_path() -> None:
    assert max_drawdown([100, -250]) == pytest.approx(250)
    assert max_drawdown([]) == 0
    assert max_drawdown([10, 5]) == 0


def test_profit_factor_is_missing_when_there_is_no_loss() -> None:
    stats = performance_stats([10.0], [8.0])
    assert stats["profit_factor"] is None
    assert "not calculable" in str(stats["profit_factor_note"]).lower()
    both = performance_stats([10.0, -4.0], [8.0, -5.0])
    assert both["profit_factor"] == pytest.approx(8 / 5)
    assert both["win_rate"] == pytest.approx(0.5)


def test_transaction_charges_match_the_assumption_formula() -> None:
    costs = CostAssumptions(
        brokerage_inr_per_order=20,
        gst_pct=18,
        exchange_charge_pct=0.003,
        stt_sell_pct=0.025,
        stamp_duty_buy_pct=0.003,
        slippage_pct=0,
    )
    charges = transaction_charges(100, 100, 10, "BUY", costs)
    brokerage = 40
    exchange = 2000 * 0.00003
    stt = 1000 * 0.00025
    stamp = 1000 * 0.00003
    gst = (brokerage + exchange) * 0.18
    assert charges == pytest.approx(brokerage + exchange + stt + stamp + gst)
    money = trade_pnl("BUY", 100, 100, 10, costs)
    assert money["gross_pnl"] == pytest.approx(0)
    assert money["slippage_cost"] == pytest.approx(0)
    assert money["net_pnl"] == pytest.approx(-charges)


def test_slippage_and_charges_reduce_a_winning_long() -> None:
    money = trade_pnl("BUY", 100, 101, 10, CostAssumptions())
    assert money["gross_pnl"] == pytest.approx(10)
    assert money["net_pnl"] < money["gross_pnl"]
    assert money["slippage_cost"] > 0
    assert money["transaction_cost"] > 0


def test_slippage_is_counted_once() -> None:
    costs = CostAssumptions(brokerage_inr_per_order=0, gst_pct=0, exchange_charge_pct=0, stt_sell_pct=0, stamp_duty_buy_pct=0)
    money = trade_pnl("BUY", 100, 101, 10, costs)
    assert money["net_pnl"] == pytest.approx(money["gross_pnl"] - money["slippage_cost"])


def test_short_profit_is_positive_when_price_falls() -> None:
    money = trade_pnl(
        "SHORT-SELL",
        100,
        99,
        10,
        CostAssumptions(slippage_pct=0, brokerage_inr_per_order=0, gst_pct=0, exchange_charge_pct=0, stt_sell_pct=0, stamp_duty_buy_pct=0),
    )
    assert money["gross_pnl"] == pytest.approx(10)
    assert money["net_pnl"] == pytest.approx(10)


def test_slippage_is_adverse() -> None:
    assert apply_slippage(100, "BUY", is_entry=True, slippage_pct=0.05) == pytest.approx(100.05)
    assert apply_slippage(100, "BUY", is_entry=False, slippage_pct=0.05) == pytest.approx(99.95)
    assert apply_slippage(100, "SHORT-SELL", is_entry=True, slippage_pct=0.05) == pytest.approx(99.95)
    assert apply_slippage(100, "SHORT-SELL", is_entry=False, slippage_pct=0.05) == pytest.approx(100.05)


def test_stop_and_target_in_one_candle_uses_the_stop() -> None:
    reason, price = resolve_exit("BUY", stop=99.5, target=101.0, bar={"open": 100.0, "high": 102.0, "low": 99.0})
    assert reason == "stop_loss"
    assert price == pytest.approx(99.5)


def test_gap_through_the_stop_fills_at_the_open() -> None:
    reason, price = resolve_exit("BUY", stop=99.5, target=101.0, bar={"open": 99.0, "high": 99.2, "low": 98.5})
    assert reason == "stop_loss_gap"
    assert price == pytest.approx(99.0)


def test_target_alone_fills_at_the_target() -> None:
    reason, price = resolve_exit("BUY", stop=99.5, target=101.0, bar={"open": 100.0, "high": 101.2, "low": 99.8})
    assert reason == "target"
    assert price == pytest.approx(101)


def test_short_exits_mirror_the_long_exits() -> None:
    assert resolve_exit("SHORT-SELL", stop=101.0, target=99.0, bar={"open": 100.0, "high": 101.5, "low": 98.5}) == ("stop_loss", 101.0)
    assert resolve_exit("SHORT-SELL", stop=101.0, target=99.0, bar={"open": 98.0, "high": 98.5, "low": 97.5}) == ("target_gap", 98.0)


def test_exit_reasons_have_plain_names() -> None:
    assert "stop-loss" in exit_reason_text("stop_loss")
    assert "end of the session" in exit_reason_text("session_close")


def test_cost_docstring_does_not_pretend_to_be_a_zerodha_tariff() -> None:
    text = (CostAssumptions.__doc__ or "").lower()
    assert "zerodha" in text
    assert "not the official" in text or "not a" in text


def test_negative_cost_is_rejected() -> None:
    with pytest.raises(ValueError, match="slippage"):
        CostAssumptions(slippage_pct=-0.1)
