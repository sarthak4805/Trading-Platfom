"""The NSE stock lists the desk can scan.

NIFTY 50, NIFTY 100, and NIFTY 500 members and their industries come from NSE's
index files, saved in ``nse_indices.csv``. The F&O list is not in that file: it
comes from Zerodha's stock futures at login, so it is always current.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

INDEX_FILE = Path(__file__).with_name("nse_indices.csv")
INDEX_FILE_DATE = "27 Oct 2026"

LIST_CHOICES = ("All", "NIFTY 50", "NIFTY 100", "NIFTY F&O", "NIFTY 500")
DEFAULT_LIST = "NIFTY 100"
FO_LISTS = ("All", "NIFTY F&O")
OTHER_SECTOR = "Other"

UNIVERSE_DISCLAIMER = (
    f"NIFTY 50, NIFTY 100, and NIFTY 500 members and industries come from NSE's index files saved on "
    f"{INDEX_FILE_DATE}. NSE reviews the indices twice a year, so refresh nse_indices.csv after a review. "
    "The F&O list comes from Zerodha's stock futures at login; F&O stocks outside NIFTY 500 get the "
    f"sector '{OTHER_SECTOR}'. Symbols Zerodha does not recognise are left out and listed in the sidebar."
)


@dataclass(frozen=True)
class Stock:
    """One tracked name."""

    symbol: str
    name: str
    sector: str


def _read_index_file() -> tuple[tuple[Stock, ...], frozenset[str], frozenset[str]]:
    stocks, nifty50, nifty100 = [], set(), set()
    with INDEX_FILE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stock = Stock(row["symbol"], row["name"], row["industry"])
            stocks.append(stock)
            if row["nifty50"] == "1":
                nifty50.add(stock.symbol)
            if row["nifty100"] == "1":
                nifty100.add(stock.symbol)
    if len({stock.symbol for stock in stocks}) != len(stocks):
        raise RuntimeError(f"{INDEX_FILE.name} contains a duplicate symbol.")
    return tuple(stocks), frozenset(nifty50), frozenset(nifty100)


NIFTY500, _NIFTY50, _NIFTY100 = _read_index_file()


def index_stocks(choice: str) -> list[Stock]:
    """Members of an NSE index list. 'All' and 'NIFTY F&O' start from NIFTY 500 until F&O names are known."""
    if choice == "NIFTY 50":
        return [stock for stock in NIFTY500 if stock.symbol in _NIFTY50]
    if choice == "NIFTY 100":
        return [stock for stock in NIFTY500 if stock.symbol in _NIFTY100]
    return list(NIFTY500)


def with_fo(choice: str, fo_names: Mapping[str, str]) -> list[Stock]:
    """'NIFTY F&O' or 'All' once Zerodha's F&O stocks are known. ``fo_names`` maps symbol to company name."""
    known = {stock.symbol: stock for stock in NIFTY500}
    fo = [known.get(symbol) or Stock(symbol, name or symbol, OTHER_SECTOR) for symbol, name in sorted(fo_names.items())]
    if choice == "NIFTY F&O":
        return fo
    return list(NIFTY500) + [stock for stock in fo if stock.symbol not in known]
