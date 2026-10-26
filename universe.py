"""The 100 NSE large-cap names the desk tracks.

These symbols, company names, and sectors are a working list of well-known NSE
large caps. They are not a verified copy of the official NIFTY 100. Confirm
constituents and sectors against the current NSE index factsheet.
"""

from __future__ import annotations

from dataclasses import dataclass


UNIVERSE_DISCLAIMER = (
    "This list is a working sample of well-known NSE large-cap names. It is not a "
    "verified copy of the current official NIFTY 100 constituents, and the sector "
    "labels are an illustrative mapping until checked against the NSE index factsheet. "
    "Symbols Zerodha does not recognise are left out and listed in the sidebar."
)


@dataclass(frozen=True)
class Stock:
    """One tracked name."""

    symbol: str
    name: str
    sector: str


# (symbol, company name, illustrative sector)
_ROWS: tuple[tuple[str, str, str], ...] = (
    ("RELIANCE", "Reliance Industries", "Energy"),
    ("TCS", "Tata Consultancy Services", "Information Technology"),
    ("HDFCBANK", "HDFC Bank", "Financial Services"),
    ("BHARTIARTL", "Bharti Airtel", "Telecommunication"),
    ("ICICIBANK", "ICICI Bank", "Financial Services"),
    ("SBIN", "State Bank of India", "Financial Services"),
    ("INFY", "Infosys", "Information Technology"),
    ("LICI", "Life Insurance Corporation of India", "Financial Services"),
    ("ITC", "ITC", "FMCG"),
    ("HINDUNILVR", "Hindustan Unilever", "FMCG"),
    ("LT", "Larsen & Toubro", "Construction"),
    ("BAJFINANCE", "Bajaj Finance", "Financial Services"),
    ("HCLTECH", "HCL Technologies", "Information Technology"),
    ("MARUTI", "Maruti Suzuki India", "Automobile"),
    ("SUNPHARMA", "Sun Pharmaceutical Industries", "Healthcare"),
    ("KOTAKBANK", "Kotak Mahindra Bank", "Financial Services"),
    ("M&M", "Mahindra & Mahindra", "Automobile"),
    ("AXISBANK", "Axis Bank", "Financial Services"),
    ("ULTRACEMCO", "UltraTech Cement", "Cement"),
    ("NTPC", "NTPC", "Power"),
    ("BAJAJFINSV", "Bajaj Finserv", "Financial Services"),
    ("TITAN", "Titan Company", "Consumer Durables"),
    ("ADANIENT", "Adani Enterprises", "Diversified"),
    ("ONGC", "Oil and Natural Gas Corporation", "Energy"),
    ("POWERGRID", "Power Grid Corporation of India", "Power"),
    ("TATAMOTORS", "Tata Motors", "Automobile"),
    ("ADANIPORTS", "Adani Ports and Special Economic Zone", "Infrastructure"),
    ("WIPRO", "Wipro", "Information Technology"),
    ("COALINDIA", "Coal India", "Energy"),
    ("JSWSTEEL", "JSW Steel", "Metals & Mining"),
    ("ASIANPAINT", "Asian Paints", "Consumer Durables"),
    ("NESTLEIND", "Nestle India", "FMCG"),
    ("HINDALCO", "Hindalco Industries", "Metals & Mining"),
    ("JIOFIN", "Jio Financial Services", "Financial Services"),
    ("BEL", "Bharat Electronics", "Capital Goods"),
    ("TRENT", "Trent", "Consumer Services"),
    ("GRASIM", "Grasim Industries", "Cement"),
    ("DLF", "DLF", "Realty"),
    ("HAL", "Hindustan Aeronautics", "Capital Goods"),
    ("VEDL", "Vedanta", "Metals & Mining"),
    ("IOC", "Indian Oil Corporation", "Energy"),
    ("SBILIFE", "SBI Life Insurance", "Financial Services"),
    ("DIVISLAB", "Divi's Laboratories", "Healthcare"),
    ("EICHERMOT", "Eicher Motors", "Automobile"),
    ("BAJAJ-AUTO", "Bajaj Auto", "Automobile"),
    ("SHRIRAMFIN", "Shriram Finance", "Financial Services"),
    ("CIPLA", "Cipla", "Healthcare"),
    ("BPCL", "Bharat Petroleum Corporation", "Energy"),
    ("TECHM", "Tech Mahindra", "Information Technology"),
    ("TATACONSUM", "Tata Consumer Products", "FMCG"),
    ("APOLLOHOSP", "Apollo Hospitals Enterprise", "Healthcare"),
    ("HEROMOTOCO", "Hero MotoCorp", "Automobile"),
    ("INDUSINDBK", "IndusInd Bank", "Financial Services"),
    ("DRREDDY", "Dr. Reddy's Laboratories", "Healthcare"),
    ("BRITANNIA", "Britannia Industries", "FMCG"),
    ("HINDZINC", "Hindustan Zinc", "Metals & Mining"),
    ("TATAPOWER", "Tata Power", "Power"),
    ("ADANIGREEN", "Adani Green Energy", "Power"),
    ("PIDILITIND", "Pidilite Industries", "Chemicals"),
    ("GODREJCP", "Godrej Consumer Products", "FMCG"),
    ("CHOLAFIN", "Cholamandalam Investment and Finance", "Financial Services"),
    ("AMBUJACEM", "Ambuja Cements", "Cement"),
    ("BANKBARODA", "Bank of Baroda", "Financial Services"),
    ("GAIL", "GAIL (India)", "Energy"),
    ("DABUR", "Dabur India", "FMCG"),
    ("HAVELLS", "Havells India", "Consumer Durables"),
    ("SIEMENS", "Siemens", "Capital Goods"),
    ("DMART", "Avenue Supermarts", "Consumer Services"),
    ("ABB", "ABB India", "Capital Goods"),
    ("VBL", "Varun Beverages", "FMCG"),
    ("MOTHERSON", "Samvardhana Motherson International", "Automobile"),
    ("LODHA", "Macrotech Developers", "Realty"),
    ("ZYDUSLIFE", "Zydus Lifesciences", "Healthcare"),
    ("CANBK", "Canara Bank", "Financial Services"),
    ("NAUKRI", "Info Edge (India)", "Consumer Services"),
    ("IRFC", "Indian Railway Finance Corporation", "Financial Services"),
    ("PFC", "Power Finance Corporation", "Financial Services"),
    ("RECLTD", "REC", "Financial Services"),
    ("POLYCAB", "Polycab India", "Capital Goods"),
    ("TORNTPHARM", "Torrent Pharmaceuticals", "Healthcare"),
    ("CGPOWER", "CG Power and Industrial Solutions", "Capital Goods"),
    ("TVSMOTOR", "TVS Motor Company", "Automobile"),
    ("MAXHEALTH", "Max Healthcare Institute", "Healthcare"),
    ("LTIM", "LTIMindtree", "Information Technology"),
    ("INDIGO", "InterGlobe Aviation", "Consumer Services"),
    ("JINDALSTEL", "Jindal Steel & Power", "Metals & Mining"),
    ("TATACOMM", "Tata Communications", "Telecommunication"),
    ("COLPAL", "Colgate-Palmolive (India)", "FMCG"),
    ("MUTHOOTFIN", "Muthoot Finance", "Financial Services"),
    ("HDFCLIFE", "HDFC Life Insurance", "Financial Services"),
    ("PIIND", "PI Industries", "Chemicals"),
    ("UPL", "UPL", "Chemicals"),
    ("SRF", "SRF", "Chemicals"),
    ("TATASTEEL", "Tata Steel", "Metals & Mining"),
    ("MPHASIS", "Mphasis", "Information Technology"),
    ("OFSS", "Oracle Financial Services Software", "Information Technology"),
    ("HINDPETRO", "Hindustan Petroleum Corporation", "Energy"),
    ("NHPC", "NHPC", "Power"),
    ("VOLTAS", "Voltas", "Consumer Durables"),
    ("BERGEPAINT", "Berger Paints India", "Consumer Durables"),
)


def _build_universe() -> tuple[Stock, ...]:
    stocks = tuple(Stock(symbol=symbol, name=name, sector=sector) for symbol, name, sector in _ROWS)
    symbols = [stock.symbol for stock in stocks]
    if len(symbols) != len(set(symbols)):
        raise RuntimeError("The universe contains a duplicate symbol.")
    if len(stocks) != 100:
        raise RuntimeError(f"The universe has {len(stocks)} names; this build expects 100.")
    return stocks


UNIVERSE: tuple[Stock, ...] = _build_universe()
_BY_SYMBOL: dict[str, Stock] = {stock.symbol: stock for stock in UNIVERSE}


def get_stock(symbol: str) -> Stock:
    """Return one tracked stock, or raise a helpful error."""
    try:
        return _BY_SYMBOL[symbol]
    except KeyError as exc:
        examples = ", ".join(stock.symbol for stock in UNIVERSE[:6])
        raise KeyError(f"Unknown symbol '{symbol}'. Examples of tracked symbols: {examples}.") from exc


def all_sectors() -> list[str]:
    """Sorted unique illustrative sector labels."""
    return sorted({stock.sector for stock in UNIVERSE})
