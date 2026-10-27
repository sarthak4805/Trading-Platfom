from universe import LIST_CHOICES, NIFTY500, OTHER_SECTOR, index_stocks, with_fo


def test_index_lists_nest_and_have_sectors():
    nifty50 = {stock.symbol for stock in index_stocks("NIFTY 50")}
    nifty100 = {stock.symbol for stock in index_stocks("NIFTY 100")}
    nifty500 = {stock.symbol for stock in index_stocks("NIFTY 500")}
    assert len(nifty50) == 50 and len(nifty100) == 100 and len(nifty500) >= 495
    assert nifty50 <= nifty100 <= nifty500
    assert all(stock.sector and stock.name for stock in NIFTY500)
    assert LIST_CHOICES == ("All", "NIFTY 50", "NIFTY 100", "NIFTY F&O", "NIFTY 500")


def test_fo_lists_keep_nse_sectors_and_label_outsiders():
    inside = NIFTY500[0]
    fo = with_fo("NIFTY F&O", {inside.symbol: "ignored", "NEWCO": "New Company"})
    assert [stock.symbol for stock in fo] == sorted([inside.symbol, "NEWCO"])
    by_symbol = {stock.symbol: stock for stock in fo}
    assert by_symbol[inside.symbol] == inside
    assert by_symbol["NEWCO"].sector == OTHER_SECTOR and by_symbol["NEWCO"].name == "New Company"
    everything = with_fo("All", {inside.symbol: "", "NEWCO": ""})
    assert len(everything) == len(NIFTY500) + 1 and everything[-1].symbol == "NEWCO"
