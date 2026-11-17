import numpy as np
import pandas as pd
import pytest

from core.data_loader import load_news_csv, load_prices_csv
from core.market_data import MarketDataError, fetch_yfinance, parse_tickers, standardize_prices


def fake_multi(tickers, **kw):
    idx = pd.bdate_range("2024-01-01", periods=30, name="Date")
    cols = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Volume"], tickers])
    return pd.DataFrame(np.random.default_rng(0).uniform(10, 20, (30, len(cols))), index=idx, columns=cols)


def test_parse_tickers():
    assert parse_tickers("aapl, msft  aapl") == ["AAPL", "MSFT"]
    with pytest.raises(MarketDataError):
        parse_tickers("bad$ticker")
    with pytest.raises(MarketDataError):
        parse_tickers("  ")


def test_fetch_mocked_multiindex():
    df, rep = fetch_yfinance(["AAPL", "MSFT"], "2024-01-01", "2024-03-01", downloader=fake_multi)
    assert set(df["Ticker"]) == {"AAPL", "MSFT"} and rep["adjusted"] and rep["retrieved_at_utc"]
    assert list(df.columns) == ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]


def test_fetch_empty_and_errors():
    with pytest.raises(MarketDataError, match="no data"):
        fetch_yfinance(["ZZZZ"], "2024-01-01", "2024-03-01", downloader=lambda *a, **k: pd.DataFrame())
    def boom(*a, **k):
        raise RuntimeError("429 Too Many Requests")
    with pytest.raises(MarketDataError, match="rate limit"):
        fetch_yfinance(["AAPL"], "2024-01-01", "2024-03-01", downloader=boom, sleep=lambda s: None)
    with pytest.raises(MarketDataError):
        fetch_yfinance(["AAPL"], "2024-03-01", "2024-01-01", downloader=boom)


def test_standardize_duplicates_and_volume():
    df = pd.DataFrame({"Date": ["2024-01-02", "2024-01-02", "2024-01-03"], "Ticker": ["a", "a", "a"],
                       "Open": [1, 1, 1], "High": [1, 1, 1], "Low": [1, 1, 1], "Close": [1, 2, 3],
                       "Volume": [np.nan, 5, 5]})
    out = standardize_prices(df)
    assert len(out) == 2 and out["Ticker"].iloc[0] == "A" and out["Close"].iloc[0] == 2
    with pytest.raises(MarketDataError):
        standardize_prices(df.drop(columns=["Close"]))


def test_csv_roundtrip(tmp_path, prices, news):
    p, n = tmp_path / "p.csv", tmp_path / "n.csv"
    prices.to_csv(p, index=False)
    news.to_csv(n, index=False)
    assert len(load_prices_csv(p)) == len(prices) and len(load_news_csv(n)) == len(news)
    with pytest.raises(MarketDataError):
        load_prices_csv(tmp_path / "missing.csv")
