"""CSV import/export for prices and news."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.market_data import MarketDataError, standardize_prices

NEWS_COLUMNS = ["Date", "Ticker", "Headline", "Source"]


def load_prices_csv(path: str | Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise MarketDataError(f"File not found: {path}") from exc
    except Exception as exc:
        raise MarketDataError(f"Could not read CSV: {exc}") from exc
    return standardize_prices(df)


def load_news_csv(path: str | Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise MarketDataError(f"Could not read news CSV: {exc}") from exc
    missing = [c for c in NEWS_COLUMNS if c not in df.columns]
    if missing:
        raise MarketDataError(f"News CSV missing columns: {missing}")
    df = df[NEWS_COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Headline"] = df["Headline"].astype(str).str.strip()
    return df.dropna(subset=["Date"]).query("Headline != ''").reset_index(drop=True)


def export_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
