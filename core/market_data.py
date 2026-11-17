"""Market data retrieval (Yahoo Finance), validation and synthetic demo data.

Internal format (long): Date, Ticker, Open, High, Low, Close, Volume.
Yahoo Finance is a free, best-effort source: data may be delayed, revised or
incomplete. It is NOT guaranteed real-time data.
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
_PRICE_COLS = ["Open", "High", "Low", "Close"]
_TICKER_RE = re.compile(r"^[A-Z0-9.\-^=]{1,15}$")
VALID_INTERVALS = ["1d", "5d", "1wk", "1mo"]


class MarketDataError(Exception):
    """Raised for any recoverable market-data problem (shown in the GUI)."""


def parse_tickers(text: str) -> list[str]:
    """'aapl, msft nvda' -> ['AAPL','MSFT','NVDA'] (deduplicated, validated)."""
    out: list[str] = []
    for tok in re.split(r"[,\s;]+", text.strip().upper()):
        if not tok:
            continue
        if not _TICKER_RE.match(tok):
            raise MarketDataError(f"Invalid ticker symbol: '{tok}'")
        if tok not in out:
            out.append(tok)
    if not out:
        raise MarketDataError("Enter at least one ticker symbol.")
    return out


def standardize_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean a long price table into the internal format."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise MarketDataError(f"Missing required columns: {missing}")
    out = df[REQUIRED_COLUMNS].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()
    for c in _PRICE_COLS + ["Volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["Date", "Close"])
    out = out[out["Close"] > 0]
    for c in ["Open", "High", "Low"]:           # fill rare missing OHL with Close
        out[c] = out[c].fillna(out["Close"])
    out["Volume"] = out["Volume"].fillna(0.0).clip(lower=0.0)  # missing volume -> 0
    out = out.drop_duplicates(subset=["Date", "Ticker"], keep="last")
    out = out.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    if out.empty:
        raise MarketDataError("No valid price rows after cleaning.")
    return out


def _yf_to_long(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Convert yfinance output (MultiIndex or flat columns) to long format."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    frames = []
    if isinstance(raw.columns, pd.MultiIndex):
        field_level = next((i for i in range(raw.columns.nlevels)
                            if "Close" in set(raw.columns.get_level_values(i))), None)
        if field_level is None:
            raise MarketDataError("Unexpected data format returned by Yahoo Finance.")
        tick_level = 1 - field_level if raw.columns.nlevels == 2 else field_level
        for t in dict.fromkeys(raw.columns.get_level_values(tick_level)):
            sub = raw.xs(t, axis=1, level=tick_level).copy()
            sub["Ticker"] = str(t).upper()
            frames.append(sub.reset_index())
    else:  # flat columns: single ticker
        sub = raw.copy()
        sub["Ticker"] = tickers[0]
        frames.append(sub.reset_index())
    long = pd.concat(frames, ignore_index=True)
    long = long.rename(columns={long.columns[0]: "Date"}) if "Date" not in long.columns else long
    for c in REQUIRED_COLUMNS:
        if c not in long.columns:
            long[c] = np.nan
    return long[REQUIRED_COLUMNS]


def fetch_yfinance(tickers: list[str], start: str, end: str, interval: str = "1d",
                   downloader: Optional[Callable] = None, retries: int = 2,
                   sleep: Callable[[float], None] = time.sleep) -> tuple[pd.DataFrame, dict]:
    """Download prices. `downloader` is injectable so tests never touch the network.

    Returns (clean long DataFrame, report dict). Prices are split/dividend
    ADJUSTED (auto_adjust=True) - documented in the report.
    """
    if interval not in VALID_INTERVALS:
        raise MarketDataError(f"Unsupported interval '{interval}'. Use one of {VALID_INTERVALS}.")
    if pd.Timestamp(start) >= pd.Timestamp(end):
        raise MarketDataError("Start date must be before end date.")
    if downloader is None:
        import yfinance as yf
        downloader = yf.download
    raw, last_err = None, None
    for attempt in range(retries + 1):
        try:
            raw = downloader(tickers, start=start, end=end, interval=interval,
                             auto_adjust=True, progress=False)
            last_err = None
            break
        except Exception as exc:  # network / rate-limit / library errors
            last_err = exc
            if attempt < retries:
                sleep(1.5 * (attempt + 1))
    if last_err is not None:
        msg = str(last_err)
        if "429" in msg or "rate" in msg.lower():
            raise MarketDataError("Yahoo Finance rate limit reached. Wait a few minutes and retry, "
                                  "or use a CSV / synthetic dataset.") from last_err
        raise MarketDataError(f"Download failed (network or Yahoo error): {msg}") from last_err
    long = _yf_to_long(raw, tickers)
    if long.empty:
        raise MarketDataError("Yahoo Finance returned no data (invalid tickers, dates or no network).")
    clean = standardize_prices(long)
    returned = sorted(clean["Ticker"].unique())
    report = {
        "source": "Yahoo Finance (yfinance)",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "requested": tickers, "returned": returned,
        "missing_tickers": [t for t in tickers if t not in returned],
        "rows": len(clean), "interval": interval, "adjusted": True,
        "note": "Best-effort free data; not guaranteed real-time or error-free.",
    }
    return clean, report


# ----------------------------------------------------------------- synthetic
SYNTHETIC_TICKERS = ["ALPH", "BRVO", "CHRL", "DLTA", "ECHO", "FXTR", "GLDN", "HRZN"]


def generate_synthetic_prices(n_days: int = 1000, tickers: Optional[list[str]] = None,
                              seed: int = 7, end: str = "2025-12-31") -> pd.DataFrame:
    """Clearly-fictional prices (geometric random walk + mild momentum/volatility clustering).

    NOT real market data. Used for offline demos and tests.
    """
    rng = np.random.default_rng(seed)
    tickers = tickers or SYNTHETIC_TICKERS
    dates = pd.bdate_range(end=end, periods=n_days)
    market = rng.normal(0.0003, 0.008, n_days)
    rows = []
    for i, t in enumerate(tickers):
        drift = rng.uniform(-0.0001, 0.0008)
        beta = rng.uniform(0.6, 1.4)
        base_vol = rng.uniform(0.008, 0.02)
        ret = np.zeros(n_days)
        vol = base_vol
        for k in range(1, n_days):
            vol = float(np.clip(0.9 * vol + 0.1 * base_vol + 0.15 * abs(ret[k - 1]), 0.004, 0.04))
            ret[k] = float(np.clip(drift + beta * market[k] + 0.06 * ret[k - 1] + rng.normal(0, vol), -0.12, 0.12))
        close = rng.uniform(40, 400) * np.exp(np.cumsum(ret))
        openp = close * (1 + rng.normal(0, 0.002, n_days))
        high = np.maximum(openp, close) * (1 + np.abs(rng.normal(0, 0.004, n_days)))
        low = np.minimum(openp, close) * (1 - np.abs(rng.normal(0, 0.004, n_days)))
        volume = rng.lognormal(15 + 0.3 * i, 0.3, n_days) * (1 + 20 * np.abs(ret))
        rows.append(pd.DataFrame({"Date": dates, "Ticker": t, "Open": openp, "High": high,
                                  "Low": low, "Close": close, "Volume": volume.round()}))
    return standardize_prices(pd.concat(rows, ignore_index=True))


_POS = ["{t} (synthetic) reports record quarterly demand", "{t} (synthetic) raises full-year outlook",
        "Analysts upgrade {t} (synthetic) on strong product pipeline"]
_NEG = ["{t} (synthetic) faces supply chain disruption", "{t} (synthetic) cuts guidance amid weak demand",
        "Regulators open inquiry into {t} (synthetic)"]
_NEU = ["{t} (synthetic) announces routine leadership update", "{t} (synthetic) to present at industry conference"]


def generate_synthetic_news(prices: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    """Fictional headlines (clearly tagged 'synthetic'), ~2 per ticker per month."""
    rng = np.random.default_rng(seed)
    rows = []
    for t, g in prices.groupby("Ticker"):
        dates = g["Date"].sort_values().to_numpy()
        for d in dates[::10]:
            pool = [_POS, _NEG, _NEU][rng.choice(3, p=[0.4, 0.3, 0.3])]
            rows.append({"Date": pd.Timestamp(d), "Ticker": t,
                         "Headline": pool[rng.integers(len(pool))].format(t=t),
                         "Source": "SYNTHETIC"})
    return pd.DataFrame(rows).sort_values(["Ticker", "Date"]).reset_index(drop=True)
