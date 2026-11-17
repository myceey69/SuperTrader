"""Combine ANN predicted returns with sentiment.

final = (1-w)*return_score + w*sentiment_score
 return_score    = min-max of predicted returns across the CURRENT candidate set
                   (uses only today's predictions -> no look-ahead); 0.5 if all equal.
 sentiment_score = (s + 1) / 2  (maps [-1,1] -> [0,1])
Raw predicted returns are kept in their own column. Ties: higher raw predicted
return, then ticker alphabetical (deterministic), and flagged in `Tied`.
"""
from __future__ import annotations

import pandas as pd

from core.sentiment_analyzer import SentimentResult


def rank_stocks(predictions: dict[str, float], sentiments: dict[str, SentimentResult],
                weight: float = 0.25, top_n: int = 3) -> pd.DataFrame:
    if not predictions:
        raise ValueError("No predictions to rank")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("sentiment weight must be in [0, 1]")
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    r = pd.Series(predictions, dtype=float)
    span = r.max() - r.min()
    ret_score = (r - r.min()) / span if span > 0 else pd.Series(0.5, index=r.index)
    rows = []
    for t in r.index:
        s = sentiments.get(t)
        sv = s.sentiment if s else 0.0
        rows.append({"Ticker": t, "Predicted Return": r[t], "Sentiment": sv,
                     "Sentiment Mode": s.mode if s else "fallback-disabled",
                     "Sentiment Reason": s.reason if s else "No sentiment data (neutral).",
                     "Return Score": ret_score[t], "Sentiment Score": (sv + 1) / 2})
    df = pd.DataFrame(rows)
    df["Final Score"] = (1 - weight) * df["Return Score"] + weight * df["Sentiment Score"]
    df = df.sort_values(["Final Score", "Predicted Return", "Ticker"], ascending=[False, False, True]).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    df["Tied"] = df["Final Score"].round(12).duplicated(keep=False)
    df["Selected"] = df["Rank"] <= min(top_n, len(df))
    return df
