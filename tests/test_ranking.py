import pytest

from core.sentiment_analyzer import SentimentResult
from core.stock_ranker import rank_stocks


def S(t, s):
    return SentimentResult(t, s, "r", "gemini")


def test_formula_and_selection():
    df = rank_stocks({"A": 0.10, "B": 0.00, "C": -0.05}, {"A": S("A", 0), "B": S("B", 1), "C": S("C", -1)}, 0.25, 2)
    a = df.set_index("Ticker").loc["A"]
    assert a["Return Score"] == 1.0 and a["Sentiment Score"] == 0.5
    assert a["Final Score"] == pytest.approx(0.75 * 1.0 + 0.25 * 0.5)
    assert list(df[df.Selected].Ticker) == ["A", "B"] and list(df.Rank) == [1, 2, 3]
    assert df.set_index("Ticker").loc["C", "Predicted Return"] == -0.05      # raw kept separate


def test_weight_extremes_and_ties_and_missing():
    p = {"A": 0.02, "B": 0.01}
    sent = {"A": S("A", -1), "B": S("B", 1)}
    assert rank_stocks(p, sent, 0.0, 1).iloc[0].Ticker == "A"
    assert rank_stocks(p, sent, 1.0, 1).iloc[0].Ticker == "B"
    t = rank_stocks({"B": 0.01, "A": 0.01}, {}, 0.25, 1)             # equal + missing sentiment
    assert list(t.Ticker) == ["A", "B"] and t.Tied.all() and (t["Return Score"] == 0.5).all()
    with pytest.raises(ValueError):
        rank_stocks(p, sent, 1.5, 1)
    assert rank_stocks(p, sent, 0.5, 99).Selected.sum() == 2
