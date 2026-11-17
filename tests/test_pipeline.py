import json

import pytest

from config import PipelineConfig
from core.pipeline import PortfolioPipeline


def fake_gemini(prompt, system, json_mode):
    if json_mode:
        t = prompt.split("\n")[0].split(": ")[1]
        return json.dumps({"ticker": t, "sentiment": 0.4, "reason": "Mocked positive."})
    return "Mock explanation using the supplied numbers."


def cfg(**kw):
    base = dict(epochs=8, rf_trees=20, top_n=2, cash=20_000.0, fee=2.0, holdings={"ALPH": 10})
    base.update(kw)
    return PipelineConfig(**base)


def test_full_pipeline_offline(prices, news):
    msgs = []
    p = PortfolioPipeline(cfg(), progress_cb=lambda s, f, m: msgs.append(s))
    p.set_data(prices, news, "synthetic")
    p.run_all()
    assert {"prepare", "train", "sentiment", "rank", "allocate", "plan", "explain", "done"} <= set(msgs)
    assert len(p.comparison) == 3 and p.sentiment_status == "disabled"
    assert (p.ranking["Sentiment Mode"] == "fallback-disabled").all() and p.ranking.Selected.sum() == 2
    assert p.plan.status == "optimal" and p.plan.final_cash >= 0
    assert p.plan.final_holdings == {t: p.allocation.target_shares.get(t, 0) for t in p.plan.final_holdings}
    assert "deterministic" in p.explanation["source"] and p.explanation["prompt"]


def test_pipeline_with_mocked_gemini(prices, news):
    p = PortfolioPipeline(cfg(use_gemini=True), generate_fn=fake_gemini)
    p.set_data(prices, news, "synthetic")
    p.run_all()
    assert (p.ranking["Sentiment Mode"] == "gemini").all() or (p.ranking["Sentiment Mode"] != "fallback-disabled").all()
    assert p.explanation["source"] == "gemini" and "Mock explanation" in p.explanation["text"]
    assert "predicted return" in p.explanation["prompt"] and "TRADE PLAN" in p.explanation["prompt"]


def test_requires_data_and_small_splits(prices):
    with pytest.raises(RuntimeError):
        PortfolioPipeline(cfg()).prepare()
    p = PortfolioPipeline(cfg(train_end="2024-01-01", val_end="2024-01-05"))
    p.set_data(prices)
    with pytest.raises(ValueError):
        p.prepare()
