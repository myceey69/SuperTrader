"""Stateful pipeline orchestrating all stages (backend only - no GUI imports).

Each stage method can be called on its own (GUI buttons) or via run_all().
progress_cb(stage_name, fraction_0_to_1, message) is optional.
"""
from __future__ import annotations

from datetime import datetime
from typing import Callable, Optional

import numpy as np
import pandas as pd

from config import FEATURE_COLUMNS, LABEL_COLUMN, PipelineConfig
from core import explanation_generator as expl
from core.data_splitter import SplitResult, assert_no_label_leakage, chronological_split, default_split_dates
from core.feature_engineering import build_features
from core.gemini_client import GenerateFn
from core.model_evaluation import compare_models
from core.neural_network import ANNRegressor
from core.portfolio_allocator import allocate_portfolio
from core.preprocessing import FeatureScaler
from core.random_forest import MeanReturnBaseline, build_random_forest
from core.sentiment_analyzer import SentimentAnalyzer
from core.stock_ranker import rank_stocks
from core.trade_planner import plan_trades

ProgressCb = Callable[[str, float, str], None]
STAGES = ["prepare", "train", "sentiment", "rank", "allocate", "plan", "explain"]
_ATTRS = {"prepare": ["features", "split", "scaler"], "train": ["ann", "rf", "baseline", "comparison"],
          "sentiment": ["sentiment_results", "as_of"], "rank": ["ranking", "snapshot"],
          "allocate": ["allocation"], "plan": ["plan"], "explain": ["explanation"]}
_DOWNSTREAM = {"prepare": STAGES, "train": ["train", "rank", "allocate", "plan", "explain"],
               "sentiment": ["sentiment", "rank", "allocate", "plan", "explain"],
               "rank": ["rank", "allocate", "plan", "explain"], "allocate": ["allocate", "plan", "explain"],
               "plan": ["plan", "explain"], "explain": ["explain"]}


class PortfolioPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None, progress_cb: Optional[ProgressCb] = None,
                 generate_fn: Optional[GenerateFn] = None):
        self.cfg = config or PipelineConfig()
        self.progress_cb = progress_cb
        self.generate_fn = generate_fn          # injected fake in tests; None -> real SDK when enabled
        self.prices: Optional[pd.DataFrame] = None
        self.news: Optional[pd.DataFrame] = None
        self.data_source: str = "none"
        self.features = self.split = self.scaler = self.ann = None
        self.rf = self.baseline = None
        self.comparison = self.ranking = self.plan = self.sentiment_results = None
        self.sentiment_status = "disabled"
        self.allocation = self.explanation = None
        self.snapshot: Optional[dict] = None
        self.as_of = None
        self.current_predictions: dict[str, float] = {}

    # ------------------------------------------------------------------ utils
    def _p(self, stage: str, frac: float, msg: str) -> None:
        if self.progress_cb:
            self.progress_cb(stage, frac, msg)

    def set_data(self, prices: pd.DataFrame, news: Optional[pd.DataFrame] = None, source: str = "custom") -> None:
        self.prices, self.data_source = prices, source
        if news is not None:
            self.news = news
        self.invalidate("prepare")

    def invalidate(self, stage: str) -> None:
        """Drop results of `stage` and everything that depends on it."""
        for st in _DOWNSTREAM[stage]:
            for attr in _ATTRS[st]:
                setattr(self, attr, None)
        if "sentiment" in _DOWNSTREAM[stage]:
            self.sentiment_status = "disabled"

    def run_stage(self, stage: str, ann_progress=None, should_stop=None) -> None:
        self.invalidate(stage)
        if stage == "train":
            self.train(ann_progress, should_stop)
        else:
            getattr(self, "plan_trades" if stage == "plan" else stage)()

    # ----------------------------------------------------------------- stages
    def prepare(self) -> None:
        if self.prices is None:
            raise RuntimeError("Load or download price data first.")
        c = self.cfg
        self._p("prepare", 0.0, "Engineering features and forward-return labels...")
        self.features = build_features(self.prices, c.horizon)
        ts, vs = (pd.Timestamp(c.train_end), pd.Timestamp(c.val_end)) if c.train_end and c.val_end \
            else default_split_dates(self.features)
        self.split = chronological_split(self.features, ts, vs, c.embargo, c.horizon)
        assert_no_label_leakage(self.split)
        for name, df in (("train", self.split.train), ("validation", self.split.val), ("test", self.split.test)):
            if len(df) < 20:
                raise ValueError(f"The {name} split has only {len(df)} rows - need more data or different split dates.")
        self.scaler = FeatureScaler().fit(self.split.train)      # TRAIN ONLY
        self._p("prepare", 1.0, f"Prepared {len(self.features)} feature rows.")

    def train(self, ann_progress: Optional[Callable[[int, int, float, float], None]] = None,
              should_stop: Optional[Callable[[], bool]] = None) -> None:
        if self.split is None:
            self.prepare()
        c, sp = self.cfg, self.split
        xtr, xva, xte = (self.scaler.transform(d) for d in (sp.train, sp.val, sp.test))
        ytr, yva, yte = (d[LABEL_COLUMN].to_numpy() for d in (sp.train, sp.val, sp.test))

        def cb(ep, total, tr, va):
            self._p("train", ep / total, f"ANN epoch {ep}/{total}  train {tr:.4f}  val {va:.4f}")
            if ann_progress:
                ann_progress(ep, total, tr, va)

        self.ann = ANNRegressor(c.hidden_layers, c.learning_rate, c.epochs, c.batch_size, c.seed)
        self.ann.fit(xtr, ytr, xva, yva, cb, should_stop)
        self._p("train", 1.0, "Training baseline and random forest...")
        self.baseline = MeanReturnBaseline().fit(xtr, ytr)
        self.rf = build_random_forest(c.rf_trees, c.seed).fit(xtr, ytr)
        models = {"Mean-return baseline": self.baseline, "Random Forest": self.rf, "PyTorch ANN": self.ann}
        self.comparison = compare_models({k: m.predict(xva) for k, m in models.items()}, yva,
                                         {k: m.predict(xte) for k, m in models.items()}, yte)

    def _latest_rows(self) -> pd.DataFrame:
        """Most recent feature row per ticker (features only use past data; label may be NaN)."""
        return self.features.sort_values("Date").groupby("Ticker").tail(1).sort_values("Ticker").reset_index(drop=True)

    def sentiment(self) -> None:
        if self.features is None:
            self.prepare()
        latest = self._latest_rows()
        self.as_of = latest["Date"].max()
        analyzer = SentimentAnalyzer(self.cfg.use_gemini, self.generate_fn)
        self._p("sentiment", 0.0, "Scoring headlines...")
        self.sentiment_results = analyzer.analyze(self.news, list(latest["Ticker"]), self.as_of, self.cfg.news_window_days)
        self.sentiment_status = analyzer.status
        self._p("sentiment", 1.0, f"Sentiment status: {analyzer.status}")

    def rank(self) -> None:
        if self.ann is None:
            self.train()
        if self.sentiment_results is None:
            self.sentiment()
        latest = self._latest_rows()
        preds = self.ann.predict(self.scaler.transform(latest))
        self.current_predictions = dict(zip(latest["Ticker"], preds))
        self.snapshot = {"timestamp": datetime.now().isoformat(timespec="seconds"), "as_of": self.as_of,
                         "prices": dict(zip(latest["Ticker"], latest["Close"]))}
        self.ranking = rank_stocks(self.current_predictions, {r.ticker: r for r in self.sentiment_results},
                                   self.cfg.sentiment_weight, self.cfg.top_n)
        self._p("rank", 1.0, f"Top pick: {self.ranking.iloc[0]['Ticker']}")

    def allocate(self) -> None:
        if self.ranking is None:
            self.rank()
        sel = self.ranking[self.ranking["Selected"]]
        self.allocation = allocate_portfolio(list(sel["Ticker"]), self.snapshot["prices"], self.cfg.holdings,
                                             self.cfg.cash, self.cfg.fee, dict(zip(sel["Ticker"], sel["Final Score"])),
                                             self.cfg.allocation_method)
        self._p("allocate", 1.0, f"Target uses capital {self.allocation.capital:,.2f}")

    def plan_trades(self) -> None:
        if self.allocation is None:
            self.allocate()
        self.plan = plan_trades(self.cfg.holdings, self.cfg.cash, self.allocation.target_shares,
                                self.snapshot["prices"], self.cfg.fee, self.cfg.planner_mode, self.cfg.max_states)
        self._p("plan", 1.0, f"Plan {self.plan.status}: {len(self.plan.trades)} trades, fees {self.plan.total_fee:.2f}")

    def explain(self) -> None:
        if self.plan is None:
            self.plan_trades()
        gen = None
        if self.cfg.use_gemini:
            gen = self.generate_fn
            if gen is None:
                try:
                    from core.gemini_client import make_generate_fn
                    gen = make_generate_fn()
                except Exception:
                    gen = None
        summary = {"as_of": str(self.as_of.date()), "weight": self.cfg.sentiment_weight, "top_n": self.cfg.top_n,
                   "cash": self.cfg.cash, "fee": self.cfg.fee, "capital": self.allocation.capital,
                   "ranking": self.ranking.to_dict("records"), "allocation": self.allocation.table.to_dict("records"),
                   "plan_status": self.plan.status, "n_trades": len(self.plan.trades), "total_fee": self.plan.total_fee,
                   "final_cash": self.plan.final_cash, "trades": self.plan.trades.to_dict("records")}
        self.explanation = expl.explain(summary, gen)
        self._p("explain", 1.0, f"Explanation source: {self.explanation['source']}")

    def run_all(self, ann_progress=None, should_stop=None) -> "PortfolioPipeline":
        self.invalidate("prepare")
        self.prepare()
        self.train(ann_progress, should_stop)
        self.sentiment()
        self.rank()
        self.allocate()
        self.plan_trades()
        self.explain()
        self._p("done", 1.0, "Pipeline complete.")
        return self
