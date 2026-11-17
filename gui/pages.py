"""All dashboard pages. Pages only read/write the app's PortfolioPipeline; no ML logic here."""
from __future__ import annotations

import tkinter as tk
from datetime import date
from tkinter import filedialog

import customtkinter as ctk
import numpy as np
import pandas as pd

from config import DATA_DIR, OUTPUT_DIR, PipelineConfig
from core.data_loader import export_csv, load_news_csv, load_prices_csv
from core.market_data import (VALID_INTERVALS, MarketDataError, fetch_yfinance, generate_synthetic_news,
                              generate_synthetic_prices, parse_tickers)
from gui import theme as T
from gui.widgets import ChartFrame, DataTable, MetricCard, button, labeled_entry


def parse_holdings(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for tok in [x for x in text.replace(";", ",").split(",") if x.strip()]:
        try:
            t, q = tok.split(":")
            q = int(q)
        except ValueError:
            raise ValueError(f"Bad holding '{tok.strip()}'. Use TICKER:SHARES, e.g. ALPH:10, BRVO:5")
        if q < 0:
            raise ValueError("Holdings cannot be negative (no short selling)")
        if q:
            out[t.strip().upper()] = q
    return out


def _f(var: tk.StringVar, name: str, cast=float, lo=None, hi=None):
    try:
        v = cast(var.get().strip())
    except ValueError:
        raise ValueError(f"'{name}' must be a valid {cast.__name__}")
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise ValueError(f"'{name}' must be between {lo} and {hi}")
    return v


class Page(ctk.CTkScrollableFrame):
    title = ""
    subtitle = ""

    def __init__(self, master, app):
        super().__init__(master, fg_color=T.BG, corner_radius=0)
        self.app = app
        ctk.CTkLabel(self, text=self.title, font=(T.FONT, 24, "bold"), text_color=T.TEXT).pack(anchor="w", padx=20, pady=(16, 0))
        if self.subtitle:
            ctk.CTkLabel(self, text=self.subtitle, font=(T.FONT, 12), text_color=T.MUTED, wraplength=900, justify="left").pack(anchor="w", padx=20, pady=(0, 8))
        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.pack(fill="both", expand=True, padx=20, pady=8)
        self.body.grid_columnconfigure(0, weight=1)
        self.build()

    def row(self) -> ctk.CTkFrame:
        f = ctk.CTkFrame(self.body, fg_color="transparent")
        f.pack(fill="x", pady=6)
        return f

    def note(self, text: str = "", color: str = T.MUTED) -> ctk.CTkLabel:
        lab = ctk.CTkLabel(self.body, text=text, text_color=color, font=(T.FONT, 12), wraplength=950, justify="left", anchor="w")
        lab.pack(fill="x", pady=4)
        return lab

    def build(self) -> None: ...
    def apply_config(self, cfg: PipelineConfig) -> None: ...
    def refresh(self) -> None: ...

    @property
    def pipe(self):
        return self.app.pipe


# ------------------------------------------------------------------ Dashboard
class DashboardPage(Page):
    title = "Dashboard"
    subtitle = "Overview of the latest pipeline run. Use RUN FULL PIPELINE (sidebar) to execute every stage."

    def build(self) -> None:
        grid = ctk.CTkFrame(self.body, fg_color="transparent")
        grid.pack(fill="x")
        self.cards = {}
        for i, name in enumerate(["Data Source", "Available Cash", "Model Status", "Selected Stocks", "Total Trading Fees", "Gemini Status"]):
            c = MetricCard(grid, name)
            c.grid(row=i // 3, column=i % 3, padx=6, pady=6, sticky="nsew")
            grid.grid_columnconfigure(i % 3, weight=1)
            self.cards[name] = c
        self.note("Predicted returns are model estimates, not guaranteed outcomes. Synthetic data is fictional and never real market data.")
        self.checklist = self.note("")

    def refresh(self) -> None:
        p, c = self.pipe, self.cards
        c["Data Source"].set(f"{p.data_source}  ({0 if p.prices is None else p.prices.Ticker.nunique()} tickers)")
        c["Available Cash"].set(f"${p.cfg.cash:,.2f}")
        if p.comparison is not None:
            ann = p.comparison.set_index("Model").loc["PyTorch ANN"]
            c["Model Status"].set(f"ANN trained (test RMSE {ann['Test RMSE']:.4f})", T.GREEN)
        else:
            c["Model Status"].set("Not trained", T.AMBER)
        sel = [] if p.ranking is None else list(p.ranking[p.ranking.Selected].Ticker)
        c["Selected Stocks"].set(", ".join(sel) or "-")
        c["Total Trading Fees"].set("-" if p.plan is None else f"${p.plan.total_fee:,.2f}")
        mode = "ON" if p.cfg.use_gemini else "OFF (neutral fallback)"
        c["Gemini Status"].set(f"{mode} - {p.sentiment_status}", T.GREEN if p.cfg.use_gemini and p.sentiment_status.startswith(("conn", "ready")) else T.MUTED)
        done = {"Features/split": p.split is not None, "ANN + comparison": p.comparison is not None, "Sentiment": p.sentiment_results is not None,
                "Ranking": p.ranking is not None, "Allocation": p.allocation is not None, "Trade plan": p.plan is not None, "Explanation": p.explanation is not None}
        self.checklist.configure(text="   ".join(("[x] " if v else "[ ] ") + k for k, v in done.items()))


# ------------------------------------------------------------- Market Explorer
class MarketExplorerPage(Page):
    title = "Market Explorer"
    subtitle = "Download Yahoo Finance history (free, best-effort, not guaranteed real-time), import a CSV, or use offline synthetic data."

    def build(self) -> None:
        r = self.row()
        f, self.tickers = labeled_entry(r, "Tickers", "AAPL, MSFT, NVDA", 220); f.pack(side="left", padx=4)
        f, self.start = labeled_entry(r, "Start (YYYY-MM-DD)", "2020-01-01"); f.pack(side="left", padx=4)
        f, self.end = labeled_entry(r, "End (YYYY-MM-DD)", date.today().isoformat()); f.pack(side="left", padx=4)
        fi = ctk.CTkFrame(r, fg_color="transparent"); fi.pack(side="left", padx=4)
        ctk.CTkLabel(fi, text="Interval", text_color=T.MUTED, font=(T.FONT, 11)).pack(anchor="w")
        self.interval = tk.StringVar(value="1d")
        ctk.CTkOptionMenu(fi, values=VALID_INTERVALS, variable=self.interval, width=90, fg_color=T.CARD, button_color=T.CARD_HI).pack()
        r2 = self.row()
        button(r2, "Fetch Market Data", self.fetch, T.GREEN).pack(side="left", padx=4)
        button(r2, "Load CSV", self.load_csv).pack(side="left", padx=4)
        button(r2, "Use Synthetic Data", self.use_synthetic, T.CARD_HI).pack(side="left", padx=4)
        button(r2, "Export CSV", self.export, T.CARD_HI).pack(side="left", padx=4)
        self.status = self.note("")
        self.chart = ChartFrame(self.body, 4.2); self.chart.pack(fill="x", pady=6)
        self.table = DataTable(self.body, 9); self.table.pack(fill="x", pady=6)

    def fetch(self) -> None:
        try:
            tickers = parse_tickers(self.tickers.get())
        except MarketDataError as e:
            return self.app.show_error(str(e))
        s, e_, iv = self.start.get().strip(), self.end.get().strip(), self.interval.get()
        self.app.run_task(lambda: fetch_yfinance(tickers, s, e_, iv), "Downloading from Yahoo Finance...", self._fetched)

    def _fetched(self, res) -> None:
        df, rep = res
        self.pipe.set_data(df, None, "Yahoo Finance")
        msg = f"{rep['rows']} rows, tickers {rep['returned']} | retrieved {rep['retrieved_at_utc']} UTC | adjusted prices"
        if rep["missing_tickers"]:
            msg += f" | NO DATA for {rep['missing_tickers']}"
        self.status.configure(text=msg)
        self.app.refresh_all()

    def load_csv(self) -> None:
        path = filedialog.askopenfilename(initialdir=DATA_DIR, filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            self.pipe.set_data(load_prices_csv(path), None, f"CSV: {path.split('/')[-1].split(chr(92))[-1]}")
        except MarketDataError as e:
            return self.app.show_error(str(e))
        self.status.configure(text="CSV loaded.")
        self.app.refresh_all()

    def use_synthetic(self) -> None:
        self.app.load_synthetic()
        self.status.configure(text="Synthetic (fictional) data loaded - NOT real market data.")

    def export(self) -> None:
        if self.pipe.prices is None:
            return self.app.show_error("No data to export.")
        path = filedialog.asksaveasfilename(initialdir=OUTPUT_DIR, defaultextension=".csv", initialfile="prices_export.csv")
        if path:
            export_csv(self.pipe.prices, path)
            self.status.configure(text=f"Exported to {path}")

    def refresh(self) -> None:
        df = self.pipe.prices
        if df is None:
            return
        ax_p, ax_v = self.chart.reset(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        for t, g in list(df.groupby("Ticker"))[:8]:
            ax_p.plot(g["Date"], g["Close"], label=t, linewidth=1.2)
        ax_p.set_title("Close price" + ("  (SYNTHETIC)" if "ynthetic" in self.pipe.data_source else ""))
        self.chart.legend(ax_p)
        first = df["Ticker"].iloc[0]
        g = df[df["Ticker"] == first]
        ax_v.bar(g["Date"], g["Volume"], color=T.ACCENT, width=1.0)
        ax_v.set_title(f"Volume - {first}", fontsize=9)
        self.chart.draw()
        self.table.set(df.tail(100).iloc[::-1].reset_index(drop=True))


# ------------------------------------------------------------- Data Preparation
class DataPrepPage(Page):
    title = "Data Preparation"
    subtitle = ("Features use only past data; the label is the forward return Close[t+h]/Close[t]-1. Chronological split with an "
                "embargo (>= horizon) so no label window crosses a split boundary. Scaler is fit on the training split only.")

    def build(self) -> None:
        r = self.row()
        f, self.horizon = labeled_entry(r, "Horizon (days)", "20", 90); f.pack(side="left", padx=4)
        f, self.embargo = labeled_entry(r, "Embargo (days)", "20", 90); f.pack(side="left", padx=4)
        f, self.train_end = labeled_entry(r, "Validation starts (blank=auto)", "", 190); f.pack(side="left", padx=4)
        f, self.val_end = labeled_entry(r, "Test starts (blank=auto)", "", 190); f.pack(side="left", padx=4)
        button(r, "Prepare Data", lambda: self.app.run_stage("prepare", "Preparing data..."), T.GREEN).pack(side="left", padx=12, pady=(18, 0))
        self.info = self.note("")
        ctk.CTkLabel(self.body, text="Chronological splits", text_color=T.TEXT, font=(T.FONT, 14, "bold")).pack(anchor="w")
        self.split_t = DataTable(self.body, 3); self.split_t.pack(fill="x", pady=4)
        ctk.CTkLabel(self.body, text="Normalization statistics (training data only)", text_color=T.TEXT, font=(T.FONT, 14, "bold")).pack(anchor="w")
        self.scale_t = DataTable(self.body, 9); self.scale_t.pack(fill="x", pady=4)
        ctk.CTkLabel(self.body, text="Engineered features + forward-return label (latest rows)", text_color=T.TEXT, font=(T.FONT, 14, "bold")).pack(anchor="w")
        self.feat_t = DataTable(self.body, 8); self.feat_t.pack(fill="x", pady=4)

    def apply_config(self, cfg) -> None:
        cfg.horizon = _f(self.horizon, "Horizon", int, 1, 250)
        cfg.embargo = _f(self.embargo, "Embargo", int, 0, 500)
        if cfg.embargo < cfg.horizon:
            raise ValueError("Embargo must be >= horizon to prevent label leakage.")
        a, b = self.train_end.get().strip(), self.val_end.get().strip()
        if bool(a) != bool(b):
            raise ValueError("Set both split dates or leave both blank.")
        if a:
            try:
                pd.Timestamp(a), pd.Timestamp(b)
            except ValueError:
                raise ValueError("Split dates must be YYYY-MM-DD.")
        cfg.train_end, cfg.val_end = a or None, b or None

    def refresh(self) -> None:
        p = self.pipe
        if p.split is None:
            return
        s = p.split
        self.info.configure(text=f"Embargo {s.embargo} trading days | horizon {s.horizon} | validation starts {s.train_end.date()} | "
                                 f"test starts {s.val_end.date()} | excluded observations: {s.excluded_total} | feature rows: {len(p.features)}")
        self.split_t.set(s.summary)
        self.scale_t.set(p.scaler.stats())
        self.feat_t.set(p.features.tail(60).drop(columns=["label_end_date"]).iloc[::-1].reset_index(drop=True))


# ------------------------------------------------------------------ Neural net
class NeuralNetPage(Page):
    title = "Neural Network"
    subtitle = "PyTorch feedforward regressor (Input -> 64 -> ReLU -> 32 -> ReLU -> 1), Adam + MSE on the continuous forward return."

    def build(self) -> None:
        r = self.row()
        f, self.hidden = labeled_entry(r, "Hidden layers", "64, 32"); f.pack(side="left", padx=4)
        f, self.lr = labeled_entry(r, "Learning rate", "0.001"); f.pack(side="left", padx=4)
        f, self.epochs = labeled_entry(r, "Epochs", "100", 80); f.pack(side="left", padx=4)
        f, self.batch = labeled_entry(r, "Batch size", "32", 80); f.pack(side="left", padx=4)
        f, self.seed = labeled_entry(r, "Seed", "42", 70); f.pack(side="left", padx=4)
        button(r, "Train Models", lambda: self.app.run_stage("train", "Training ANN + baselines..."), T.GREEN).pack(side="left", padx=12, pady=(18, 0))
        self.info = self.note("Training runs in a background thread; progress is shown in the sidebar.")
        self.chart = ChartFrame(self.body, 3.6); self.chart.pack(fill="x", pady=6)

    def apply_config(self, cfg) -> None:
        try:
            cfg.hidden_layers = tuple(int(x) for x in self.hidden.get().replace(";", ",").split(",") if x.strip())
        except ValueError:
            raise ValueError("Hidden layers must be integers like 64, 32")
        if not cfg.hidden_layers or min(cfg.hidden_layers) < 1 or max(cfg.hidden_layers) > 1024:
            raise ValueError("Hidden layer sizes must be between 1 and 1024")
        cfg.learning_rate = _f(self.lr, "Learning rate", float, 1e-6, 1.0)
        cfg.epochs = _f(self.epochs, "Epochs", int, 1, 2000)
        cfg.batch_size = _f(self.batch, "Batch size", int, 1, 4096)
        cfg.seed = _f(self.seed, "Seed", int, 0, 2**31 - 1)

    def refresh(self) -> None:
        a = self.pipe.ann
        if a is None:
            return
        h = a.history
        ax = self.chart.reset()
        ax.plot(range(1, len(h.train_loss) + 1), h.train_loss, label="Training loss", color=T.ACCENT)
        ax.plot(range(1, len(h.val_loss) + 1), h.val_loss, label="Validation loss", color=T.AMBER)
        ax.axvline(h.best_epoch, color=T.GREEN, linestyle="--", linewidth=1, label=f"Best epoch {h.best_epoch}")
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (standardized target)"); ax.set_title("Training vs validation loss")
        self.chart.legend(ax); self.chart.draw()
        self.info.configure(text=f"Trained {len(h.train_loss)} epochs; weights restored from best validation epoch {h.best_epoch} "
                                 f"(val loss {min(h.val_loss):.4f}). Architecture: {a.hidden}.")


# ------------------------------------------------------------ Model comparison
class ComparisonPage(Page):
    title = "Model Comparison"
    subtitle = ("Mean-return baseline vs Random Forest *regressor* (Assignment 2's classifier reconstructed as a regressor for a fair "
                "continuous-return comparison) vs PyTorch ANN. Validation data selects; the test set is the final evaluation.")

    def build(self) -> None:
        self.table = DataTable(self.body, 4); self.table.pack(fill="x", pady=6)
        self.verdict = self.note("")
        self.chart = ChartFrame(self.body, 3.0); self.chart.pack(fill="x", pady=6)

    def refresh(self) -> None:
        c = self.pipe.comparison
        if c is None:
            return
        self.table.set(c, formats={"Best on Validation (RMSE)": lambda v: "yes" if v else ""})
        t = c.set_index("Model")
        best_val = t["Val RMSE"].idxmin()
        ann_better = t.loc["PyTorch ANN", "Test RMSE"] < min(t.loc["Mean-return baseline", "Test RMSE"], t.loc["Random Forest", "Test RMSE"])
        self.verdict.configure(text=f"Best validation RMSE: {best_val}. On the test set the ANN "
                                    f"{'beats' if ann_better else 'does NOT beat'} both the baseline and the random forest (RMSE). "
                                    "Daily-return forecasting is hard: R2 near or below 0 is normal and no superiority is claimed unless shown.")
        ax = self.chart.reset()
        x = np.arange(len(t))
        ax.bar(x - 0.2, t["Val RMSE"], 0.4, label="Validation RMSE", color=T.ACCENT)
        ax.bar(x + 0.2, t["Test RMSE"], 0.4, label="Test RMSE", color=T.AMBER)
        ax.set_xticks(x); ax.set_xticklabels(t.index, fontsize=9); ax.set_title("RMSE by model (lower is better)")
        self.chart.legend(ax); self.chart.draw()


# ------------------------------------------------------------------ Sentiment
class SentimentPage(Page):
    title = "News Sentiment"
    subtitle = "Gemini scores headlines from -1 (negative) to +1 (positive). Headlines are treated as untrusted data. Gemini OFF -> neutral 0.0 fallback (clearly labelled)."

    def build(self) -> None:
        r = self.row()
        self.use_gemini = tk.BooleanVar(value=False)
        ctk.CTkSwitch(r, text="Enable Gemini", variable=self.use_gemini, command=self._toggle, progress_color=T.GREEN).pack(side="left", padx=6)
        f, self.window = labeled_entry(r, "News window (days)", "30", 120); f.pack(side="left", padx=10)
        button(r, "Load News CSV", self.load_news).pack(side="left", padx=4, pady=(18, 0))
        button(r, "Analyze Sentiment", lambda: self.app.run_stage("sentiment", "Scoring headlines..."), T.GREEN).pack(side="left", padx=4, pady=(18, 0))
        self.status = self.note("API key is read from .env (GEMINI_API_KEY); it is never shown or stored in the app.")
        self.table = DataTable(self.body, 8); self.table.pack(fill="x", pady=6)
        self.table.on_select(self._show_headlines)
        self.heads = ctk.CTkTextbox(self.body, height=170, fg_color=T.CARD, text_color=T.TEXT)
        self.heads.pack(fill="x", pady=6)
        self._results = []

    def _toggle(self) -> None:
        self.pipe.invalidate("sentiment")
        self.app.refresh_all()

    def load_news(self) -> None:
        path = filedialog.askopenfilename(initialdir=DATA_DIR, filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            self.pipe.news = load_news_csv(path)
        except MarketDataError as e:
            return self.app.show_error(str(e))
        self.pipe.invalidate("sentiment")
        self.status.configure(text=f"Loaded {len(self.pipe.news)} headlines.")
        self.app.refresh_all()

    def apply_config(self, cfg) -> None:
        cfg.use_gemini = bool(self.use_gemini.get())
        cfg.news_window_days = _f(self.window, "News window", int, 1, 3650)

    def _show_headlines(self, idx: int) -> None:
        self.heads.configure(state="normal"); self.heads.delete("1.0", "end")
        if idx < len(self._results):
            r = self._results[idx]
            self.heads.insert("end", f"{r.ticker}: {r.reason}\n\nHeadlines used ({len(r.headlines)}):\n" + "\n".join(f"- {h}" for h in r.headlines))
        self.heads.configure(state="disabled")

    def refresh(self) -> None:
        p = self.pipe
        news = "no news loaded" if p.news is None else f"{len(p.news)} headlines loaded"
        self.status.configure(text=f"Gemini: {'ON' if p.cfg.use_gemini else 'OFF'} | status: {p.sentiment_status} | {news}")
        self._results = p.sentiment_results or []
        if self._results:
            self.table.set(pd.DataFrame([{"Ticker": r.ticker, "Sentiment": r.sentiment, "Mode": r.mode, "Reason": r.reason} for r in self._results]), color_col="Sentiment")


# -------------------------------------------------------------------- Ranking
class RankingPage(Page):
    title = "Stock Ranking"
    subtitle = "final = (1-w) x normalized predicted return + w x normalized sentiment. Return scores are min-max scaled across today's candidates only (no look-ahead)."

    def build(self) -> None:
        r = self.row()
        self.w = tk.DoubleVar(value=0.25)
        fw = ctk.CTkFrame(r, fg_color="transparent"); fw.pack(side="left", padx=6)
        self.wlabel = ctk.CTkLabel(fw, text="Sentiment weight w = 0.25", text_color=T.MUTED, font=(T.FONT, 11)); self.wlabel.pack(anchor="w")
        ctk.CTkSlider(fw, from_=0, to=1, number_of_steps=20, variable=self.w, width=240, command=lambda v: self.wlabel.configure(text=f"Sentiment weight w = {v:.2f}")).pack()
        f, self.topn = labeled_entry(r, "Top N stocks", "3", 80); f.pack(side="left", padx=12)
        button(r, "Rank Stocks", lambda: self.app.run_stage("rank", "Ranking stocks..."), T.GREEN).pack(side="left", padx=4, pady=(18, 0))
        self.table = DataTable(self.body, 8); self.table.pack(fill="x", pady=6)
        self.chart = ChartFrame(self.body, 3.0); self.chart.pack(fill="x", pady=6)

    def apply_config(self, cfg) -> None:
        cfg.sentiment_weight = round(float(self.w.get()), 4)
        cfg.top_n = _f(self.topn, "Top N", int, 1, 500)

    def refresh(self) -> None:
        r = self.pipe.ranking
        if r is None:
            return
        view = r[["Rank", "Ticker", "Predicted Return", "Sentiment", "Sentiment Mode", "Final Score", "Selected", "Tied", "Sentiment Reason"]]
        self.table.set(view, color_col="Predicted Return", formats={"Predicted Return": lambda v: f"{v:+.3%}", "Sentiment": lambda v: f"{v:+.2f}"})
        ax = self.chart.reset()
        ax.bar(r["Ticker"], r["Final Score"], color=[T.GREEN if s else T.CARD_HI for s in r["Selected"]])
        ax.set_title("Final score (green = selected)"); self.chart.draw()


# ------------------------------------------------------------------ Portfolio
class PortfolioPage(Page):
    title = "Portfolio Allocation"
    subtitle = "Deterministic whole-share allocation (Gemini never sets quantities). No borrowing, no shorting; fees are reserved so the target is reachable."

    def build(self) -> None:
        r = self.row()
        f, self.cash = labeled_entry(r, "Available cash ($)", "10000", 130); f.pack(side="left", padx=4)
        f, self.fee = labeled_entry(r, "Fee per trade ($)", "1.00", 110); f.pack(side="left", padx=4)
        f, self.holdings = labeled_entry(r, "Current holdings (TICKER:SHARES, ...)", "ALPH:10, BRVO:5", 300); f.pack(side="left", padx=4)
        fm = ctk.CTkFrame(r, fg_color="transparent"); fm.pack(side="left", padx=4)
        ctk.CTkLabel(fm, text="Method", text_color=T.MUTED, font=(T.FONT, 11)).pack(anchor="w")
        self.method = tk.StringVar(value="equal")
        ctk.CTkOptionMenu(fm, values=["equal", "score"], variable=self.method, width=100, fg_color=T.CARD, button_color=T.CARD_HI).pack()
        button(r, "Allocate Portfolio", lambda: self.app.run_stage("allocate", "Allocating..."), T.GREEN).pack(side="left", padx=12, pady=(18, 0))
        self.summary = self.note("")
        self.table = DataTable(self.body, 6); self.table.pack(fill="x", pady=6)
        self.chart = ChartFrame(self.body, 3.0); self.chart.pack(fill="x", pady=6)

    def apply_config(self, cfg) -> None:
        cfg.cash = _f(self.cash, "Cash", float, 0)
        cfg.fee = _f(self.fee, "Fee", float, 0)
        cfg.holdings = parse_holdings(self.holdings.get())
        cfg.allocation_method = self.method.get()

    def refresh(self) -> None:
        a = self.pipe.allocation
        if a is None:
            return
        self.summary.configure(text=f"Capital (cash + holdings at snapshot prices): ${a.capital:,.2f} | estimated cash after trades & fees: ${a.estimated_final_cash:,.2f}"
                                    f" | price snapshot {self.pipe.snapshot['timestamp']} (data as of {self.pipe.as_of.date()})")
        self.table.set(a.table, formats={"Weight": lambda v: f"{v:.1%}", "Allocation %": lambda v: f"{v:.2f}%"})
        ax = self.chart.reset()
        ax.pie(a.table["Target Value"].clip(lower=0.001), labels=a.table["Ticker"], autopct="%1.0f%%", textprops={"color": T.TEXT, "fontsize": 9})
        ax.set_title("Target allocation (by value)"); ax.grid(False); self.chart.draw()


# ---------------------------------------------------------------- Trade planner
class TradePage(Page):
    title = "Trade Planner (Dijkstra)"
    subtitle = ("Minimum-fee sequence of buys/sells from current holdings to the target. State = (holdings, cash); edge cost = fee. "
                "'direct' = one trade per changed ticker (fast, provably fee-optimal); 'toward'/'full' search more moves (slower).")

    def build(self) -> None:
        r = self.row()
        fm = ctk.CTkFrame(r, fg_color="transparent"); fm.pack(side="left", padx=4)
        ctk.CTkLabel(fm, text="Move mode", text_color=T.MUTED, font=(T.FONT, 11)).pack(anchor="w")
        self.mode = tk.StringVar(value="direct")
        ctk.CTkOptionMenu(fm, values=["direct", "toward", "full"], variable=self.mode, width=100, fg_color=T.CARD, button_color=T.CARD_HI).pack()
        f, self.max_states = labeled_entry(r, "Max states", "200000", 110); f.pack(side="left", padx=8)
        button(r, "Plan Trades", lambda: self.app.run_stage("plan", "Running Dijkstra..."), T.GREEN).pack(side="left", padx=8, pady=(18, 0))
        self.summary = self.note("")
        ctk.CTkLabel(self.body, text="Starting vs target portfolio", text_color=T.TEXT, font=(T.FONT, 14, "bold")).pack(anchor="w")
        self.hold_t = DataTable(self.body, 5); self.hold_t.pack(fill="x", pady=4)
        ctk.CTkLabel(self.body, text="Transaction sequence", text_color=T.TEXT, font=(T.FONT, 14, "bold")).pack(anchor="w")
        self.trade_t = DataTable(self.body, 8); self.trade_t.pack(fill="x", pady=4)

    def apply_config(self, cfg) -> None:
        cfg.planner_mode = self.mode.get()
        cfg.max_states = _f(self.max_states, "Max states", int, 10, 5_000_000)

    def refresh(self) -> None:
        p = self.pipe.plan
        if p is None:
            return
        col = T.GREEN if p.status == "optimal" else T.RED if p.status == "unreachable" else T.AMBER
        self.summary.configure(text=f"Status: {p.status.upper()} - {p.message} | explored states: {p.explored_states} | transactions: {len(p.trades)} | "
                                    f"total fees: ${p.total_fee:,.2f} | remaining cash: ${p.final_cash:,.2f}", text_color=col)
        tick = sorted(set(p.start_holdings) | set(p.target_holdings))
        self.hold_t.set(pd.DataFrame({"Ticker": tick, "Start Shares": [p.start_holdings.get(t, 0) for t in tick],
                                      "Target Shares": [p.target_holdings.get(t, 0) for t in tick],
                                      "Final Shares": [p.final_holdings.get(t, 0) for t in tick]}))
        self.trade_t.set(p.trades)


# ----------------------------------------------------------------- Explanation
class ExplanationPage(Page):
    title = "AI Explanation"
    subtitle = "All numbers are computed in Python first, then passed to Gemini (if enabled) which only narrates them. With Gemini off, a deterministic explanation is shown."

    def build(self) -> None:
        r = self.row()
        button(r, "Generate Explanation", lambda: self.app.run_stage("explain", "Generating explanation..."), T.GREEN, 190).pack(side="left", padx=4)
        self.source = self.note("")
        ctk.CTkLabel(self.body, text="Exact prompt sent to Gemini", text_color=T.TEXT, font=(T.FONT, 14, "bold")).pack(anchor="w")
        self.prompt = ctk.CTkTextbox(self.body, height=260, fg_color=T.CARD, text_color=T.TEXT); self.prompt.pack(fill="x", pady=4)
        ctk.CTkLabel(self.body, text="Explanation", text_color=T.TEXT, font=(T.FONT, 14, "bold")).pack(anchor="w")
        self.text = ctk.CTkTextbox(self.body, height=200, fg_color=T.CARD, text_color=T.TEXT, wrap="word"); self.text.pack(fill="x", pady=4)

    def refresh(self) -> None:
        e = self.pipe.explanation
        if e is None:
            return
        self.source.configure(text=f"Source: {e['source']}")
        for box, val in ((self.prompt, e["prompt"]), (self.text, e["text"])):
            box.configure(state="normal"); box.delete("1.0", "end"); box.insert("end", val); box.configure(state="disabled")
