"""Main window: sidebar navigation, page container, background worker + progress.

Threading rule: worker threads NEVER touch widgets. They push messages into a
queue.Queue; the Tk main thread polls it every 100 ms and updates the GUI.
"""
from __future__ import annotations

import queue
import threading
import traceback
from tkinter import messagebox
from typing import Callable, Optional

import customtkinter as ctk

from config import DATA_DIR, PipelineConfig
from core.data_loader import load_news_csv, load_prices_csv
from core.market_data import generate_synthetic_news, generate_synthetic_prices
from core.pipeline import PortfolioPipeline
from gui import theme as T
from gui.pages import (ComparisonPage, DashboardPage, DataPrepPage, ExplanationPage, MarketExplorerPage, NeuralNetPage,
                       PortfolioPage, RankingPage, SentimentPage, TradePage)
from gui.widgets import button, setup_styles

NAV = [("Dashboard", DashboardPage), ("Market Explorer", MarketExplorerPage), ("Data Preparation", DataPrepPage),
       ("Neural Network", NeuralNetPage), ("Model Comparison", ComparisonPage), ("News Sentiment", SentimentPage),
       ("Stock Ranking", RankingPage), ("Portfolio Allocation", PortfolioPage), ("Trade Planner", TradePage),
       ("AI Explanation", ExplanationPage)]


class AppWindow(ctk.CTk):
    def __init__(self) -> None:
        ctk.set_appearance_mode("dark")
        super().__init__(fg_color=T.BG)
        self.title("AI Portfolio Optimizer - Intelligent Trade Planning")
        self.geometry("1440x900")
        self.minsize(1000, 650)
        setup_styles()
        self.q: queue.Queue = queue.Queue()
        self.busy = False
        self.pipe = PortfolioPipeline(PipelineConfig(), progress_cb=lambda s, f, m: self.q.put(("progress", s, f, m)))
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self.container = ctk.CTkFrame(self, fg_color=T.BG, corner_radius=0)
        self.container.grid(row=0, column=1, sticky="nsew")
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        self.pages: dict[str, ctk.CTkFrame] = {}
        for name, cls in NAV:
            self.pages[name] = cls(self.container, self)
        self.load_synthetic()
        self.show_page("Dashboard")
        self.after(100, self._poll)

    # ------------------------------------------------------------- layout
    def _build_sidebar(self) -> None:
        sb = ctk.CTkFrame(self, width=230, fg_color=T.PANEL, corner_radius=0)
        sb.grid(row=0, column=0, sticky="nsw")
        sb.grid_propagate(False)
        ctk.CTkLabel(sb, text="AI PORTFOLIO\nOPTIMIZER", font=(T.FONT, 18, "bold"), text_color=T.GREEN, justify="left").pack(anchor="w", padx=18, pady=(20, 14))
        self.nav_buttons = {}
        for name, _ in NAV:
            b = ctk.CTkButton(sb, text=name, anchor="w", fg_color="transparent", hover_color=T.CARD_HI, text_color=T.TEXT,
                              corner_radius=8, height=34, font=(T.FONT, 13), command=lambda n=name: self.show_page(n))
            b.pack(fill="x", padx=10, pady=2)
            self.nav_buttons[name] = b
        ctk.CTkFrame(sb, fg_color="transparent").pack(fill="both", expand=True)
        self.run_btn = button(sb, "RUN FULL PIPELINE", self.run_full, T.GREEN, 200, height=46)
        self.run_btn.pack(padx=14, pady=(8, 8))
        self.progress = ctk.CTkProgressBar(sb, progress_color=T.GREEN, fg_color=T.CARD)
        self.progress.set(0)
        self.progress.pack(fill="x", padx=14, pady=4)
        self.stage_lab = ctk.CTkLabel(sb, text="Idle", font=(T.FONT, 12, "bold"), text_color=T.TEXT)
        self.stage_lab.pack(anchor="w", padx=14)
        self.status_lab = ctk.CTkLabel(sb, text="", font=(T.FONT, 11), text_color=T.MUTED, wraplength=200, justify="left")
        self.status_lab.pack(anchor="w", padx=14, pady=(0, 16))

    def show_page(self, name: str) -> None:
        for n, b in self.nav_buttons.items():
            b.configure(fg_color=T.CARD_HI if n == name else "transparent")
        for n, page in self.pages.items():     # grid_forget/grid: tkraise() does not work on scrollable frames
            page.grid_forget()
        self.pages[name].grid(row=0, column=0, sticky="nsew")
        self.pages[name].refresh()

    def refresh_all(self) -> None:
        for p in self.pages.values():
            try:
                p.refresh()
            except Exception:   # a display problem must never crash the app
                traceback.print_exc()

    # -------------------------------------------------------------- data
    def load_synthetic(self) -> None:
        try:
            prices, news = load_prices_csv(DATA_DIR / "sample_prices.csv"), load_news_csv(DATA_DIR / "sample_news.csv")
        except Exception:
            prices = generate_synthetic_prices()
            news = generate_synthetic_news(prices)
        self.pipe.set_data(prices, news, "Synthetic demo data (fictional)")
        self.refresh_all()

    # ------------------------------------------------------------ workers
    def collect_config(self) -> PipelineConfig:
        cfg = PipelineConfig()
        for p in self.pages.values():
            p.apply_config(cfg)
        self.pipe.cfg = cfg
        return cfg

    def show_error(self, msg: str) -> None:
        self.stage_lab.configure(text="Error", text_color=T.RED)
        self.status_lab.configure(text=msg[:200], text_color=T.RED)
        messagebox.showerror("Error", msg)

    def run_task(self, fn: Callable, label: str, on_done: Optional[Callable] = None) -> None:
        if self.busy:
            self.status_lab.configure(text="A task is already running.", text_color=T.AMBER)
            return
        self.busy = True
        self.run_btn.configure(state="disabled")
        self.progress.set(0)
        self.stage_lab.configure(text=label, text_color=T.TEXT)
        self.status_lab.configure(text="", text_color=T.MUTED)

        def work() -> None:
            try:
                self.q.put(("done", fn(), on_done))
            except Exception as exc:
                self.q.put(("error", exc, traceback.format_exc()))

        threading.Thread(target=work, daemon=True).start()

    def run_stage(self, stage: str, label: str) -> None:
        try:
            self.collect_config()
        except ValueError as e:
            return self.show_error(str(e))
        if self.pipe.prices is None:
            return self.show_error("Load or download price data first (Market Explorer).")
        self.run_task(lambda: self.pipe.run_stage(stage), label)

    def run_full(self) -> None:
        try:
            self.collect_config()
        except ValueError as e:
            return self.show_error(str(e))
        if self.pipe.prices is None:
            return self.show_error("Load or download price data first (Market Explorer).")
        self.run_task(lambda: self.pipe.run_all(), "Running full pipeline...")

    def _poll(self) -> None:
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    _, stage, frac, text = msg
                    self.stage_lab.configure(text=f"Stage: {stage}", text_color=T.TEXT)
                    self.status_lab.configure(text=text, text_color=T.MUTED)
                    self.progress.set(min(max(frac, 0), 1))
                elif kind == "done":
                    self.busy = False
                    self.run_btn.configure(state="normal")
                    self.progress.set(1)
                    self.stage_lab.configure(text="Complete", text_color=T.GREEN)
                    if msg[2]:
                        msg[2](msg[1])
                    self.refresh_all()
                elif kind == "error":
                    self.busy = False
                    self.run_btn.configure(state="normal")
                    self.progress.set(0)
                    print(msg[2])
                    self.show_error(f"{type(msg[1]).__name__}: {msg[1]}")
        except queue.Empty:
            pass
        self.after(100, self._poll)
