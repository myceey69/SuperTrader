"""Reusable dark-themed widgets: MetricCard, DataTable, ChartFrame, form helpers."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

import customtkinter as ctk
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from gui import theme as T  # noqa: E402


def setup_styles() -> None:
    st = ttk.Style()
    st.theme_use("clam")
    st.configure("Dark.Treeview", background=T.CARD, fieldbackground=T.CARD, foreground=T.TEXT,
                 rowheight=26, borderwidth=0, font=(T.FONT, 10))
    st.configure("Dark.Treeview.Heading", background=T.CARD_HI, foreground=T.TEXT, relief="flat", font=(T.FONT, 10, "bold"))
    st.map("Dark.Treeview", background=[("selected", T.ACCENT)], foreground=[("selected", "white")])
    st.configure("Vertical.TScrollbar", background=T.CARD_HI, troughcolor=T.CARD, bordercolor=T.CARD, arrowcolor=T.TEXT)
    st.configure("Horizontal.TScrollbar", background=T.CARD_HI, troughcolor=T.CARD, bordercolor=T.CARD, arrowcolor=T.TEXT)


class MetricCard(ctk.CTkFrame):
    def __init__(self, master, title: str, value: str = "-"):
        super().__init__(master, fg_color=T.CARD, corner_radius=12)
        ctk.CTkLabel(self, text=title.upper(), text_color=T.MUTED, font=(T.FONT, 11)).pack(anchor="w", padx=14, pady=(10, 0))
        self.value = ctk.CTkLabel(self, text=value, text_color=T.TEXT, font=(T.FONT, 20, "bold"), wraplength=260, justify="left")
        self.value.pack(anchor="w", padx=14, pady=(2, 12))

    def set(self, text: str, color: str = T.TEXT) -> None:
        self.value.configure(text=text, text_color=color)


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    if isinstance(v, (pd.Timestamp,)):
        return v.strftime("%Y-%m-%d") if not pd.isna(v) else "-"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.4f}" if abs(v) < 100 else f"{v:,.2f}"
    return str(v)


class DataTable(ctk.CTkFrame):
    """Treeview in a rounded card. Row text is green/red by the sign of `color_col`."""

    def __init__(self, master, height: int = 8):
        super().__init__(master, fg_color=T.CARD, corner_radius=12)
        self.tree = ttk.Treeview(self, show="headings", height=height, style="Dark.Treeview", selectmode="browse")
        vs = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        hs = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        self.tree.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=(6, 0))
        vs.grid(row=0, column=1, sticky="ns", pady=(6, 0))
        hs.grid(row=1, column=0, sticky="ew", padx=(6, 0))
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.tree.tag_configure("pos", foreground=T.GREEN)
        self.tree.tag_configure("neg", foreground=T.RED)
        self._df: Optional[pd.DataFrame] = None

    def set(self, df: Optional[pd.DataFrame], color_col: Optional[str] = None, formats: Optional[dict] = None) -> None:
        self.tree.delete(*self.tree.get_children())
        self._df = df
        if df is None or df.empty and len(df.columns) == 0:
            self.tree["columns"] = ()
            return
        cols = [str(c) for c in df.columns]
        self.tree["columns"] = cols
        formats = formats or {}
        for c in cols:
            self.tree.heading(c, text=c)
            width = max([len(c)] + [len(_fmt(v)) for v in df[c].head(200)]) * 8 + 20
            self.tree.column(c, width=min(max(width, 70), 520), anchor="w", stretch=True)
        for _, row in df.iterrows():
            vals = [formats[c](row[c]) if c in formats else _fmt(row[c]) for c in df.columns]
            tag = ()
            if color_col is not None and color_col in df.columns and isinstance(row[color_col], (int, float, np.number)):
                tag = ("pos",) if row[color_col] > 0 else ("neg",) if row[color_col] < 0 else ()
            self.tree.insert("", "end", values=vals, tags=tag)

    def on_select(self, callback: Callable[[int], None]) -> None:
        self.tree.bind("<<TreeviewSelect>>", lambda e: callback(self.tree.index(self.tree.selection()[0])) if self.tree.selection() else None)


class ChartFrame(ctk.CTkFrame):
    def __init__(self, master, height_in: float = 3.2):
        super().__init__(master, fg_color=T.CARD, corner_radius=12)
        self.fig = Figure(figsize=(7, height_in), dpi=100, facecolor=T.CARD)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().configure(bg=T.CARD, highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

    def reset(self, nrows: int = 1, ncols: int = 1, **kw):
        self.fig.clear()
        axes = self.fig.subplots(nrows, ncols, **kw)
        for ax in np.atleast_1d(axes).ravel():
            self.style(ax)
        return axes

    @staticmethod
    def style(ax) -> None:
        ax.set_facecolor(T.PANEL)
        for sp in ax.spines.values():
            sp.set_color(T.CARD_HI)
        ax.tick_params(colors=T.MUTED, labelsize=8)
        ax.grid(color=T.CARD_HI, linewidth=0.5, alpha=0.7)
        ax.title.set_color(T.TEXT)
        ax.xaxis.label.set_color(T.MUTED)
        ax.yaxis.label.set_color(T.MUTED)

    def legend(self, ax) -> None:
        leg = ax.legend(facecolor=T.CARD, edgecolor=T.CARD_HI, fontsize=8, loc="best")
        for t in leg.get_texts():
            t.set_color(T.TEXT)

    def draw(self) -> None:
        self.fig.tight_layout()
        self.canvas.draw_idle()


def labeled_entry(parent, label: str, default: str = "", width: int = 110) -> tuple[ctk.CTkFrame, tk.StringVar]:
    var = tk.StringVar(value=default)
    f = ctk.CTkFrame(parent, fg_color="transparent")
    ctk.CTkLabel(f, text=label, text_color=T.MUTED, font=(T.FONT, 11)).pack(anchor="w")
    ctk.CTkEntry(f, textvariable=var, width=width, fg_color=T.CARD, border_color=T.CARD_HI).pack(anchor="w")
    return f, var


def button(parent, text: str, command, color: str = T.ACCENT, width: int = 150, **kw) -> ctk.CTkButton:
    return ctk.CTkButton(parent, text=text, command=command, fg_color=color, hover_color=T.CARD_HI,
                         corner_radius=8, width=width, font=(T.FONT, 12, "bold"), **kw)
