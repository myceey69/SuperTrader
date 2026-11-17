"""Central configuration: paths, feature list and the PipelineConfig dataclass."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
load_dotenv(BASE_DIR / ".env")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def gemini_api_key() -> str:
    """API key is read from the environment / .env only. Never hardcoded."""
    return os.getenv("GEMINI_API_KEY", "").strip()


FEATURE_COLUMNS = [
    "daily_return", "return_5d", "return_20d", "ma5_ratio", "ma20_ratio",
    "volatility_20d", "volume_ratio_5d", "volume_ratio_20d", "momentum_10d",
]
LABEL_COLUMN = "forward_return"


@dataclass
class PipelineConfig:
    # data / labels
    horizon: int = 20
    embargo: int = 20
    train_end: Optional[str] = None   # first date of validation split (None -> 60th pct of dates)
    val_end: Optional[str] = None     # first date of test split (None -> 80th pct of dates)
    # ANN
    hidden_layers: tuple[int, ...] = (64, 32)
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    seed: int = 42
    rf_trees: int = 100
    # sentiment / ranking
    use_gemini: bool = False
    sentiment_weight: float = 0.25
    top_n: int = 3
    news_window_days: int = 30
    # portfolio / trading
    cash: float = 10_000.0
    fee: float = 1.0
    holdings: dict[str, int] = field(default_factory=dict)
    allocation_method: str = "equal"   # "equal" | "score"
    planner_mode: str = "direct"       # "direct" | "toward" | "full"
    max_states: int = 200_000
