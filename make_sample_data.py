"""Regenerate the clearly-labelled SYNTHETIC demo datasets in data/."""
from config import DATA_DIR
from core.data_loader import export_csv
from core.market_data import generate_synthetic_news, generate_synthetic_prices

if __name__ == "__main__":
    prices = generate_synthetic_prices()
    export_csv(prices, DATA_DIR / "sample_prices.csv")
    export_csv(generate_synthetic_news(prices), DATA_DIR / "sample_news.csv")
    print("Wrote synthetic data (NOT real market data) to", DATA_DIR)
