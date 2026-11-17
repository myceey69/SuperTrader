import pytest
from core.market_data import generate_synthetic_news, generate_synthetic_prices


@pytest.fixture(scope="session")
def prices():
    return generate_synthetic_prices(n_days=500, tickers=["ALPH", "BRVO", "CHRL", "DLTA"], seed=3)


@pytest.fixture(scope="session")
def news(prices):
    return generate_synthetic_news(prices)
