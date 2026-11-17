import pytest

from core.portfolio_allocator import allocate_portfolio
from core.trade_planner import direct_plan_feasible

PR = {"A": 100.0, "B": 33.0, "C": 250.0, "OLD": 10.0}


def test_equal_allocation_constraints():
    tp = allocate_portfolio(["A", "B"], PR, {}, 10_000, 5.0)
    assert tp.estimated_final_cash >= 0 and all(isinstance(v, int) and v >= 0 for v in tp.target_shares.values())
    spent = sum(q * PR[t] for t, q in tp.target_shares.items())
    assert spent + 2 * 5.0 <= 10_000
    assert direct_plan_feasible({}, 10_000, tp.target_shares, PR, 5.0)
    assert tp.estimated_final_cash < 100            # leftover cash topped up (cheaper than one A share)


def test_existing_holdings_and_sale_of_unselected():
    tp = allocate_portfolio(["A"], PR, {"OLD": 100}, 0.0, 2.0)
    assert tp.capital == 1000.0 and "OLD" not in tp.target_shares   # unselected holding -> target 0 -> planner sells it
    assert tp.target_shares["A"] == 9 and tp.estimated_final_cash >= 0


def test_fees_can_force_fewer_shares():
    tp = allocate_portfolio(["A"], PR, {}, 100.0, 1.0)      # 100 cash: a share costs 100 + fee
    assert tp.target_shares.get("A", 0) == 0


def test_validation():
    with pytest.raises(ValueError):
        allocate_portfolio([], PR, {}, 1, 1)
    with pytest.raises(ValueError):
        allocate_portfolio(["ZZ"], PR, {}, 1, 1)
