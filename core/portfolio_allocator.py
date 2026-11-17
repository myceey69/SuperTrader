"""Deterministic capital allocation (Gemini never touches these numbers).

capital = cash + market value of current holdings (snapshot prices); the target
holds only the selected stocks. Shares = floor(capital * weight / price), then
reduced until the plan is feasible (cash >= 0 incl. fees, sells-first order) and
finally greedy top-up with leftover cash in rank order.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.trade_planner import direct_plan_feasible


@dataclass
class TargetPortfolio:
    target_shares: dict[str, int]
    table: pd.DataFrame
    capital: float
    estimated_final_cash: float


def allocate_portfolio(selected: list[str], prices: dict[str, float], current_holdings: dict[str, int],
                       cash: float, fee: float, scores: dict[str, float] | None = None,
                       method: str = "equal") -> TargetPortfolio:
    if not selected:
        raise ValueError("No stocks selected")
    if cash < 0 or fee < 0:
        raise ValueError("cash and fee must be non-negative")
    for t in set(selected) | set(current_holdings):
        if t not in prices:
            raise ValueError(f"No price snapshot for {t}")
    if method == "score" and scores and sum(max(scores.get(t, 0), 0) for t in selected) > 0:
        tot = sum(max(scores[t], 0) for t in selected)
        weights = {t: max(scores[t], 0) / tot for t in selected}
    else:
        weights = {t: 1 / len(selected) for t in selected}
    capital = cash + sum(q * prices[t] for t, q in current_holdings.items())
    target = {t: int(capital * weights[t] // prices[t]) for t in selected}

    def feasible(tg):
        return direct_plan_feasible(current_holdings, cash, tg, prices, fee)

    guard = 0
    while not feasible(target) and guard < 100_000:           # shed shares from the largest position
        held = [t for t in selected if target[t] > 0]
        if not held:
            break
        big = max(held, key=lambda t: (target[t] * prices[t], t))
        target[big] -= 1
        guard += 1
    changed = True
    while changed:                                            # top-up with leftover cash
        changed = False
        for t in selected:
            target[t] += 1
            if feasible(target):
                changed = True
            else:
                target[t] -= 1
    spent = sum(target[t] * prices[t] for t in selected)
    n_trades = sum(1 for t in set(selected) | set(current_holdings) if target.get(t, 0) != current_holdings.get(t, 0))
    final_cash = capital - spent - n_trades * fee
    rows = [{"Ticker": t, "Weight": weights[t], "Price": prices[t], "Current Shares": current_holdings.get(t, 0),
             "Target Shares": target[t], "Target Value": target[t] * prices[t],
             "Allocation %": 100 * target[t] * prices[t] / capital if capital else 0.0} for t in selected]
    return TargetPortfolio({t: q for t, q in target.items() if q > 0},
                           pd.DataFrame(rows), capital, final_cash)
