"""Dijkstra minimum-fee trade planner (heapq).

STATE  = (holdings tuple in sorted-ticker order, cash in integer cents).
EDGE   = one valid buy/sell of k whole shares of one ticker, cost = fixed fee.
GOAL   = holdings == target exactly.  Cash/prices use integer cents (exact math).
Assumptions: whole shares, no shorting, no borrowing, fixed price snapshot,
sell proceeds usable immediately, cash >= 0 after every trade (fee included).

Move modes (the trade-off between exactness and speed):
  full   : every k, holdings bounded to [0, max(start,target)] per ticker.
           Exact for the bounded model; state space grows quickly (small cases).
  toward : any k, but never moves a ticker away from / past its target.
  direct : one trade per changed ticker straight to target (default).
           Justification: every changed ticker needs >= 1 trade, so
           fee*changed_tickers is a lower bound; direct reaches it whenever ANY
           valid plan exists (sell everything first, largest net proceeds first).
           Tests compare 'direct' against 'full' on small cases.
Priority = (total fee, number of trades, state) -> deterministic; zero fees terminate
because visited states are never re-expanded. Complexity: O((V+E) log V).
If `max_states` is hit before the goal, status is 'incomplete' (NOT claimed optimal).
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class PlanResult:
    status: str                      # optimal | unreachable | incomplete
    mode: str
    trades: pd.DataFrame
    total_fee: float
    explored_states: int
    final_holdings: dict[str, int]
    final_cash: float
    message: str = ""
    start_holdings: dict[str, int] = field(default_factory=dict)
    target_holdings: dict[str, int] = field(default_factory=dict)


def _cents(x: float) -> int:
    return int(round(float(x) * 100))


def _moves(h, t, cash, price, fee, mode, bound):
    """Yield (index, signed_shares, new_cash) for valid moves, deterministic order."""
    for i in range(len(h)):
        cur, tgt = h[i], t[i]
        if mode == "direct":
            ks = [tgt - cur] if cur != tgt else []
        elif mode == "toward":
            d = tgt - cur
            ks = list(range(1, d + 1)) if d > 0 else list(range(-1, d - 1, -1)) if d < 0 else []
        else:  # full
            ks = list(range(-cur, 0)) + list(range(1, bound[i] - cur + 1))
        for k in ks:
            new_cash = cash - k * price[i] - fee
            if new_cash >= 0 and cur + k >= 0:
                yield i, k, new_cash


def plan_trades(start_holdings: dict[str, int], start_cash: float, target_holdings: dict[str, int],
                prices: dict[str, float], fee: float, mode: str = "direct",
                max_states: int = 200_000) -> PlanResult:
    if mode not in ("direct", "toward", "full"):
        raise ValueError("mode must be direct, toward or full")
    if fee < 0 or start_cash < 0:
        raise ValueError("fee and cash must be non-negative")
    tickers = sorted(set(start_holdings) | set(target_holdings))
    for t in tickers:
        if t not in prices or prices[t] <= 0:
            raise ValueError(f"Missing/invalid price for {t}")
    if any(v < 0 for v in list(start_holdings.values()) + list(target_holdings.values())):
        raise ValueError("Holdings cannot be negative (no short selling)")
    s = tuple(int(start_holdings.get(t, 0)) for t in tickers)
    g = tuple(int(target_holdings.get(t, 0)) for t in tickers)
    price = [_cents(prices[t]) for t in tickers]
    fee_c, cash_c = _cents(fee), _cents(start_cash)
    bound = [max(a, b) for a, b in zip(s, g)]
    start = (s, cash_c)

    def build(status, path, explored, msg=""):
        rows, cash = [], cash_c
        for step, (i, k) in enumerate(path, 1):
            cash -= k * price[i] + fee_c
            rows.append({"Step": step, "Action": "BUY" if k > 0 else "SELL", "Ticker": tickers[i],
                         "Shares": abs(k), "Price": price[i] / 100, "Trade Value": abs(k) * price[i] / 100,
                         "Fee": fee_c / 100, "Remaining Cash": cash / 100})
        hold = list(s)
        for i, k in path:
            hold[i] += k
        return PlanResult(status, mode, pd.DataFrame(rows, columns=["Step", "Action", "Ticker", "Shares", "Price",
                          "Trade Value", "Fee", "Remaining Cash"]), len(path) * fee_c / 100, explored,
                          dict(zip(tickers, hold)), cash / 100, msg, dict(zip(tickers, s)), dict(zip(tickers, g)))

    if s == g:
        return build("optimal", [], 0, "Portfolio already matches the target.")
    dist = {start: (0, 0)}
    pred: dict = {}
    heap = [(0, 0, start)]
    visited: set = set()
    while heap:
        cost, steps, state = heapq.heappop(heap)
        if state in visited:
            continue
        visited.add(state)
        if state[0] == g:                                   # reconstruct path
            path, cur = [], state
            while cur != start:
                prev, mv = pred[cur]
                path.append(mv)
                cur = prev
            path.reverse()
            return build("optimal", path, len(visited), f"Optimal ({mode} moves).")
        if len(visited) >= max_states:
            return build("incomplete", [], len(visited),
                         f"Search stopped at {max_states} states - optimality NOT established.")
        h, cash = state
        for i, k, new_cash in _moves(h, g, cash, price, fee_c, mode, bound):
            nh = h[:i] + (h[i] + k,) + h[i + 1:]
            ns, nd = (nh, new_cash), (cost + fee_c, steps + 1)
            if ns not in visited and nd < dist.get(ns, (float("inf"), 0)):
                dist[ns] = nd
                pred[ns] = (state, (i, k))
                heapq.heappush(heap, (nd[0], nd[1], ns))
    return build("unreachable", [], len(visited), "Target cannot be reached with the available cash and fees.")


def direct_plan_feasible(start_holdings, start_cash, target_holdings, prices, fee) -> bool:
    """Fast check used by the allocator: sells first (largest net proceeds first), then buys."""
    tickers = sorted(set(start_holdings) | set(target_holdings))
    cash = _cents(start_cash)
    fee_c = _cents(fee)
    deltas = {t: int(target_holdings.get(t, 0)) - int(start_holdings.get(t, 0)) for t in tickers}
    sells = sorted((t for t in tickers if deltas[t] < 0), key=lambda t: (-(-deltas[t] * _cents(prices[t])), t))
    buys = [t for t in tickers if deltas[t] > 0]
    for t in sells + buys:
        cash -= deltas[t] * _cents(prices[t]) + fee_c
        if cash < 0:
            return False
    return True
