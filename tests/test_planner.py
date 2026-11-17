import itertools
from collections import deque

import pytest

from core.trade_planner import plan_trades

PR = {"A": 10.0, "B": 20.0, "C": 5.0}


def reference_min_trades(start, cash, target, prices, fee):
    """Independent BFS over the bounded 'full' model (levels = number of trades)."""
    tick = sorted(set(start) | set(target))
    bound = [max(start.get(t, 0), target.get(t, 0)) for t in tick]
    s0 = (tuple(start.get(t, 0) for t in tick), round(cash * 100))
    goal = tuple(target.get(t, 0) for t in tick)
    seen, q = {s0}, deque([(s0, 0)])
    while q:
        (h, c), d = q.popleft()
        if h == goal:
            return d
        for i, t in enumerate(tick):
            for nh_i in range(bound[i] + 1):
                k = nh_i - h[i]
                if k == 0:
                    continue
                nc = c - k * round(prices[t] * 100) - round(fee * 100)
                if nc < 0:
                    continue
                ns = (h[:i] + (nh_i,) + h[i + 1:], nc)
                if ns not in seen:
                    seen.add(ns)
                    q.append((ns, d + 1))
    return None


def test_simple_sell_then_buy():
    r = plan_trades({"A": 5}, 0.0, {"B": 2}, PR, 1.0)
    assert r.status == "optimal" and list(r.trades.Action) == ["SELL", "BUY"]      # must sell first
    assert r.total_fee == 2.0 and r.final_holdings == {"A": 0, "B": 2} and r.final_cash == pytest.approx(50 - 40 - 2)
    assert r.trades["Remaining Cash"].min() >= 0


def test_unreachable_when_fees_exceed_cash():
    r = plan_trades({"A": 1}, 0.0, {"B": 1}, PR, 8.0)         # sell nets 2, buy needs 28
    assert r.status == "unreachable" and r.trades.empty


def test_zero_fee_terminates_and_is_exact():
    r = plan_trades({"A": 3}, 0.0, {"A": 1, "B": 1}, PR, 0.0)
    assert r.status == "optimal" and r.total_fee == 0 and r.final_cash == pytest.approx(0.0)


def test_already_at_target():
    r = plan_trades({"A": 2}, 5.0, {"A": 2}, PR, 1.0)
    assert r.status == "optimal" and r.trades.empty and r.explored_states == 0


def test_incomplete_is_never_called_optimal():
    r = plan_trades({"A": 4, "B": 3}, 100.0, {"C": 20}, PR, 1.0, mode="full", max_states=5)
    assert r.status == "incomplete"


def test_deterministic():
    a = plan_trades({"A": 5, "B": 2}, 10.0, {"C": 9}, PR, 1.0)
    b = plan_trades({"A": 5, "B": 2}, 10.0, {"C": 9}, PR, 1.0)
    assert a.trades.equals(b.trades)


@pytest.mark.parametrize("start,cash,target,fee", [
    ({"A": 2}, 0.0, {"B": 1}, 1.0), ({"A": 3, "B": 1}, 2.0, {"C": 6}, 2.0), ({"A": 1}, 3.0, {"A": 2, "B": 1}, 1.5),
    ({"A": 2}, 0.0, {"A": 2, "B": 1}, 1.0), ({"B": 2}, 1.0, {"A": 1, "C": 2}, 3.0), ({"A": 1}, 0.0, {"B": 1}, 9.0),
])
def test_matches_exhaustive_reference(start, cash, target, fee):
    ref = reference_min_trades(start, cash, target, PR, fee)
    for mode in ("direct", "toward", "full"):
        r = plan_trades(start, cash, target, PR, fee, mode=mode)
        if ref is None:
            assert r.status == "unreachable", mode
        else:
            assert r.status == "optimal" and r.total_fee == pytest.approx(ref * fee), (mode, ref)
            assert r.final_holdings == {t: target.get(t, 0) for t in r.final_holdings}
            assert r.final_cash >= 0
