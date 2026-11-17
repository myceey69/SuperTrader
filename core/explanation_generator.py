"""Natural-language explanation. All numbers are computed first in Python and passed
explicitly; Gemini only narrates them."""
from __future__ import annotations

from typing import Optional

from core.gemini_client import GenerateFn

SYSTEM = ("You explain a finished portfolio plan to a university student. Use ONLY the numbers given; "
          "never invent, round differently or recompute figures. Do not give personal financial advice. "
          "Predicted returns are estimates, not guaranteed outcomes.")


def build_prompt(s: dict) -> str:
    lines = [f"Analysis date: {s['as_of']}  |  Sentiment weight: {s['weight']}  |  Top N: {s['top_n']}",
             f"Starting cash: {s['cash']:.2f}  |  Fee per trade: {s['fee']:.2f}", "", "RANKING (best first):"]
    for r in s["ranking"]:
        lines.append(f"- #{r['Rank']} {r['Ticker']}: predicted return {r['Predicted Return']:.4%}, sentiment "
                     f"{r['Sentiment']:+.2f} ({r['Sentiment Mode']}), final score {r['Final Score']:.4f}, "
                     f"selected={r['Selected']}")
    lines += ["", "TARGET ALLOCATION:"]
    for r in s["allocation"]:
        lines.append(f"- {r['Ticker']}: {r['Target Shares']} shares @ {r['Price']:.2f} = {r['Target Value']:.2f} "
                     f"({r['Allocation %']:.2f}% of capital {s['capital']:.2f})")
    lines += ["", f"TRADE PLAN ({s['plan_status']}, {s['n_trades']} trades, total fees {s['total_fee']:.2f}, "
              f"remaining cash {s['final_cash']:.2f}):"]
    for r in s["trades"]:
        lines.append(f"- Step {r['Step']}: {r['Action']} {r['Shares']} {r['Ticker']} @ {r['Price']:.2f}, fee {r['Fee']:.2f}")
    lines += ["", "Explain in about 200 words: why these stocks were selected, how predicted returns and sentiment "
              "influenced ranking, how capital was allocated, which trades are needed and what they cost."]
    return "\n".join(lines)


def deterministic_explanation(s: dict) -> str:
    sel = [r for r in s["ranking"] if r["Selected"]]
    parts = [f"Selected {len(sel)} stocks: " + ", ".join(f"{r['Ticker']} (predicted {r['Predicted Return']:.2%}, "
             f"sentiment {r['Sentiment']:+.2f}, score {r['Final Score']:.3f})" for r in sel) + ".",
             f"Final score = {1 - s['weight']:.2f} x normalized predicted return + {s['weight']:.2f} x normalized sentiment.",
             f"Capital of {s['capital']:.2f} was split across them using whole shares; "
             + ", ".join(f"{r['Ticker']}: {r['Target Shares']} shares" for r in s["allocation"]) + ".",
             f"The plan needs {s['n_trades']} trades costing {s['total_fee']:.2f} in fees; remaining cash {s['final_cash']:.2f}.",
             "Predicted returns are model estimates, not guaranteed outcomes."]
    return " ".join(parts)


def explain(s: dict, generate_fn: Optional[GenerateFn]) -> dict:
    prompt = build_prompt(s)
    if generate_fn is None:
        return {"prompt": prompt, "text": deterministic_explanation(s), "source": "deterministic (Gemini disabled)"}
    try:
        return {"prompt": prompt, "text": generate_fn(prompt, SYSTEM, False), "source": "gemini"}
    except Exception as exc:
        return {"prompt": prompt, "text": deterministic_explanation(s),
                "source": f"deterministic (Gemini error: {str(exc)[:80]})"}
