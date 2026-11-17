# AI-Powered Portfolio Optimization and Intelligent Trade Planning System (DRAFT v0.1)

Desktop app (CustomTkinter, dark theme): market data -> features -> chronological split + embargo -> train-only scaling ->
PyTorch ANN / Random Forest / baseline -> Gemini sentiment -> ranking -> allocation -> Dijkstra trade plan -> explanation.

## Windows install & run
```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      (optional: add GEMINI_API_KEY; the app works fully offline without it)
python main.py
```
Click **RUN FULL PIPELINE** (sidebar). It starts on fictional synthetic data from `data/` (regenerate: `python make_sample_data.py`).
Tests: `python -m pytest -q`

## Assumptions (Assignments 1 and 2 were not available - reconstructed from the Assignment 3 text)
- Random forest is a *regressor* on the same continuous 20-day forward return (fair comparison).
- Momentum = 10-day rate of change. Zero/missing volume -> volume ratio 1.0 (neutral).
- Validation selects the ANN checkpoint (best epoch); the test set is only used for final metrics.
- Return scores = min-max across *today's* candidates (no look-ahead); sentiment score = (s+1)/2; 0.5 if all returns equal.
- Capital = cash + market value of current holdings at the price snapshot; unselected holdings are sold.
- Prices are rounded to cents inside the planner (integer-cent arithmetic). Fee = fixed amount per trade.
- Planner `direct` mode (default): one trade per changed ticker; a lower bound argument shows it is fee-optimal whenever any valid plan exists. `toward`/`full` search more moves; `full` is exact only inside bounds [0, max(start,target)] and may stop as "incomplete" at the state cap (never reported as optimal).
- Gemini model name comes from `GEMINI_MODEL` in `.env` (default `gemini-2.5-flash`): VERIFY it is current/free-tier for your account.

## Layout
`core/` backend (no GUI imports) | `gui/` CustomTkinter pages | `tests/` pytest | `data/` synthetic CSVs

## prompt 
You are a senior Python/ML engineer and CS professor. Attached is AI_Portfolio_Optimizer_draft.zip, a working but unfinished
Assignment 3 project (CustomTkinter dark GUI, yfinance, PyTorch ANN, Random Forest, Gemini sentiment, ranking, allocation,
Dijkstra trade planner). Read README.md first. Do NOT rewrite from scratch: extend the existing modules, keep interfaces
stable, keep core/ free of GUI imports. Install the dependencies, run `python -m pytest -q` (49 tests currently pass), then
work through the list below. After each item run the tests; do not claim anything works unless you executed it, and report
what you could not verify (e.g. no network or no Windows).

1. Training UX: stream per-epoch loss to the Neural Network page live (use the queue; widgets updated only from the main
   thread), add a Stop button (the pipeline already supports should_stop), and add early stopping with patience.
2. Honest evaluation: add a rolling/expanding walk-forward backtest on the test period (rank by predicted return, report
   the realized forward return of the top-N vs equal-weight vs baseline), and per-ticker metrics. Show it on Model Comparison.
3. Historical sentiment evaluation without future headlines (use only headlines dated <= the prediction date), with a test.
4. Optional online news provider behind a clean interface (NewsProvider), with a mocked test. News CSV stays the default.
5. Gemini hardening: verify the google-genai usage and the model name against current docs, add retry/backoff for 429s,
   a "Test connection" button on the News Sentiment page, and tests with mocked failures.
6. Planner: add a performance/limits test for larger portfolios (direct mode with 15+ tickers, toward mode with the state cap),
   show explored-state counts and an "incomplete" warning in the GUI, and document the complexity and the optimality proof in docs/.
7. GUI polish: small-window readability (down to ~1000x650), a table sort-by-column, a currency/percent formatting pass,
   error banners instead of only message boxes, and a Windows-friendly file picker default folder.
8. Tests: add GUI smoke tests that run under xvfb-run (skipped on Windows), a test that the embargo/leakage check fails when
   the embargo is too small, and a real-Windows checklist in the README.
9. Docs: finish the README (architecture diagram, API key setup, demo walkthrough for a university presentation), plus
   docs/algorithms.md covering for each algorithm: what it does, why, how, time complexity, how correctness is verified.
10. Final: tick the acceptance checklist from the Master Prompt (only items actually verified), regenerate the zip.
