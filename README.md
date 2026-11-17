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
