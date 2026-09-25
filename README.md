# AI Investor V6

Phone-friendly strategy research laboratory, realistic backtesting and paper trading.

## V6 additions
- Four independent strategy families: Trend, Momentum, Trend + Momentum, Mean Reversion
- Proper buy-and-hold benchmark
- Normalised $10,000 equity curves
- CAGR, total return, maximum drawdown
- Win rate, trade count, average win/loss, profit factor
- Time-in-market measurement
- Brokerage and slippage assumptions
- Chronological Training / Validation / Out-of-sample robustness view
- Research dashboard
- Paper trader kept separate from research

## Important
V6 does not automatically choose a strategy based on historical return. Historical performance can be affected by data quality, survivorship bias, transaction assumptions and parameter selection. Results are not guarantees of future performance.

## Deploy
Replace `app.py`, `requirements.txt`, and `README.md` in the existing `RSingh1984/AI-Investor` repository on branch `main`.
Main Streamlit file: `app.py`.
