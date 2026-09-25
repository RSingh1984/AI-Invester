# AI Investor V8

Phone-friendly Streamlit portfolio regime and risk research laboratory.

## V8 additions
- Portfolio-level backtest across the watchlist
- Fixed Bull / Range / Bear regime engine
- Regime-based strategy selection without historical winner-picking
- ATR-based stop and 3R target defaults
- Risk-based position sizing
- Maximum position, exposure and simultaneous-position controls
- Conservative handling when stop and target are both touched
- Brokerage and slippage assumptions
- Portfolio CAGR, drawdown, trade count, win rate and profit factor
- Walk-forward Training / Validation / Out-of-sample portfolio robustness
- Current research dashboard showing regime and rule output
- Paper-only signal scan using the same fixed rules
- Learning explanations for every major component

## Default risk settings
- Starting capital: $10,000
- Risk per trade: 0.75% of equity
- Maximum position: 25% of equity
- Maximum invested exposure: 80%
- Maximum simultaneous positions: 3
- Stop: 2 ATR
- Profit target: 3R
- Brokerage: $6.50 per transaction
- Slippage: 0.10% per transaction

## Important
Backtests are historical simulations. They do not guarantee future returns. V8 is educational and has no broker connection or real-money execution.

## Deploy
Replace `app.py`, `requirements.txt`, and `README.md` in the existing GitHub repository, then allow Streamlit Cloud to redeploy.
