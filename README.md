# AI Investor V9.7

V9.7 is an educational stock-market research and paper-trading laboratory.

## V9.7 changes
- Keeps **V9.5 as the fixed control**.
- Replaces the V9.6 score with a genuinely variable **0–10 trigger-quality score**.
- Separates **regime/setup gates** from trigger scoring.
- Scores four independent dimensions:
  - Trend: 0–2
  - Momentum: 0–2
  - Volatility: 0–2
  - Price action: 0–4
- Treats **20-day breakout** and **pullback recovery** as distinct setup types for diagnostics.
- Uses fixed, pre-declared trigger thresholds; they are not tuned to maximize the historical backtest.
- Keeps the experimental **2.5 ATR stop** and **3R target** from V9.6 so the entry architecture is the main change.
- Keeps the portfolio drawdown brake: default 10% peak-to-trough drawdown, pausing new entries for 20 trading days.
- Keeps risk-based position sizing, 0.75% default risk per trade, 60% max exposure and 3 simultaneous positions.
- Keeps chronological walk-forward testing with training, validation and held-out out-of-sample periods.
- Adds diagnostics by setup, score bucket, trigger component and exit reason.

## Important
This project is **paper-only**. It has no broker connection, does not guarantee returns, and historical backtests do not establish future profitability.
