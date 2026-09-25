# AI Investor V7

Phone-friendly Streamlit research, robustness and paper-trading laboratory.

## V7 additions
- Training / Validation / Out-of-sample robustness scorecard
- Fixed, user-visible research thresholds to reduce hindsight bias
- OOS CAGR, maximum drawdown, trade count, win rate and profit factor checks
- Training-to-OOS CAGR stability ratio
- Cross-asset consistency summary
- Strategy-by-asset research status: descriptive only, not a forecast or recommendation
- Training vs Validation vs OOS CAGR comparison
- Carries forward V6 strategy lab, paper trader, research, portfolio and journal
- No broker connection and no real-money execution

## Important
Backtests are historical simulations and do not guarantee future returns. The scorecard is an educational research aid. It does not automatically select a strategy for real-money trading.

## Deploy
Replace `app.py`, `requirements.txt`, and `README.md` in the existing GitHub repository, then let Streamlit Cloud redeploy.
