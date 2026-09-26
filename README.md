# AI Investor V10

Educational stock-market research and paper-trading app built with Streamlit.

V10 keeps the V9.7 control for comparison and changes the V9.8 entry architecture:
- fast single-day clean-breakout path (no two-day waiting rule)
- breakout + pullback recovery path
- entry-quality score separates actual setup quality from conditions that were nearly always true in V9.8
- realistic brokerage and slippage
- fixed risk sizing, exposure limits and drawdown brake
- chronological walk-forward testing with no parameter tuning on held-out slices

Paper trading only. No broker connection and no guaranteed returns.
