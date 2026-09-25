# AI Investor V8.1

Phone-friendly Streamlit research and paper-trading laboratory.

## V8.1 focus
V8.1 keeps the V8 fixed regime engine and adds diagnostics rather than tuning parameters to historical results.

- Portfolio regime engine: Bull / Range / Bear
- Risk-based position sizing from ATR stop distance
- Maximum position, exposure and simultaneous-position controls
- Conservative stop/target handling
- Brokerage and slippage assumptions
- Walk-forward robustness slices
- P/L by ticker, selected strategy, entry regime and exit reason
- Cost sensitivity: stated costs vs zero-cost counterfactual
- Regime-exit sensitivity: normal exits vs stops/targets only
- Largest-loss inspection
- Paper trader and research dashboard
- No broker connection and no real-money execution

## Important
Backtests are historical simulations. They do not guarantee future returns. V8.1 does not automatically tune itself to maximise historical performance.
