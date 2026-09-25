# AI Investor V9

Phone-friendly Streamlit research and paper-trading laboratory.

## V9 purpose
V9 responds to the V8.1 diagnostic findings:
- reduce unnecessary turnover
- require stronger entry confirmation
- remove routine regime exits
- add minimum holding discipline
- add post-exit cooldown
- cap portfolio exposure
- keep realistic brokerage and slippage in the main test

## Default settings
- Starting paper capital: $10,000
- Watchlist: BHP.AX, CBA.AX, CSL.AX, VAS.AX
- Brokerage: $6.50 per transaction
- Slippage: 0.10%
- Risk per trade: 0.75% of equity
- Max position: 25%
- Max exposure: 60%
- Max simultaneous positions: 3
- Stop: 2 ATR
- Target: 3R
- Minimum hold: 5 trading days
- Cooldown after exit: 10 trading days
- Maximum hold: 90 trading days

## Important
This is an educational research and paper-trading tool. It has no broker connection and does not guarantee returns. Historical backtests are not forecasts.

Signals are calculated using a day's closing information and executed at the next trading day's open in the backtest to reduce same-bar look-ahead.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
