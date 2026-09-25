# AI Investor V2 — Phone-Friendly Paper Trading Prototype

V2 adds:
- Automatic paper BUY candidates from a simple rules engine
- Automatic paper SELL when profit target or stop-loss is reached
- Position sizing
- Portfolio tracking
- Trade journal
- Research dashboard
- Learning explanations

## Run
pip install -r requirements.txt
streamlit run app.py

For phone use, host the app on a service that provides a public HTTPS URL, then open that URL in Chrome on Android.

## Important
This version does NOT place real trades. It is a research/education prototype. The rules are deliberately simple and should be evaluated with substantial paper-trading data before considering any real-money integration.
