
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone

st.set_page_config(page_title="AI Investor V2", page_icon="📈", layout="wide")

STARTING_CASH = 10_000.0

st.title("📈 AI Investor V2")
st.caption("Automatic paper-trading laboratory • $10,000 virtual capital • No real-money execution")

# ---------- State ----------
defaults = {
    "cash": STARTING_CASH,
    "positions": {},
    "trades": [],
    "equity_curve": [],
    "last_scan": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------- Data ----------
@st.cache_data(ttl=900)
def get_history(ticker, period="2y"):
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

def get_price(ticker):
    df = get_history(ticker, "5d")
    if df.empty:
        return None
    return float(df["Close"].iloc[-1])

def analyse(ticker):
    df = get_history(ticker, "2y")
    if df.empty or len(df) < 220:
        return None
    close = df["Close"]
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    rsi_delta = close.diff()
    gain = rsi_delta.clip(lower=0).rolling(14).mean()
    loss = (-rsi_delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    last = float(close.iloc[-1])
    s50 = float(sma50.iloc[-1])
    s200 = float(sma200.iloc[-1])
    rsi_last = float(rsi.iloc[-1])
    ret_6m = float(close.iloc[-1] / close.iloc[-126] - 1) if len(close) > 126 else np.nan
    score = 0
    reasons = []
    if last > s200:
        score += 1; reasons.append("price is above the 200-day average")
    if s50 > s200:
        score += 1; reasons.append("50-day average is above the 200-day average")
    if 45 <= rsi_last <= 70:
        score += 1; reasons.append("RSI is in a non-extreme momentum range")
    if ret_6m > 0:
        score += 1; reasons.append("six-month price trend is positive")
    signal = "WATCH"
    if score >= 3:
        signal = "PAPER BUY CANDIDATE"
    elif score <= 1:
        signal = "AVOID / WAIT"
    return {
        "ticker": ticker, "price": last, "sma50": s50, "sma200": s200,
        "rsi": rsi_last, "return_6m": ret_6m, "score": score,
        "signal": signal, "reasons": reasons, "history": df
    }

def portfolio_value():
    total = st.session_state.cash
    for ticker, pos in st.session_state.positions.items():
        px = get_price(ticker)
        if px is not None:
            total += pos["shares"] * px
    return total

def record_equity():
    st.session_state.equity_curve.append(
        [datetime.now(timezone.utc).isoformat(), portfolio_value()]
    )

def buy(ticker, amount, target_pct, stop_pct):
    px = get_price(ticker)
    if px is None or amount <= 0 or amount > st.session_state.cash:
        return False, "Insufficient paper cash or unavailable price."
    shares = int(amount // px)
    if shares < 1:
        return False, "Position size is too small for one whole share."
    cost = shares * px
    pos = st.session_state.positions.get(ticker, {
        "shares": 0, "avg_price": 0.0, "target_pct": target_pct, "stop_pct": stop_pct
    })
    old_cost = pos["shares"] * pos["avg_price"]
    pos["avg_price"] = (old_cost + cost) / (pos["shares"] + shares)
    pos["shares"] += shares
    pos["target_pct"] = target_pct
    pos["stop_pct"] = stop_pct
    st.session_state.positions[ticker] = pos
    st.session_state.cash -= cost
    st.session_state.trades.append([
        datetime.now().strftime("%Y-%m-%d %H:%M"), "BUY", ticker, shares, px, 0.0,
        "Rule-based paper entry"
    ])
    record_equity()
    return True, f"Bought {shares} shares of {ticker} at ${px:,.2f}."

def auto_exit():
    events = []
    for ticker, pos in list(st.session_state.positions.items()):
        px = get_price(ticker)
        if px is None:
            continue
        change = px / pos["avg_price"] - 1
        if change >= pos["target_pct"] or change <= -pos["stop_pct"]:
            proceeds = px * pos["shares"]
            pnl = (px - pos["avg_price"]) * pos["shares"]
            reason = "Profit target reached" if change >= pos["target_pct"] else "Stop-loss reached"
            st.session_state.cash += proceeds
            st.session_state.trades.append([
                datetime.now().strftime("%Y-%m-%d %H:%M"), "SELL", ticker,
                pos["shares"], px, pnl, reason
            ])
            events.append(f"{reason}: sold {ticker} at ${px:,.2f} ({change*100:+.1f}%).")
            del st.session_state.positions[ticker]
    if events:
        record_equity()
    return events

# ---------- Sidebar controls ----------
st.sidebar.header("⚙️ Paper-trading rules")
profit_target = st.sidebar.slider("Profit target", 1, 50, 15) / 100
stop_loss = st.sidebar.slider("Stop-loss", 1, 30, 8) / 100
position_pct = st.sidebar.slider("Maximum position size", 1, 25, 10) / 100

st.sidebar.markdown("---")
st.sidebar.write("**Starting capital:** $10,000")
st.sidebar.write("**Real trades:** Disabled")

# ---------- Dashboard ----------
auto_events = auto_exit()
if auto_events:
    for e in auto_events:
        st.sidebar.success(e)

value = portfolio_value()
pnl = value - STARTING_CASH
c1,c2,c3,c4 = st.columns(4)
c1.metric("Portfolio", f"${value:,.2f}")
c2.metric("Total P/L", f"${pnl:+,.2f}", f"{pnl/STARTING_CASH*100:+.2f}%")
c3.metric("Cash", f"${st.session_state.cash:,.2f}")
c4.metric("Open positions", len(st.session_state.positions))

tabs = st.tabs(["🤖 Auto Trader", "🔎 Research", "💼 Portfolio", "📚 Learn"])

with tabs[0]:
    st.subheader("Automatic paper trader")
    st.write("The engine only uses virtual money. It scans your watchlist, applies the rules above, and records paper trades.")
    watchlist = st.text_input("Watchlist (comma-separated)", "BHP.AX,CBA.AX,CSL.AX,VAS.AX")
    if st.button("Run automatic paper scan", type="primary"):
        tickers = [x.strip().upper() for x in watchlist.split(",") if x.strip()]
        results = []
        for t in tickers:
            a = analyse(t)
            if a:
                results.append(a)
                if a["score"] >= 3 and t not in st.session_state.positions:
                    budget = min(st.session_state.cash, STARTING_CASH * position_pct)
                    if budget >= a["price"]:
                        ok, msg = buy(t, budget, profit_target, stop_loss)
                        if ok:
                            st.success(f"🤖 {msg}")
        st.session_state.last_scan = datetime.now().strftime("%Y-%m-%d %H:%M")
        if results:
            st.dataframe(pd.DataFrame([{
                "Ticker": r["ticker"], "Price": round(r["price"],2),
                "Score": r["score"], "6M %": round(r["return_6m"]*100,1),
                "RSI": round(r["rsi"],1), "Signal": r["signal"]
            } for r in results]), use_container_width=True, hide_index=True)
        else:
            st.warning("No usable market data was returned.")
    if st.session_state.last_scan:
        st.caption(f"Last scan: {st.session_state.last_scan}")

with tabs[1]:
    st.subheader("Research")
    ticker = st.text_input("Ticker to research", "BHP.AX").upper().strip()
    if st.button("Analyse stock"):
        a = analyse(ticker)
        if not a:
            st.error("Not enough market data.")
        else:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Price", f"${a['price']:,.2f}")
            c2.metric("6-month return", f"{a['return_6m']*100:+.1f}%")
            c3.metric("RSI", f"{a['rsi']:.1f}")
            c4.metric("Rule score", f"{a['score']}/4")
            h = a["history"].copy()
            h["SMA50"] = h["Close"].rolling(50).mean()
            h["SMA200"] = h["Close"].rolling(200).mean()
            st.line_chart(h[["Close","SMA50","SMA200"]])
            st.markdown(f"### Current rule signal: **{a['signal']}**")
            for r in a["reasons"]:
                st.write("• " + r)
            st.info("This is a mechanical research signal, not a guarantee or a personal investment recommendation.")

with tabs[2]:
    st.subheader("Paper portfolio")
    if st.session_state.positions:
        rows = []
        for t,p in st.session_state.positions.items():
            px = get_price(t) or p["avg_price"]
            mv = px * p["shares"]
            rows.append({
                "Ticker": t, "Shares": p["shares"], "Avg cost": p["avg_price"],
                "Price": px, "Market value": mv,
                "P/L": (px-p["avg_price"])*p["shares"],
                "Target": p["target_pct"]*100, "Stop": p["stop_pct"]*100
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No open paper positions.")

    if st.session_state.trades:
        st.markdown("### Trade journal")
        st.dataframe(pd.DataFrame(st.session_state.trades,
            columns=["Time","Action","Ticker","Shares","Price","P/L","Reason"]),
            use_container_width=True, hide_index=True)

    if st.button("Reset paper account"):
        st.session_state.cash = STARTING_CASH
        st.session_state.positions = {}
        st.session_state.trades = []
        st.session_state.equity_curve = []
        st.rerun()

with tabs[3]:
    st.subheader("Learn while you trade")
    st.markdown("""
**Profit target:** the price gain at which our paper rules close a position.

**Stop-loss:** a predefined loss level at which the paper system exits.

**Position sizing:** controls how much of the portfolio is exposed to one investment.

**RSI:** a momentum indicator. It can help describe whether recent price movement is relatively strong or weak, but it is not a prediction machine.

**Moving average:** an average of prices over a chosen period. Comparing short and long averages is one way to study trend.

**Backtesting:** testing rules against historical data. A good historical result does not guarantee future performance.
""")
    st.warning("Educational software only. Historical and simulated results are not guarantees of future returns.")

st.divider()
st.caption("AI Investor V2 • Paper trading only • No broker connection • No real-money execution")
