import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="AI Investor V4", page_icon="📈", layout="wide")

# =========================================================
# DATA + INDICATORS
# =========================================================
@st.cache_data(ttl=900)
def load_history(ticker, period="5y"):
    df = yf.download(ticker, period=period, interval="1d",
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def add_indicators(df):
    x = df.copy()
    c = x["Close"].astype(float)
    x["SMA20"] = c.rolling(20).mean()
    x["SMA50"] = c.rolling(50).mean()
    x["SMA200"] = c.rolling(200).mean()
    x["RSI"] = calc_rsi(c)
    x["Return20"] = c.pct_change(20) * 100
    x["Return126"] = c.pct_change(126) * 100
    x["Vol20"] = c.pct_change().rolling(20).std() * np.sqrt(252) * 100
    return x

def score_row(row):
    score = 0
    reasons, warnings = [], []
    price = row["Close"]

    if price > row["SMA50"]:
        score += 1; reasons.append("Price is above the 50-day average.")
    else:
        warnings.append("Price is below the 50-day average.")

    if row["SMA50"] > row["SMA200"]:
        score += 1; reasons.append("50-day average is above the 200-day average.")
    else:
        warnings.append("50-day average is below the 200-day average.")

    if row["Return126"] > 10:
        score += 1; reasons.append(f"6-month momentum is positive ({row['Return126']:.1f}%).")
    elif row["Return126"] < 0:
        warnings.append(f"6-month momentum is negative ({row['Return126']:.1f}%).")

    if 45 <= row["RSI"] <= 65:
        score += 1; reasons.append(f"RSI is in the model's moderate zone ({row['RSI']:.1f}).")
    elif row["RSI"] > 70:
        warnings.append(f"RSI is high ({row['RSI']:.1f}).")
    elif row["RSI"] < 30:
        warnings.append(f"RSI is low ({row['RSI']:.1f}); low RSI is not automatically bullish.")

    if price > row["SMA20"]:
        score += 1; reasons.append("Price is above the 20-day average.")
    else:
        warnings.append("Price is below the 20-day average.")

    if row["Vol20"] < 35:
        score += 1; reasons.append(f"Recent annualised volatility is {row['Vol20']:.1f}%.")
    else:
        warnings.append(f"Recent annualised volatility is high ({row['Vol20']:.1f}%).")

    return score, reasons, warnings

def latest_analysis(ticker):
    df = add_indicators(load_history(ticker, "2y"))
    if df.empty or len(df) < 220:
        return None
    r = df.iloc[-1]
    score, reasons, warnings = score_row(r)
    signal = "BUY CANDIDATE" if score >= st.session_state.min_score else (
        "WATCH" if score == st.session_state.min_score - 1 else "WAIT"
    )
    return {
        "Ticker": ticker, "Price": float(r["Close"]), "Score": int(score),
        "6M %": float(r["Return126"]), "1M %": float(r["Return20"]),
        "RSI": float(r["RSI"]), "Vol %": float(r["Vol20"]),
        "SMA20": float(r["SMA20"]), "SMA50": float(r["SMA50"]),
        "SMA200": float(r["SMA200"]), "Signal": signal,
        "Reasons": reasons, "Warnings": warnings
    }

# =========================================================
# FUNDAMENTALS
# =========================================================
@st.cache_data(ttl=3600)
def fundamental_snapshot(ticker):
    try:
        info = yf.Ticker(ticker).info
        fields = {
            "Market cap": info.get("marketCap"),
            "P/E": info.get("trailingPE"),
            "Forward P/E": info.get("forwardPE"),
            "EPS": info.get("trailingEps"),
            "Revenue growth": info.get("revenueGrowth"),
            "Earnings growth": info.get("earningsGrowth"),
            "Profit margin": info.get("profitMargins"),
            "Debt/equity": info.get("debtToEquity"),
            "ROE": info.get("returnOnEquity"),
            "Dividend yield": info.get("dividendYield"),
        }
        return fields
    except Exception as e:
        return {"Error": str(e)}

def fmt_metric(k, v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if k in ["Revenue growth", "Earnings growth", "Profit margin", "ROE", "Dividend yield"]:
        try: return f"{float(v)*100:.1f}%"
        except: return str(v)
    if k == "Market cap":
        try:
            n = float(v)
            return f"${n/1e9:.2f}B" if n >= 1e9 else f"${n/1e6:.1f}M"
        except: return str(v)
    try:
        return f"{float(v):.2f}"
    except:
        return str(v)

# =========================================================
# BACKTEST
# =========================================================
def backtest(ticker, target_pct, stop_pct, min_score, max_position_pct, years):
    raw = load_history(ticker, f"{years}y")
    if raw.empty or len(raw) < 260:
        return None

    df = add_indicators(raw).dropna().copy()
    cash = 10000.0
    shares = 0
    entry = None
    trades = []
    equity_curve = []

    for idx, row in df.iterrows():
        price = float(row["Close"])

        # Exit existing position first
        if shares > 0:
            if price >= entry * (1 + target_pct/100):
                pnl = (price - entry) * shares
                cash += shares * price
                trades.append(pnl)
                shares = 0
                entry = None
            elif price <= entry * (1 - stop_pct/100):
                pnl = (price - entry) * shares
                cash += shares * price
                trades.append(pnl)
                shares = 0
                entry = None

        # Entry
        if shares == 0:
            score, _, _ = score_row(row)
            if score >= min_score:
                max_value = cash * max_position_pct/100
                qty = int(max_value // price)
                if qty >= 1:
                    shares = qty
                    entry = price
                    cash -= qty * price

        equity = cash + shares * price
        equity_curve.append((idx, equity))

    # Close at end for a complete realised result
    if shares > 0:
        last_price = float(df["Close"].iloc[-1])
        pnl = (last_price - entry) * shares
        cash += shares * last_price
        trades.append(pnl)
        shares = 0

    final_equity = cash
    start = 10000.0
    total_return = (final_equity/start - 1) * 100

    curve = pd.Series([v for _, v in equity_curve],
                      index=[d for d, _ in equity_curve])
    drawdown = curve / curve.cummax() - 1
    max_dd = float(drawdown.min() * 100) if not drawdown.empty else 0.0

    wins = [x for x in trades if x > 0]
    losses = [x for x in trades if x <= 0]
    win_rate = len(wins)/len(trades)*100 if trades else 0.0
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    profit_factor = sum(wins)/abs(sum(losses)) if losses and sum(losses) != 0 else np.inf if wins else 0.0

    return {
        "Ticker": ticker, "Final": final_equity, "Return %": total_return,
        "Trades": len(trades), "Win rate %": win_rate,
        "Max drawdown %": max_dd, "Avg win": avg_win,
        "Avg loss": avg_loss, "Profit factor": profit_factor,
        "curve": curve
    }

# =========================================================
# SESSION STATE
# =========================================================
if "cash" not in st.session_state: st.session_state.cash = 10000.0
if "positions" not in st.session_state: st.session_state.positions = {}
if "trades" not in st.session_state: st.session_state.trades = []

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙️ V4 Settings")
watch_text = st.sidebar.text_input("Watchlist", "BHP.AX,CBA.AX,CSL.AX,VAS.AX")
tickers = [x.strip().upper() for x in watch_text.split(",") if x.strip()]
st.session_state.profit_target = st.sidebar.slider("Paper profit target %", 3, 50, 15)
st.session_state.stop_loss = st.sidebar.slider("Paper stop-loss %", 1, 30, 8)
st.session_state.max_position = st.sidebar.slider("Maximum position %", 1, 25, 10)
st.session_state.min_score = st.sidebar.slider("Minimum BUY score", 3, 6, 4)
st.session_state.backtest_years = st.sidebar.selectbox("Backtest period", [2,3,5], index=2)

if st.sidebar.button("🔄 Reset paper account"):
    st.session_state.cash = 10000.0
    st.session_state.positions = {}
    st.session_state.trades = []
    st.rerun()

st.sidebar.caption("Educational paper trading only. Historical results do not guarantee future results.")

# =========================================================
# HEADER
# =========================================================
st.title("📈 AI Investor V4")
st.caption("Research + backtesting + paper trading • $10,000 virtual capital • No broker connection")

st.info("V4 is designed to test the strategy before real money is considered. The backtest is historical research, not a prediction.")

tabs = st.tabs(["🤖 Paper Trader","🧪 Backtest","📊 Research","💼 Portfolio","🧠 Learn","🧾 Journal"])

# =========================================================
# PAPER TRADER
# =========================================================
with tabs[0]:
    st.subheader("Automatic paper trader")

    if st.button("🔎 Run V4 paper scan", type="primary"):
        results = []
        for ticker in tickers:
            try:
                a = latest_analysis(ticker)
                if a: results.append(a)
            except Exception as e:
                st.warning(f"{ticker}: {e}")

        # exits
        for ticker in list(st.session_state.positions):
            pos = st.session_state.positions[ticker]
            a = next((x for x in results if x["Ticker"] == ticker), None)
            if not a: continue
            price = a["Price"]
            if price >= pos["target"] or price <= pos["stop"]:
                pnl = (price-pos["entry"])*pos["shares"]
                st.session_state.cash += pos["shares"]*price
                target_hit = price >= pos["target"]
                action = "SELL / TARGET" if target_hit else "SELL / STOP"
                st.session_state.trades.append({
                    "Time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "Action": action, "Ticker": ticker, "Shares": pos["shares"],
                    "Entry": pos["entry"], "Exit": price, "P/L": pnl
                })
                del st.session_state.positions[ticker]
                st.success(f"{'🎯' if target_hit else '🛑'} {action}: {ticker}, P/L ${pnl:+.2f}")

        # buys
        portfolio_value = st.session_state.cash + sum(
            p["shares"] * next((a["Price"] for a in results if a["Ticker"] == t), p["entry"])
            for t,p in st.session_state.positions.items()
        )
        max_value = portfolio_value * st.session_state.max_position/100

        for a in results:
            ticker, price = a["Ticker"], a["Price"]
            if ticker in st.session_state.positions or a["Score"] < st.session_state.min_score:
                continue
            qty = int(min(st.session_state.cash // price, max_value // price))
            if qty < 1: continue
            target = price*(1+st.session_state.profit_target/100)
            stop = price*(1-st.session_state.stop_loss/100)
            risk = (price-stop)*qty
            st.session_state.cash -= qty*price
            st.session_state.positions[ticker] = {
                "shares": qty, "entry": price, "target": target,
                "stop": stop, "risk": risk
            }
            st.session_state.trades.append({
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Action": "PAPER BUY", "Ticker": ticker, "Shares": qty,
                "Entry": price, "Exit": "", "P/L": ""
            })
            st.success(f"🤖 Paper BUY: {qty} {ticker} @ ${price:.2f} | Planned stop risk ${risk:.2f}")

        if results:
            st.dataframe(pd.DataFrame([{
                "Ticker":a["Ticker"], "Price":round(a["Price"],2),
                "Score":f'{a["Score"]}/6', "6M %":round(a["6M %"],1),
                "RSI":round(a["RSI"],1), "Vol %":round(a["Vol %"],1),
                "Signal":a["Signal"]
            } for a in results]), use_container_width=True)

            st.markdown("### 🧠 Why?")
            for a in results:
                with st.expander(f'{a["Ticker"]} — {a["Signal"]} — {a["Score"]}/6'):
                    for x in a["Reasons"]: st.write("• "+x)
                    for x in a["Warnings"]: st.write("⚠️ "+x)

# =========================================================
# BACKTEST
# =========================================================
with tabs[1]:
    st.subheader("Historical strategy test")
    st.write("This simulates the current rule set on historical daily prices. It is not a guarantee of future performance.")

    if st.button("🧪 Run backtest on watchlist"):
        results = []
        curves = {}
        for ticker in tickers:
            try:
                r = backtest(ticker, st.session_state.profit_target,
                             st.session_state.stop_loss, st.session_state.min_score,
                             st.session_state.max_position, st.session_state.backtest_years)
                if r:
                    results.append(r)
                    curves[ticker] = r["curve"]
            except Exception as e:
                st.warning(f"{ticker}: {e}")

        if results:
            table = pd.DataFrame([{
                "Ticker":r["Ticker"], "Final $":round(r["Final"],2),
                "Strategy return %":round(r["Return %"],2),
                "Trades":r["Trades"], "Win rate %":round(r["Win rate %"],1),
                "Max drawdown %":round(r["Max drawdown %"],1),
                "Avg win $":round(r["Avg win"],2),
                "Avg loss $":round(r["Avg loss"],2),
                "Profit factor":("∞" if np.isinf(r["Profit factor"]) else round(r["Profit factor"],2))
            } for r in results])
            st.dataframe(table, use_container_width=True)

            st.markdown("### Equity curves")
            for ticker, curve in curves.items():
                st.line_chart(curve.rename(ticker))

            st.warning("Do not select a strategy because of one attractive backtest. Look for consistency, drawdown, trade count and sensitivity to settings.")

# =========================================================
# RESEARCH
# =========================================================
with tabs[2]:
    st.subheader("Research dashboard")
    for ticker in tickers:
        a = latest_analysis(ticker)
        if not a: continue
        with st.expander(f'{ticker} — ${a["Price"]:.2f} — {a["Signal"]}'):
            c = st.columns(6)
            c[0].metric("Score",f'{a["Score"]}/6')
            c[1].metric("6M",f'{a["6M %"]:+.1f}%')
            c[2].metric("1M",f'{a["1M %"]:+.1f}%')
            c[3].metric("RSI",f'{a["RSI"]:.1f}')
            c[4].metric("Volatility",f'{a["Vol %"]:.1f}%')
            c[5].metric("Price",f'${a["Price"]:.2f}')
            st.write(f'SMA20 ${a["SMA20"]:.2f} | SMA50 ${a["SMA50"]:.2f} | SMA200 ${a["SMA200"]:.2f}')
            st.write("**Technical factors:**")
            for x in a["Reasons"]: st.write("• "+x)
            for x in a["Warnings"]: st.write("⚠️ "+x)

            st.write("**Fundamental snapshot:**")
            f = fundamental_snapshot(ticker)
            if "Error" in f:
                st.caption("Fundamental data was unavailable from the data provider for this ticker.")
            else:
                fc = st.columns(5)
                keys = list(f.keys())
                for i,k in enumerate(keys):
                    fc[i % 5].metric(k, fmt_metric(k, f[k]))

# =========================================================
# PORTFOLIO
# =========================================================
with tabs[3]:
    st.subheader("Paper portfolio")
    total = st.session_state.cash
    rows=[]
    for ticker,pos in st.session_state.positions.items():
        a = latest_analysis(ticker)
        current = a["Price"] if a else pos["entry"]
        value = current*pos["shares"]
        unreal = (current-pos["entry"])*pos["shares"]
        total += value
        rows.append({
            "Ticker":ticker,"Shares":pos["shares"],"Entry":round(pos["entry"],2),
            "Current":round(current,2),"Unrealised P/L":round(unreal,2),
            "Target":round(pos["target"],2),"Stop":round(pos["stop"],2),
            "Risk at stop":round(pos["risk"],2)
        })
    c=st.columns(3)
    c[0].metric("Portfolio",f"${total:,.2f}")
    c[1].metric("Total P/L",f"${total-10000:+,.2f}")
    c[2].metric("Cash",f"${st.session_state.cash:,.2f}")
    st.dataframe(pd.DataFrame(rows), use_container_width=True) if rows else st.info("No open paper positions.")

# =========================================================
# LEARN
# =========================================================
with tabs[4]:
    st.subheader("What V4 is testing")
    st.markdown("""
### 1. Technical signals
The model checks moving averages, momentum, RSI and volatility.

### 2. Fundamentals
Where the provider has data, V4 displays valuation, growth, profitability, leverage and dividend fields. Missing data is shown as unavailable rather than guessed.

### 3. Backtesting
Backtesting asks: **“If these rules had been followed on historical daily prices, what would have happened?”**

It does **not** prove the strategy will work in the future.

### 4. Risk
Every new paper position has a maximum planned loss based on the stop level and position size.

### 5. What we are NOT doing
We are not claiming a guaranteed profit, not treating a score as a probability of success, and not connecting a broker.

### Next research discipline
Run the backtest across several stocks and settings, then keep the strategy unchanged for a paper-trading period. This helps reduce the temptation to change the rules after seeing individual outcomes.
""")

# =========================================================
# JOURNAL
# =========================================================
with tabs[5]:
    st.subheader("Trade journal")
    if st.session_state.trades:
        st.dataframe(pd.DataFrame(st.session_state.trades), use_container_width=True)
    else:
        st.info("No trades recorded yet.")

st.divider()
st.caption("AI Investor V4 • Educational research and paper trading only • Market data via Yahoo Finance/yfinance • No broker connection")
