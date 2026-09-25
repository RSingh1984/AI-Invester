import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="AI Investor V3", page_icon="📈", layout="wide")

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=300)
def load_history(ticker):
    df = yf.download(ticker, period="2y", interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

def analyse(ticker):
    df = load_history(ticker)
    if df.empty or len(df) < 220:
        return None
    close = df["Close"].astype(float)
    price = float(close.iloc[-1])
    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1])
    rv = float(rsi(close).iloc[-1])
    six = (price / float(close.iloc[-126]) - 1) * 100
    one = (price / float(close.iloc[-21]) - 1) * 100
    vol = float(close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100)

    score, reasons, warnings = 0, [], []
    if price > sma50:
        score += 1; reasons.append("Price is above the 50-day average.")
    else: warnings.append("Price is below the 50-day average.")
    if sma50 > sma200:
        score += 1; reasons.append("50-day trend is above the 200-day trend.")
    else: warnings.append("50-day trend is below the 200-day trend.")
    if six > 10:
        score += 1; reasons.append(f"6-month momentum is positive at {six:.1f}%.")
    elif six < 0:
        warnings.append(f"6-month momentum is negative at {six:.1f}%.")
    if 45 <= rv <= 65:
        score += 1; reasons.append(f"RSI is in a moderate zone ({rv:.1f}).")
    elif rv > 70:
        warnings.append(f"RSI is high ({rv:.1f}).")
    elif rv < 30:
        warnings.append(f"RSI is low ({rv:.1f}); low RSI is not automatically a buy.")
    if price > sma20:
        score += 1; reasons.append("Price is above the 20-day average.")
    else: warnings.append("Price is below the 20-day average.")
    if vol < 35:
        score += 1; reasons.append(f"Recent annualised volatility is {vol:.1f}%.")
    else: warnings.append(f"Recent annualised volatility is high at {vol:.1f}%.")

    confidence = min(95, 20 + score * 12 + (5 if six > 20 else 0))
    signal = "PAPER BUY CANDIDATE" if score >= 4 else ("WATCH" if score == 3 else "AVOID / WAIT")
    return dict(Ticker=ticker, Price=price, Score=score, Confidence=confidence,
                SixM=six, OneM=one, RSI=rv, Vol=vol, SMA20=sma20,
                SMA50=sma50, SMA200=sma200, Signal=signal,
                Reasons=reasons, Warnings=warnings)

def init():
    if "cash" not in st.session_state: st.session_state.cash = 10000.0
    if "positions" not in st.session_state: st.session_state.positions = {}
    if "trades" not in st.session_state: st.session_state.trades = []
    if "v2_imported" not in st.session_state: st.session_state.v2_imported = False

def log(action, ticker, shares, price, reason, target="", stop="", pnl=""):
    st.session_state.trades.append({
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Action": action, "Ticker": ticker, "Shares": shares,
        "Price": round(price, 2), "Value": round(shares * price, 2),
        "Reason": reason, "Target": target, "Stop": stop, "P/L": pnl
    })

def import_v2():
    if st.session_state.v2_imported: return
    target_pct = st.session_state.profit_target
    stop_pct = st.session_state.stop_loss
    for ticker, shares, entry in [("BHP.AX",16,60.72), ("CSL.AX",5,176.95)]:
        target = entry * (1 + target_pct/100)
        stop = entry * (1 - stop_pct/100)
        st.session_state.positions[ticker] = {
            "shares": shares, "entry": entry, "target": target, "stop": stop,
            "risk": (entry-stop)*shares, "confidence": "V2"
        }
        st.session_state.cash -= shares * entry
        log("V2 IMPORT", ticker, shares, entry, "Carried forward from V2.", round(target,2), round(stop,2), "")
    st.session_state.v2_imported = True

def current_value():
    total = st.session_state.cash
    for ticker, pos in st.session_state.positions.items():
        a = analyse(ticker)
        total += pos["shares"] * (a["Price"] if a else pos["entry"])
    return total

init()
st.sidebar.header("⚙️ V3 Settings")
watch = st.sidebar.text_input("Watchlist", "BHP.AX,CBA.AX,CSL.AX,VAS.AX")
tickers = [x.strip().upper() for x in watch.split(",") if x.strip()]
st.session_state.profit_target = st.sidebar.slider("Profit target %", 3, 50, 15)
st.session_state.stop_loss = st.sidebar.slider("Stop-loss %", 1, 30, 8)
max_pct = st.sidebar.slider("Maximum position %", 1, 25, 10)
min_score = st.sidebar.slider("Minimum BUY score", 3, 6, 4)
if st.sidebar.button("🔄 Reset paper account"):
    for k in ["cash","positions","trades","v2_imported"]:
        st.session_state.pop(k, None)
    st.rerun()
st.sidebar.caption("Paper money only. Rule-based signals cannot guarantee profits.")

st.title("📈 AI Investor V3")
st.caption("Automatic paper-trading laboratory • $10,000 virtual capital • No broker connection • No real-money execution")

if not st.session_state.v2_imported:
    st.info("Your two V2 paper positions can be carried into V3.")
    if st.button("➡️ Continue my V2 experiment"):
        import_v2()
        st.rerun()

pv = current_value()
pnl = pv - 10000
cols = st.columns(4)
cols[0].metric("Portfolio", f"${pv:,.2f}")
cols[1].metric("Total P/L", f"${pnl:+,.2f}", f"{pnl/100:+.2f}%")
cols[2].metric("Cash", f"${st.session_state.cash:,.2f}")
cols[3].metric("Open positions", len(st.session_state.positions))

tabs = st.tabs(["🤖 Auto Trader","📊 Research","💼 Portfolio","📚 Learn","🧾 Journal"])

with tabs[0]:
    st.subheader("Automatic paper trader")
    st.write("V3 adds trend, momentum, RSI, volatility, position sizing, risk budgeting and explanations.")
    if st.button("🔎 Run V3 automatic paper scan", type="primary"):
        results = []
        for ticker in tickers:
            try:
                a = analyse(ticker)
                if a: results.append(a)
            except Exception as e:
                st.warning(f"{ticker}: {e}")

        # Exit first: target or stop
        for ticker in list(st.session_state.positions):
            pos = st.session_state.positions[ticker]
            a = next((x for x in results if x["Ticker"] == ticker), None)
            if not a: continue
            price = a["Price"]
            if price >= pos["target"] or price <= pos["stop"]:
                target_hit = price >= pos["target"]
                proceeds = pos["shares"] * price
                trade_pnl = (price - pos["entry"]) * pos["shares"]
                st.session_state.cash += proceeds
                del st.session_state.positions[ticker]
                action = "SELL / TARGET" if target_hit else "SELL / STOP"
                log(action, ticker, pos["shares"], price,
                    "Profit target reached." if target_hit else "Stop-loss reached.",
                    round(pos["target"],2), round(pos["stop"],2), round(trade_pnl,2))
                (st.success if target_hit else st.warning)(
                    f"{'🎯' if target_hit else '🛑'} Sold {pos['shares']} {ticker} at ${price:.2f}. P/L ${trade_pnl:+.2f}"
                )

        # New buys
        portfolio_now = current_value()
        max_value = portfolio_now * max_pct/100
        for a in results:
            ticker, price = a["Ticker"], a["Price"]
            if ticker in st.session_state.positions or a["Score"] < min_score: continue
            shares = int(min(st.session_state.cash // price, max_value // price))
            if shares < 1: continue
            target = price * (1 + st.session_state.profit_target/100)
            stop = price * (1 - st.session_state.stop_loss/100)
            risk = (price-stop)*shares
            reason = " ".join(a["Reasons"])
            st.session_state.cash -= shares*price
            st.session_state.positions[ticker] = {
                "shares": shares, "entry": price, "target": target, "stop": stop,
                "risk": risk, "confidence": f"{a['Confidence']}%"
            }
            log("PAPER BUY", ticker, shares, price, reason, round(target,2), round(stop,2), "")
            st.success(f"🤖 Bought {shares} shares of {ticker} at ${price:.2f}. Planned stop-risk: ${risk:.2f}")

        if results:
            table = pd.DataFrame([{
                "Ticker":a["Ticker"],"Price":round(a["Price"],2),"Score":f'{a["Score"]}/6',
                "Confidence":f'{a["Confidence"]}%',"6M %":round(a["SixM"],1),
                "RSI":round(a["RSI"],1),"Vol %":round(a["Vol"],1),"Signal":a["Signal"]
            } for a in results])
            st.dataframe(table, use_container_width=True)
            st.markdown("### 🧠 Decision explanations")
            for a in results:
                with st.expander(f'{a["Ticker"]} — {a["Signal"]} — score {a["Score"]}/6'):
                    st.write(f'**Rule-based signal strength:** {a["Confidence"]}% (not a probability of profit)')
                    if a["Reasons"]:
                        st.write("**Positive factors:**")
                        for x in a["Reasons"]: st.write("• "+x)
                    if a["Warnings"]:
                        st.write("**Caution factors:**")
                        for x in a["Warnings"]: st.write("• "+x)

with tabs[1]:
    st.subheader("Research dashboard")
    for ticker in tickers:
        a = analyse(ticker)
        if a:
            with st.expander(f'{ticker} — ${a["Price"]:.2f} — {a["Signal"]}'):
                c = st.columns(6)
                c[0].metric("Score",f'{a["Score"]}/6'); c[1].metric("Confidence",f'{a["Confidence"]}%')
                c[2].metric("6M",f'{a["SixM"]:+.1f}%'); c[3].metric("1M",f'{a["OneM"]:+.1f}%')
                c[4].metric("RSI",f'{a["RSI"]:.1f}'); c[5].metric("Volatility",f'{a["Vol"]:.1f}%')
                st.write(f'SMA20 ${a["SMA20"]:.2f} | SMA50 ${a["SMA50"]:.2f} | SMA200 ${a["SMA200"]:.2f}')
                for x in a["Reasons"]: st.write("• "+x)
                for x in a["Warnings"]: st.write("⚠️ "+x)

with tabs[2]:
    st.subheader("Open positions")
    rows=[]
    for ticker,pos in st.session_state.positions.items():
        a=analyse(ticker); current=a["Price"] if a else pos["entry"]
        rows.append({
            "Ticker":ticker,"Shares":pos["shares"],"Entry":round(pos["entry"],2),
            "Current":round(current,2),
            "Unrealised P/L":round((current-pos["entry"])*pos["shares"],2),
            "Target":round(pos["target"],2),"Stop":round(pos["stop"],2),
            "Risk at stop":round(pos["risk"],2),"Confidence":pos["confidence"]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True) if rows else st.info("No open paper positions.")
    st.write(f'Default target: **+{st.session_state.profit_target}%** | Default stop: **-{st.session_state.stop_loss}%**')

with tabs[3]:
    st.subheader("Learn while you invest")
    st.markdown("""
**SMA20 / SMA50 / SMA200:** moving averages used to describe short-, medium- and long-term price trends.

**RSI:** a momentum indicator. A low RSI does not automatically mean a stock is cheap, and a high RSI does not automatically mean it must fall.

**Score:** a checklist combining trend, momentum, RSI zone and volatility. It is not a prediction.

**Stop-loss:** a predefined exit level used to limit the planned loss.

**Profit target:** a predefined level where the paper trader takes profit.

**Risk budget:** the dollar amount that could be lost if the position reaches the stop.

V3 is an educational paper-trading experiment. It cannot guarantee profits.
""")

with tabs[4]:
    st.subheader("Trade journal")
    if st.session_state.trades:
        st.dataframe(pd.DataFrame(st.session_state.trades), use_container_width=True)
    else:
        st.info("No trades recorded yet.")

st.divider()
st.caption("AI Investor V3 • Educational paper trading only • Yahoo Finance data via yfinance • No broker connection")
