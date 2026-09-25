import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="AI Investor V5", page_icon="📈", layout="wide")

# ============================================================
# DATA
# ============================================================
@st.cache_data(ttl=900)
def load_history(ticker, period="10y"):
    df = yf.download(ticker, period=period, interval="1d",
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

def rsi(close, period=14):
    d = close.diff()
    gain = d.clip(lower=0).rolling(period).mean()
    loss = (-d.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def indicators(df):
    x = df.copy()
    c = x["Close"].astype(float)
    x["SMA20"] = c.rolling(20).mean()
    x["SMA50"] = c.rolling(50).mean()
    x["SMA200"] = c.rolling(200).mean()
    x["RSI"] = rsi(c)
    x["Ret20"] = c.pct_change(20) * 100
    x["Ret126"] = c.pct_change(126) * 100
    x["Vol20"] = c.pct_change().rolling(20).std() * np.sqrt(252) * 100
    return x

def signal_score(row):
    score = 0
    if row["Close"] > row["SMA50"]: score += 1
    if row["SMA50"] > row["SMA200"]: score += 1
    if row["Ret126"] > 10: score += 1
    if 45 <= row["RSI"] <= 65: score += 1
    if row["Close"] > row["SMA20"]: score += 1
    if row["Vol20"] < 35: score += 1
    return score

# ============================================================
# REALISTIC BACKTEST
# Signal is calculated at close; entry occurs next trading day.
# Exit target/stop is evaluated using next-day OHLC.
# If target and stop are both touched on one day, stop is
# conservatively assumed to occur first.
# Costs are applied on both entry and exit.
# ============================================================
def realistic_backtest(ticker, years, target_pct, stop_pct, min_score,
                       position_pct, brokerage, slippage_pct, initial=10000):
    raw = load_history(ticker, f"{years}y")
    if raw.empty or len(raw) < 260:
        return None

    df = indicators(raw).dropna().copy()
    cash = float(initial)
    shares = 0
    entry_price = None
    entry_date = None
    trades = []
    equity_points = []

    dates = list(df.index)

    for i in range(len(df) - 1):
        today = df.iloc[i]
        tomorrow = df.iloc[i + 1]
        today_date = dates[i]
        next_date = dates[i + 1]

        # Mark portfolio at today's close before next-day execution.
        mark = cash + (shares * float(today["Close"]) if shares else 0)
        equity_points.append((today_date, mark))

        # Manage an existing position using tomorrow's OHLC.
        if shares > 0:
            o = float(tomorrow["Open"])
            h = float(tomorrow["High"])
            l = float(tomorrow["Low"])

            target = entry_price * (1 + target_pct / 100)
            stop = entry_price * (1 - stop_pct / 100)

            exit_price = None
            exit_reason = None

            # Gap through stop/target gets next open.
            if o <= stop:
                exit_price = o
                exit_reason = "STOP (gap)"
            elif o >= target:
                exit_price = o
                exit_reason = "TARGET (gap)"
            # Intraday: conservative assumption if both levels hit.
            elif l <= stop and h >= target:
                exit_price = stop
                exit_reason = "STOP (both touched)"
            elif l <= stop:
                exit_price = stop
                exit_reason = "STOP"
            elif h >= target:
                exit_price = target
                exit_reason = "TARGET"

            if exit_price is not None:
                gross = (exit_price - entry_price) * shares
                sell_cost = max(float(brokerage), exit_price * shares * slippage_pct / 100)
                proceeds = exit_price * shares - sell_cost
                cash += proceeds
                net_pnl = gross - (entry_price * shares * slippage_pct / 100) - float(brokerage) - sell_cost + float(brokerage)
                # Simpler and explicit: net P/L from cash flows.
                buy_cash = -(entry_price * shares + max(float(brokerage), entry_price * shares * slippage_pct / 100))
                net_pnl = (exit_price * shares - sell_cost) + buy_cash

                trades.append({
                    "Entry date": entry_date,
                    "Exit date": next_date,
                    "Entry": entry_price,
                    "Exit": exit_price,
                    "Shares": shares,
                    "P/L": net_pnl,
                    "Reason": exit_reason
                })
                shares = 0
                entry_price = None
                entry_date = None

        # If flat, today's close signal creates a next-day order.
        if shares == 0:
            score = signal_score(today)
            if score >= min_score:
                next_open = float(tomorrow["Open"])
                max_value = cash * position_pct / 100
                qty = int(max_value // next_open)
                if qty >= 1:
                    buy_value = qty * next_open
                    buy_cost = max(float(brokerage), buy_value * slippage_pct / 100)
                    total = buy_value + buy_cost
                    if total <= cash:
                        cash -= total
                        shares = qty
                        entry_price = next_open
                        entry_date = next_date

    # Close any remaining position at final close.
    if shares > 0:
        final_date = dates[-1]
        final_price = float(df["Close"].iloc[-1])
        sell_cost = max(float(brokerage), final_price * shares * slippage_pct / 100)
        cash += final_price * shares - sell_cost
        buy_value = entry_price * shares
        buy_cost = max(float(brokerage), buy_value * slippage_pct / 100)
        net_pnl = (final_price * shares - sell_cost) - (buy_value + buy_cost)
        trades.append({
            "Entry date": entry_date,
            "Exit date": final_date,
            "Entry": entry_price,
            "Exit": final_price,
            "Shares": shares,
            "P/L": net_pnl,
            "Reason": "END OF TEST"
        })
        shares = 0

    equity_points.append((dates[-1], cash))
    curve = pd.Series([v for _, v in equity_points],
                      index=[d for d, _ in equity_points]).sort_index()
    curve = curve[~curve.index.duplicated(keep="last")]

    final_value = float(cash)
    total_return = (final_value / initial - 1) * 100
    years_actual = max((curve.index[-1] - curve.index[0]).days / 365.25, 0.01)
    cagr = ((final_value / initial) ** (1 / years_actual) - 1) * 100

    dd = curve / curve.cummax() - 1
    max_dd = float(dd.min() * 100)

    pnl = [t["P/L"] for t in trades]
    wins = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p <= 0]
    win_rate = len(wins) / len(pnl) * 100 if pnl else 0
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else (np.inf if wins else 0)

    return {
        "Ticker": ticker,
        "Final": final_value,
        "Return %": total_return,
        "CAGR %": cagr,
        "Trades": len(trades),
        "Win rate %": win_rate,
        "Max DD %": max_dd,
        "Profit factor": pf,
        "Avg win": np.mean(wins) if wins else 0,
        "Avg loss": np.mean(losses) if losses else 0,
        "curve": curve,
        "trades": pd.DataFrame(trades)
    }

def buy_hold(ticker, years, initial=10000):
    raw = load_history(ticker, f"{years}y")
    if raw.empty:
        return None
    close = raw["Close"].astype(float)
    first = float(close.iloc[0])
    last = float(close.iloc[-1])
    final = initial * last / first
    total = (final / initial - 1) * 100
    yrs = max((close.index[-1] - close.index[0]).days / 365.25, 0.01)
    cagr = ((final / initial) ** (1 / yrs) - 1) * 100
    curve = initial * close / first
    dd = curve / curve.cummax() - 1
    return {"Ticker": ticker, "Final": final, "Return %": total,
            "CAGR %": cagr, "Max DD %": float(dd.min()*100), "curve": curve}

# ============================================================
# PAPER ACCOUNT
# ============================================================
if "cash" not in st.session_state: st.session_state.cash = 10000.0
if "positions" not in st.session_state: st.session_state.positions = {}
if "trades" not in st.session_state: st.session_state.trades = []

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("⚙️ V5 Settings")
watch_text = st.sidebar.text_input("Watchlist", "BHP.AX,CBA.AX,CSL.AX,VAS.AX")
tickers = [x.strip().upper() for x in watch_text.split(",") if x.strip()]

st.session_state.target = st.sidebar.slider("Profit target %", 3, 50, 15)
st.session_state.stop = st.sidebar.slider("Stop-loss %", 1, 30, 8)
st.session_state.position = st.sidebar.slider("Maximum position %", 1, 25, 10)
st.session_state.min_score = st.sidebar.slider("Minimum BUY score", 3, 6, 4)

st.sidebar.subheader("Backtest realism")
years = st.sidebar.selectbox("Historical period", [2, 3, 5, 10], index=2)
brokerage = st.sidebar.number_input("Brokerage per transaction ($)", 0.0, 50.0, 6.50, 0.50)
slippage = st.sidebar.number_input("Slippage % per transaction", 0.0, 1.0, 0.10, 0.05)

st.sidebar.caption("Paper trading only. Historical results are not guarantees.")

if st.sidebar.button("🔄 Reset paper account"):
    st.session_state.cash = 10000.0
    st.session_state.positions = {}
    st.session_state.trades = []
    st.rerun()

# ============================================================
# HEADER
# ============================================================
st.title("📈 AI Investor V5")
st.caption("Realistic historical testing + paper trading • $10,000 virtual capital • No broker connection")

st.info("V5 uses next-day execution, transaction costs and slippage in the historical test. This reduces some common backtest distortions, but it does not make historical results predictive.")

tabs = st.tabs(["🧪 Backtest Lab", "🤖 Paper Trader", "📊 Research", "💼 Portfolio", "🧠 Learn", "🧾 Journal"])

# ============================================================
# BACKTEST LAB
# ============================================================
with tabs[0]:
    st.subheader("Realistic strategy backtest")
    st.write("Signal at today's close → order executed at the next trading day's open. Target/stop are tested against the next day's OHLC.")

    if st.button("🧪 Run realistic backtest", type="primary"):
        strategy = []
        hold = []
        curves = {}

        for ticker in tickers:
            try:
                r = realistic_backtest(
                    ticker, years, st.session_state.target, st.session_state.stop,
                    st.session_state.min_score, st.session_state.position,
                    brokerage, slippage
                )
                b = buy_hold(ticker, years)
                if r:
                    strategy.append(r)
                    curves[ticker + " strategy"] = r["curve"]
                if b:
                    hold.append(b)
                    curves[ticker + " buy & hold"] = b["curve"]
            except Exception as e:
                st.warning(f"{ticker}: {e}")

        if strategy:
            st.markdown("### Strategy results")
            rows = []
            for r in strategy:
                rows.append({
                    "Ticker": r["Ticker"],
                    "Final $": round(r["Final"], 2),
                    "Return %": round(r["Return %"], 2),
                    "CAGR %": round(r["CAGR %"], 2),
                    "Trades": r["Trades"],
                    "Win rate %": round(r["Win rate %"], 1),
                    "Max DD %": round(r["Max DD %"], 1),
                    "Profit factor": "∞" if np.isinf(r["Profit factor"]) else round(r["Profit factor"], 2)
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            st.markdown("### Strategy vs buy & hold")
            comparison = []
            for r in strategy:
                b = next((x for x in hold if x["Ticker"] == r["Ticker"]), None)
                comparison.append({
                    "Ticker": r["Ticker"],
                    "Strategy CAGR %": round(r["CAGR %"], 2),
                    "Buy & hold CAGR %": round(b["CAGR %"], 2) if b else None,
                    "Strategy Max DD %": round(r["Max DD %"], 1),
                    "Buy & hold Max DD %": round(b["Max DD %"], 1) if b else None,
                })
            st.dataframe(pd.DataFrame(comparison), use_container_width=True)

            st.markdown("### Equity curves")
            for ticker in tickers:
                if ticker + " strategy" in curves and ticker + " buy & hold" in curves:
                    chart = pd.concat(
                        [curves[ticker + " strategy"].rename("Strategy"),
                         curves[ticker + " buy & hold"].rename("Buy & Hold")],
                        axis=1
                    ).ffill()
                    st.line_chart(chart)

            st.markdown("### Interpretation")
            st.write(
                "The key comparison is not simply the highest return. Look at return, CAGR, "
                "maximum drawdown, number of trades and whether results remain reasonable after "
                "costs. A strategy that only works for one stock or one period may be fragile."
            )

# ============================================================
# PAPER TRADER
# ============================================================
with tabs[1]:
    st.subheader("Automatic paper trader")
    st.write("The paper engine keeps the V4 rule set but uses the same target, stop and risk controls.")

    if st.button("🔎 Run V5 paper scan"):
        results = []
        for ticker in tickers:
            try:
                raw = indicators(load_history(ticker, "2y")).dropna()
                if raw.empty: continue
                row = raw.iloc[-1]
                score = signal_score(row)
                results.append({
                    "Ticker": ticker, "Price": float(row["Close"]), "Score": score,
                    "6M %": float(row["Ret126"]), "RSI": float(row["RSI"]),
                    "Vol %": float(row["Vol20"])
                })
            except Exception as e:
                st.warning(f"{ticker}: {e}")

        # exits
        for ticker in list(st.session_state.positions):
            p = st.session_state.positions[ticker]
            a = next((x for x in results if x["Ticker"] == ticker), None)
            if not a: continue
            px = a["Price"]
            if px >= p["target"] or px <= p["stop"]:
                pnl = (px - p["entry"]) * p["shares"]
                st.session_state.cash += p["shares"] * px
                st.session_state.trades.append({
                    "Time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "Action": "SELL", "Ticker": ticker, "Shares": p["shares"],
                    "Entry": p["entry"], "Exit": px, "P/L": pnl
                })
                del st.session_state.positions[ticker]
                st.success(f"SELL {ticker} @ ${px:.2f} | P/L ${pnl:+.2f}")

        portfolio = st.session_state.cash
        for t,p in st.session_state.positions.items():
            a = next((x for x in results if x["Ticker"] == t), None)
            portfolio += p["shares"] * (a["Price"] if a else p["entry"])
        max_value = portfolio * st.session_state.position / 100

        for a in results:
            if a["Ticker"] in st.session_state.positions or a["Score"] < st.session_state.min_score:
                continue
            qty = int(min(st.session_state.cash // a["Price"], max_value // a["Price"]))
            if qty < 1: continue
            target = a["Price"] * (1 + st.session_state.target/100)
            stop = a["Price"] * (1 - st.session_state.stop/100)
            risk = (a["Price"] - stop) * qty
            st.session_state.cash -= qty * a["Price"]
            st.session_state.positions[a["Ticker"]] = {
                "shares": qty, "entry": a["Price"], "target": target,
                "stop": stop, "risk": risk
            }
            st.session_state.trades.append({
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Action": "PAPER BUY", "Ticker": a["Ticker"], "Shares": qty,
                "Entry": a["Price"], "Exit": "", "P/L": ""
            })
            st.success(f"BUY {qty} {a['Ticker']} @ ${a['Price']:.2f} | stop risk ${risk:.2f}")

        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True)

# ============================================================
# RESEARCH
# ============================================================
with tabs[2]:
    st.subheader("Research")
    for ticker in tickers:
        raw = indicators(load_history(ticker, "2y")).dropna()
        if raw.empty: continue
        row = raw.iloc[-1]
        score = signal_score(row)
        with st.expander(f"{ticker} — ${float(row['Close']):.2f} — score {score}/6"):
            c = st.columns(6)
            c[0].metric("Score", f"{score}/6")
            c[1].metric("6M", f"{float(row['Ret126']):+.1f}%")
            c[2].metric("1M", f"{float(row['Ret20']):+.1f}%")
            c[3].metric("RSI", f"{float(row['RSI']):.1f}")
            c[4].metric("Volatility", f"{float(row['Vol20']):.1f}%")
            c[5].metric("Price", f"${float(row['Close']):.2f}")
            st.write(f"SMA20 ${float(row['SMA20']):.2f} | SMA50 ${float(row['SMA50']):.2f} | SMA200 ${float(row['SMA200']):.2f}")

# ============================================================
# PORTFOLIO
# ============================================================
with tabs[3]:
    st.subheader("Paper portfolio")
    total = st.session_state.cash
    rows = []
    for ticker, p in st.session_state.positions.items():
        raw = indicators(load_history(ticker, "2y")).dropna()
        current = float(raw["Close"].iloc[-1]) if not raw.empty else p["entry"]
        value = current * p["shares"]
        total += value
        rows.append({
            "Ticker": ticker, "Shares": p["shares"],
            "Entry": round(p["entry"],2), "Current": round(current,2),
            "Unrealised P/L": round((current-p["entry"])*p["shares"],2),
            "Target": round(p["target"],2), "Stop": round(p["stop"],2),
            "Risk at stop": round(p["risk"],2)
        })
    c=st.columns(3)
    c[0].metric("Portfolio", f"${total:,.2f}")
    c[1].metric("P/L", f"${total-10000:+,.2f}")
    c[2].metric("Cash", f"${st.session_state.cash:,.2f}")
    if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else: st.info("No open paper positions.")

# ============================================================
# LEARN
# ============================================================
with tabs[4]:
    st.subheader("How to read V5")
    st.markdown("""
**Next-day execution:** V5 does not buy at the same closing price that created the signal. It waits for the next trading session's open.

**Brokerage:** a fixed dollar transaction cost is charged on each entry and exit.

**Slippage:** V5 adds a small execution-cost assumption to entry and exit. Real slippage varies with liquidity and market conditions.

**Buy & hold:** a simple benchmark that invests the same starting capital and holds the stock for the test period.

**CAGR:** annualised growth rate over the test period.

**Maximum drawdown:** the largest peak-to-trough decline in the simulated equity curve.

**Profit factor:** gross winning P/L divided by gross losing P/L. Above 1 means winning P/L exceeded losing P/L in that simulation; it is not a guarantee.

**Important:** Historical backtests can be affected by survivorship bias, data quality, corporate actions, parameter selection and other limitations. They should be treated as research, not proof of future performance.
""")

# ============================================================
# JOURNAL
# ============================================================
with tabs[5]:
    st.subheader("Trade journal")
    if st.session_state.trades:
        st.dataframe(pd.DataFrame(st.session_state.trades), use_container_width=True)
    else:
        st.info("No paper trades recorded yet.")

st.divider()
st.caption("AI Investor V5 • Educational research and paper trading only • No broker connection • No guaranteed returns")
