
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="AI Investor V7", page_icon="📈", layout="wide")

# ---------------- DATA ----------------
@st.cache_data(ttl=900)
def load_history(ticker, period="12y"):
    df = yf.download(ticker, period=period, interval="1d",
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

def add_indicators(df):
    x = df.copy()
    c = x["Close"].astype(float)
    x["SMA20"] = c.rolling(20).mean()
    x["SMA50"] = c.rolling(50).mean()
    x["SMA200"] = c.rolling(200).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["RSI"] = 100 - (100 / (1 + rs))
    x["MOM20"] = c.pct_change(20) * 100
    x["MOM63"] = c.pct_change(63) * 100
    x["MOM126"] = c.pct_change(126) * 100
    x["VOL20"] = c.pct_change().rolling(20).std() * np.sqrt(252) * 100
    x["ATR14"] = (x["High"] - x["Low"]).rolling(14).mean()
    return x

# ---------------- STRATEGIES ----------------
def strategy_signal(row, name):
    if name == "Trend":
        return bool(row["Close"] > row["SMA50"] and row["SMA50"] > row["SMA200"])
    if name == "Momentum":
        return bool(row["MOM63"] > 0 and row["MOM126"] > 0)
    if name == "Trend + Momentum":
        return bool(row["Close"] > row["SMA50"] and row["SMA50"] > row["SMA200"]
                    and row["MOM63"] > 0 and row["MOM126"] > 0)
    if name == "Mean Reversion":
        return bool(row["RSI"] < 35 and row["Close"] < row["SMA20"])
    return False

def run_strategy(df, strategy_name, initial=10000, position_pct=100,
                 brokerage=6.50, slippage_pct=0.10):
    d = add_indicators(df).dropna().copy()
    if len(d) < 250:
        return None

    cash = float(initial)
    shares = 0
    entry_price = None
    entry_date = None
    trades = []
    equity = []

    dates = list(d.index)
    for i in range(len(d) - 1):
        row = d.iloc[i]
        nxt = d.iloc[i + 1]
        dt = dates[i]
        ndt = dates[i + 1]

        # mark-to-market
        equity.append((dt, cash + shares * float(row["Close"])))

        # Exit when signal turns off; execute next open.
        if shares > 0 and not strategy_signal(row, strategy_name):
            px = float(nxt["Open"])
            value = shares * px
            cost = float(brokerage) + value * slippage_pct / 100
            cash += value - cost
            buy_cost = float(brokerage) + entry_price * shares * slippage_pct / 100
            pnl = (px * shares - cost) - (entry_price * shares + buy_cost)
            trades.append({
                "Entry date": entry_date, "Exit date": ndt,
                "Entry": entry_price, "Exit": px, "Shares": shares,
                "P/L": pnl
            })
            shares = 0
            entry_price = None
            entry_date = None

        # Enter when signal is on; execute next open.
        if shares == 0 and strategy_signal(row, strategy_name):
            px = float(nxt["Open"])
            max_value = cash * position_pct / 100
            qty = int(max_value // px)
            if qty >= 1:
                value = qty * px
                cost = float(brokerage) + value * slippage_pct / 100
                if value + cost <= cash:
                    cash -= value + cost
                    shares = qty
                    entry_price = px
                    entry_date = ndt

    # Close at final close for an apples-to-apples ending value.
    if shares > 0:
        px = float(d["Close"].iloc[-1])
        dt = dates[-1]
        value = shares * px
        sell_cost = float(brokerage) + value * slippage_pct / 100
        buy_cost = float(brokerage) + entry_price * shares * slippage_pct / 100
        cash += value - sell_cost
        pnl = (value - sell_cost) - (entry_price * shares + buy_cost)
        trades.append({
            "Entry date": entry_date, "Exit date": dt,
            "Entry": entry_price, "Exit": px, "Shares": shares,
            "P/L": pnl
        })
        shares = 0

    equity.append((dates[-1], cash))
    curve = pd.Series([v for _, v in equity],
                      index=[d for d, _ in equity]).sort_index()
    curve = curve[~curve.index.duplicated(keep="last")]
    curve = curve.reindex(pd.date_range(curve.index.min(), curve.index.max(), freq="B")).ffill()

    final = float(curve.iloc[-1])
    total_return = (final / initial - 1) * 100
    years = max((curve.index[-1] - curve.index[0]).days / 365.25, 0.01)
    cagr = ((final / initial) ** (1 / years) - 1) * 100
    dd = curve / curve.cummax() - 1
    max_dd = float(dd.min() * 100)

    pnl = [t["P/L"] for t in trades]
    wins = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p <= 0]
    win_rate = 100 * len(wins) / len(pnl) if pnl else 0
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_win / gross_loss if gross_loss else (np.inf if gross_win else 0)
    exposure_days = 0
    for t in trades:
        exposure_days += max((pd.Timestamp(t["Exit date"]) - pd.Timestamp(t["Entry date"])).days, 0)
    time_in_market = 100 * exposure_days / max((curve.index[-1] - curve.index[0]).days, 1)

    return {
        "Strategy": strategy_name, "Final $": final, "Return %": total_return,
        "CAGR %": cagr, "Trades": len(trades), "Win rate %": win_rate,
        "Max DD %": max_dd, "Profit factor": pf,
        "Avg win $": np.mean(wins) if wins else 0,
        "Avg loss $": np.mean(losses) if losses else 0,
        "Time in market %": time_in_market,
        "curve": curve, "trades": pd.DataFrame(trades)
    }

def buy_hold(df, initial=10000):
    c = df["Close"].astype(float).dropna()
    if len(c) < 2:
        return None
    curve = initial * c / float(c.iloc[0])
    final = float(curve.iloc[-1])
    total = (final / initial - 1) * 100
    years = max((curve.index[-1] - curve.index[0]).days / 365.25, 0.01)
    cagr = ((final / initial) ** (1 / years) - 1) * 100
    dd = curve / curve.cummax() - 1
    return {"Final $": final, "Return %": total, "CAGR %": cagr,
            "Max DD %": float(dd.min()*100), "curve": curve}

def run_period(df, start, end, strategy_name, **kwargs):
    d = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    return run_strategy(d, strategy_name, **kwargs) if len(d) else None

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ V7 Settings")
watch_text = st.sidebar.text_input("Watchlist", "BHP.AX,CBA.AX,CSL.AX,VAS.AX")
tickers = [x.strip().upper() for x in watch_text.split(",") if x.strip()]
years = st.sidebar.selectbox("Test period", [5, 7, 10], index=0)
brokerage = st.sidebar.number_input("Brokerage / transaction ($)", 0.0, 50.0, 6.50, 0.50)
slippage = st.sidebar.number_input("Slippage / transaction (%)", 0.0, 1.0, 0.10, 0.05)
position_pct = st.sidebar.slider("Position size (% of cash)", 10, 100, 100, 10)

if "paper_cash" not in st.session_state:
    st.session_state.paper_cash = 10000.0
if "paper_positions" not in st.session_state:
    st.session_state.paper_positions = {}
if "paper_trades" not in st.session_state:
    st.session_state.paper_trades = []

if "v7_robustness" not in st.session_state:
    st.session_state.v7_robustness = pd.DataFrame()


# ---------------- HEADER ----------------
st.title("📈 AI Investor V7")
st.caption("Strategy research laboratory + realistic historical testing + paper trading")
st.info("V6 is a research tool. Historical simulations are not forecasts or guarantees. Avoid changing rules repeatedly just to improve a past result.")

tabs = st.tabs(["🧪 Strategy Lab", "🧠 Robustness", "🤖 Paper Trader", "📊 Research", "💼 Portfolio", "📚 Learn", "🧾 Journal"])

# ---------------- STRATEGY LAB ----------------
with tabs[0]:
    st.subheader("Compare independent strategy families")
    st.write("All entries use next-day open execution. Costs and slippage are included.")

    if st.button("🧪 Run V7 strategy comparison", type="primary"):
        all_rows = []
        benchmark_rows = []
        chart_data = {}

        for ticker in tickers:
            raw = load_history(ticker, f"{years+2}y")
            if raw.empty:
                continue
            raw = raw.sort_index()
            for name in ["Trend", "Momentum", "Trend + Momentum", "Mean Reversion"]:
                r = run_strategy(raw.tail(int(years*252 + 50)), name,
                                 initial=10000, position_pct=position_pct,
                                 brokerage=brokerage, slippage_pct=slippage)
                if r:
                    all_rows.append({"Ticker": ticker, **{k:v for k,v in r.items() if k not in ["curve","trades"]}})
                    chart_data[f"{ticker} — {name}"] = r["curve"]
            b = buy_hold(raw.tail(int(years*252 + 50)))
            if b:
                benchmark_rows.append({"Ticker": ticker, "Buy & Hold Final $": b["Final $"],
                                        "Buy & Hold Return %": b["Return %"],
                                        "Buy & Hold CAGR %": b["CAGR %"],
                                        "Buy & Hold Max DD %": b["Max DD %"]})
                chart_data[f"{ticker} — Buy & Hold"] = b["curve"]

        if all_rows:
            result_df = pd.DataFrame(all_rows)
            st.session_state.v7_results = result_df
            st.session_state.v7_bench = pd.DataFrame(benchmark_rows)
            st.session_state.v7_curves = chart_data

    if "v7_results" in st.session_state:
        df = st.session_state.v7_results.copy()
        display = df.copy()
        display["Final $"] = display["Final $"].round(2)
        for col in ["Return %","CAGR %","Win rate %","Max DD %","Avg win $","Avg loss $","Time in market %"]:
            display[col] = display[col].round(2)
        display["Profit factor"] = display["Profit factor"].apply(lambda x: "∞" if np.isinf(x) else round(x,2))
        st.dataframe(display, use_container_width=True)

        st.subheader("Buy & hold benchmark")
        b = st.session_state.v7_bench.copy()
        for col in b.columns[1:]:
            b[col] = b[col].round(2)
        st.dataframe(b, use_container_width=True)

        st.subheader("Normalised $10,000 equity curves")
        st.caption("Each line starts at $10,000. This makes strategy paths comparable without mixing raw stock prices.")
        for ticker in tickers:
            names = [c for c in st.session_state.v7_curves if c.startswith(ticker + " — ")]
            if names:
                chart = pd.concat([st.session_state.v7_curves[n].rename(n.split(" — ")[1]) for n in names], axis=1).ffill()
                st.line_chart(chart)

# ---------------- ROBUSTNESS ----------------
with tabs[1]:
    st.subheader("🧠 Robustness scorecard")
    st.write("V7 keeps the chronological Training → Validation → Out-of-sample split, then adds transparent consistency checks. Nothing is automatically tuned to maximise a historical result.")

    st.markdown("""
**How to read the scorecard**
- **Training** is the earliest historical slice.
- **Validation** is the middle slice.
- **Out-of-sample (OOS)** is the latest held-back slice.
- V7 reports whether each metric passes simple, predefined research checks. These are **not forecasts** and are not guarantees of future returns.
""")

    with st.expander("⚙️ V7 research checks", expanded=False):
        st.write("These thresholds are fixed before looking at the results, to reduce hindsight bias.")
        c1, c2, c3, c4 = st.columns(4)
        min_trades = c1.number_input("Minimum OOS trades", 1, 100, 8, 1)
        min_pf = c2.number_input("Minimum OOS profit factor", 0.1, 5.0, 1.0, 0.1)
        min_cagr = c3.number_input("Minimum OOS CAGR %", -20.0, 50.0, 0.0, 0.5)
        max_dd_limit = c4.number_input("Maximum OOS drawdown %", -100.0, 0.0, -25.0, 1.0)

    if st.button("🔬 Run V7 robustness scorecard", type="primary"):
        rows = []
        for ticker in tickers:
            raw = load_history(ticker, f"{years+2}y").sort_index()
            if len(raw) < 500:
                continue
            n = len(raw)
            cut1 = int(n * 0.55)
            cut2 = int(n * 0.775)
            periods = [
                ("Training", raw.iloc[:cut1]),
                ("Validation", raw.iloc[cut1:cut2]),
                ("Out-of-sample", raw.iloc[cut2:])
            ]
            for period_name, part in periods:
                for strategy in ["Trend", "Momentum", "Trend + Momentum", "Mean Reversion"]:
                    r = run_strategy(part, strategy, initial=10000,
                                     position_pct=position_pct,
                                     brokerage=brokerage, slippage_pct=slippage)
                    if r:
                        rows.append({
                            "Ticker": ticker, "Period": period_name,
                            "Strategy": strategy,
                            "CAGR %": r["CAGR %"], "Max DD %": r["Max DD %"],
                            "Trades": r["Trades"], "Win rate %": r["Win rate %"],
                            "Profit factor": r["Profit factor"]
                        })

        if rows:
            rr = pd.DataFrame(rows)
            st.session_state.v7_robustness = rr

    rr = st.session_state.v7_robustness
    if not rr.empty:
        st.subheader("Full robustness results")
        st.dataframe(rr.round(2), use_container_width=True)

        # Per ticker + strategy scorecard
        score_rows = []
        for (ticker, strategy), g in rr.groupby(["Ticker", "Strategy"]):
            piv = g.set_index("Period")
            tr = piv.loc["Training"] if "Training" in piv.index else None
            va = piv.loc["Validation"] if "Validation" in piv.index else None
            oo = piv.loc["Out-of-sample"] if "Out-of-sample" in piv.index else None
            if tr is None or va is None or oo is None:
                continue

            oos_checks = {
                "OOS CAGR check": oo["CAGR %"] >= min_cagr,
                "OOS drawdown check": oo["Max DD %"] >= max_dd_limit,
                "OOS trade-count check": oo["Trades"] >= min_trades,
                "OOS profit-factor check": (oo["Profit factor"] >= min_pf) if np.isfinite(oo["Profit factor"]) else True,
            }
            # Stability is descriptive: how much of training CAGR remains in OOS.
            if abs(tr["CAGR %"]) > 0.25:
                stability_ratio = oo["CAGR %"] / tr["CAGR %"]
            else:
                stability_ratio = np.nan

            passed = sum(bool(v) for v in oos_checks.values())
            if passed == 4 and oo["CAGR %"] > 0:
                status = "Consistent checks"
            elif passed >= 2:
                status = "Mixed / investigate"
            else:
                status = "Weak OOS checks"

            score_rows.append({
                "Ticker": ticker, "Strategy": strategy,
                "Training CAGR %": tr["CAGR %"],
                "Validation CAGR %": va["CAGR %"],
                "OOS CAGR %": oo["CAGR %"],
                "OOS Max DD %": oo["Max DD %"],
                "OOS Trades": oo["Trades"],
                "OOS Win rate %": oo["Win rate %"],
                "OOS Profit factor": oo["Profit factor"],
                "OOS / Training CAGR": stability_ratio,
                "Checks passed": f"{passed}/4",
                "Research status": status
            })

        score = pd.DataFrame(score_rows)
        if not score.empty:
            st.subheader("Strategy-by-asset scorecard")
            view = score.copy()
            numeric_cols = [c for c in view.columns if c not in ["Ticker","Strategy","Checks passed","Research status"]]
            for c in numeric_cols:
                view[c] = pd.to_numeric(view[c], errors="coerce").round(2)
            st.dataframe(view, use_container_width=True)

            st.subheader("Cross-asset consistency")
            agg = score.groupby("Strategy").agg(
                Assets=("Ticker", "count"),
                Positive_OOS_CAGR=("OOS CAGR %", lambda x: int((x > 0).sum())),
                Avg_OOS_CAGR=("OOS CAGR %", "mean"),
                Median_OOS_CAGR=("OOS CAGR %", "median"),
                Avg_OOS_Max_DD=("OOS Max DD %", "mean"),
                Median_OOS_PF=("OOS Profit factor", "median"),
            ).reset_index()
            agg["Positive OOS CAGR %"] = 100 * agg["Positive_OOS_CAGR"] / agg["Assets"]
            st.dataframe(agg.round(2), use_container_width=True)

            st.info("V7 does not automatically select a strategy. A 'Consistent checks' label only means the strategy met the displayed, pre-set research thresholds on the historical OOS slice for that asset. It is not a recommendation or a prediction.")

            st.subheader("Training → Validation → OOS CAGR")
            chart = score.set_index(["Ticker", "Strategy"])[["Training CAGR %", "Validation CAGR %", "OOS CAGR %"]]
            st.bar_chart(chart)

            st.subheader("Why this matters")
            st.write("A large training result followed by weak OOS results is a warning that the historical pattern may not generalise. V7 makes that comparison explicit instead of selecting a strategy from the highest historical return alone.")
    else:
        st.info("Run the V7 robustness scorecard to generate the Training, Validation and Out-of-sample comparison.")

# ---------------- PAPER TRADER ----------------
with tabs[2]:
    st.subheader("Paper trader")
    st.write("Paper trading remains separate from the research lab. It never places real orders.")
    if st.button("🔎 Run paper scan"):
        results = []
        for ticker in tickers:
            raw = add_indicators(load_history(ticker, "2y")).dropna()
            if raw.empty: continue
            r = raw.iloc[-1]
            signals = {s: strategy_signal(r, s) for s in ["Trend","Momentum","Trend + Momentum","Mean Reversion"]}
            score = sum(signals.values())
            results.append({"Ticker": ticker, "Price": float(r["Close"]),
                            "Strategy signals": score, **signals,
                            "RSI": float(r["RSI"]), "6M %": float(r["MOM126"]),
                            "Vol %": float(r["VOL20"])})
        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True)
            st.info("Paper BUY/SELL automation is intentionally conservative in V6: use the strategy lab first to choose rules based on robustness, then we can promote a tested rule set into the paper engine.")

# ---------------- RESEARCH ----------------
with tabs[3]:
    st.subheader("Research dashboard")
    for ticker in tickers:
        raw = add_indicators(load_history(ticker, "2y")).dropna()
        if raw.empty: continue
        r = raw.iloc[-1]
        with st.expander(f"{ticker} — ${float(r['Close']):.2f}"):
            cols = st.columns(6)
            cols[0].metric("1M momentum", f"{r['MOM20']:+.1f}%")
            cols[1].metric("3M momentum", f"{r['MOM63']:+.1f}%")
            cols[2].metric("6M momentum", f"{r['MOM126']:+.1f}%")
            cols[3].metric("RSI", f"{r['RSI']:.1f}")
            cols[4].metric("Volatility", f"{r['VOL20']:.1f}%")
            cols[5].metric("SMA50", f"${r['SMA50']:.2f}")
            st.write(f"SMA20: ${r['SMA20']:.2f} | SMA200: ${r['SMA200']:.2f}")

# ---------------- PORTFOLIO ----------------
with tabs[4]:
    st.subheader("Paper portfolio")
    total = st.session_state.paper_cash
    rows = []
    for ticker, p in st.session_state.paper_positions.items():
        raw = add_indicators(load_history(ticker, "2y")).dropna()
        px = float(raw["Close"].iloc[-1]) if not raw.empty else p["entry"]
        value = p["shares"] * px
        total += value
        rows.append({"Ticker":ticker, "Shares":p["shares"],
                     "Entry":p["entry"], "Current":px,
                     "Unrealised P/L":(px-p["entry"])*p["shares"]})
    c = st.columns(3)
    c[0].metric("Portfolio", f"${total:,.2f}")
    c[1].metric("P/L", f"${total-10000:+,.2f}")
    c[2].metric("Cash", f"${st.session_state.paper_cash:,.2f}")
    if rows: st.dataframe(pd.DataFrame(rows).round(2), use_container_width=True)
    else: st.info("No paper positions.")

# ---------------- LEARN ----------------
with tabs[5]:
    st.subheader("V7 concepts")
    st.markdown("""
**Trend:** buys when price and moving-average structure indicate an upward trend.

**Momentum:** buys when recent medium-term returns are positive.

**Trend + Momentum:** requires both conditions.

**Mean reversion:** looks for oversold/pullback conditions. These signals can be early or remain weak for long periods.

**Out-of-sample:** data held back from the earlier portions of the test. It is useful for checking whether a rule behaves similarly on unseen historical data.

**CAGR:** annualised historical growth rate.

**Maximum drawdown:** largest peak-to-trough decline in the simulated equity curve.

**Profit factor:** gross winning P/L divided by gross losing P/L. It is descriptive and sample-dependent.

**Why V7 does not auto-pick a winner:** choosing whichever historical strategy has the highest past return can create overfitting. V7 instead makes pre-set OOS checks and cross-asset consistency visible. A research status is descriptive, not a forecast.
""")

# ---------------- JOURNAL ----------------
with tabs[6]:
    st.subheader("Paper trade journal")
    if st.session_state.paper_trades:
        st.dataframe(pd.DataFrame(st.session_state.paper_trades), use_container_width=True)
    else:
        st.info("No paper trades recorded.")

st.divider()
st.caption("AI Investor V7 • Educational research and paper trading only • No broker connection • No guaranteed returns")
