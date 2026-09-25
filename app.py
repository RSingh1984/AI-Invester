import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass

st.set_page_config(page_title="AI Investor V8", page_icon="📈", layout="wide")

WATCHLIST_DEFAULT = "BHP.AX,CBA.AX,CSL.AX,VAS.AX"
STRATEGIES = ["Trend", "Momentum", "Trend + Momentum", "Mean Reversion"]

# ---------------- DATA ----------------
@st.cache_data(ttl=900)
def load_history(ticker, period="12y"):
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    needed = ["Open", "High", "Low", "Close"]
    return df.dropna(subset=[c for c in needed if c in df.columns]).sort_index()


def add_indicators(df):
    x = df.copy()
    c = x["Close"].astype(float)
    h = x["High"].astype(float)
    l = x["Low"].astype(float)
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
    prev_close = c.shift(1)
    tr = pd.concat([(h-l), (h-prev_close).abs(), (l-prev_close).abs()], axis=1).max(axis=1)
    x["ATR14"] = tr.rolling(14).mean()
    x["ATR_PCT"] = x["ATR14"] / c * 100
    return x.dropna()


# ---------------- RULE ENGINE ----------------
def classify_regime(row):
    close = float(row["Close"])
    s20, s50, s200 = float(row["SMA20"]), float(row["SMA50"]), float(row["SMA200"])
    mom63 = float(row["MOM63"])
    if close > s50 > s200 and mom63 > 0:
        return "Bull trend"
    if close < s50 < s200 and mom63 < 0:
        return "Bear trend"
    return "Range / transition"


def strategy_signal(row, name):
    if name == "Trend":
        return bool(row["Close"] > row["SMA50"] and row["SMA50"] > row["SMA200"])
    if name == "Momentum":
        return bool(row["MOM63"] > 0 and row["MOM126"] > 0)
    if name == "Trend + Momentum":
        return bool(row["Close"] > row["SMA50"] and row["SMA50"] > row["SMA200"] and row["MOM63"] > 0 and row["MOM126"] > 0)
    if name == "Mean Reversion":
        return bool(row["RSI"] < 35 and row["Close"] < row["SMA20"])
    return False


def regime_signal(row):
    regime = classify_regime(row)
    if regime == "Bull trend":
        # Require both trend and medium-term momentum.
        return bool(row["Close"] > row["SMA50"] > row["SMA200"] and row["MOM63"] > 0 and row["MOM126"] > 0), "Trend + Momentum"
    if regime == "Range / transition":
        # Mean reversion only when the pullback is relatively controlled.
        return bool(row["RSI"] < 35 and row["Close"] < row["SMA20"] and row["MOM20"] > -12), "Mean Reversion"
    return False, "Cash / defensive"


# ---------------- METRICS ----------------
def metrics_from_curve(curve, trades, initial=10000):
    curve = pd.Series(curve).dropna()
    if curve.empty:
        return None
    curve = curve[~curve.index.duplicated(keep="last")]
    final = float(curve.iloc[-1])
    total = (final / initial - 1) * 100
    years = max((curve.index[-1] - curve.index[0]).days / 365.25, 0.01)
    cagr = ((final / initial) ** (1 / years) - 1) * 100
    dd = curve / curve.cummax() - 1
    max_dd = float(dd.min() * 100)
    pnl = [float(t["P/L"]) for t in trades]
    wins = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p <= 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_win / gross_loss if gross_loss else (np.inf if gross_win else 0)
    return {
        "Final $": final, "Return %": total, "CAGR %": cagr,
        "Trades": len(pnl), "Win rate %": 100*len(wins)/len(pnl) if pnl else 0,
        "Max DD %": max_dd, "Profit factor": pf,
        "Avg win $": np.mean(wins) if wins else 0,
        "Avg loss $": np.mean(losses) if losses else 0,
    }


def buy_hold_curve(data, initial=10000):
    c = data["Close"].astype(float).dropna()
    return initial * c / float(c.iloc[0]) if len(c) else pd.Series(dtype=float)


# ---------------- PORTFOLIO BACKTEST ----------------
def run_portfolio(data_map, initial=10000, risk_pct=0.75, max_pos_pct=25,
                  max_exposure_pct=80, stop_atr=2.0, target_r=3.0,
                  brokerage=6.50, slippage_pct=0.10, max_positions=3):
    prepared = {t: add_indicators(df) for t, df in data_map.items() if not df.empty}
    prepared = {t: d for t, d in prepared.items() if len(d) >= 260}
    if not prepared:
        return None

    dates = sorted(set().union(*[set(d.index) for d in prepared.values()]))
    dates = [pd.Timestamp(x) for x in dates]
    cash = float(initial)
    positions = {}
    trades = []
    curve = []
    exposure_series = []

    for i, dt in enumerate(dates[:-1]):
        next_dt = dates[i+1]
        # Only process if next day exists for a ticker.
        # First mark current equity.
        equity = cash
        for t, p in positions.items():
            d = prepared[t]
            if dt in d.index:
                equity += p["shares"] * float(d.loc[dt, "Close"])
            else:
                equity += p["shares"] * p["last_price"]
        curve.append((dt, equity))
        exposure_series.append((dt, 100 * (equity-cash) / equity if equity else 0))

        # Exit/stop/target checks at next day's OHLC. Conservative: if stop and target both touched, stop wins.
        for t in list(positions.keys()):
            d = prepared[t]
            if next_dt not in d.index:
                continue
            row = d.loc[next_dt]
            p = positions[t]
            low, high, op = float(row["Low"]), float(row["High"]), float(row["Open"])
            exit_px, reason = None, None
            if low <= p["stop"] and high >= p["target"]:
                exit_px, reason = p["stop"], "Stop (both touched)"
            elif low <= p["stop"]:
                exit_px, reason = p["stop"], "Stop"
            elif high >= p["target"]:
                exit_px, reason = p["target"], "Target"
            else:
                prev = d.loc[dt]
                signal, _ = regime_signal(prev)
                if not signal:
                    exit_px, reason = op, "Regime exit"
            if exit_px is not None:
                gross = p["shares"] * exit_px
                sell_cost = brokerage + gross * slippage_pct / 100
                cash += gross - sell_cost
                buy_cost = p["entry_value"] * slippage_pct / 100 + brokerage
                pnl = (gross - sell_cost) - (p["entry_value"] + buy_cost)
                trades.append({"Ticker": t, "Entry date": p["entry_date"], "Exit date": next_dt,
                               "Entry": p["entry_price"], "Exit": exit_px, "Shares": p["shares"],
                               "P/L": pnl, "Reason": reason, "Strategy": p["strategy"]})
                del positions[t]

        # New entries based on today's close, executed at next day's open.
        if len(positions) < max_positions:
            candidates = []
            for t, d in prepared.items():
                if t in positions or dt not in d.index or next_dt not in d.index:
                    continue
                row = d.loc[dt]
                signal, strategy = regime_signal(row)
                if not signal:
                    continue
                # Rank candidates by a simple fixed score; no historical tuning.
                score = float(row["MOM63"]) + float(row["MOM126"]) - max(float(row["VOL20"])-35, 0)*0.2
                candidates.append((score, t, strategy))
            candidates.sort(reverse=True)
            for _, t, strategy in candidates:
                if len(positions) >= max_positions:
                    break
                d = prepared[t]
                row = d.loc[dt]
                nxt = d.loc[next_dt]
                px = float(nxt["Open"])
                atr = float(row["ATR14"])
                if not np.isfinite(px) or not np.isfinite(atr) or atr <= 0:
                    continue
                stop = px - stop_atr * atr
                risk_per_share = px - stop
                if risk_per_share <= 0:
                    continue
                current_equity = cash + sum(p["shares"] * (float(prepared[t2].loc[dt,"Close"]) if dt in prepared[t2].index else p["last_price"]) for t2,p in positions.items())
                risk_budget = current_equity * risk_pct / 100
                qty_risk = int(risk_budget // risk_per_share)
                max_value = min(current_equity * max_pos_pct/100, current_equity * max_exposure_pct/100 - sum(p["shares"]*p["entry_price"] for p in positions.values()))
                qty_cap = int(max(max_value, 0) // px)
                qty = min(qty_risk, qty_cap)
                if qty < 1:
                    continue
                value = qty * px
                buy_cost = brokerage + value * slippage_pct / 100
                if value + buy_cost > cash:
                    continue
                cash -= value + buy_cost
                positions[t] = {
                    "shares": qty, "entry_price": px, "entry_value": value,
                    "entry_date": next_dt, "stop": stop, "target": px + target_r * risk_per_share,
                    "strategy": strategy, "last_price": px
                }

        for t, p in positions.items():
            if next_dt in prepared[t].index:
                p["last_price"] = float(prepared[t].loc[next_dt, "Close"])

    # Close any remaining positions at their final available close.
    for t, p in list(positions.items()):
        d = prepared[t]
        last_dt = d.index[-1]
        px = float(d["Close"].iloc[-1])
        gross = p["shares"] * px
        sell_cost = brokerage + gross * slippage_pct / 100
        cash += gross - sell_cost
        buy_cost = p["entry_value"] * slippage_pct / 100 + brokerage
        pnl = (gross - sell_cost) - (p["entry_value"] + buy_cost)
        trades.append({"Ticker": t, "Entry date": p["entry_date"], "Exit date": last_dt,
                       "Entry": p["entry_price"], "Exit": px, "Shares": p["shares"],
                       "P/L": pnl, "Reason": "End of test", "Strategy": p["strategy"]})
        del positions[t]

    if not curve:
        return None
    curve.append((max(d.index[-1] for d in prepared.values()), cash))
    curve_s = pd.Series({pd.Timestamp(k): v for k,v in curve}).sort_index()
    curve_s = curve_s[~curve_s.index.duplicated(keep="last")]
    curve_s = curve_s.reindex(pd.date_range(curve_s.index.min(), curve_s.index.max(), freq="B")).ffill()
    m = metrics_from_curve(curve_s, trades, initial)
    m["curve"] = curve_s
    m["trades_df"] = pd.DataFrame(trades)
    m["avg exposure %"] = float(np.mean([x[1] for x in exposure_series])) if exposure_series else 0
    return m


def split_data(data_map, fractions=(0.55, 0.225, 0.225)):
    out = {}
    for t, df in data_map.items():
        n = len(df)
        a = int(n*fractions[0]); b = int(n*(fractions[0]+fractions[1]))
        out[t] = (df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy())
    return out


# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ V8 Settings")
watch_text = st.sidebar.text_input("Watchlist", WATCHLIST_DEFAULT)
tickers = [x.strip().upper() for x in watch_text.split(",") if x.strip()]
years = st.sidebar.selectbox("Test period", [5, 7, 10], index=0)
brokerage = st.sidebar.number_input("Brokerage / transaction ($)", 0.0, 50.0, 6.50, 0.50)
slippage = st.sidebar.number_input("Slippage / transaction (%)", 0.0, 1.0, 0.10, 0.05)
risk_pct = st.sidebar.slider("Risk per trade (% equity)", 0.25, 2.0, 0.75, 0.25)
max_pos_pct = st.sidebar.slider("Max position (% equity)", 5, 50, 25, 5)
max_exposure_pct = st.sidebar.slider("Max invested exposure (%)", 25, 100, 80, 5)
max_positions = st.sidebar.slider("Max simultaneous positions", 1, 4, 3, 1)
stop_atr = st.sidebar.slider("Stop distance (ATR)", 1.0, 4.0, 2.0, 0.25)
target_r = st.sidebar.slider("Profit target (R)", 1.0, 5.0, 3.0, 0.5)

if "v8_portfolio" not in st.session_state:
    st.session_state.v8_portfolio = None
if "v8_strategy" not in st.session_state:
    st.session_state.v8_strategy = pd.DataFrame()

st.title("📈 AI Investor V8")
st.caption("Portfolio regime engine + risk-based backtesting + paper-trading laboratory")
st.info("V8 is an educational research and paper-trading tool. It does not guarantee returns and has no broker connection.")

tabs = st.tabs(["🏦 Portfolio Lab", "🔬 Robustness", "🤖 Paper Trader", "📊 Research", "💼 Portfolio", "📚 Learn"])

# ---------------- PORTFOLIO LAB ----------------
with tabs[0]:
    st.subheader("V8 portfolio regime engine")
    st.write("The engine first classifies each asset as Bull trend, Range / transition, or Bear trend. It then uses a fixed strategy rule and sizes positions from stop distance and portfolio risk.")
    if st.button("🚀 Run V8 portfolio backtest", type="primary"):
        data_map = {}
        for ticker in tickers:
            raw = load_history(ticker, f"{years+2}y")
            if not raw.empty:
                data_map[ticker] = raw.tail(int(years*252 + 260))
        if data_map:
            result = run_portfolio(data_map, initial=10000, risk_pct=risk_pct, max_pos_pct=max_pos_pct,
                                   max_exposure_pct=max_exposure_pct, stop_atr=stop_atr, target_r=target_r,
                                   brokerage=brokerage, slippage_pct=slippage, max_positions=max_positions)
            st.session_state.v8_portfolio = result
            # Individual regime strategy diagnostics on the same test period.
            rows=[]
            for t, raw in data_map.items():
                d=add_indicators(raw)
                for strat in STRATEGIES:
                    # Use a simple fixed strategy simulation from V7 logic for reference.
                    cash=10000.0; shares=0; entry=None; trades=[]; eq=[]
                    for i in range(len(d)-1):
                        row=d.iloc[i]; nxt=d.iloc[i+1]
                        eq.append((d.index[i], cash+shares*float(row['Close'])))
                        sig=strategy_signal(row,strat)
                        if shares and not sig:
                            px=float(nxt['Open']); value=shares*px; cost=brokerage+value*slippage/100
                            buycost=brokerage+entry[0]*shares*slippage/100
                            cash += value-cost; trades.append({'P/L':(value-cost)-(entry[0]*shares+buycost)})
                            shares=0; entry=None
                        if not shares and sig:
                            px=float(nxt['Open']); qty=int((cash*max_pos_pct/100)//px); value=qty*px; cost=brokerage+value*slippage/100
                            if qty>=1 and value+cost<=cash:
                                cash-=value+cost; shares=qty; entry=(px,d.index[i+1])
                    if shares:
                        px=float(d['Close'].iloc[-1]); value=shares*px; cost=brokerage+value*slippage/100; buycost=brokerage+entry[0]*shares*slippage/100
                        cash+=value-cost; trades.append({'P/L':(value-cost)-(entry[0]*shares+buycost)})
                    eq.append((d.index[-1],cash))
                    curve=pd.Series({k:v for k,v in eq}).sort_index(); curve=curve.reindex(pd.date_range(curve.index.min(),curve.index.max(),freq='B')).ffill()
                    mm=metrics_from_curve(curve,trades,10000)
                    if mm: rows.append({'Ticker':t,'Strategy':strat,**{k:mm[k] for k in ['Final $','Return %','CAGR %','Trades','Win rate %','Max DD %','Profit factor']}})
            st.session_state.v8_strategy=pd.DataFrame(rows)

    if st.session_state.v8_portfolio:
        r=st.session_state.v8_portfolio
        c=st.columns(5)
        c[0].metric("Final portfolio", f"${r['Final $']:,.2f}")
        c[1].metric("Return", f"{r['Return %']:+.2f}%")
        c[2].metric("CAGR", f"{r['CAGR %']:+.2f}%")
        c[3].metric("Max drawdown", f"{r['Max DD %']:.2f}%")
        c[4].metric("Profit factor", "∞" if np.isinf(r['Profit factor']) else f"{r['Profit factor']:.2f}")
        st.write(f"Trades: **{r['Trades']}** | Win rate: **{r['Win rate %']:.1f}%** | Average exposure: **{r['avg exposure %']:.1f}%**")
        st.subheader("Portfolio equity curve")
        st.line_chart(r["curve"])
        st.subheader("Trade log")
        if not r["trades_df"].empty:
            st.dataframe(r["trades_df"].round(2), use_container_width=True)
        st.subheader("Fixed strategy diagnostics")
        if not st.session_state.v8_strategy.empty:
            st.dataframe(st.session_state.v8_strategy.round(2), use_container_width=True)
        st.caption("The diagnostics are descriptive comparisons. The portfolio engine itself uses the fixed regime rules shown in Learn; it does not pick the historical winner.")

# ---------------- ROBUSTNESS ----------------
with tabs[1]:
    st.subheader("🔬 V8 walk-forward robustness")
    st.write("This section evaluates the same fixed regime engine separately on chronological slices. No automatic parameter tuning is performed.")
    if st.button("🔬 Run V8 robustness test", type="primary"):
        rows=[]
        for ticker in tickers:
            raw=load_history(ticker,f"{years+2}y")
            if raw.empty: continue
            parts=split_data({ticker: raw.tail(int(years*252+260))})[ticker]
            for name,part in zip(["Training","Validation","Out-of-sample"],parts):
                # Single-asset portfolio simulation; still uses risk-based sizing.
                res=run_portfolio({ticker:part},initial=10000,risk_pct=risk_pct,max_pos_pct=max_pos_pct,
                                  max_exposure_pct=100,stop_atr=stop_atr,target_r=target_r,
                                  brokerage=brokerage,slippage_pct=slippage,max_positions=1)
                if res:
                    rows.append({'Ticker':ticker,'Period':name,'CAGR %':res['CAGR %'],'Max DD %':res['Max DD %'],
                                 'Trades':res['Trades'],'Win rate %':res['Win rate %'],'Profit factor':res['Profit factor']})
        if rows:
            st.session_state.v8_robust=pd.DataFrame(rows)
    if "v8_robust" in st.session_state and not st.session_state.v8_robust.empty:
        rr=st.session_state.v8_robust.copy()
        st.dataframe(rr.round(2),use_container_width=True)
        st.subheader("OOS summary")
        oos=rr[rr['Period']=='Out-of-sample']
        if not oos.empty:
            st.dataframe(oos.round(2),use_container_width=True)
            st.info("OOS is a historical held-back slice, not a forecast. A positive OOS result does not establish future profitability.")

# ---------------- PAPER TRADER ----------------
with tabs[2]:
    st.subheader("🤖 V8 paper trader")
    st.write("Paper only. No broker connection. The same fixed regime rules are used to produce watch/entry/exit information.")
    if st.button("🔎 Run V8 paper scan", type="primary"):
        rows=[]
        for ticker in tickers:
            d=add_indicators(load_history(ticker,"2y"))
            if d.empty: continue
            r=d.iloc[-1]
            signal,strategy=regime_signal(r)
            regime=classify_regime(r)
            stop=float(r['Close']-stop_atr*r['ATR14'])
            target=float(r['Close']+target_r*(r['Close']-stop))
            rows.append({'Ticker':ticker,'Price':float(r['Close']),'Regime':regime,'Action':'PAPER BUY CANDIDATE' if signal else 'WAIT / CASH',
                         'Rule':strategy,'RSI':float(r['RSI']),'6M %':float(r['MOM126']),'Vol %':float(r['VOL20']),
                         'Stop':stop if signal else np.nan,'Target':target if signal else np.nan})
        if rows:
            st.dataframe(pd.DataFrame(rows).round(2),use_container_width=True)
            st.info("A paper-buy candidate is a rules-based signal, not a promise of profit. Review the research and risk settings before treating it as a learning exercise.")

# ---------------- RESEARCH ----------------
with tabs[3]:
    st.subheader("📊 Current research dashboard")
    for ticker in tickers:
        d=add_indicators(load_history(ticker,"2y"))
        if d.empty: continue
        r=d.iloc[-1]; regime=classify_regime(r); signal,strategy=regime_signal(r)
        with st.expander(f"{ticker} — ${float(r['Close']):.2f} — {regime}"):
            c=st.columns(6)
            c[0].metric("Regime",regime)
            c[1].metric("1M momentum",f"{r['MOM20']:+.1f}%")
            c[2].metric("3M momentum",f"{r['MOM63']:+.1f}%")
            c[3].metric("6M momentum",f"{r['MOM126']:+.1f}%")
            c[4].metric("RSI",f"{r['RSI']:.1f}")
            c[5].metric("Volatility",f"{r['VOL20']:.1f}%")
            st.write(f"SMA20 ${r['SMA20']:.2f} | SMA50 ${r['SMA50']:.2f} | SMA200 ${r['SMA200']:.2f} | ATR14 ${r['ATR14']:.2f}")
            st.write(f"Rule output: **{strategy}** — {'entry candidate' if signal else 'cash/defensive'}")

# ---------------- PORTFOLIO ----------------
with tabs[4]:
    st.subheader("💼 Paper portfolio")
    st.info("The V8 portfolio tab is a simulation dashboard. It does not persist trades between browser sessions and does not connect to a broker.")
    if st.session_state.v8_portfolio:
        r=st.session_state.v8_portfolio
        st.metric("Latest simulated portfolio value",f"${r['Final $']:,.2f}")
        if not r['trades_df'].empty:
            st.dataframe(r['trades_df'].round(2),use_container_width=True)
    else:
        st.write("Run the V8 portfolio backtest first.")

# ---------------- LEARN ----------------
with tabs[5]:
    st.subheader("📚 What V8 is teaching you")
    st.markdown("""
### 1. Regime first
V8 classifies each asset using fixed price, moving-average and momentum conditions:
- **Bull trend:** Close > SMA50 > SMA200 and positive 3M momentum.
- **Range / transition:** neither strong bull nor strong bear.
- **Bear trend:** Close < SMA50 < SMA200 and negative 3M momentum.

### 2. Strategy follows the regime
- Bull trend → Trend + Momentum.
- Range / transition → Mean Reversion only when the pullback is relatively controlled.
- Bear trend → Cash / defensive.

### 3. Risk-based position sizing
Instead of investing a fixed 100% of cash, V8 calculates share quantity from the distance between entry and the ATR-based stop. This makes the dollar risk more comparable between volatile and less-volatile assets.

### 4. Portfolio limits
V8 can cap the maximum position, total invested exposure and number of simultaneous positions. These are risk controls, not return guarantees.

### 5. Stop and target
The default stop is **2 ATR** below entry and the target is **3R** above entry. If both are touched in the same day, the simulator assumes the stop happened first. This is intentionally conservative.

### 6. Walk-forward / OOS
The latest historical slice is kept separate from earlier slices. It helps us see whether the fixed rules behave consistently across different historical periods.

### 7. What V8 does NOT do
It does not promise profit, automatically optimise parameters to the past, or place real-money trades. Historical backtests cannot establish what will happen in the future.
""")

st.divider()
st.caption("AI Investor V8 • Educational research and paper trading only • No broker connection • No guaranteed returns")
