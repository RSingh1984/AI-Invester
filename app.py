
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="AI Investor V9.4", page_icon="📈", layout="wide")

WATCHLIST_DEFAULT = "BHP.AX,CBA.AX,CSL.AX,VAS.AX"

# V9.3 is deliberately a fixed research system, not an optimizer.
# It targets the V8.1 diagnostic findings: too much turnover, weak entries,
# and frequent regime exits. Entries use current-bar information and execute
# at the next trading day's open. Exits are stop/target, or a fixed max hold.

@st.cache_data(ttl=900)
def load_history(ticker, period="7y"):
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    needed = ["Open", "High", "Low", "Close"]
    if not all(c in df.columns for c in needed):
        return pd.DataFrame()
    df = df.dropna(subset=needed).sort_index()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")]
    return df


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

    if "Volume" in x.columns:
        vol = pd.to_numeric(x["Volume"], errors="coerce")
        x["VOL_RATIO"] = vol / vol.rolling(20).mean()
    else:
        x["VOL_RATIO"] = np.nan

    prev_close = c.shift(1)
    tr = pd.concat(
        [(h - l), (h - prev_close).abs(), (l - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    x["ATR14"] = tr.rolling(14).mean()
    x["ATR_PCT"] = x["ATR14"] / c * 100

    return x.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["SMA20", "SMA50", "SMA200", "RSI", "MOM20", "MOM63", "MOM126", "VOL20", "ATR14", "ATR_PCT"]
    )


def classify_regime(row):
    close = float(row["Close"])
    s50 = float(row["SMA50"])
    s200 = float(row["SMA200"])
    mom63 = float(row["MOM63"])
    if close > s50 > s200 and mom63 > 0:
        return "Bull trend"
    if close < s50 < s200 and mom63 < 0:
        return "Bear trend"
    return "Range / transition"


def entry_score(row):
    """Fixed, transparent confirmation score. No historical optimization."""
    regime = classify_regime(row)
    if regime != "Bull trend":
        return 0, ["Not bull regime"]

    checks = []
    checks.append(("Price > SMA50", float(row["Close"]) > float(row["SMA50"])))
    checks.append(("SMA50 > SMA200", float(row["SMA50"]) > float(row["SMA200"])))
    checks.append(("3M momentum > 0", float(row["MOM63"]) > 0))
    checks.append(("6M momentum > 0", float(row["MOM126"]) > 0))
    checks.append(("RSI 50–70", 50 <= float(row["RSI"]) <= 70))
    checks.append(("1M momentum > 0", float(row["MOM20"]) > 0))
    checks.append(("Volatility <= 6%", float(row["ATR_PCT"]) <= 6))
    if pd.notna(row.get("VOL_RATIO", np.nan)):
        checks.append(("Volume >= 0.8x avg", float(row["VOL_RATIO"]) >= 0.8))

    score = sum(1 for _, ok in checks if ok)
    failed = [name for name, ok in checks if not ok]
    return score, failed


def entry_checks(row):
    """Return the fixed confirmation components for diagnostic attribution."""
    regime = classify_regime(row)
    checks = {
        "Bull regime": regime == "Bull trend",
        "Price > SMA50": float(row["Close"]) > float(row["SMA50"]),
        "SMA50 > SMA200": float(row["SMA50"]) > float(row["SMA200"]),
        "3M momentum > 0": float(row["MOM63"]) > 0,
        "6M momentum > 0": float(row["MOM126"]) > 0,
        "RSI 50–70": 50 <= float(row["RSI"]) <= 70,
        "1M momentum > 0": float(row["MOM20"]) > 0,
        "Volatility <= 6%": float(row["ATR_PCT"]) <= 6,
    }
    if pd.notna(row.get("VOL_RATIO", np.nan)):
        checks["Volume >= 0.8x avg"] = float(row["VOL_RATIO"]) >= 0.8
    return checks


def is_entry_candidate(row, min_score=6):
    score, _ = entry_score(row)
    # If volume is unavailable, the score is out of 7 instead of 8.
    available_checks = 8 if pd.notna(row.get("VOL_RATIO", np.nan)) else 7
    threshold = min_score if available_checks == 8 else min(5, available_checks)
    return classify_regime(row) == "Bull trend" and score >= threshold


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
        "Final $": final,
        "Return %": total,
        "CAGR %": cagr,
        "Trades": len(pnl),
        "Win rate %": 100 * len(wins) / len(pnl) if pnl else 0,
        "Max DD %": max_dd,
        "Profit factor": pf,
        "Avg win $": np.mean(wins) if wins else 0,
        "Avg loss $": np.mean(losses) if losses else 0,
    }


def safe_row(df, dt):
    if df.empty:
        return None
    try:
        pos = df.index.get_indexer([pd.Timestamp(dt)])[0]
    except Exception:
        return None
    return df.iloc[pos] if pos >= 0 else None


def safe_close(df, dt, fallback=np.nan):
    row = safe_row(df, dt)
    if row is None:
        return fallback
    try:
        return float(row["Close"])
    except Exception:
        return fallback


def prior_two_rows(df, dt):
    """Return the two most recent trading rows at or before dt, without get_loc failures."""
    if df.empty:
        return None, None
    try:
        ts = pd.Timestamp(dt)
        pos = int(df.index.searchsorted(ts, side="right")) - 1
        if pos < 0:
            return None, None
        current = df.iloc[pos]
        previous = df.iloc[pos - 1] if pos >= 1 else None
        return current, previous
    except Exception:
        return None, None


def run_v9(
    data_map,
    initial=10000,
    risk_pct=0.75,
    max_pos_pct=25,
    max_exposure_pct=60,
    stop_atr=2.0,
    target_r=3.0,
    brokerage=6.50,
    slippage_pct=0.10,
    max_positions=3,
    min_hold_days=5,
    cooldown_days=10,
    max_hold_days=90,
    eval_start=None,
    eval_end=None,
):
    prepared = {t: add_indicators(df) for t, df in data_map.items() if not df.empty}
    prepared = {t: d for t, d in prepared.items() if len(d) >= 260}
    if not prepared:
        return None

    dates = sorted(set().union(*[set(d.index) for d in prepared.values()]))
    dates = [pd.Timestamp(x) for x in dates]

    cash = float(initial)
    positions = {}
    last_exit_idx = {}
    trades = []
    curve_points = []
    exposure_points = []

    for i, dt in enumerate(dates[:-1]):
        next_dt = dates[i + 1]

        equity = cash
        invested = 0.0
        for t, p in positions.items():
            px = safe_close(prepared[t], dt, p["last_price"])
            equity += p["shares"] * px
            invested += p["shares"] * px
        if (eval_start is None or dt >= pd.Timestamp(eval_start)) and (eval_end is None or dt <= pd.Timestamp(eval_end)):
            curve_points.append((dt, equity))
            exposure_points.append((dt, 100 * invested / equity if equity else 0))

        # -------- exits --------
        if (eval_start is None or dt >= pd.Timestamp(eval_start)) and (eval_end is None or next_dt <= pd.Timestamp(eval_end)):
            for t in list(positions.keys()):
                d = prepared[t]
                row_next = safe_row(d, next_dt)
                if row_next is None:
                    continue

                p = positions[t]
                low = float(row_next["Low"])
                high = float(row_next["High"])
                held = i - p["entry_index"] + 1

                exit_px = None
                reason = None

                # Conservative same-day collision rule: stop first.
                if low <= p["stop"] and high >= p["target"]:
                    exit_px, reason = p["stop"], "Stop (both touched)"
                elif low <= p["stop"]:
                    exit_px, reason = p["stop"], "Stop"
                elif high >= p["target"]:
                    exit_px, reason = p["target"], "Target"
                elif held >= min_hold_days:
                    prev_row, prev_prev = prior_two_rows(d, dt)
                    confirmed_break = (
                        prev_row is not None
                        and prev_prev is not None
                        and pd.notna(prev_row.get("SMA50"))
                        and pd.notna(prev_prev.get("SMA50"))
                        and float(prev_row["Close"]) < float(prev_row["SMA50"])
                        and float(prev_prev["Close"]) < float(prev_prev["SMA50"])
                    )
                    if confirmed_break:
                        exit_px, reason = float(row_next["Open"]), "Confirmed trend break"
                    elif held >= max_hold_days:
                        exit_px, reason = float(row_next["Open"]), "Max hold"

                if exit_px is not None:
                    gross = p["shares"] * exit_px
                    sell_cost = brokerage + gross * slippage_pct / 100
                    buy_cost = brokerage + p["entry_value"] * slippage_pct / 100
                    cash += gross - sell_cost
                    pnl = (gross - sell_cost) - (p["entry_value"] + buy_cost)

                    trades.append({
                        "Ticker": t,
                        "Entry date": p["entry_date"],
                        "Exit date": next_dt,
                        "Entry": p["entry_price"],
                        "Exit": exit_px,
                        "Shares": p["shares"],
                        "P/L": pnl,
                        "Reason": reason,
                        "Entry score": p["score"],
                        "Entry regime": p["entry_regime"],
                        "Entry value": p["entry_value"],
                        "Brokerage total": 2 * brokerage,
                        "Slippage total": p["entry_value"] * slippage_pct / 100 + gross * slippage_pct / 100,
                        "Hold days": held,
                        "entry_checks": p.get("entry_checks", {}),
                    })
                    del positions[t]
                    last_exit_idx[t] = i + 1

        # -------- entries --------
        slots = max_positions - len(positions)
        entries_allowed = (eval_start is None or dt >= pd.Timestamp(eval_start)) and (eval_end is None or next_dt <= pd.Timestamp(eval_end))
        if slots > 0 and entries_allowed:
            candidates = []
            for t, d in prepared.items():
                if t in positions or dt not in d.index or next_dt not in d.index:
                    continue

                if t in last_exit_idx and (i - last_exit_idx[t]) < cooldown_days:
                    continue

                row = safe_row(d, dt)
                nxt = safe_row(d, next_dt)
                if row is None or nxt is None:
                    continue

                if not is_entry_candidate(row):
                    continue

                score, failed = entry_score(row)
                # Rank only by current signal strength, not by historical outcome.
                strength = (
                    score,
                    float(row["MOM63"]),
                    float(row["MOM126"]),
                    -float(row["ATR_PCT"]),
                )
                candidates.append((strength, t, row, nxt, score))

            candidates.sort(reverse=True, key=lambda x: x[0])

            for _, t, row, nxt, score in candidates[:slots]:
                px = float(nxt["Open"])
                atr = float(row["ATR14"])
                if not np.isfinite(px) or not np.isfinite(atr) or atr <= 0:
                    continue

                stop = px - stop_atr * atr
                risk_per_share = px - stop
                if risk_per_share <= 0:
                    continue

                current_equity = cash
                gross_existing = 0.0
                for t2, p2 in positions.items():
                    px2 = safe_close(prepared[t2], dt, p2["last_price"])
                    current_equity += p2["shares"] * px2
                    gross_existing += p2["shares"] * px2

                risk_budget = current_equity * risk_pct / 100
                qty_risk = int(risk_budget // risk_per_share)

                exposure_room = current_equity * max_exposure_pct / 100 - gross_existing
                max_value = min(current_equity * max_pos_pct / 100, max(exposure_room, 0))
                qty_cap = int(max_value // px)

                qty = min(qty_risk, qty_cap)
                if qty < 1:
                    continue

                value = qty * px
                buy_cost = brokerage + value * slippage_pct / 100
                if value + buy_cost > cash:
                    continue

                cash -= value + buy_cost
                positions[t] = {
                    "shares": qty,
                    "entry_price": px,
                    "entry_value": value,
                    "entry_date": next_dt,
                    "entry_index": i + 1,
                    "stop": stop,
                    "target": px + target_r * risk_per_share,
                    "score": score,
                    "entry_regime": classify_regime(row),
                    "entry_checks": entry_checks(row),
                    "last_price": px,
                }

        for t, p in positions.items():
            px = safe_close(prepared[t], next_dt, p["last_price"])
            if np.isfinite(px):
                p["last_price"] = px

    # End-of-evaluation close.
    for t, p in list(positions.items()):
        d = prepared[t]
        cutoff = pd.Timestamp(eval_end) if eval_end is not None else d.index[-1]
        eligible = d.index[d.index <= cutoff]
        if len(eligible) == 0:
            continue
        last_dt = eligible[-1]
        px = float(d.loc[last_dt, "Close"])
        gross = p["shares"] * px
        sell_cost = brokerage + gross * slippage_pct / 100
        buy_cost = brokerage + p["entry_value"] * slippage_pct / 100
        cash += gross - sell_cost
        pnl = (gross - sell_cost) - (p["entry_value"] + buy_cost)
        held = max(1, len(d.loc[p["entry_date"]:last_dt]) - 1)

        trades.append({
            "Ticker": t,
            "Entry date": p["entry_date"],
            "Exit date": last_dt,
            "Entry": p["entry_price"],
            "Exit": px,
            "Shares": p["shares"],
            "P/L": pnl,
            "Reason": "End of test",
            "Entry score": p["score"],
            "Entry regime": p["entry_regime"],
            "Entry value": p["entry_value"],
            "Brokerage total": 2 * brokerage,
            "Slippage total": p["entry_value"] * slippage_pct / 100 + gross * slippage_pct / 100,
            "Hold days": held,
            "entry_checks": p.get("entry_checks", {}),
        })

    if not curve_points:
        return None

    final_curve_dates = [d.index[d.index <= pd.Timestamp(eval_end)][-1] for d in prepared.values() if eval_end is not None and len(d.index[d.index <= pd.Timestamp(eval_end)]) > 0]
    if eval_end is None:
        final_dt = max(d.index[-1] for d in prepared.values())
    else:
        final_dt = max(final_curve_dates) if final_curve_dates else None
    if final_dt is not None:
        curve_points.append((final_dt, cash))
    curve = pd.Series({pd.Timestamp(k): v for k, v in curve_points}).sort_index()
    curve = curve[~curve.index.duplicated(keep="last")]
    curve = curve.reindex(pd.date_range(curve.index.min(), curve.index.max(), freq="B")).ffill()

    m = metrics_from_curve(curve, trades, initial)
    m["curve"] = curve
    m["trades_df"] = pd.DataFrame(trades)
    m["avg exposure %"] = float(np.mean([v for _, v in exposure_points])) if exposure_points else 0
    return m


def build_data(tickers, years):
    data_map = {}
    for ticker in tickers:
        raw = load_history(ticker, f"{years + 2}y")
        if not raw.empty:
            data_map[ticker] = raw.tail(int(years * 252 + 260))
    return data_map


def fmt_pf(x):
    return "∞" if np.isinf(x) else f"{x:.2f}"


# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ V9.4 Settings")
watch_text = st.sidebar.text_input("Watchlist", WATCHLIST_DEFAULT)
tickers = [x.strip().upper() for x in watch_text.split(",") if x.strip()]
years = st.sidebar.selectbox("Test period", [5, 7, 10], index=0)
brokerage = st.sidebar.number_input("Brokerage / transaction ($)", 0.0, 50.0, 6.50, 0.50)
slippage = st.sidebar.number_input("Slippage / transaction (%)", 0.0, 1.0, 0.10, 0.05)
risk_pct = st.sidebar.slider("Risk per trade (% equity)", 0.25, 2.0, 0.75, 0.25)
max_pos_pct = st.sidebar.slider("Max position (% equity)", 5, 50, 25, 5)
max_exposure_pct = st.sidebar.slider("Max invested exposure (%)", 25, 100, 60, 5)
max_positions = st.sidebar.slider("Max simultaneous positions", 1, 4, 3, 1)
stop_atr = st.sidebar.slider("Stop distance (ATR)", 1.0, 4.0, 2.0, 0.25)
target_r = st.sidebar.slider("Profit target (R)", 1.0, 5.0, 3.0, 0.5)
min_hold_days = st.sidebar.slider("Minimum hold (days)", 1, 20, 5, 1)
cooldown_days = st.sidebar.slider("Cooldown after exit (days)", 0, 30, 10, 1)
max_hold_days = st.sidebar.slider("Maximum hold (days)", 30, 180, 90, 10)

if "v93_result" not in st.session_state:
    st.session_state.v93_result = None
if "v93_data" not in st.session_state:
    st.session_state.v93_data = {}
if "v93_diag" not in st.session_state:
    st.session_state.v93_diag = None

st.title("📈 AI Investor V9.4")
st.caption("Confirmed-entry + lower-turnover + risk-controlled paper-trading research laboratory")
st.info(
    "V9 is an educational research and paper-trading system. It does not guarantee returns, "
    "does not predict the future, and has no broker connection. It is intentionally fixed rather "
    "than automatically tuned to historical results."
)

tabs = st.tabs([
    "🏦 Portfolio Lab", "🔬 Robustness", "🧪 Diagnostics",
    "🤖 Paper Trader", "📊 Research", "💼 Portfolio", "📚 Learn"
])

# ---------------- PORTFOLIO LAB ----------------
with tabs[0]:
    st.subheader("V9.3 entry-quality portfolio engine")
    st.write(
        "V9 addresses the V8.1 diagnostic findings by requiring several current-market confirmations, "
        "removing routine regime exits, adding a minimum hold, adding a post-exit cooldown, and limiting "
        "total exposure. Signals are calculated on one day's close and executed at the next day's open."
    )

    if st.button("🚀 Run V9.4 portfolio backtest", type="primary"):
        data_map = build_data(tickers, years)
        st.session_state.v93_data = data_map
        st.session_state.v93_result = run_v9(
            data_map,
            initial=10000,
            risk_pct=risk_pct,
            max_pos_pct=max_pos_pct,
            max_exposure_pct=max_exposure_pct,
            stop_atr=stop_atr,
            target_r=target_r,
            brokerage=brokerage,
            slippage_pct=slippage,
            max_positions=max_positions,
            min_hold_days=min_hold_days,
            cooldown_days=cooldown_days,
            max_hold_days=max_hold_days,
        )

    r = st.session_state.v93_result
    if r:
        c = st.columns(5)
        c[0].metric("Final portfolio", f"${r['Final $']:,.2f}")
        c[1].metric("Return", f"{r['Return %']:+.2f}%")
        c[2].metric("CAGR", f"{r['CAGR %']:+.2f}%")
        c[3].metric("Max drawdown", f"{r['Max DD %']:.2f}%")
        c[4].metric("Profit factor", fmt_pf(r["Profit factor"]))
        st.write(
            f"Trades: **{r['Trades']}** | Win rate: **{r['Win rate %']:.1f}%** | "
            f"Average exposure: **{r['avg exposure %']:.1f}%** | "
            f"Average win: **${r['Avg win $']:.2f}** | Average loss: **${r['Avg loss $']:.2f}**"
        )
        st.subheader("Portfolio equity curve")
        st.line_chart(r["curve"])
        st.subheader("Trade log")
        if not r["trades_df"].empty:
            st.dataframe(r["trades_df"].round(2), use_container_width=True)

        st.subheader("Exit breakdown")
        if not r["trades_df"].empty:
            ex = r["trades_df"].groupby("Reason").agg(
                Trades=("P/L", "count"),
                Gross_PnL=("P/L", "sum"),
                Avg_PnL=("P/L", "mean"),
                Win_rate=("P/L", lambda s: 100 * (s > 0).mean()),
            ).reset_index()
            st.dataframe(ex.round(2), use_container_width=True)

# ---------------- ROBUSTNESS ----------------
with tabs[1]:
    st.subheader("🔬 V9.3 walk-forward robustness")
    st.write(
        "The historical data is split chronologically into Training, Validation and Out-of-sample slices. "
        "The same fixed V9 rules are used in every slice; no slice is used to tune parameters."
    )
    if st.button("🔬 Run V9.3 robustness test", type="primary"):
        rows = []
        for ticker in tickers:
            raw = load_history(ticker, f"{years + 2}y")
            if raw.empty:
                continue
            raw = raw.tail(int(years * 252 + 260))
            n = len(raw)
            a = int(n * 0.55)
            b = int(n * 0.775)
            parts = [
                ("Training", raw.index[0], raw.index[a - 1]),
                ("Validation", raw.index[a], raw.index[b - 1]),
                ("Out-of-sample", raw.index[b], raw.index[-1]),
            ]
            for name, period_start, period_end in parts:
                # Use the full history for indicator warm-up, but evaluate only inside this period.
                res = run_v9(
                    {ticker: raw},
                    initial=10000,
                    risk_pct=risk_pct,
                    max_pos_pct=max_pos_pct,
                    max_exposure_pct=max_exposure_pct,
                    stop_atr=stop_atr,
                    target_r=target_r,
                    brokerage=brokerage,
                    slippage_pct=slippage,
                    max_positions=1,
                    min_hold_days=min_hold_days,
                    cooldown_days=cooldown_days,
                    max_hold_days=max_hold_days,
                    eval_start=period_start,
                    eval_end=period_end,
                )
                if res:
                    rows.append({
                        "Ticker": ticker,
                        "Period": name,
                        "CAGR %": res["CAGR %"],
                        "Max DD %": res["Max DD %"],
                        "Trades": res["Trades"],
                        "Win rate %": res["Win rate %"],
                        "Profit factor": res["Profit factor"],
                    })
        st.session_state.v93_robust = pd.DataFrame(rows)

    rr = st.session_state.get("v93_robust", pd.DataFrame())
    if not rr.empty:
        st.dataframe(rr.round(2), use_container_width=True)
        oos = rr[rr["Period"] == "Out-of-sample"]
        if not oos.empty:
            st.subheader("Out-of-sample summary")
            st.dataframe(oos.round(2), use_container_width=True)
            st.info("Out-of-sample is a historical held-back slice. It is not a forecast.")

# ---------------- DIAGNOSTICS ----------------
with tabs[2]:
    st.subheader("🧪 V9.4 diagnostic laboratory")
    st.write(
        "The diagnostics test whether V9 actually reduced turnover and whether its losses, if any, "
        "are concentrated in particular stocks, entry scores, or exit reasons."
    )

    if st.button("🧪 Run V9.4 diagnostics", type="primary"):
        data_map = st.session_state.v93_data or build_data(tickers, years)
        st.session_state.v93_data = data_map
        base = run_v9(
            data_map,
            initial=10000,
            risk_pct=risk_pct,
            max_pos_pct=max_pos_pct,
            max_exposure_pct=max_exposure_pct,
            stop_atr=stop_atr,
            target_r=target_r,
            brokerage=brokerage,
            slippage_pct=slippage,
            max_positions=max_positions,
            min_hold_days=min_hold_days,
            cooldown_days=cooldown_days,
            max_hold_days=max_hold_days,
        )
        zero_cost = run_v9(
            data_map,
            initial=10000,
            risk_pct=risk_pct,
            max_pos_pct=max_pos_pct,
            max_exposure_pct=max_exposure_pct,
            stop_atr=stop_atr,
            target_r=target_r,
            brokerage=0,
            slippage_pct=0,
            max_positions=max_positions,
            min_hold_days=min_hold_days,
            cooldown_days=cooldown_days,
            max_hold_days=max_hold_days,
        )
        st.session_state.v93_diag = (base, zero_cost)

    diag = st.session_state.v93_diag
    if diag:
        base, zero = diag
        st.markdown("### 1. V9 baseline")
        c = st.columns(5)
        c[0].metric("Final", f"${base['Final $']:,.2f}")
        c[1].metric("CAGR", f"{base['CAGR %']:+.2f}%")
        c[2].metric("Max DD", f"{base['Max DD %']:.2f}%")
        c[3].metric("Trades", f"{base['Trades']}")
        c[4].metric("Profit factor", fmt_pf(base["Profit factor"]))

        st.markdown("### 2. Cost sensitivity")
        cost_df = pd.DataFrame([
            {
                "Test": "Current costs",
                "Final $": base["Final $"],
                "Return %": base["Return %"],
                "CAGR %": base["CAGR %"],
                "Max DD %": base["Max DD %"],
                "Trades": base["Trades"],
                "Profit factor": base["Profit factor"],
            },
            {
                "Test": "Zero brokerage + zero slippage",
                "Final $": zero["Final $"],
                "Return %": zero["Return %"],
                "CAGR %": zero["CAGR %"],
                "Max DD %": zero["Max DD %"],
                "Trades": zero["Trades"],
                "Profit factor": zero["Profit factor"],
            },
        ])
        st.dataframe(cost_df.round(2), use_container_width=True)

        trades = base["trades_df"].copy()
        if not trades.empty:
            st.markdown("### 3. P/L by stock")
            by_stock = trades.groupby("Ticker").agg(
                Trades=("P/L", "count"),
                Gross_PnL=("P/L", "sum"),
                Avg_PnL=("P/L", "mean"),
                Win_rate=("P/L", lambda s: 100 * (s > 0).mean()),
            ).reset_index()
            st.dataframe(by_stock.round(2), use_container_width=True)

            st.markdown("### 4. P/L by entry score")
            by_score = trades.groupby("Entry score").agg(
                Trades=("P/L", "count"),
                Gross_PnL=("P/L", "sum"),
                Avg_PnL=("P/L", "mean"),
                Win_rate=("P/L", lambda s: 100 * (s > 0).mean()),
            ).reset_index()
            st.dataframe(by_score.round(2), use_container_width=True)

            st.markdown("### 5. Exit reasons")
            by_reason = trades.groupby("Reason").agg(
                Trades=("P/L", "count"),
                Gross_PnL=("P/L", "sum"),
                Avg_PnL=("P/L", "mean"),
                Win_rate=("P/L", lambda s: 100 * (s > 0).mean()),
            ).reset_index()
            st.dataframe(by_reason.round(2), use_container_width=True)

            st.markdown("### 6. Largest losses")
            cols = [
                "Ticker", "Entry date", "Exit date", "Entry", "Exit",
                "Shares", "P/L", "Reason", "Entry score", "Hold days"
            ]
            st.dataframe(
                trades.sort_values("P/L").head(15)[cols].round(2),
                use_container_width=True,
            )

            st.markdown("### 7. Largest winners")
            st.dataframe(
                trades.sort_values("P/L", ascending=False).head(15)[cols].round(2),
                use_container_width=True,
            )


        st.markdown("### 8. Entry-quality component diagnostics")
        st.caption("This section describes the fixed entry confirmations actually present on each historical trade. It does not optimize or choose a winner.")
        component_rows = []
        for _, tr in trades.iterrows():
            checks = tr.get("entry_checks", {})
            if isinstance(checks, dict):
                for name, passed in checks.items():
                    component_rows.append({"Component": name, "Passed": bool(passed), "P/L": float(tr["P/L"])})
        if component_rows:
            cf = pd.DataFrame(component_rows)
            summary = cf.groupby(["Component", "Passed"]).agg(
                Trades=("P/L", "count"),
                Gross_PnL=("P/L", "sum"),
                Avg_PnL=("P/L", "mean"),
                Win_rate=("P/L", lambda s: 100 * (s > 0).mean()),
            ).reset_index()
            st.dataframe(summary.round(2), use_container_width=True)
            st.info("Component results are descriptive only. They are not used to automatically tune the strategy.")

# ---------------- PAPER TRADER ----------------
with tabs[3]:
    st.subheader("🤖 V9.4 paper trader")
    st.write(
        "Paper-only scanner. It uses the same fixed entry confirmation rules as the backtest. "
        "It does not place real trades."
    )
    if st.button("🔎 Run V9 paper scan", type="primary"):
        rows = []
        for ticker in tickers:
            d = add_indicators(load_history(ticker, "2y"))
            if d.empty:
                continue
            r = d.iloc[-1]
            score, failed = entry_score(r)
            regime = classify_regime(r)
            candidate = is_entry_candidate(r)
            stop = float(r["Close"] - stop_atr * r["ATR14"])
            target = float(r["Close"] + target_r * (r["Close"] - stop))
            rows.append({
                "Ticker": ticker,
                "Price": float(r["Close"]),
                "Regime": regime,
                "Score": score,
                "Action": "PAPER BUY CANDIDATE" if candidate else "WAIT / CASH",
                "RSI": float(r["RSI"]),
                "3M %": float(r["MOM63"]),
                "6M %": float(r["MOM126"]),
                "ATR %": float(r["ATR_PCT"]),
                "Stop": stop if candidate else np.nan,
                "Target": target if candidate else np.nan,
                "Why not": ", ".join(failed[:4]) if not candidate else "",
            })
        st.session_state.v93_scan = pd.DataFrame(rows)

    scan = st.session_state.get("v93_scan", pd.DataFrame())
    if not scan.empty:
        st.dataframe(scan.round(2), use_container_width=True)
        st.info("A candidate is only a rules-based paper signal. It is not a guarantee or a recommendation.")

# ---------------- RESEARCH ----------------
with tabs[4]:
    st.subheader("📊 Current research dashboard")
    for ticker in tickers:
        d = add_indicators(load_history(ticker, "2y"))
        if d.empty:
            continue
        r = d.iloc[-1]
        regime = classify_regime(r)
        score, failed = entry_score(r)
        candidate = is_entry_candidate(r)
        with st.expander(f"{ticker} — ${float(r['Close']):.2f} — {regime}"):
            c = st.columns(6)
            c[0].metric("Regime", regime)
            c[1].metric("Entry score", f"{score}/8")
            c[2].metric("1M momentum", f"{r['MOM20']:+.1f}%")
            c[3].metric("3M momentum", f"{r['MOM63']:+.1f}%")
            c[4].metric("6M momentum", f"{r['MOM126']:+.1f}%")
            c[5].metric("RSI", f"{r['RSI']:.1f}")
            st.write(
                f"SMA20 ${r['SMA20']:.2f} | SMA50 ${r['SMA50']:.2f} | "
                f"SMA200 ${r['SMA200']:.2f} | ATR14 ${r['ATR14']:.2f} | "
                f"ATR% {r['ATR_PCT']:.2f}%"
            )
            st.write(
                f"Rule output: **{'PAPER BUY CANDIDATE' if candidate else 'WAIT / CASH'}**"
            )
            if not candidate:
                st.caption("Failed confirmations: " + ", ".join(failed))

# ---------------- PORTFOLIO ----------------
with tabs[5]:
    st.subheader("💼 Paper portfolio")
    st.info(
        "V9 is paper-only and does not persist trades as a broker account. "
        "Use the Trade log to study entries, exits, costs and risk."
    )
    r = st.session_state.v93_result
    if r:
        st.metric("Latest simulated portfolio value", f"${r['Final $']:,.2f}")
        if not r["trades_df"].empty:
            st.dataframe(r["trades_df"].round(2), use_container_width=True)
    else:
        st.write("Run the V9 portfolio backtest first.")

# ---------------- LEARN ----------------
with tabs[6]:
    st.subheader("📚 What V9 is teaching you")
    st.markdown("""
### 1. V9 does not have to trade
The system can remain in cash when the evidence is weak. Fewer trades can be useful when transaction costs are meaningful.

### 2. Entry confirmation
A candidate must be in a Bull trend and satisfy several fixed confirmations:
- price above SMA50
- SMA50 above SMA200
- positive 3-month momentum
- positive 6-month momentum
- RSI between 50 and 70
- positive 1-month momentum
- ATR volatility no higher than 6%
- volume at least 0.8× its 20-day average when volume is available

The score is a transparent current-market rule. It is not trained to pick a historical winner.

### 3. No routine regime exit
V8.1 showed that frequent regime exits added substantial turnover. V9 therefore does not automatically sell merely because the regime label changes.

### 4. Stops and targets
The default stop is 2 ATR below entry and the target is 3R above entry. If both are touched in one day, the simulator assumes the stop happened first.

### 5. Holding discipline
V9 adds a minimum holding period, a cooldown after an exit, and a maximum holding period. These are designed to reduce repeated in-and-out trading.

### 6. Position sizing
Position size is determined by the amount of account equity at risk and the distance to the stop. A maximum position and maximum total exposure also apply.

### 7. Walk-forward testing
Training, Validation and Out-of-sample slices are tested chronologically with the same fixed rules. No historical slice automatically changes the rules.

### 8. Entry-quality diagnostics
V9.4 records which fixed confirmations were present on each trade so we can study whether the current entry logic is behaving consistently. The diagnostics do not automatically select a historical winner.

### 9. Important limitation
Backtests are historical simulations. They cannot establish future returns, and real execution can differ because of spreads, liquidity, taxes, corporate actions, gaps and other market effects.
""")

st.divider()
st.caption("AI Investor V9.4 • Educational research and paper trading only • No broker connection • No guaranteed returns")
