
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="AI Investor V11", page_icon="📈", layout="wide")

WATCHLIST_DEFAULT = "BHP.AX,CBA.AX,CSL.AX,VAS.AX"

# V9.7 is deliberately a fixed research system, not an optimizer.
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

    # V9.7 price-action confirmation: a fresh breakout or a pullback recovery.
    # These are calculated from information available at the signal close only.
    x["PREV_CLOSE"] = c.shift(1)
    x["PREV_SMA20"] = x["SMA20"].shift(1)
    x["PRIOR_20D_HIGH"] = h.rolling(20).max().shift(1)
    x["BREAKOUT20"] = c > x["PRIOR_20D_HIGH"]
    x["PULLBACK_RECOVERY"] = (
        (x["PREV_CLOSE"] <= x["PREV_SMA20"])
        & (c > x["SMA20"])
        & (c > x["PREV_CLOSE"])
    )
    x["PRICE_CONFIRMATION"] = x["BREAKOUT20"] | x["PULLBACK_RECOVERY"]

    # V9.7 trigger-strength features. These are descriptive measurements,
    # not historical-performance optimizers.
    x["SMA50_SLOPE20"] = x["SMA50"].pct_change(20) * 100
    x["CLOSE_LOCATION"] = (c - l) / (h - l).replace(0, np.nan)
    x["BREAKOUT_EXTENSION"] = (c / x["PRIOR_20D_HIGH"] - 1) * 100

    return x.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["SMA20", "SMA50", "SMA200", "RSI", "MOM20", "MOM63", "MOM126", "VOL20", "ATR14", "ATR_PCT", "PREV_CLOSE", "PREV_SMA20", "PRIOR_20D_HIGH", "SMA50_SLOPE20", "CLOSE_LOCATION", "BREAKOUT_EXTENSION"]
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


def price_confirmation(row):
    """Return the fixed V9.7 price-action trigger and its setup type."""
    breakout = bool(row.get("BREAKOUT20", False))
    recovery = bool(row.get("PULLBACK_RECOVERY", False))
    if breakout and recovery:
        return True, "Breakout + pullback recovery"
    if breakout:
        return True, "20-day breakout"
    if recovery:
        return True, "Pullback recovery"
    return False, "No fresh price confirmation"


def entry_score(row):
    """Fixed V9.7 entry-quality score; it describes current setup strength only."""
    regime = classify_regime(row)
    if regime != "Bull trend":
        return 0, ["Not bull regime"]

    checks = [
        ("RSI 50–70", 50 <= float(row["RSI"]) <= 70),
        ("1M momentum > 0", float(row["MOM20"]) > 0),
        ("Volatility <= 6%", float(row["ATR_PCT"]) <= 6),
    ]
    confirmed, setup = price_confirmation(row)
    checks.append(("Fresh price confirmation", confirmed))
    score = sum(1 for _, ok in checks if ok)
    failed = [name for name, ok in checks if not ok]
    if confirmed:
        failed = [x for x in failed if x != "Fresh price confirmation"]
    return score, failed


def entry_checks(row):
    """Return fixed V9.7 entry components for descriptive diagnostics."""
    regime = classify_regime(row)
    confirmed, setup = price_confirmation(row)
    return {
        "Bull regime": regime == "Bull trend",
        "Price > SMA50": float(row["Close"]) > float(row["SMA50"]),
        "SMA50 > SMA200": float(row["SMA50"]) > float(row["SMA200"]),
        "3M momentum > 0": float(row["MOM63"]) > 0,
        "RSI 50–70": 50 <= float(row["RSI"]) <= 70,
        "1M momentum > 0": float(row["MOM20"]) > 0,
        "Volatility <= 6%": float(row["ATR_PCT"]) <= 6,
        "Fresh price confirmation": confirmed,
        "20-day breakout": bool(row.get("BREAKOUT20", False)),
        "Pullback recovery": bool(row.get("PULLBACK_RECOVERY", False)),
    }


def is_entry_candidate(row, min_score=4):
    score, _ = entry_score(row)
    return (
        classify_regime(row) == "Bull trend"
        and score >= min_score
        and price_confirmation(row)[0]
        and 50 <= float(row["RSI"]) <= 70
        and float(row["MOM20"]) > 0
        and float(row["ATR_PCT"]) <= 6
    )


def v97_setup(row):
    breakout = bool(row.get("BREAKOUT20", False))
    recovery = bool(row.get("PULLBACK_RECOVERY", False))
    if breakout and recovery:
        return "Breakout + pullback recovery"
    if breakout:
        return "20-day breakout"
    if recovery:
        return "Pullback recovery"
    return "No setup"


def v97_entry_components(row):
    """V9.7 genuinely variable 0-10 trigger score.

    Regime is a gate, not a score. Setup is a gate, while trigger quality is
    measured independently across trend, momentum, volatility and price action.
    """
    close = float(row["Close"])
    sma50 = float(row["SMA50"])
    sma200 = float(row["SMA200"])
    slope50 = float(row["SMA50_SLOPE20"])
    mom20 = float(row["MOM20"])
    mom63 = float(row["MOM63"])
    atr_pct = float(row["ATR_PCT"])

    trend = int(sma50 > sma200) + int(slope50 > 0)
    momentum = int(mom20 > 1.0) + int(mom63 > 5.0)
    volatility = 2 if atr_pct <= 5.0 else (1 if atr_pct <= 7.0 else 0)

    setup = v97_setup(row)
    loc = float(row["CLOSE_LOCATION"])
    volume_ratio = float(row["VOL_RATIO"]) if pd.notna(row.get("VOL_RATIO", np.nan)) else np.nan

    if setup in ("20-day breakout", "Breakout + pullback recovery"):
        extension = float(row["BREAKOUT_EXTENSION"])
        volume_ok = pd.notna(volume_ratio) and volume_ratio >= 1.10
        price_action = (
            int(extension >= 0.5) +
            int(volume_ok) +
            int(loc >= 0.65) +
            int(0.0 <= extension <= 3.0)
        )
    elif setup == "Pullback recovery":
        prev_close = float(row["PREV_CLOSE"])
        reclaim = (close / float(row["SMA20"]) - 1) * 100
        day_gain = (close / prev_close - 1) * 100
        price_action = (
            int(reclaim >= 0.25) +
            int(day_gain >= 0.5) +
            int(loc >= 0.60) +
            int(mom20 > 0)
        )
    else:
        price_action = 0

    total = trend + momentum + volatility + price_action
    return {
        "Trend quality (0-2)": trend,
        "Momentum quality (0-2)": momentum,
        "Volatility score (0-2)": volatility,
        "Trigger quality (0-3)": price_action,
        "Entry quality score (0-9)": total,
    }


def v97_entry_score(row):
    parts = v97_entry_components(row)
    return int(parts["Entry quality score (0-9)"]), parts


def v97_is_entry_candidate(row, min_score=6):
    score, _ = v97_entry_score(row)
    return (
        classify_regime(row) == "Bull trend"
        and v97_setup(row) != "No setup"
        and score >= min_score
    )


def v97_entry_checks(row):
    score, parts = v97_entry_score(row)
    setup = v97_setup(row)
    return {
        **parts,
        "Bull regime": classify_regime(row) == "Bull trend",
        "Setup present": setup != "No setup",
        "Setup": setup,
        "SMA50 slope > 0": float(row["SMA50_SLOPE20"]) > 0,
        "MOM20 > 1%": float(row["MOM20"]) > 1,
        "MOM63 > 5%": float(row["MOM63"]) > 5,
        "ATR <= 7%": float(row["ATR_PCT"]) <= 7,
        "Close location": float(row["CLOSE_LOCATION"]),
        "Breakout extension %": float(row["BREAKOUT_EXTENSION"]),
        "Volume ratio": float(row["VOL_RATIO"]) if pd.notna(row.get("VOL_RATIO", np.nan)) else np.nan,
    }

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


def run_v95_control(
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
                        "setup": p.get("setup", ""),
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
                # Rank only by current setup quality, not by historical outcome.
                confirmed, setup = price_confirmation(row)
                setup_priority = 1 if setup in ("20-day breakout", "Breakout + pullback recovery") else 0
                strength = (
                    score,
                    setup_priority,
                    float(row["MOM20"]),
                    float(row["MOM63"]),
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
                    "setup": price_confirmation(row)[1],
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
            "setup": p.get("setup", ""),
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



def run_v97(
    data_map,
    initial=10000,
    risk_pct=0.75,
    max_pos_pct=25,
    max_exposure_pct=60,
    stop_atr=2.5,
    target_r=3.0,
    brokerage=6.50,
    slippage_pct=0.10,
    max_positions=3,
    cooldown_days=10,
    max_hold_days=90,
    drawdown_brake_pct=10.0,
    brake_days=20,
    eval_start=None,
    eval_end=None,
):
    """V9.7 setup/trigger architecture with V9.5 control and V9.7 risk-brake structure."""
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
    peak_equity = float(initial)
    prior_drawdown = 0.0
    brake_until_idx = -1
    brake_events = 0

    def in_eval(dt, next_dt=None):
        return (eval_start is None or dt >= pd.Timestamp(eval_start)) and (eval_end is None or (next_dt is not None and next_dt <= pd.Timestamp(eval_end)))

    for i, dt in enumerate(dates[:-1]):
        next_dt = dates[i + 1]
        equity = cash
        invested = 0.0
        for t, p in positions.items():
            px = safe_close(prepared[t], dt, p["last_price"])
            equity += p["shares"] * px
            invested += p["shares"] * px

        if in_eval(dt, next_dt):
            curve_points.append((dt, equity))
            exposure_points.append((dt, 100 * invested / equity if equity else 0))

        # Portfolio-level drawdown brake: trigger only when crossing the threshold.
        if equity > peak_equity:
            peak_equity = equity
        drawdown = (equity / peak_equity - 1) * 100 if peak_equity else 0
        if drawdown <= -drawdown_brake_pct and prior_drawdown > -drawdown_brake_pct:
            brake_until_idx = max(brake_until_idx, i + brake_days)
            brake_events += 1
        prior_drawdown = drawdown

        # -------- exits --------
        if in_eval(dt, next_dt):
            for t in list(positions.keys()):
                d = prepared[t]
                row_next = safe_row(d, next_dt)
                if row_next is None:
                    continue
                p = positions[t]
                low, high = float(row_next["Low"]), float(row_next["High"])
                held = i - p["entry_index"] + 1
                exit_px = None
                reason = None
                if low <= p["stop"] and high >= p["target"]:
                    exit_px, reason = p["stop"], "Stop (both touched)"
                elif low <= p["stop"]:
                    exit_px, reason = p["stop"], "Stop"
                elif high >= p["target"]:
                    exit_px, reason = p["target"], "Target"
                elif held >= 5:
                    prev_row, prev_prev = prior_two_rows(d, dt)
                    confirmed_break = (
                        prev_row is not None and prev_prev is not None
                        and pd.notna(prev_row.get("SMA50")) and pd.notna(prev_prev.get("SMA50"))
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
                        "Ticker": t, "Entry date": p["entry_date"], "Exit date": next_dt,
                        "Entry": p["entry_price"], "Exit": exit_px, "Shares": p["shares"], "P/L": pnl,
                        "Reason": reason, "Entry score": p["score"], "Entry regime": p["entry_regime"],
                        "Entry value": p["entry_value"], "Brokerage total": 2 * brokerage,
                        "Slippage total": p["entry_value"] * slippage_pct / 100 + gross * slippage_pct / 100,
                        "Hold days": held, "entry_checks": p.get("entry_checks", {}), "setup": p.get("setup", ""),
                    })
                    del positions[t]
                    last_exit_idx[t] = i + 1

        # -------- entries --------
        slots = max_positions - len(positions)
        entries_allowed = in_eval(dt, next_dt) and i >= brake_until_idx
        if slots > 0 and entries_allowed:
            candidates = []
            for t, d in prepared.items():
                if t in positions or dt not in d.index or next_dt not in d.index:
                    continue
                if t in last_exit_idx and (i - last_exit_idx[t]) < cooldown_days:
                    continue
                row = safe_row(d, dt)
                nxt = safe_row(d, next_dt)
                if row is None or nxt is None or not v97_is_entry_candidate(row):
                    continue
                score, parts = v97_entry_score(row)
                confirmed, setup = v10_price_confirmation(row)
                strength = (score, parts["Trigger quality (0-3)"], parts["Momentum quality (0-2)"], parts["Trend quality (0-2)"], -float(row["ATR_PCT"]), float(row["MOM20"]))
                candidates.append((strength, t, row, nxt, score, parts))
            candidates.sort(reverse=True, key=lambda x: x[0])
            for _, t, row, nxt, score, parts in candidates[:slots]:
                px = float(nxt["Open"])
                atr = float(row["ATR14"])
                if not np.isfinite(px) or not np.isfinite(atr) or atr <= 0:
                    continue
                stop = px - stop_atr * atr
                risk_per_share = px - stop
                risk_budget = equity * risk_pct / 100
                qty_risk = int(risk_budget // risk_per_share)
                gross_existing = sum(
                    p2["shares"] * safe_close(prepared[t2], dt, p2["last_price"])
                    for t2, p2 in positions.items()
                )
                current_equity = equity
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
                    "shares": qty, "entry_price": px, "entry_value": value, "entry_date": next_dt,
                    "entry_index": i + 1, "stop": stop, "target": px + target_r * risk_per_share,
                    "score": score, "entry_regime": classify_regime(row),
                    "entry_checks": v97_entry_checks(row), "setup": v97_setup(row), "last_price": px,
                }

        for t, p in positions.items():
            px = safe_close(prepared[t], next_dt, p["last_price"])
            if np.isfinite(px):
                p["last_price"] = px

    # End-of-test close
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
            "Ticker": t, "Entry date": p["entry_date"], "Exit date": last_dt,
            "Entry": p["entry_price"], "Exit": px, "Shares": p["shares"], "P/L": pnl,
            "Reason": "End of test", "Entry score": p["score"], "Entry regime": p["entry_regime"],
            "Entry value": p["entry_value"], "Brokerage total": 2 * brokerage,
            "Slippage total": p["entry_value"] * slippage_pct / 100 + gross * slippage_pct / 100,
            "Hold days": held, "entry_checks": p.get("entry_checks", {}), "setup": p.get("setup", ""),
        })

    if not curve_points:
        return None
    final_curve_dates = [d.index[d.index <= pd.Timestamp(eval_end)][-1] for d in prepared.values() if eval_end is not None and len(d.index[d.index <= pd.Timestamp(eval_end)]) > 0]
    final_dt = max(d.index[-1] for d in prepared.values()) if eval_end is None else (max(final_curve_dates) if final_curve_dates else None)
    if final_dt is not None:
        curve_points.append((final_dt, cash))
    curve = pd.Series({pd.Timestamp(k): v for k, v in curve_points}).sort_index()
    curve = curve[~curve.index.duplicated(keep="last")]
    curve = curve.reindex(pd.date_range(curve.index.min(), curve.index.max(), freq="B")).ffill()
    m = metrics_from_curve(curve, trades, initial)
    m["curve"] = curve
    m["trades_df"] = pd.DataFrame(trades)
    m["avg exposure %"] = float(np.mean([v for _, v in exposure_points])) if exposure_points else 0
    m["brake events"] = brake_events
    m["brake days"] = brake_days
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




def add_v10_indicators(df):
    """V10 entry features: clean breakout continuation or pullback recovery.

    V10 keeps the fast path. It does not require a two-day wait. Instead, a
    breakout must prove basic quality at the signal close; otherwise the
    pullback path can enter after a recent breakout resets and recovers.
    """
    x = add_indicators(df).copy()
    x["BREAKOUT20"] = x["Close"] > x["PRIOR_20D_HIGH"]
    x["PREV_BREAKOUT20"] = x["BREAKOUT20"].shift(1).fillna(False).astype(bool)
    x["RECENT_BREAKOUT_10"] = x["BREAKOUT20"].shift(1).rolling(10, min_periods=1).max().fillna(0).astype(bool)
    x["BREAKOUT_EXTENSION"] = (x["Close"] / x["PRIOR_20D_HIGH"] - 1) * 100
    x["CLOSE_LOCATION"] = (x["Close"] - x["Low"]) / (x["High"] - x["Low"]).replace(0, np.nan)

    vol = pd.to_numeric(x.get("Volume", pd.Series(index=x.index, dtype=float)), errors="coerce")
    x["VOL_RATIO"] = vol / vol.rolling(20).mean() if "Volume" in x.columns else np.nan

    # A recent breakout followed by a short reset and recovery is the second,
    # faster alternative to chasing an extended breakout.
    x["PULLBACK_RECOVERY"] = (
        x["RECENT_BREAKOUT_10"]
        & (x["PREV_CLOSE"] <= x["PREV_SMA20"])
        & (x["Close"] > x["SMA20"])
        & (x["Close"] > x["PREV_CLOSE"])
    )
    return x.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["SMA20","SMA50","SMA200","RSI","MOM20","MOM63","MOM126","VOL20","ATR14","ATR_PCT","PREV_CLOSE","PREV_SMA20","PRIOR_20D_HIGH","SMA50_SLOPE20","CLOSE_LOCATION","BREAKOUT_EXTENSION"]
    )


def v10_setup(row):
    breakout = bool(row.get("BREAKOUT20", False))
    pullback = bool(row.get("PULLBACK_RECOVERY", False))
    ext = float(row["BREAKOUT_EXTENSION"])
    loc = float(row["CLOSE_LOCATION"])
    vol = float(row["VOL_RATIO"]) if pd.notna(row.get("VOL_RATIO", np.nan)) else np.nan

    # Fast path: one clean breakout is enough. We do not wait for a second day.
    clean_breakout = breakout and 0.0 <= ext <= 3.0 and loc >= 0.65 and (pd.isna(vol) or vol >= 1.0)
    if clean_breakout:
        return "Clean breakout"
    if pullback:
        return "Breakout + pullback recovery"
    return "No setup"


def v10_price_confirmation(row):
    setup = v10_setup(row)
    return setup != "No setup", setup


def v10_entry_components(row):
    """V10 quality score: separates setup quality from gates that were always true in V9.8."""
    sma50 = float(row["SMA50"]); sma200 = float(row["SMA200"])
    slope50 = float(row["SMA50_SLOPE20"])
    mom20 = float(row["MOM20"]); mom63 = float(row["MOM63"])
    atr_pct = float(row["ATR_PCT"]); rsi = float(row["RSI"])
    loc = float(row["CLOSE_LOCATION"]); ext = float(row["BREAKOUT_EXTENSION"])
    vol = float(row["VOL_RATIO"]) if pd.notna(row.get("VOL_RATIO", np.nan)) else np.nan
    setup = v10_setup(row)

    # These are strength measures, not simple Bull-regime gates.
    trend = int(sma50 > sma200) + int(slope50 > 0)
    momentum = int(mom20 > 1.0) + int(mom63 > 5.0)

    if setup == "Clean breakout":
        trigger = (
            int(loc >= 0.65) +
            int(0.0 <= ext <= 3.0) +
            int(pd.isna(vol) or vol >= 1.0)
        )
    elif setup == "Breakout + pullback recovery":
        day_gain = (float(row["Close"]) / float(row["PREV_CLOSE"]) - 1) * 100
        reclaim = (float(row["Close"]) / float(row["SMA20"]) - 1) * 100
        trigger = (
            int(reclaim >= 0.15) +
            int(day_gain >= 0.25) +
            int(loc >= 0.55)
        )
    else:
        trigger = 0

    rsi_quality = int(52 <= rsi <= 68)
    volatility_quality = int(atr_pct <= 7.0)
    total = trend + momentum + trigger + rsi_quality + volatility_quality
    return {
        "Trend quality (0-2)": trend,
        "Momentum quality (0-2)": momentum,
        "Trigger quality (0-3)": trigger,
        "RSI quality (0-1)": rsi_quality,
        "Volatility quality (0-1)": volatility_quality,
        "Entry quality score (0-9)": total,
    }


def v10_entry_score(row):
    parts = v10_entry_components(row)
    return int(parts["Entry quality score (0-9)"]), parts


def v10_is_entry_candidate(row, min_score=6):
    score, _ = v10_entry_score(row)
    setup = v10_setup(row)
    return (
        classify_regime(row) == "Bull trend"
        and setup != "No setup"
        and score >= min_score
        and 48 <= float(row["RSI"]) <= 72
        and float(row["MOM20"]) > 0
        and float(row["MOM63"]) > 0
        and float(row["ATR_PCT"]) <= 7.0
    )


def v10_entry_checks(row):
    score, parts = v10_entry_score(row)
    return {
        **parts,
        "Bull regime": classify_regime(row) == "Bull trend",
        "Setup present": v10_setup(row) != "No setup",
        "Setup": v10_setup(row),
        "RSI 48-72": 48 <= float(row["RSI"]) <= 72,
        "MOM20 > 0%": float(row["MOM20"]) > 0,
        "MOM63 > 0%": float(row["MOM63"]) > 0,
        "ATR <= 7%": float(row["ATR_PCT"]) <= 7,
        "Close location": float(row["CLOSE_LOCATION"]),
        "Breakout extension %": float(row["BREAKOUT_EXTENSION"]),
        "Volume ratio": float(row["VOL_RATIO"]) if pd.notna(row.get("VOL_RATIO", np.nan)) else np.nan,
        "Recent breakout": bool(row.get("RECENT_BREAKOUT_10", False)),
    }


# ---------------- V11 entry-quality model ----------------
def add_v11_indicators(df):
    """V11 features focused on entry quality rather than repeated regime gates.

    The model keeps V10's two setup paths but measures independent qualities:
    trend slope, momentum strength/acceleration, breakout/pullback trigger quality,
    volume confirmation and distance from SMA20. All values use signal-day data only.
    """
    x = add_v10_indicators(df).copy()
    x["PREV_MOM20"] = x["MOM20"].shift(1)
    x["MOM20_ACCEL"] = x["MOM20"] - x["PREV_MOM20"]
    x["DIST_SMA20"] = (x["Close"] / x["SMA20"] - 1) * 100
    x["DAY_GAIN"] = (x["Close"] / x["PREV_CLOSE"] - 1) * 100
    x["SMA50_SLOPE20"] = x["SMA50"].pct_change(20) * 100
    return x.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["PREV_MOM20", "MOM20_ACCEL", "DIST_SMA20", "DAY_GAIN"]
    )


def v11_setup(row):
    """Two fixed setup paths, with an anti-chase distance check."""
    breakout = bool(row.get("BREAKOUT20", False))
    pullback = bool(row.get("PULLBACK_RECOVERY", False))
    ext = float(row["BREAKOUT_EXTENSION"])
    loc = float(row["CLOSE_LOCATION"])
    vol = float(row["VOL_RATIO"]) if pd.notna(row.get("VOL_RATIO", np.nan)) else np.nan
    dist = float(row["DIST_SMA20"])

    clean_breakout = (
        breakout and 0.0 <= ext <= 3.0 and loc >= 0.65
        and (pd.isna(vol) or vol >= 1.0) and dist <= 8.0
    )
    if clean_breakout:
        return "Clean breakout"
    if pullback and dist <= 8.0:
        return "Breakout + pullback recovery"
    return "No setup"


def v11_entry_components(row):
    """Fixed 0-10 score made from setup-quality measures.

    Unlike V10, no points are awarded merely because the market is in a Bull
    regime. Regime remains a gate; the score is reserved for qualities that can
    vary meaningfully between otherwise eligible entries.
    """
    sma50 = float(row["SMA50"]); sma200 = float(row["SMA200"])
    slope50 = float(row["SMA50_SLOPE20"])
    mom20 = float(row["MOM20"]); mom63 = float(row["MOM63"])
    prev_mom20 = float(row["PREV_MOM20"])
    atr_pct = float(row["ATR_PCT"])
    loc = float(row["CLOSE_LOCATION"]); ext = float(row["BREAKOUT_EXTENSION"])
    vol = float(row["VOL_RATIO"]) if pd.notna(row.get("VOL_RATIO", np.nan)) else np.nan
    setup = v11_setup(row)

    # Trend quality: direction + meaningful slope, rather than just Bull regime.
    trend = int(sma50 > sma200) + int(slope50 > 0.5)

    # Momentum quality: strength + acceleration.
    momentum = int(mom20 > 1.5) + int((mom20 > 0) and (mom20 > prev_mom20) and (mom63 > 0))

    if setup == "Clean breakout":
        trigger = (
            int(0.0 <= ext <= 1.5) +
            int(loc >= 0.75) +
            int(pd.isna(vol) or vol >= 1.20)
        )
    elif setup == "Breakout + pullback recovery":
        reclaim = float(row["Close"] / row["SMA20"] - 1) * 100
        trigger = (
            int(reclaim >= 0.50) +
            int(float(row["DAY_GAIN"]) >= 0.75) +
            int(loc >= 0.70)
        )
    else:
        trigger = 0

    # One point for sensible volatility; avoid rewarding extremely quiet or
    # extremely volatile entries through the score itself.
    volatility = int(3.0 <= atr_pct <= 7.0)
    # One point for not chasing too far above the 20-day average.
    location = int(0.0 <= float(row["DIST_SMA20"]) <= 6.0)

    total = trend + momentum + trigger + volatility + location
    return {
        "Trend quality (0-2)": trend,
        "Momentum quality (0-2)": momentum,
        "Trigger quality (0-3)": trigger,
        "Volatility quality (0-1)": volatility,
        "Entry location quality (0-1)": location,
        "Entry quality score (0-10)": total,
    }


def v11_entry_score(row):
    parts = v11_entry_components(row)
    return int(parts["Entry quality score (0-10)"]), parts


def v11_is_entry_candidate(row, min_score=6):
    score, _ = v11_entry_score(row)
    setup = v11_setup(row)
    return (
        classify_regime(row) == "Bull trend"
        and setup != "No setup"
        and score >= min_score
        and 48 <= float(row["RSI"]) <= 72
        and float(row["MOM20"]) > 0
        and float(row["MOM63"]) > 0
        and float(row["ATR_PCT"]) <= 7.0
        and float(row["DIST_SMA20"]) <= 8.0
    )


def v11_entry_checks(row):
    score, parts = v11_entry_score(row)
    return {
        **parts,
        "Bull regime": classify_regime(row) == "Bull trend",
        "Setup present": v11_setup(row) != "No setup",
        "Setup": v11_setup(row),
        "RSI 48-72": 48 <= float(row["RSI"]) <= 72,
        "MOM20 > 0%": float(row["MOM20"]) > 0,
        "MOM63 > 0%": float(row["MOM63"]) > 0,
        "ATR <= 7%": float(row["ATR_PCT"]) <= 7,
        "Distance from SMA20 %": float(row["DIST_SMA20"]),
        "Momentum acceleration %": float(row["MOM20_ACCEL"]),
        "Close location": float(row["CLOSE_LOCATION"]),
        "Breakout extension %": float(row["BREAKOUT_EXTENSION"]),
        "Volume ratio": float(row["VOL_RATIO"]) if pd.notna(row.get("VOL_RATIO", np.nan)) else np.nan,
    }

def run_v10(
    data_map,
    initial=10000,
    risk_pct=0.75,
    max_pos_pct=25,
    max_exposure_pct=60,
    stop_atr=2.5,
    target_r=3.0,
    brokerage=6.50,
    slippage_pct=0.10,
    max_positions=3,
    cooldown_days=10,
    max_hold_days=90,
    drawdown_brake_pct=10.0,
    brake_days=20,
    eval_start=None,
    eval_end=None,
):
    """V10 setup/trigger architecture with V9.5 control and V10 risk-brake structure."""
    prepared = {t: add_v10_indicators(df) for t, df in data_map.items() if not df.empty}
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
    peak_equity = float(initial)
    prior_drawdown = 0.0
    brake_until_idx = -1
    brake_events = 0

    def in_eval(dt, next_dt=None):
        return (eval_start is None or dt >= pd.Timestamp(eval_start)) and (eval_end is None or (next_dt is not None and next_dt <= pd.Timestamp(eval_end)))

    for i, dt in enumerate(dates[:-1]):
        next_dt = dates[i + 1]
        equity = cash
        invested = 0.0
        for t, p in positions.items():
            px = safe_close(prepared[t], dt, p["last_price"])
            equity += p["shares"] * px
            invested += p["shares"] * px

        if in_eval(dt, next_dt):
            curve_points.append((dt, equity))
            exposure_points.append((dt, 100 * invested / equity if equity else 0))

        # Portfolio-level drawdown brake: trigger only when crossing the threshold.
        if equity > peak_equity:
            peak_equity = equity
        drawdown = (equity / peak_equity - 1) * 100 if peak_equity else 0
        if drawdown <= -drawdown_brake_pct and prior_drawdown > -drawdown_brake_pct:
            brake_until_idx = max(brake_until_idx, i + brake_days)
            brake_events += 1
        prior_drawdown = drawdown

        # -------- exits --------
        if in_eval(dt, next_dt):
            for t in list(positions.keys()):
                d = prepared[t]
                row_next = safe_row(d, next_dt)
                if row_next is None:
                    continue
                p = positions[t]
                low, high = float(row_next["Low"]), float(row_next["High"])
                held = i - p["entry_index"] + 1
                exit_px = None
                reason = None
                if low <= p["stop"] and high >= p["target"]:
                    exit_px, reason = p["stop"], "Stop (both touched)"
                elif low <= p["stop"]:
                    exit_px, reason = p["stop"], "Stop"
                elif high >= p["target"]:
                    exit_px, reason = p["target"], "Target"
                elif held >= 5:
                    prev_row, prev_prev = prior_two_rows(d, dt)
                    confirmed_break = (
                        prev_row is not None and prev_prev is not None
                        and pd.notna(prev_row.get("SMA50")) and pd.notna(prev_prev.get("SMA50"))
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
                        "Ticker": t, "Entry date": p["entry_date"], "Exit date": next_dt,
                        "Entry": p["entry_price"], "Exit": exit_px, "Shares": p["shares"], "P/L": pnl,
                        "Reason": reason, "Entry score": p["score"], "Entry regime": p["entry_regime"],
                        "Entry value": p["entry_value"], "Brokerage total": 2 * brokerage,
                        "Slippage total": p["entry_value"] * slippage_pct / 100 + gross * slippage_pct / 100,
                        "Hold days": held, "entry_checks": p.get("entry_checks", {}), "setup": p.get("setup", ""),
                    })
                    del positions[t]
                    last_exit_idx[t] = i + 1

        # -------- entries --------
        slots = max_positions - len(positions)
        entries_allowed = in_eval(dt, next_dt) and i >= brake_until_idx
        if slots > 0 and entries_allowed:
            candidates = []
            for t, d in prepared.items():
                if t in positions or dt not in d.index or next_dt not in d.index:
                    continue
                if t in last_exit_idx and (i - last_exit_idx[t]) < cooldown_days:
                    continue
                row = safe_row(d, dt)
                nxt = safe_row(d, next_dt)
                if row is None or nxt is None or not v10_is_entry_candidate(row):
                    continue
                score, parts = v10_entry_score(row)
                confirmed, setup = v10_price_confirmation(row)
                strength = (score, parts["Trigger quality (0-3)"], parts["Momentum quality (0-2)"], parts["Trend quality (0-2)"], -float(row["ATR_PCT"]), float(row["MOM20"]))
                candidates.append((strength, t, row, nxt, score, parts))
            candidates.sort(reverse=True, key=lambda x: x[0])
            for _, t, row, nxt, score, parts in candidates[:slots]:
                px = float(nxt["Open"])
                atr = float(row["ATR14"])
                if not np.isfinite(px) or not np.isfinite(atr) or atr <= 0:
                    continue
                stop = px - stop_atr * atr
                risk_per_share = px - stop
                risk_budget = equity * risk_pct / 100
                qty_risk = int(risk_budget // risk_per_share)
                gross_existing = sum(
                    p2["shares"] * safe_close(prepared[t2], dt, p2["last_price"])
                    for t2, p2 in positions.items()
                )
                current_equity = equity
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
                    "shares": qty, "entry_price": px, "entry_value": value, "entry_date": next_dt,
                    "entry_index": i + 1, "stop": stop, "target": px + target_r * risk_per_share,
                    "score": score, "entry_regime": classify_regime(row),
                    "entry_checks": v10_entry_checks(row), "setup": v10_setup(row), "last_price": px,
                }

        for t, p in positions.items():
            px = safe_close(prepared[t], next_dt, p["last_price"])
            if np.isfinite(px):
                p["last_price"] = px

    # End-of-test close
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
            "Ticker": t, "Entry date": p["entry_date"], "Exit date": last_dt,
            "Entry": p["entry_price"], "Exit": px, "Shares": p["shares"], "P/L": pnl,
            "Reason": "End of test", "Entry score": p["score"], "Entry regime": p["entry_regime"],
            "Entry value": p["entry_value"], "Brokerage total": 2 * brokerage,
            "Slippage total": p["entry_value"] * slippage_pct / 100 + gross * slippage_pct / 100,
            "Hold days": held, "entry_checks": p.get("entry_checks", {}), "setup": p.get("setup", ""),
        })

    if not curve_points:
        return None
    final_curve_dates = [d.index[d.index <= pd.Timestamp(eval_end)][-1] for d in prepared.values() if eval_end is not None and len(d.index[d.index <= pd.Timestamp(eval_end)]) > 0]
    final_dt = max(d.index[-1] for d in prepared.values()) if eval_end is None else (max(final_curve_dates) if final_curve_dates else None)
    if final_dt is not None:
        curve_points.append((final_dt, cash))
    curve = pd.Series({pd.Timestamp(k): v for k, v in curve_points}).sort_index()
    curve = curve[~curve.index.duplicated(keep="last")]
    curve = curve.reindex(pd.date_range(curve.index.min(), curve.index.max(), freq="B")).ffill()
    m = metrics_from_curve(curve, trades, initial)
    m["curve"] = curve
    m["trades_df"] = pd.DataFrame(trades)
    m["avg exposure %"] = float(np.mean([v for _, v in exposure_points])) if exposure_points else 0
    m["brake events"] = brake_events
    m["brake days"] = brake_days
    return m

def run_v11(
    data_map,
    initial=10000,
    risk_pct=0.75,
    max_pos_pct=25,
    max_exposure_pct=60,
    stop_atr=3.0,
    target_r=3.0,
    brokerage=6.50,
    slippage_pct=0.10,
    max_positions=3,
    cooldown_days=10,
    max_hold_days=90,
    drawdown_brake_pct=10.0,
    brake_days=20,
    eval_start=None,
    eval_end=None,
):
    """V11 entry-quality architecture with V9.5 control and V10 risk-brake structure."""
    prepared = {t: add_v11_indicators(df) for t, df in data_map.items() if not df.empty}
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
    peak_equity = float(initial)
    prior_drawdown = 0.0
    brake_until_idx = -1
    brake_events = 0

    def in_eval(dt, next_dt=None):
        return (eval_start is None or dt >= pd.Timestamp(eval_start)) and (eval_end is None or (next_dt is not None and next_dt <= pd.Timestamp(eval_end)))

    for i, dt in enumerate(dates[:-1]):
        next_dt = dates[i + 1]
        equity = cash
        invested = 0.0
        for t, p in positions.items():
            px = safe_close(prepared[t], dt, p["last_price"])
            equity += p["shares"] * px
            invested += p["shares"] * px

        if in_eval(dt, next_dt):
            curve_points.append((dt, equity))
            exposure_points.append((dt, 100 * invested / equity if equity else 0))

        # Portfolio-level drawdown brake: trigger only when crossing the threshold.
        if equity > peak_equity:
            peak_equity = equity
        drawdown = (equity / peak_equity - 1) * 100 if peak_equity else 0
        if drawdown <= -drawdown_brake_pct and prior_drawdown > -drawdown_brake_pct:
            brake_until_idx = max(brake_until_idx, i + brake_days)
            brake_events += 1
        prior_drawdown = drawdown

        # -------- exits --------
        if in_eval(dt, next_dt):
            for t in list(positions.keys()):
                d = prepared[t]
                row_next = safe_row(d, next_dt)
                if row_next is None:
                    continue
                p = positions[t]
                low, high = float(row_next["Low"]), float(row_next["High"])
                held = i - p["entry_index"] + 1
                exit_px = None
                reason = None
                if low <= p["stop"] and high >= p["target"]:
                    exit_px, reason = p["stop"], "Stop (both touched)"
                elif low <= p["stop"]:
                    exit_px, reason = p["stop"], "Stop"
                elif high >= p["target"]:
                    exit_px, reason = p["target"], "Target"
                elif held >= 5:
                    prev_row, prev_prev = prior_two_rows(d, dt)
                    confirmed_break = (
                        prev_row is not None and prev_prev is not None
                        and pd.notna(prev_row.get("SMA50")) and pd.notna(prev_prev.get("SMA50"))
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
                        "Ticker": t, "Entry date": p["entry_date"], "Exit date": next_dt,
                        "Entry": p["entry_price"], "Exit": exit_px, "Shares": p["shares"], "P/L": pnl,
                        "Reason": reason, "Entry score": p["score"], "Entry regime": p["entry_regime"],
                        "Entry value": p["entry_value"], "Brokerage total": 2 * brokerage,
                        "Slippage total": p["entry_value"] * slippage_pct / 100 + gross * slippage_pct / 100,
                        "Hold days": held, "entry_checks": p.get("entry_checks", {}), "setup": p.get("setup", ""),
                    })
                    del positions[t]
                    last_exit_idx[t] = i + 1

        # -------- entries --------
        slots = max_positions - len(positions)
        entries_allowed = in_eval(dt, next_dt) and i >= brake_until_idx
        if slots > 0 and entries_allowed:
            candidates = []
            for t, d in prepared.items():
                if t in positions or dt not in d.index or next_dt not in d.index:
                    continue
                if t in last_exit_idx and (i - last_exit_idx[t]) < cooldown_days:
                    continue
                row = safe_row(d, dt)
                nxt = safe_row(d, next_dt)
                if row is None or nxt is None or not v11_is_entry_candidate(row):
                    continue
                score, parts = v11_entry_score(row)
                setup = v11_setup(row); confirmed = setup != "No setup"
                strength = (score, parts["Trigger quality (0-3)"], parts["Momentum quality (0-2)"], parts["Trend quality (0-2)"], -float(row["ATR_PCT"]), float(row["MOM20"]))
                candidates.append((strength, t, row, nxt, score, parts))
            candidates.sort(reverse=True, key=lambda x: x[0])
            for _, t, row, nxt, score, parts in candidates[:slots]:
                px = float(nxt["Open"])
                atr = float(row["ATR14"])
                if not np.isfinite(px) or not np.isfinite(atr) or atr <= 0:
                    continue
                stop = px - stop_atr * atr
                risk_per_share = px - stop
                risk_budget = equity * risk_pct / 100
                qty_risk = int(risk_budget // risk_per_share)
                gross_existing = sum(
                    p2["shares"] * safe_close(prepared[t2], dt, p2["last_price"])
                    for t2, p2 in positions.items()
                )
                current_equity = equity
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
                    "shares": qty, "entry_price": px, "entry_value": value, "entry_date": next_dt,
                    "entry_index": i + 1, "stop": stop, "target": px + target_r * risk_per_share,
                    "score": score, "entry_regime": classify_regime(row),
                    "entry_checks": v11_entry_checks(row), "setup": v11_setup(row), "last_price": px,
                }

        for t, p in positions.items():
            px = safe_close(prepared[t], next_dt, p["last_price"])
            if np.isfinite(px):
                p["last_price"] = px

    # End-of-test close
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
            "Ticker": t, "Entry date": p["entry_date"], "Exit date": last_dt,
            "Entry": p["entry_price"], "Exit": px, "Shares": p["shares"], "P/L": pnl,
            "Reason": "End of test", "Entry score": p["score"], "Entry regime": p["entry_regime"],
            "Entry value": p["entry_value"], "Brokerage total": 2 * brokerage,
            "Slippage total": p["entry_value"] * slippage_pct / 100 + gross * slippage_pct / 100,
            "Hold days": held, "entry_checks": p.get("entry_checks", {}), "setup": p.get("setup", ""),
        })

    if not curve_points:
        return None
    final_curve_dates = [d.index[d.index <= pd.Timestamp(eval_end)][-1] for d in prepared.values() if eval_end is not None and len(d.index[d.index <= pd.Timestamp(eval_end)]) > 0]
    final_dt = max(d.index[-1] for d in prepared.values()) if eval_end is None else (max(final_curve_dates) if final_curve_dates else None)
    if final_dt is not None:
        curve_points.append((final_dt, cash))
    curve = pd.Series({pd.Timestamp(k): v for k, v in curve_points}).sort_index()
    curve = curve[~curve.index.duplicated(keep="last")]
    curve = curve.reindex(pd.date_range(curve.index.min(), curve.index.max(), freq="B")).ffill()
    m = metrics_from_curve(curve, trades, initial)
    m["curve"] = curve
    m["trades_df"] = pd.DataFrame(trades)
    m["avg exposure %"] = float(np.mean([v for _, v in exposure_points])) if exposure_points else 0
    m["brake events"] = brake_events
    m["brake days"] = brake_days
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




# ---------------- V12 pullback-confirmation model ----------------
def add_v12_indicators(df):
    """V12 focuses on a proven-breakout pullback and recovery, not breakout chasing.

    A setup must have broken a 20-day high recently, then pulled back toward
    SMA20, and now reclaim SMA20 with a strong recovery day. All calculations
    use signal-day data only; execution remains at the next trading day's open.
    """
    x=add_v11_indicators(df).copy()
    x["BREAKOUT20"] = x["Close"] > x["PRIOR_20D_HIGH"]
    x["RECENT_BREAKOUT_15"] = x["BREAKOUT20"].shift(1).rolling(15, min_periods=1).max().fillna(0).astype(bool)
    # The pullback must occur before the recovery signal. Use the prior 5 bars only.
    touch = ((x["Low"] <= x["SMA20"]) | (x["Close"] <= x["SMA20"])).astype(int)
    x["PULLBACK_TOUCH_5"] = touch.shift(1).rolling(5, min_periods=1).max().fillna(0).astype(bool)
    x["SMA20_RECLAIM"] = (x["Close"] > x["SMA20"]) & (x["PREV_CLOSE"] <= x["PREV_SMA20"])
    x["RECOVERY_DAY"] = (x["Close"] > x["PREV_CLOSE"]) & (x["Close"] > x["High"].shift(1))
    x["DIST_SMA20"] = (x["Close"] / x["SMA20"] - 1) * 100
    x["DAY_GAIN"] = (x["Close"] / x["PREV_CLOSE"] - 1) * 100
    x["CLOSE_LOCATION"] = (x["Close"] - x["Low"]) / (x["High"] - x["Low"]).replace(0,np.nan)
    x["SMA50_SLOPE20"] = x["SMA50"].pct_change(20) * 100
    x["ATR_PCT"] = x["ATR14"] / x["Close"] * 100
    x["VOL_RATIO"] = pd.to_numeric(x.get("Volume", pd.Series(index=x.index,dtype=float)),errors="coerce") / pd.to_numeric(x.get("Volume", pd.Series(index=x.index,dtype=float)),errors="coerce").rolling(20).mean() if "Volume" in x.columns else np.nan
    return x.replace([np.inf,-np.inf],np.nan).dropna(subset=["DIST_SMA20","DAY_GAIN","CLOSE_LOCATION","SMA50_SLOPE20"])


def v12_setup(row):
    if bool(row.get("RECENT_BREAKOUT_15",False)) and bool(row.get("PULLBACK_TOUCH_5",False)) and bool(row.get("SMA20_RECLAIM",False)) and bool(row.get("RECOVERY_DAY",False)):
        return "Breakout pullback recovery"
    return "No setup"


def v12_entry_components(row):
    """Descriptive 0-8 setup-quality score; it is not a historical optimizer."""
    trend=int(float(row["SMA50"]) > float(row["SMA200"])) + int(float(row["SMA50_SLOPE20"]) > 0)
    momentum=int(float(row["MOM20"]) > 0) + int(float(row["MOM63"]) > 0)
    recovery=int(bool(row.get("SMA20_RECLAIM",False))) + int(bool(row.get("RECOVERY_DAY",False)))
    risk=int(float(row["ATR_PCT"]) <= 7.0)
    location=int(0.25 <= float(row["DIST_SMA20"]) <= 4.5)
    total=trend+momentum+recovery+risk+location
    return {
        "Trend quality (0-2)":trend,
        "Momentum quality (0-2)":momentum,
        "Recovery quality (0-2)":recovery,
        "Volatility quality (0-1)":risk,
        "Entry location quality (0-1)":location,
        "Entry quality score (0-8)":total,
    }


def v12_entry_score(row):
    p=v12_entry_components(row)
    return int(p["Entry quality score (0-8)"]),p


def v12_is_entry_candidate(row):
    setup=v12_setup(row)
    score,_=v12_entry_score(row)
    # Core setup gates: no score-chasing. The score is descriptive only.
    return (
        classify_regime(row)=="Bull trend"
        and setup!="No setup"
        and float(row["ATR_PCT"])<=7.0
        and float(row["MOM63"])>0
        and 48<=float(row["RSI"])<=72
        and 0.25<=float(row["DIST_SMA20"])<=4.5
        and float(row["CLOSE_LOCATION"])>=0.65
        and score>=6
    )


def v12_entry_checks(row):
    score,parts=v12_entry_score(row)
    return {
        **parts,
        "Bull regime":classify_regime(row)=="Bull trend",
        "Recent breakout (15d)":bool(row.get("RECENT_BREAKOUT_15",False)),
        "Pullback touch (5d)":bool(row.get("PULLBACK_TOUCH_5",False)),
        "SMA20 reclaim":bool(row.get("SMA20_RECLAIM",False)),
        "Recovery day":bool(row.get("RECOVERY_DAY",False)),
        "RSI 48-72":48<=float(row["RSI"])<=72,
        "MOM20 > 0%":float(row["MOM20"])>0,
        "MOM63 > 0%":float(row["MOM63"])>0,
        "ATR <= 7%":float(row["ATR_PCT"])<=7,
        "Distance from SMA20 %":float(row["DIST_SMA20"]),
        "Close location":float(row["CLOSE_LOCATION"]),
        "Volume ratio":float(row["VOL_RATIO"]) if pd.notna(row.get("VOL_RATIO",np.nan)) else np.nan,
    }


def run_v12(data_map, initial=10000, risk_pct=0.75, max_pos_pct=25, max_exposure_pct=60,
            stop_atr=3.0, target_r=3.0, brokerage=6.50, slippage_pct=0.10,
            max_positions=3, cooldown_days=10, max_hold_days=90,
            drawdown_brake_pct=10.0, brake_days=20, eval_start=None, eval_end=None):
    """V12: pullback-confirmation entries with V11 risk controls."""
    prepared={t:add_v12_indicators(df) for t,df in data_map.items() if not df.empty}
    prepared={t:d for t,d in prepared.items() if len(d)>=260}
    if not prepared: return None
    dates=sorted(set().union(*[set(d.index) for d in prepared.values()]))
    dates=[pd.Timestamp(x) for x in dates]
    cash=float(initial); positions={}; last_exit_idx={}; trades=[]; curve_points=[]; exposure_points=[]
    peak_equity=float(initial); prior_drawdown=0.0; brake_until_idx=-1; brake_events=0
    def in_eval(dt,next_dt=None):
        return (eval_start is None or dt>=pd.Timestamp(eval_start)) and (eval_end is None or (next_dt is not None and next_dt<=pd.Timestamp(eval_end)))
    for i,dt in enumerate(dates[:-1]):
        next_dt=dates[i+1]
        equity=cash; invested=0.0
        for t,p in positions.items():
            px=safe_close(prepared[t],dt,p["last_price"]); equity += p["shares"]*px; invested += p["shares"]*px
        if in_eval(dt,next_dt):
            curve_points.append((dt,equity)); exposure_points.append(100*invested/equity if equity else 0)
        if equity>peak_equity: peak_equity=equity
        drawdown=(equity/peak_equity-1)*100 if peak_equity else 0
        if drawdown<=-drawdown_brake_pct and prior_drawdown>-drawdown_brake_pct:
            brake_until_idx=max(brake_until_idx,i+brake_days); brake_events+=1
        prior_drawdown=drawdown

        # Exits: protect capital with the same V11 framework.
        if in_eval(dt,next_dt):
            for t in list(positions.keys()):
                d=prepared[t]; row_next=safe_row(d,next_dt)
                if row_next is None: continue
                p=positions[t]; low=float(row_next["Low"]); high=float(row_next["High"]); held=i-p["entry_index"]+1
                exit_px=None; reason=None
                if low<=p["stop"] and high>=p["target"]: exit_px,reason=p["stop"],"Stop (both touched)"
                elif low<=p["stop"]: exit_px,reason=p["stop"],"Stop"
                elif high>=p["target"]: exit_px,reason=p["target"],"Target"
                elif held>=5:
                    prev_row,prev_prev=prior_two_rows(d,dt)
                    confirmed=(prev_row is not None and prev_prev is not None and pd.notna(prev_row.get("SMA50")) and pd.notna(prev_prev.get("SMA50")) and float(prev_row["Close"])<float(prev_row["SMA50"]) and float(prev_prev["Close"])<float(prev_prev["SMA50"]))
                    if confirmed: exit_px,reason=float(row_next["Open"]),"Confirmed trend break"
                    elif held>=max_hold_days: exit_px,reason=float(row_next["Open"]),"Max hold"
                if exit_px is not None:
                    gross=p["shares"]*exit_px; sell_cost=brokerage+gross*slippage_pct/100; buy_cost=brokerage+p["entry_value"]*slippage_pct/100
                    cash += gross-sell_cost; pnl=(gross-sell_cost)-(p["entry_value"]+buy_cost)
                    trades.append({"Ticker":t,"Entry date":p["entry_date"],"Exit date":next_dt,"Entry":p["entry_price"],"Exit":exit_px,"Shares":p["shares"],"P/L":pnl,"Reason":reason,"Entry score":p["score"],"Entry regime":p["entry_regime"],"Entry value":p["entry_value"],"Brokerage total":2*brokerage,"Slippage total":p["entry_value"]*slippage_pct/100+gross*slippage_pct/100,"Hold days":held,"entry_checks":p.get("entry_checks",{}),"setup":p.get("setup","")})
                    del positions[t]; last_exit_idx[t]=i+1

        slots=max_positions-len(positions); entries_allowed=in_eval(dt,next_dt) and i>=brake_until_idx
        if slots>0 and entries_allowed:
            candidates=[]
            for t,d in prepared.items():
                if t in positions or dt not in d.index or next_dt not in d.index: continue
                if t in last_exit_idx and (i-last_exit_idx[t])<cooldown_days: continue
                row=safe_row(d,dt); nxt=safe_row(d,next_dt)
                if row is None or nxt is None or not v12_is_entry_candidate(row): continue
                score,parts=v12_entry_score(row)
                # Ranking is by current setup strength only; no historical P/L input.
                strength=(score,parts["Recovery quality (0-2)"],parts["Momentum quality (0-2)"],parts["Trend quality (0-2)"],-float(row["ATR_PCT"]),-abs(float(row["DIST_SMA20"])-2.0))
                candidates.append((strength,t,row,nxt,score,parts))
            candidates.sort(reverse=True,key=lambda x:x[0])
            for _,t,row,nxt,score,parts in candidates[:slots]:
                px=float(nxt["Open"]); atr=float(row["ATR14"])
                if not np.isfinite(px) or not np.isfinite(atr) or atr<=0: continue
                stop=px-stop_atr*atr; risk_per_share=px-stop
                if risk_per_share<=0: continue
                risk_budget=equity*risk_pct/100; qty_risk=int(risk_budget//risk_per_share)
                gross_existing=sum(p2["shares"]*safe_close(prepared[t2],dt,p2["last_price"]) for t2,p2 in positions.items())
                exposure_room=equity*max_exposure_pct/100-gross_existing; max_value=min(equity*max_pos_pct/100,max(exposure_room,0)); qty_cap=int(max_value//px); qty=min(qty_risk,qty_cap)
                if qty<1: continue
                value=qty*px; buy_cost=brokerage+value*slippage_pct/100
                if value+buy_cost>cash: continue
                cash-=value+buy_cost
                positions[t]={"shares":qty,"entry_price":px,"entry_value":value,"entry_date":next_dt,"entry_index":i+1,"stop":stop,"target":px+target_r*risk_per_share,"score":score,"entry_regime":classify_regime(row),"entry_checks":v12_entry_checks(row),"setup":v12_setup(row),"last_price":px}
        for t,p in positions.items():
            px=safe_close(prepared[t],next_dt,p["last_price"])
            if np.isfinite(px): p["last_price"]=px

    for t,p in list(positions.items()):
        d=prepared[t]; cutoff=pd.Timestamp(eval_end) if eval_end is not None else d.index[-1]; eligible=d.index[d.index<=cutoff]
        if len(eligible)==0: continue
        last_dt=eligible[-1]; px=float(d.loc[last_dt,"Close"]); gross=p["shares"]*px; sell_cost=brokerage+gross*slippage_pct/100; buy_cost=brokerage+p["entry_value"]*slippage_pct/100
        cash+=gross-sell_cost; pnl=(gross-sell_cost)-(p["entry_value"]+buy_cost); held=max(1,len(d.loc[p["entry_date"]:last_dt])-1)
        trades.append({"Ticker":t,"Entry date":p["entry_date"],"Exit date":last_dt,"Entry":p["entry_price"],"Exit":px,"Shares":p["shares"],"P/L":pnl,"Reason":"End of test","Entry score":p["score"],"Entry regime":p["entry_regime"],"Entry value":p["entry_value"],"Brokerage total":2*brokerage,"Slippage total":p["entry_value"]*slippage_pct/100+gross*slippage_pct/100,"Hold days":held,"entry_checks":p.get("entry_checks",{}),"setup":p.get("setup","")})
    if not curve_points: return None
    final_curve_dates=[d.index[d.index<=pd.Timestamp(eval_end)][-1] for d in prepared.values() if eval_end is not None and len(d.index[d.index<=pd.Timestamp(eval_end)])>0]
    final_dt=max(d.index[-1] for d in prepared.values()) if eval_end is None else (max(final_curve_dates) if final_curve_dates else None)
    if final_dt is not None: curve_points.append((final_dt,cash))
    curve=pd.Series({pd.Timestamp(k):v for k,v in curve_points}).sort_index(); curve=curve[~curve.index.duplicated(keep="last")]; curve=curve.reindex(pd.date_range(curve.index.min(),curve.index.max(),freq="B")).ffill()
    m=metrics_from_curve(curve,trades,initial); m["curve"]=curve; m["trades_df"]=pd.DataFrame(trades); m["avg exposure %"]=float(np.mean([v for _,v in exposure_points])) if exposure_points else 0; m["brake events"]=brake_events; m["brake days"]=brake_days
    return m


# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ V12 Settings")
watch_text=st.sidebar.text_input("Watchlist",WATCHLIST_DEFAULT)
tickers=[x.strip().upper() for x in watch_text.split(",") if x.strip()]
years=st.sidebar.selectbox("Test period",[5,7,10],index=0)
brokerage=st.sidebar.number_input("Brokerage / transaction ($)",0.0,50.0,6.50,0.50)
slippage=st.sidebar.number_input("Slippage / transaction (%)",0.0,1.0,0.10,0.05)
risk_pct=st.sidebar.slider("Risk per trade (% equity)",0.25,2.0,0.75,0.25)
max_pos_pct=st.sidebar.slider("Max position (% equity)",5,50,25,5)
max_exposure_pct=st.sidebar.slider("Max invested exposure (%)",25,100,60,5)
max_positions=st.sidebar.slider("Max simultaneous positions",1,4,3,1)
stop_atr=st.sidebar.slider("V12 stop distance (ATR)",1.5,4.0,3.0,0.25)
target_r=st.sidebar.slider("Profit target (R)",1.0,5.0,3.0,0.5)
cooldown_days=st.sidebar.slider("Cooldown after exit (days)",0,30,10,1)
max_hold_days=st.sidebar.slider("Maximum hold (days)",30,180,90,10)
drawdown_brake_pct=st.sidebar.slider("Portfolio drawdown brake (%)",5.0,20.0,10.0,1.0)
brake_days=st.sidebar.slider("Brake duration (trading days)",5,60,20,5)

if "v12_data" not in st.session_state: st.session_state.v12_data={}
if "v11_control" not in st.session_state: st.session_state.v11_control=None
if "v12_result" not in st.session_state: st.session_state.v12_result=None
if "v12_diag" not in st.session_state: st.session_state.v12_diag=None
if "v12_robust" not in st.session_state: st.session_state.v12_robust=None
if "v12_scan" not in st.session_state: st.session_state.v12_scan=pd.DataFrame()

st.title("📈 AI Investor V12")
st.caption("Breakout → controlled pullback → recovery confirmation • risk-controlled paper-trading research laboratory")
st.info("V12 is an educational research and paper-trading system. It does not guarantee returns, does not predict the future, and has no broker connection. V11 is preserved as the control.")
tabs=st.tabs(["🏦 Portfolio Lab","🔬 Robustness","🧪 Diagnostics","🤖 Paper Trader","📊 Research","💼 Portfolio","📚 Learn"])

with tabs[0]:
    st.subheader("V11 control vs V12")
    st.write("V12 removes the fast clean-breakout entry path. It waits for a recent breakout, a controlled pullback toward SMA20, and a recovery confirmation before entering. Risk sizing, stops, exposure limits and the drawdown brake remain aligned with V11.")
    if st.button("🚀 Run V11 control + V12 backtests",type="primary"):
        data_map=build_data(tickers,years); st.session_state.v12_data=data_map
        st.session_state.v11_control=run_v11(data_map,initial=10000,risk_pct=risk_pct,max_pos_pct=max_pos_pct,max_exposure_pct=max_exposure_pct,stop_atr=stop_atr,target_r=target_r,brokerage=brokerage,slippage_pct=slippage,max_positions=max_positions,cooldown_days=cooldown_days,max_hold_days=max_hold_days,drawdown_brake_pct=drawdown_brake_pct,brake_days=brake_days)
        st.session_state.v12_result=run_v12(data_map,initial=10000,risk_pct=risk_pct,max_pos_pct=max_pos_pct,max_exposure_pct=max_exposure_pct,stop_atr=stop_atr,target_r=target_r,brokerage=brokerage,slippage_pct=slippage,max_positions=max_positions,cooldown_days=cooldown_days,max_hold_days=max_hold_days,drawdown_brake_pct=drawdown_brake_pct,brake_days=brake_days)
    ctrl=st.session_state.v11_control; r=st.session_state.v12_result
    if ctrl and r:
        comp=pd.DataFrame([
            {"Version":"V11 control","Final $":ctrl["Final $"],"Return %":ctrl["Return %"],"CAGR %":ctrl["CAGR %"],"Max DD %":ctrl["Max DD %"],"Trades":ctrl["Trades"],"Win rate %":ctrl["Win rate %"],"Profit factor":ctrl["Profit factor"]},
            {"Version":"V12","Final $":r["Final $"],"Return %":r["Return %"],"CAGR %":r["CAGR %"],"Max DD %":r["Max DD %"],"Trades":r["Trades"],"Win rate %":r["Win rate %"],"Profit factor":r["Profit factor"]},])
        st.dataframe(comp.round(2),use_container_width=True)
        st.caption(f"V12 risk-brake events: {r.get('brake events',0)}; each pauses new entries for {r.get('brake days',brake_days)} trading days.")
        c=st.columns(5); c[0].metric("V12 Final",f"${r['Final $']:,.2f}"); c[1].metric("V12 Return",f"{r['Return %']:+.2f}%"); c[2].metric("V12 Max DD",f"{r['Max DD %']:.2f}%"); c[3].metric("V12 Trades",str(r['Trades'])); c[4].metric("V12 Profit factor",fmt_pf(r['Profit factor']))
        st.subheader("V12 equity curve"); st.line_chart(r["curve"])
        if not r["trades_df"].empty: st.subheader("V12 trade log"); st.dataframe(r["trades_df"].round(2),use_container_width=True)

with tabs[1]:
    st.subheader("🔬 V12 walk-forward robustness")
    st.write("Training, Validation and Out-of-sample periods are chronological. The same fixed V12 rules are used in every slice; no slice is used to tune parameters.")
    if st.button("🔬 Run V12 robustness test",type="primary"):
        rows=[]
        for ticker in tickers:
            raw=load_history(ticker,f"{years+2}y")
            if raw.empty: continue
            d=add_v12_indicators(raw)
            if len(d)<260: continue
            idx=d.index; n=len(idx); a=idx[int(n*0.55)]; b=idx[int(n*0.775)]
            for label,s,e in [("Training",idx[0],a),("Validation",a,b),("Out-of-sample",b,idx[-1])]:
                m=run_v12({ticker:raw},initial=10000,risk_pct=risk_pct,max_pos_pct=max_pos_pct,max_exposure_pct=max_exposure_pct,stop_atr=stop_atr,target_r=target_r,brokerage=brokerage,slippage_pct=slippage,max_positions=max_positions,cooldown_days=cooldown_days,max_hold_days=max_hold_days,drawdown_brake_pct=drawdown_brake_pct,brake_days=brake_days,eval_start=s,eval_end=e)
                if m: rows.append({"Ticker":ticker,"Period":label,"CAGR %":m["CAGR %"],"Max DD %":m["Max DD %"],"Trades":m["Trades"],"Win rate %":m["Win rate %"],"Profit factor":m["Profit factor"],"Brake events":m.get("brake events",0)})
        st.session_state.v12_robust=pd.DataFrame(rows)
    rr=st.session_state.v12_robust
    if rr is not None and not rr.empty:
        st.dataframe(rr.round(2),use_container_width=True)
        oos=rr[rr["Period"]=="Out-of-sample"]
        if not oos.empty: st.subheader("Out-of-sample summary"); st.dataframe(oos.round(2),use_container_width=True); st.info("Out-of-sample is a historical held-back slice, not a forecast.")

with tabs[2]:
    st.subheader("🧪 V12 diagnostic laboratory")
    st.write("Diagnostics compare V11 at the same settings with V12, including cost sensitivity, stock/setup results, exit reasons and entry components.")
    if st.button("🧪 Run V12 diagnostics",type="primary"):
        data_map=st.session_state.v12_data or build_data(tickers,years); st.session_state.v12_data=data_map
        base=run_v11(data_map,initial=10000,risk_pct=risk_pct,max_pos_pct=max_pos_pct,max_exposure_pct=max_exposure_pct,stop_atr=stop_atr,target_r=target_r,brokerage=brokerage,slippage_pct=slippage,max_positions=max_positions,cooldown_days=cooldown_days,max_hold_days=max_hold_days,drawdown_brake_pct=drawdown_brake_pct,brake_days=brake_days)
        new=run_v12(data_map,initial=10000,risk_pct=risk_pct,max_pos_pct=max_pos_pct,max_exposure_pct=max_exposure_pct,stop_atr=stop_atr,target_r=target_r,brokerage=brokerage,slippage_pct=slippage,max_positions=max_positions,cooldown_days=cooldown_days,max_hold_days=max_hold_days,drawdown_brake_pct=drawdown_brake_pct,brake_days=brake_days)
        zero=run_v12(data_map,initial=10000,risk_pct=risk_pct,max_pos_pct=max_pos_pct,max_exposure_pct=max_exposure_pct,stop_atr=stop_atr,target_r=target_r,brokerage=0,slippage_pct=0,max_positions=max_positions,cooldown_days=cooldown_days,max_hold_days=max_hold_days,drawdown_brake_pct=drawdown_brake_pct,brake_days=brake_days)
        st.session_state.v12_diag=(base,new,zero)
    diag=st.session_state.v12_diag
    if diag:
        base,new,zero=diag
        comparison=pd.DataFrame([
            {"Test":"V11 control","Final $":base["Final $"],"Return %":base["Return %"],"CAGR %":base["CAGR %"],"Max DD %":base["Max DD %"],"Trades":base["Trades"],"Profit factor":base["Profit factor"]},
            {"Test":"V12 current costs","Final $":new["Final $"],"Return %":new["Return %"],"CAGR %":new["CAGR %"],"Max DD %":new["Max DD %"],"Trades":new["Trades"],"Profit factor":new["Profit factor"]},
            {"Test":"V12 zero brokerage + zero slippage","Final $":zero["Final $"],"Return %":zero["Return %"],"CAGR %":zero["CAGR %"],"Max DD %":zero["Max DD %"],"Trades":zero["Trades"],"Profit factor":zero["Profit factor"]},])
        st.dataframe(comparison.round(2),use_container_width=True)
        trades=new["trades_df"].copy()
        if not trades.empty:
            for title,col in [("P/L by stock","Ticker"),("P/L by entry quality score","Entry score"),("P/L by setup type","setup"),("Exit reasons","Reason")]:
                st.markdown(f"### {title}"); st.dataframe(trades.groupby(col).agg(Trades=("P/L","count"),Gross_PnL=("P/L","sum"),Avg_PnL=("P/L","mean"),Win_rate=("P/L",lambda s:100*(s>0).mean())).reset_index().round(2),use_container_width=True)
            cols=["Ticker","Entry date","Exit date","Entry","Exit","Shares","P/L","Reason","Entry score","Hold days","setup"]
            st.markdown("### Largest losses"); st.dataframe(trades.sort_values("P/L").head(15)[cols].round(2),use_container_width=True)
            st.markdown("### Largest winners"); st.dataframe(trades.sort_values("P/L",ascending=False).head(15)[cols].round(2),use_container_width=True)
            st.markdown("### V12 entry components")
            comp=[]
            for _,tr in trades.iterrows():
                checks=tr.get("entry_checks",{})
                if isinstance(checks,dict):
                    for name,val in checks.items(): comp.append({"Component":name,"Value":val,"P/L":float(tr["P/L"])})
            if comp:
                cf=pd.DataFrame(comp); numeric=cf[cf["Value"].apply(lambda x:isinstance(x,(int,float,np.integer,np.floating)) and not isinstance(x,(bool,np.bool_)))]
                if not numeric.empty: st.dataframe(numeric.groupby(["Component","Value"]).agg(Trades=("P/L","count"),Gross_PnL=("P/L","sum"),Avg_PnL=("P/L","mean")).reset_index().round(2),use_container_width=True)
                boolean=cf[cf["Value"].apply(lambda x:isinstance(x,(bool,np.bool_)))]
                if not boolean.empty: st.dataframe(boolean.groupby(["Component","Value"]).agg(Trades=("P/L","count"),Gross_PnL=("P/L","sum"),Avg_PnL=("P/L","mean"),Win_rate=("P/L",lambda s:100*(s>0).mean())).reset_index().round(2),use_container_width=True)

with tabs[3]:
    st.subheader("🤖 V12 paper trader")
    st.write("Paper-only scanner. V12 waits for a recent breakout, controlled pullback and recovery confirmation. It does not place real trades.")
    if st.button("🔎 Run V12 paper scan",type="primary"):
        rows=[]
        for ticker in tickers:
            d=add_v12_indicators(load_history(ticker,"2y"))
            if d.empty: continue
            r=d.iloc[-1]; score,parts=v12_entry_score(r); candidate=v12_is_entry_candidate(r); regime=classify_regime(r); setup=v12_setup(r); stop=float(r["Close"]-stop_atr*r["ATR14"]); target=float(r["Close"]+target_r*(r["Close"]-stop))
            rows.append({"Ticker":ticker,"Price":float(r["Close"]),"Regime":regime,"Score":f"{score}/8","Trend":parts["Trend quality (0-2)"],"Momentum":parts["Momentum quality (0-2)"],"Recovery":parts["Recovery quality (0-2)"],"Volatility":parts["Volatility quality (0-1)"],"Location":parts["Entry location quality (0-1)"],"Setup":setup,"Action":"PAPER BUY CANDIDATE" if candidate else "WAIT / CASH","RSI":float(r["RSI"]),"3M %":float(r["MOM63"]),"ATR %":float(r["ATR_PCT"]),"SMA20 distance %":float(r["DIST_SMA20"]),"Stop":stop if candidate else np.nan,"Target":target if candidate else np.nan})
        st.session_state.v12_scan=pd.DataFrame(rows)
    if not st.session_state.v12_scan.empty: st.dataframe(st.session_state.v12_scan.round(2),use_container_width=True); st.info("A candidate is a rules-based paper signal, not a recommendation or guarantee.")

with tabs[4]:
    st.subheader("📊 Current V12 research dashboard")
    for ticker in tickers:
        d=add_v12_indicators(load_history(ticker,"2y"))
        if d.empty: continue
        r=d.iloc[-1]; score,parts=v12_entry_score(r); candidate=v12_is_entry_candidate(r); regime=classify_regime(r); setup=v12_setup(r)
        with st.expander(f"{ticker} — ${float(r['Close']):.2f} — {regime}"):
            c=st.columns(7); c[0].metric("Regime",regime); c[1].metric("Entry quality",f"{score}/8"); c[2].metric("Setup",setup); c[3].metric("Trend",f"{parts['Trend quality (0-2)']}/2"); c[4].metric("Momentum",f"{parts['Momentum quality (0-2)']}/2"); c[5].metric("Recovery",f"{parts['Recovery quality (0-2)']}/2"); c[6].metric("Location",f"{parts['Entry location quality (0-1)']}/1")
            st.write(f"SMA20 ${r['SMA20']:.2f} | SMA50 ${r['SMA50']:.2f} | SMA200 ${r['SMA200']:.2f} | RSI {r['RSI']:.1f} | ATR% {r['ATR_PCT']:.2f}% | SMA20 distance {r['DIST_SMA20']:.2f}%")
            st.write(f"Recent breakout: **{'YES' if r['RECENT_BREAKOUT_15'] else 'NO'}** | Pullback touch: **{'YES' if r['PULLBACK_TOUCH_5'] else 'NO'}** | Reclaim: **{'YES' if r['SMA20_RECLAIM'] else 'NO'}** | Recovery day: **{'YES' if r['RECOVERY_DAY'] else 'NO'}**")
            st.write(f"Rule output: **{'PAPER BUY CANDIDATE' if candidate else 'WAIT / CASH'}**")

with tabs[5]:
    st.subheader("💼 Paper portfolio")
    st.info("V12 is paper-only and does not connect to a broker. Use the backtest and paper scanner to study the rules before any real-money use.")
    r=st.session_state.v12_result
    if r:
        st.metric("Latest V12 simulated portfolio value",f"${r['Final $']:,.2f}")
        if not r["trades_df"].empty: st.dataframe(r["trades_df"].round(2),use_container_width=True)
    else: st.write("Run the portfolio backtest first.")

with tabs[6]:
    st.subheader("📚 What V12 is teaching you")
    st.markdown("""
### 1. V11 is the control
V11 stays unchanged so we can tell whether the new entry structure actually changes results.

### 2. We stopped chasing breakouts
V12 does not buy a fresh breakout just because it looks strong. It waits for the market to test the move.

### 3. The entry sequence is explicit
**Breakout → pullback → SMA20 reclaim → recovery day → next-open entry.**

### 4. Risk remains defensive
Default risk is 0.75% per trade, maximum position 25%, maximum exposure 60%, three simultaneous positions, 3 ATR stop, 3R target, realistic brokerage/slippage and a portfolio drawdown brake.

### 5. The score is descriptive, not an optimizer
A higher score does not mean the historical sample is guaranteed to perform better. The core setup gates decide whether a trade is eligible.

### 6. Walk-forward testing comes before any real-money discussion
Training, Validation and Out-of-sample slices use the same fixed V12 rules. Historical results are not a forecast.

### 7. Paper only
This system has no broker connection and does not guarantee returns.
""")
st.divider(); st.caption("AI Investor V12 • Educational research and paper trading only • V11 control preserved • No guaranteed returns")
