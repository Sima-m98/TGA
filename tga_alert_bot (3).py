"""
TGA Golden Alignment alert bot
------------------------------
Stage 1: checks the 1H "sign" and messages you when a market turns Golden.
Stage 2: only for Golden symbols, watches the 15m and messages you when a full setup is ready
(fib pullback + reversal), with SL/TP, then follows the trade (SL/TP hits, exit warnings).

Setup:
    pip install yfinance pandas numpy requests
    set environment variables TG_TOKEN and TG_CHAT_ID  (see README steps in chat)
Run:
    python tga_alert_bot.py          # loops forever, checks every hour
    python tga_alert_bot.py --once   # single check (for cron / task scheduler)
    python tga_alert_bot.py --test   # send a test Telegram message and exit
    python tga_alert_bot.py --status # show which rules pass for each symbol (and send it to Telegram)
"""
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

# ───────────── EDIT THIS ─────────────
SYMBOLS = {
    # name shown in alert : Yahoo Finance ticker
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "WTI":    "CL=F",
    # a list = try in order, use the first ticker that returns enough 1H data
    "XAUUSD": ["XAUUSD=X", "GC=F"],   # spot first, futures as fallback
    "XAGUSD": ["XAGUSD=X", "SI=F"],
    "US100":  "NQ=F",
    "US500":  "ES=F",   # assumed "us50" means US500 (S&P). Change if not.
    "US30":   "YM=F",
}
MAX_PUSH = 3            # no entry once pushes reach this number
PIV_LEN = 5             # swing pivot length (same as Pine input)
NOTIFY_LOST = True      # also message when Golden is lost
USE_DIV_FILTER = True   # skip Golden while a 1H divergence (RTI / MACD / oscillator) is active
DIV_LOOK = 20           # a divergence stays active this many candles
# Indicator settings, keep identical to the Pine inputs so both agree
ST_LEN, ST_MULT = 10, 3.0
UT_KEY, UT_ATR = 1.0, 10
LR_LEN = 11
RTI_LEN, RTI_SENS, RTI_SIG = 100, 95, 20   # your Zeiierman RTI settings
# LuxAlgo Trendlines with Breaks Oscillator settings (Atr slope, EMA smoothing)
LUX_LEN, LUX_SLOPE, LUX_SHORT, LUX_LONG, LUX_SMOOTH, LUX_DIV_LEN = 14, 1.0, 9, 20, 10, 4
STATE_FILE = "tga_state.json"

# ---- 15m stage (runs ONLY for symbols whose 1H is Golden) ----
ENABLE_15M = True
LTF_MIN = 5              # min number of the 7 rules that must agree on the 15m (7 = strict)
# When are the 15m rules counted?
#   "at_leg_peak": at the top/bottom of the leg BEFORE the pullback (was the 15m trend real?), plus the
#                  15m structure must still be intact at the reversal. Indicators flip during a pullback,
#                  so counting them at the reversal candle almost never works.
#   "at_trigger" : count them on the reversal candle itself (very strict, rarely fires).
LTF_MODE = "at_leg_peak"
FIB_MIN, FIB_MAX = 0.382, 0.786   # pullback must reach at least FIB_MIN and not go beyond FIB_MAX
FIB_MIN_STRONG = 0.236   # a SHALLOW pullback is accepted only while the 15m MACD histogram is very strong in the trade direction
HIST_STRONG_MULT = 1.0   # "very strong" = histogram >= this x its 50-candle average size, and not shrinking
TOUCH_BARS = 5           # pullback must have been reached within the last N candles
SWING_AGE = 80           # the swing leg must be younger than this many 15m candles
SIG_COOLDOWN = 12        # min 15m candles between two setups on the same symbol
SL_BUF = 0.15            # SL buffer in ATRs beyond the swing
TP_RATIOS = (1.0,)       # trend-based fib level(s): 1.0 = a move equal to the last leg, measured from the entry
WARN_MIN = 3             # exit warnings needed to notify you
RULE_KEYS = ["Structure", "MACD", "RTI", "OscAlign", "Supertrend", "UTBot", "LinReg"]
# ──────────────────────────────────────

TG_TOKEN = os.getenv("TG_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")


# ───────────── Indicator helpers ─────────────
def rma(s, n):
    return s.ewm(alpha=1 / n, adjust=False).mean()


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def atr(df, n):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return rma(tr, n)


def rsi(close, n=14):
    d = close.diff()
    up = rma(d.clip(lower=0), n)
    dn = rma((-d).clip(lower=0), n)
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def linreg(close, n):
    x = np.arange(n)
    xm = x.mean()
    den = ((x - xm) ** 2).sum()

    def f(y):
        slope = ((x - xm) * (y - y.mean())).sum() / den
        return y.mean() + slope * (n - 1 - xm)

    return close.rolling(n).apply(f, raw=True)


def supertrend_dir(df, length, mult):
    a = atr(df, length).values
    hl2 = ((df["high"] + df["low"]) / 2).values
    c = df["close"].values
    ub, lb = hl2 + mult * a, hl2 - mult * a
    fub, flb = ub.copy(), lb.copy()
    d = np.ones(len(df), dtype=int)
    for i in range(1, len(df)):
        if np.isnan(a[i]) or np.isnan(a[i - 1]):
            continue
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        if d[i - 1] == 1:
            d[i] = -1 if c[i] < flb[i] else 1
        else:
            d[i] = 1 if c[i] > fub[i] else -1
    return d  # +1 uptrend, -1 downtrend


def ut_bot_dir(df, key, period):
    c = df["close"].values
    loss = (key * atr(df, period)).values
    x = np.zeros(len(c))
    for i in range(1, len(c)):
        p = x[i - 1]
        if np.isnan(loss[i]):
            x[i] = 0.0
            continue
        if c[i] > p and c[i - 1] > p:
            x[i] = max(p, c[i] - loss[i])
        elif c[i] < p and c[i - 1] < p:
            x[i] = min(p, c[i] + loss[i])
        elif c[i] > p:
            x[i] = c[i] - loss[i]
        else:
            x[i] = c[i] + loss[i]
    return int(np.sign(c[-1] - x[-1]))


def structure_and_pushes(df, pv):
    h, l = df["high"].values, df["low"].values
    ph1 = ph2 = pl1 = pl2 = np.nan
    push_up = push_dn = 0
    for t in range(2 * pv, len(df)):
        i = t - pv  # pivot candidate, confirmed at bar t
        win_h = h[i - pv:t + 1]
        win_l = l[i - pv:t + 1]
        if h[i] == win_h.max() and (win_h == h[i]).sum() == 1:
            ph2, ph1 = ph1, h[i]
            if ph1 > ph2:
                push_up += 1
            else:
                push_up = push_dn = 0
        if l[i] == win_l.min() and (win_l == l[i]).sum() == 1:
            pl2, pl1 = pl1, l[i]
            if pl1 < pl2:
                push_dn += 1
            else:
                push_dn = push_up = 0
    if ph1 > ph2 and pl1 > pl2:
        d = 1
    elif ph1 < ph2 and pl1 < pl2:
        d = -1
    else:
        d = 0
    return d, push_up, push_dn


def sgn(x):
    return int(np.sign(x)) if not pd.isna(x) else 0


def divergence_active(df, series_list, pv, look):
    """Returns (bearish_active, bullish_active) from price pivots vs each oscillator series."""
    h, l = df["high"].values, df["low"].values
    n = len(df)
    bear_t, bull_t = -10**9, -10**9
    for src in series_list:
        s = src.values
        pp = ps = lp = ls = np.nan
        for t in range(2 * pv, n):
            i = t - pv
            win_h, win_l = h[i - pv:t + 1], l[i - pv:t + 1]
            if h[i] == win_h.max() and (win_h == h[i]).sum() == 1:
                if not np.isnan(pp) and h[i] > pp and s[i] < ps:
                    bear_t = max(bear_t, t)
                pp, ps = h[i], s[i]
            if l[i] == win_l.min() and (win_l == l[i]).sum() == 1:
                if not np.isnan(lp) and l[i] < lp and s[i] > ls:
                    bull_t = max(bull_t, t)
                lp, ls = l[i], s[i]
    return (n - 1 - bear_t) <= look, (n - 1 - bull_t) <= look


def rti_series(close, length=None, sens=None, sig_len=None):
    """Exact port of Zeiierman's Relative Trend Index (0-100, midline 50) plus its EMA signal line."""
    length = length or RTI_LEN
    sens = sens or RTI_SENS
    sig_len = sig_len or RTI_SIG
    sd = close.rolling(2).std(ddof=0)  # Pine ta.stdev is the population stdev
    up = (close + sd).values
    lo = (close - sd).values
    n = len(close)
    out = np.full(n, np.nan)
    if n > length:
        wu = np.lib.stride_tricks.sliding_window_view(up, length)
        wl = np.lib.stride_tricks.sliding_window_view(lo, length)
        valid = ~(np.isnan(wu).any(axis=1) | np.isnan(wl).any(axis=1))
        su, sl = np.sort(wu, axis=1), np.sort(wl, axis=1)
        ui = int(math.floor(sens / 100 * length + 0.5)) - 1
        li = int(math.floor((100 - sens) / 100 * length + 0.5)) - 1
        U, L = su[:, ui], sl[:, li]
        r = (close.values[length - 1:] - L) / (U - L) * 100
        r[~valid] = np.nan
        out[length - 1:] = r
    rti = pd.Series(out, index=close.index)
    return rti, ema(rti, sig_len)


def lux_oscillator(df):
    """Port of LuxAlgo 'Trendlines with Breaks Oscillator' (Atr slope). Returns (histogram, signal EMA)."""
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    n, L = len(df), LUX_LEN
    slope_arr = (atr(df, L) / L * LUX_SLOPE).values
    sa, la = 2 / (LUX_SHORT + 1), 2 / (LUX_LONG + 1)
    upper = lower = slope_ph = slope_pl = 0.0
    upos = dnos = 0
    bullish = bearish = 0.0
    delta = np.zeros(n)
    bull_brk = np.zeros(n, dtype=bool)
    bear_brk = np.zeros(n, dtype=bool)
    for t in range(n):
        new_ph = new_pl = False
        ph = pl = 0.0
        i = t - L
        if i >= L:
            if h[i] > h[i - L:i].max() and h[i] >= h[i + 1:t + 1].max():
                new_ph, ph = True, h[i]
            if l[i] < l[i - L:i].min() and l[i] <= l[i + 1:t + 1].min():
                new_pl, pl = True, l[i]
        slope = slope_arr[t]
        slope_ph = slope if new_ph else slope_ph
        slope_pl = slope if new_pl else slope_pl
        upper = ph - slope_ph * L if new_ph else upper - slope_ph
        lower = pl + slope_pl * L if new_pl else lower + slope_pl
        prev_u, prev_d = upos, dnos
        upos = 0 if new_ph else (1 if c[t] > upper else upos)
        dnos = 0 if new_pl else (1 if c[t] < lower else dnos)
        if upos > prev_u:
            bull_brk[t] = True
            bullish += (c[t] - upper - bullish) * sa
        elif upos == 1:
            bullish += (max(c[t] - upper, 0) - bullish) * la
        else:
            bullish -= bullish * la
        if dnos > prev_d:
            bear_brk[t] = True
            bearish += (lower - c[t] - bearish) * sa
        elif dnos == 1:
            bearish += (max(lower - c[t], 0) - bearish) * la
        else:
            bearish -= bearish * la
        delta[t] = bullish - bearish
    d = pd.Series(delta, index=df.index)
    return d, ema(d, LUX_SMOOTH), bull_brk, bear_brk


def lux_divergence_active(df, data, length, look):
    """LuxAlgo's own divergence rule on the oscillator. Returns (bearish_active, bullish_active)."""
    d, c = data.values, df["close"].values
    n = len(df)
    last_h = last_hc = last_l = last_lc = np.nan
    bear_t = bull_t = -10**9
    for t in range(2 * length, n):
        i = t - length
        win = d[i - length:t + 1]
        if d[i] == win.max() and (win == d[i]).sum() == 1 and d[t] > 0:
            if not np.isnan(last_h) and d[i] < last_h and c[i] >= last_hc:
                bear_t = t
            last_h, last_hc = d[i], c[i]
        if d[i] == win.min() and (win == d[i]).sum() == 1 and d[t] < 0:
            if not np.isnan(last_l) and d[i] > last_l and c[i] <= last_lc:
                bull_t = t
            last_l, last_lc = d[i], c[i]
    return (n - 1 - bear_t) <= look, (n - 1 - bull_t) <= look


def evaluate(df):
    """Returns dict of direction ints (+1 / -1 / 0) for the 7 rules plus pushes."""
    close = df["close"]
    macd_line = ema(close, 12) - ema(close, 26)
    rti, rti_sig = rti_series(close)
    r_now, s_now = rti.iloc[-1], rti_sig.iloc[-1]
    rti_dir = 1 if (r_now > 50 and r_now > s_now) else -1 if (r_now < 50 and r_now < s_now) else 0
    osc, osc_sig, _bb, _sb = lux_oscillator(df)
    o_now, os_now = osc.iloc[-1], osc_sig.iloc[-1]
    # oscillator on the trend side of zero AND on the trend side of its signal line
    align = 1 if (o_now > 0 and o_now > os_now) else -1 if (o_now < 0 and o_now < os_now) else 0
    lr = linreg(close, LR_LEN)
    lr_sig = lr.rolling(LR_LEN).mean()
    if close.iloc[-1] > lr_sig.iloc[-1] and lr.iloc[-1] > lr_sig.iloc[-1]:
        lrd = 1
    elif close.iloc[-1] < lr_sig.iloc[-1] and lr.iloc[-1] < lr_sig.iloc[-1]:
        lrd = -1
    else:
        lrd = 0
    struct, pu, pd_ = structure_and_pushes(df, PIV_LEN)
    macd_hist = macd_line - ema(macd_line, 9)
    div_bear, div_bull = divergence_active(df, [rti, macd_hist], PIV_LEN, DIV_LOOK)
    lux_bear, lux_bull = lux_divergence_active(df, osc, LUX_DIV_LEN, DIV_LOOK)
    div_bear, div_bull = (div_bear or lux_bear), (div_bull or lux_bull)
    # exhaustion: price stalls at one level with 3+ rejection wicks (last 12 candles)
    w = df.iloc[-12:]
    a14 = atr(df, 14).iloc[-1]
    rng = w["high"] - w["low"]
    body_hi = w[["open", "close"]].max(axis=1)
    body_lo = w[["open", "close"]].min(axis=1)
    up_rej = int(((rng > 0) & ((w["high"] - body_hi) > 0.5 * rng) & ((w["high"] - w["high"].max()).abs() < 0.25 * a14)).sum())
    dn_rej = int(((rng > 0) & ((body_lo - w["low"]) > 0.5 * rng) & ((w["low"] - w["low"].min()).abs() < 0.25 * a14)).sum())
    stall = 1 if up_rej >= 3 else -1 if dn_rej >= 3 else 0
    return {
        "Structure": struct,
        "MACD": sgn(macd_line.iloc[-1]),
        "RTI": rti_dir,
        "OscAlign": align,
        "Supertrend": int(supertrend_dir(df, ST_LEN, ST_MULT)[-1]),
        "UTBot": ut_bot_dir(df, UT_KEY, UT_ATR),
        "LinReg": lrd,
        "push_up": pu,
        "push_dn": pd_,
        "stall": stall,
        "div_bear": bool(div_bear),
        "div_bull": bool(div_bull),
    }


def golden_state(r):
    keys = ["Structure", "MACD", "RTI", "OscAlign", "Supertrend", "UTBot", "LinReg"]
    if all(r[k] == 1 for k in keys) and r["push_up"] < MAX_PUSH and r["stall"] != 1 and not (USE_DIV_FILTER and r["div_bear"]):
        return "bull"
    if all(r[k] == -1 for k in keys) and r["push_dn"] < MAX_PUSH and r["stall"] != -1 and not (USE_DIV_FILTER and r["div_bull"]):
        return "bear"
    return "none"


# ───────────── Data + Telegram ─────────────
def fetch_candles(ticker, interval="1h"):
    import yfinance as yf

    minutes = {"1h": 60, "15m": 15, "1m": 1}[interval]
    period = {"1h": "60d", "15m": "30d", "1m": "5d"}[interval]
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    # drop the candle that is still forming
    if len(df) and df.index[-1] + timedelta(minutes=minutes) > pd.Timestamp.now(tz=timezone.utc):
        df = df.iloc[:-1]
    return df


def fetch_1h(ticker):
    return fetch_candles(ticker, "1h")


def get_df(ticker, interval):
    """Tries each ticker in order; returns the first with enough data, else None."""
    for tk in (ticker if isinstance(ticker, list) else [ticker]):
        try:
            cand = fetch_candles(tk, interval)
        except Exception as e:
            print(f"  {tk} {interval} failed ({e})")
            continue
        if len(cand) >= 200:
            return cand
        print(f"  {tk} {interval} has only {len(cand)} bars, trying next")
    return None


def send(msg):
    """Returns True only if Telegram accepted the message."""
    if not TG_TOKEN or not TG_CHAT_ID:
        print("[no Telegram configured]", msg)
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            data={"chat_id": TG_CHAT_ID, "text": msg},
            timeout=15,
        )
        if not r.ok:
            print("Telegram error:", r.text)
        return r.ok
    except requests.RequestException as e:
        print("Telegram request failed:", e)
        return False


def load_state():
    try:
        with open(STATE_FILE) as f:
            st = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        st = {}
    if "h1" not in st:  # migrate the old flat format
        st = {"h1": {k: v for k, v in st.items() if isinstance(v, str)}, "trades": {}, "last_sig": {}}
    st.setdefault("trades", {})
    st.setdefault("last_sig", {})
    return st


def save_state(st):
    with open(STATE_FILE, "w") as f:
        json.dump(st, f, indent=1)


def fmt(x):
    return f"{x:.5f}" if x < 5 else f"{x:.3f}" if x < 500 else f"{x:.2f}"


# ───────────── 15m: swings, entry setup, exit warnings ─────────────
def swing_state(df, pv):
    h, l = df["high"].values, df["low"].values
    st = {"ph": np.nan, "pl": np.nan, "leg_low": np.nan, "leg_high": np.nan, "ph_idx": -1, "pl_idx": -1}
    for t in range(2 * pv, len(df)):
        i = t - pv
        wh, wl = h[i - pv:t + 1], l[i - pv:t + 1]
        if h[i] == wh.max() and (wh == h[i]).sum() == 1:
            st["ph"], st["leg_low"], st["ph_idx"] = h[i], st["pl"], i
        if l[i] == wl.min() and (wl == l[i]).sum() == 1:
            st["pl"], st["leg_high"], st["pl_idx"] = l[i], st["ph"], i
    return st


def hist_strong(df, direction):
    """True when the 15m MACD histogram is on the trade side, at least as big as its 50-candle average, and not shrinking."""
    macd_l = ema(df["close"], 12) - ema(df["close"], 26)
    h = (macd_l - ema(macd_l, 9)).values
    return bool(direction * h[-1] > 0 and abs(h[-1]) >= HIST_STRONG_MULT * np.abs(h[-50:]).mean()
                and direction * h[-1] >= direction * h[-2])


def entry_signal(df, direction):
    """15m setup: confirmations + fib pullback + reversal candle closing back over the LinReg line.
    Returns (signal_dict_or_None, reason, evaluation)."""
    r = evaluate(df)
    if USE_DIV_FILTER and ((direction == 1 and r["div_bear"]) or (direction == -1 and r["div_bull"])):
        return None, "15m divergence active", r
    sw = swing_state(df, PIV_LEN)
    if LTF_MODE == "at_trigger":
        score = sum(1 for k in RULE_KEYS if r[k] == direction)
        when = "at the trigger candle"
    else:
        if r["Structure"] != direction:
            return None, "15m structure no longer intact", r
        pk = sw["ph_idx"] if direction == 1 else sw["pl_idx"]
        if pk < 150:
            return None, "not enough history before the swing peak", r
        rp = evaluate(df.iloc[:pk + 1])
        score = sum(1 for k in RULE_KEYS if rp[k] == direction)
        when = "at the swing peak"
    if score < LTF_MIN:
        return None, f"15m confirmations {score}/7 {when} (need {LTF_MIN})", r
    n = len(df)
    c, o = df["close"].values, df["open"].values
    hi, lo = df["high"].values, df["low"].values
    lr_sig = linreg(df["close"], LR_LEN).rolling(LR_LEN).mean().values
    a = atr(df, 14).iloc[-1]
    entry = c[-1]
    strong = hist_strong(df, direction)
    min_depth = FIB_MIN_STRONG if strong else FIB_MIN
    if direction == 1:
        rng = sw["ph"] - sw["leg_low"]
        if not rng > 0:
            return None, "no clear swing leg yet", r
        lvl_hi, lvl_lo = sw["ph"] - rng * min_depth, sw["ph"] - rng * FIB_MAX
        recent = lo[-TOUCH_BARS:].min()
        if not (lvl_lo <= recent <= lvl_hi and (n - 1 - sw["ph_idx"]) <= SWING_AGE):
            return None, "no fib pullback in zone", r
        if not (c[-1] > o[-1] and c[-1] > lr_sig[-1] and c[-2] <= lr_sig[-2]):
            return None, "waiting for reversal candle over the LinReg line", r
        sl = min(sw["pl"], recent) - a * SL_BUF
        if sl >= entry:
            return None, "invalid SL", r
        tps = [entry + rng * x for x in TP_RATIOS]
        depth = (sw["ph"] - recent) / rng
    else:
        rng = sw["leg_high"] - sw["pl"]
        if not rng > 0:
            return None, "no clear swing leg yet", r
        lvl_lo, lvl_hi = sw["pl"] + rng * min_depth, sw["pl"] + rng * FIB_MAX
        recent = hi[-TOUCH_BARS:].max()
        if not (lvl_lo <= recent <= lvl_hi and (n - 1 - sw["pl_idx"]) <= SWING_AGE):
            return None, "no fib pullback in zone", r
        if not (c[-1] < o[-1] and c[-1] < lr_sig[-1] and c[-2] >= lr_sig[-2]):
            return None, "waiting for reversal candle under the LinReg line", r
        sl = max(sw["ph"], recent) + a * SL_BUF
        if sl <= entry:
            return None, "invalid SL", r
        tps = [entry - rng * x for x in TP_RATIOS]
        depth = (recent - sw["pl"]) / rng
    return {"entry": float(entry), "sl": float(sl), "tps": [float(x) for x in tps],
            "depth": float(depth), "shallow": bool(depth < FIB_MIN), "score": score, "time": df.index[-1].isoformat()}, "ok", r


def exit_warnings(df, d):
    """Your manual-exit conditions, evaluated on the last closed 15m candle. d: +1 long, -1 short."""
    c, o = df["close"].values, df["open"].values
    h, l = df["high"].values, df["low"].values
    close = df["close"]
    osc, sig, bull_brk, bear_brk = lux_oscillator(df)
    osc, sig = osc.values, sig.values
    rti, rti_sig = rti_series(close)
    rti, rti_sig = rti.values, rti_sig.values
    macd_line = ema(close, 12) - ema(close, 26)
    hist = (macd_line - ema(macd_line, 9)).values
    lr_sig = linreg(close, LR_LEN).rolling(LR_LEN).mean().values
    sw = swing_state(df, PIV_LEN)
    r = evaluate(df)
    a = atr(df, 14).iloc[-1]
    out = []
    x = d  # direction multiplier: mirrors every test for shorts
    if (bear_brk[-1] if d == 1 else bull_brk[-1]):
        out.append("Oscillator dot in the opposite colour")
    if x * (osc[-2] - sig[-2]) >= 0 and x * (osc[-1] - sig[-1]) < 0:
        out.append("Oscillator crossed its signal line against the trade")
    if x * osc[-2] >= 0 and x * osc[-1] < 0:
        out.append("Oscillator crossed the zero line against the trade")
    lvl = sw["pl"] if d == 1 else sw["ph"]
    if not np.isnan(lvl) and x * (c[-2] - lvl) >= 0 and x * (c[-1] - lvl) < 0:
        out.append("Break of the 15m structure")
    if x * (rti[-2] - rti_sig[-2]) >= 0 and x * (rti[-1] - rti_sig[-1]) < 0:
        out.append("RTI crossed its signal line against the trade")
    rec = slice(-20, None)
    if d == 1:
        top = h[rec].max()
        wicks = sum(1 for i in range(-20, 0) if h[i] > l[i] and (h[i] - max(o[i], c[i])) > 0.5 * (h[i] - l[i]) and abs(h[i] - top) < 0.25 * a)
    else:
        bot = l[rec].min()
        wicks = sum(1 for i in range(-20, 0) if h[i] > l[i] and (min(o[i], c[i]) - l[i]) > 0.5 * (h[i] - l[i]) and abs(l[i] - bot) < 0.25 * a)
    if wicks >= 3:
        out.append("3+ long wicks rejecting the same level")
    if d == 1 and c[-1] < o[-1] and c[-2] > o[-2] and o[-1] >= c[-2] and c[-1] <= o[-2]:
        out.append("Opposite (bearish) engulfing candle")
    if d == -1 and c[-1] > o[-1] and c[-2] < o[-2] and o[-1] <= c[-2] and c[-1] >= o[-2]:
        out.append("Opposite (bullish) engulfing candle")
    if (r["div_bear"] if d == 1 else r["div_bull"]):
        out.append("Divergence against the trade")
    if x * (hist[-2] - hist[-3]) > 0 and x * (hist[-1] - hist[-2]) < 0:
        out.append("MACD histogram hooked against the trade")
    if x * (c[-1] - lr_sig[-1]) < 0 and x * (c[-1] - o[-1]) < 0:
        out.append("Opposite candle closed through the LinReg line")
    return out


def h1_hist_state(df1h, d):
    """'strong' if the 1H MACD histogram is on the trade side of zero and not shrinking two candles in a row."""
    close = df1h["close"]
    ml = ema(close, 12) - ema(close, 26)
    hist = (ml - ema(ml, 9)).values
    on_side = d * hist[-1] > 0
    shrinking = d * hist[-1] < d * hist[-2] < d * hist[-3]
    return "strong" if on_side and not shrinking else "weak"


def manage_trade(name, tr, df, h1_state, fast=None, df1h=None):
    """Follows an open setup. SL/TP use 1m candles when available (SL checked first).
    TP = target level OR the 15m MACD line crossing its signal line. Returns True if still open."""
    d = tr["dir"]
    label = "LONG" if d == 1 else "SHORT"
    src = fast if (fast is not None and len(fast)) else df
    since = src[src.index > pd.Timestamp(tr["time"])]
    for ts, row in since.iterrows():
        sl_hit = row["low"] <= tr["sl"] if d == 1 else row["high"] >= tr["sl"]
        if sl_hit:
            send(f"🛑 {name} {label}: stop loss level {fmt(tr['sl'])} was touched ({ts:%d %b %H:%M} UTC). Setup closed.")
            return False
        while tr["tp_hit"] < len(tr["tps"]):
            tp = tr["tps"][tr["tp_hit"]]
            if (row["high"] >= tp) if d == 1 else (row["low"] <= tp):
                tr["tp_hit"] += 1
                send(f"✅ {name} {label}: target {tr['tp_hit']} reached ({fmt(tp)}).")
            else:
                break
        if tr["tp_hit"] >= len(tr["tps"]):
            send(f"🏁 {name} {label}: all targets reached. Setup closed.")
            return False
    close = df["close"]
    ml = ema(close, 12) - ema(close, 26)
    sg = ema(ml, 9)
    if df.index[-1] > pd.Timestamp(tr["time"]) and d * (ml.iloc[-2] - sg.iloc[-2]) >= 0 and d * (ml.iloc[-1] - sg.iloc[-1]) < 0:
        send(f"🏁 {name} {label}: the 15m MACD line crossed its signal line, the trend is ending. This is your take-profit signal. Setup closed.")
        return False
    warns = exit_warnings(df, d)
    n_w = len(warns)
    aligned = h1_state == ("bull" if d == 1 else "bear")
    strength = h1_hist_state(df1h, d) if df1h is not None else ("strong" if aligned else "weak")
    exit_now = (not aligned) or strength == "weak"
    key = (n_w, exit_now)
    if n_w >= WARN_MIN and key != tuple(tr.get("last_warn", [])):
        listing = "\n".join(f"• {w}" for w in warns)
        why = []
        if strength == "weak":
            why.append("the 1H MACD histogram is weak or fading")
        if not aligned:
            why.append("1H is no longer Golden")
        if exit_now:
            msg = (f"🔴 {name} {label}: {n_w} exit warnings on 15m AND " + " and ".join(why) + f"\n{listing}\n"
                   f"Per your rule the trend is over. Your manual-exit conditions are met, review the trade now.")
        else:
            msg = (f"⚠️ {name} {label}: {n_w} exit warnings on 15m\n{listing}\n"
                   f"The 1H MACD histogram is still strong and 1H is still Golden, so per your rule this does not end the trade. Hold unless the SL is hit.")
        if send(msg):
            tr["last_warn"] = list(key)
    elif n_w < WARN_MIN:
        tr["last_warn"] = []
    return True


def scan_15m(state):
    h1, trades, last_sig = state["h1"], state["trades"], state["last_sig"]
    for name, ticker in SYMBOLS.items():
        try:
            if name in trades:
                df = get_df(ticker, "15m")
                if df is None:
                    continue
                fast = get_df(ticker, "1m")
                df1h = get_df(ticker, "1h")
                if not manage_trade(name, trades[name], df, h1.get(name, "none"), fast, df1h):
                    del trades[name]
                continue
            g = h1.get(name, "none")
            if g not in ("bull", "bear"):
                continue
            df = get_df(ticker, "15m")
            if df is None:
                print(f"{name} 15m: no data")
                continue
            sig, reason, r = entry_signal(df, 1 if g == "bull" else -1)
            print(f"{datetime.now():%H:%M} {name} 15m [{g}]: {reason}")
            if sig is None:
                continue
            if name in last_sig:
                gap = (pd.Timestamp(sig["time"]) - pd.Timestamp(last_sig[name])).total_seconds() / 900
                if gap < SIG_COOLDOWN:
                    continue
            d = 1 if g == "bull" else -1
            risk = abs(sig["entry"] - sig["sl"])
            rr = [abs(tp - sig["entry"]) / risk for tp in sig["tps"]]
            tp_lines = "\n".join(f"Target (trend-fib {TP_RATIOS[i]:g}): {fmt(tp)} (R:R {rr[i]:.1f})" for i, tp in enumerate(sig["tps"]))
            label = "LONG ▲" if d == 1 else "SHORT ▼"
            msg = (f"🎯 SETUP READY: {name} {label}\n"
                   f"1H Golden ✔ | 15m trend confirmed {sig['score']}/7 ✔ | pullback {sig['depth']:.0%} of the last leg{' (shallow, MACD histogram very strong)' if sig['shallow'] else ''} ✔\n"
                   f"No divergence ✔ | reversal candle closed over the LinReg line ✔\n\n"
                   f"Reference entry: {fmt(sig['entry'])}\n"
                   f"SL: {fmt(sig['sl'])}\n"
                   f"{tp_lines}\n"
                   f"Or take profit earlier if the 15m MACD line crosses its signal line (trend ending).\n\n"
                   f"Now: drop to the 1m and look for the hook at the level. Levels are already planned, "
                   f"if the SL is not hit, you stay in.")
            if send(msg):
                trades[name] = {"dir": d, "entry": sig["entry"], "sl": sig["sl"], "tps": sig["tps"],
                                "time": sig["time"], "tp_hit": 0, "last_warn": []}
                last_sig[name] = sig["time"]
        except Exception as e:
            print(f"{name} 15m: error {e}")


def check_all():
    state = load_state()
    h1 = state["h1"]
    for name, ticker in SYMBOLS.items():
        try:
            df = get_df(ticker, "1h")
            if df is None:
                print(f"{name}: no usable data, skipped")
                continue
            r = evaluate(df)
            new = golden_state(r)
            old = h1.get(name, "none")
            print(f"{datetime.now():%H:%M} {name}: {new}  {r}")
            if new != old:
                if new in ("bull", "bear"):
                    arrow = "▲ BULLISH" if new == "bull" else "▼ BEARISH"
                    pushes = r["push_up"] if new == "bull" else r["push_dn"]
                    delivered = send(f"🟡 {name} is GOLDEN {arrow} on 1H\nAll 7 conditions aligned. Pushes: {pushes}/{MAX_PUSH}\n"
                                     f"I'll now watch the 15m and message you only when a full setup is ready.")
                    if delivered:
                        h1[name] = new
                    else:
                        print(f"{name}: alert NOT delivered, will retry next run")
                else:
                    if NOTIFY_LOST and old in ("bull", "bear"):
                        send(f"⚪ {name} is no longer Golden on 1H (was {old}).")
                    h1[name] = new
        except Exception as e:  # keep going if one symbol fails
            print(f"{name}: error {e}")
    if ENABLE_15M:
        scan_15m(state)
    save_state(state)


def status_report():
    """Shows, per symbol, which of the 7 rules pass and what blocks Golden. Also sent to Telegram."""
    def arrow(d):
        return "▲" if d == 1 else "▼" if d == -1 else "–"

    lines = []
    for name, ticker in SYMBOLS.items():
        try:
            df = None
            for tk in (ticker if isinstance(ticker, list) else [ticker]):
                try:
                    cand = fetch_1h(tk)
                except Exception:
                    continue
                if len(cand) >= 200:
                    df = cand
                    break
            if df is None:
                lines.append(f"{name}: no data")
                continue
            r = evaluate(df)
            blockers = []
            if r["push_up"] >= MAX_PUSH or r["push_dn"] >= MAX_PUSH:
                blockers.append(f"pushes {r['push_up']}/{r['push_dn']}")
            if r["stall"] != 0:
                blockers.append("stall")
            if USE_DIV_FILTER and (r["div_bear"] or r["div_bull"]):
                blockers.append("divergence")
            lines.append(
                f"{name}: {golden_state(r).upper()} | Str{arrow(r['Structure'])} MACD{arrow(r['MACD'])} "
                f"RTI{arrow(r['RTI'])} Osc{arrow(r['OscAlign'])} ST{arrow(r['Supertrend'])} "
                f"UT{arrow(r['UTBot'])} LR{arrow(r['LinReg'])}"
                + (f" | blocked by: {', '.join(blockers)}" if blockers else "")
            )
        except Exception as e:
            lines.append(f"{name}: error {e}")
    text = "TGA status (1H, last closed candle)\n" + "\n".join(lines)
    print(text)
    send(text)


def sleep_until_next_quarter():
    now = datetime.now(timezone.utc)
    minute = (now.minute // 15 + 1) * 15
    nxt = now.replace(minute=0, second=45, microsecond=0) + timedelta(minutes=minute)
    time.sleep(max(30, (nxt - now).total_seconds()))


if __name__ == "__main__":
    if "--test" in sys.argv:
        send("✅ TGA bot test message. Telegram is working.")
    elif "--status" in sys.argv:
        status_report()
    elif "--once" in sys.argv:
        check_all()
    else:
        while True:
            check_all()
            sleep_until_next_quarter()
