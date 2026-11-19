"""
TGA Golden Alignment alert bot
------------------------------
Checks the 1H "sign" for each symbol and sends a Telegram message
when the market turns Golden (all 7 conditions aligned, pushes < 3).

Setup:
    pip install yfinance pandas numpy requests
    set environment variables TG_TOKEN and TG_CHAT_ID  (see README steps in chat)
Run:
    python tga_alert_bot.py          # loops forever, checks every hour
    python tga_alert_bot.py --once   # single check (for cron / task scheduler)
    python tga_alert_bot.py --test   # send a test Telegram message and exit
"""
import json
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
RTI_FAST, RTI_SLOW = 20, 50   # set the same two numbers in the Pine inputs
STATE_FILE = "tga_state.json"
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


def evaluate(df):
    """Returns dict of direction ints (+1 / -1 / 0) for the 7 rules plus pushes."""
    close = df["close"]
    macd_line = ema(close, 12) - ema(close, 26)
    rti = (ema(close, RTI_FAST) - ema(close, RTI_SLOW)) / atr(df, 14)
    rti_sig = ema(rti, 9)
    rti_dir = 1 if (rti.iloc[-1] > 0 and rti.iloc[-1] > rti_sig.iloc[-1]) else -1 if (rti.iloc[-1] < 0 and rti.iloc[-1] < rti_sig.iloc[-1]) else 0
    osc = ema(rsi(close, 14), 3) - 50
    osc_sig = ema(osc, 9)
    if osc.iloc[-1] > osc_sig.iloc[-1] and close.iloc[-1] > close.iloc[-4]:
        align = 1
    elif osc.iloc[-1] < osc_sig.iloc[-1] and close.iloc[-1] < close.iloc[-4]:
        align = -1
    else:
        align = 0
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
    div_bear, div_bull = divergence_active(df, [rti, macd_hist, osc], PIV_LEN, DIV_LOOK)
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
def fetch_1h(ticker):
    import yfinance as yf

    df = yf.download(ticker, period="60d", interval="1h", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    # drop the candle that is still forming
    if len(df) and df.index[-1] + timedelta(hours=1) > pd.Timestamp.now(tz=timezone.utc):
        df = df.iloc[:-1]
    return df


def send(msg):
    if not TG_TOKEN or not TG_CHAT_ID:
        print("[no Telegram configured]", msg)
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            data={"chat_id": TG_CHAT_ID, "text": msg},
            timeout=15,
        )
        if not r.ok:
            print("Telegram error:", r.text)
    except requests.RequestException as e:
        print("Telegram request failed:", e)


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f)


def check_all():
    state = load_state()
    for name, ticker in SYMBOLS.items():
        try:
            df = None
            for tk in (ticker if isinstance(ticker, list) else [ticker]):
                try:
                    cand = fetch_1h(tk)
                except Exception as e:
                    print(f"{name}: {tk} failed ({e})")
                    continue
                if len(cand) >= 120:
                    df = cand
                    break
                print(f"{name}: {tk} has only {len(cand)} bars, trying next")
            if df is None:
                print(f"{name}: no usable data, skipped")
                continue
            r = evaluate(df)
            new = golden_state(r)
            old = state.get(name, "none")
            print(f"{datetime.now():%H:%M} {name}: {new}  {r}")
            if new != old:
                if new in ("bull", "bear"):
                    arrow = "▲ BULLISH" if new == "bull" else "▼ BEARISH"
                    pushes = r["push_up"] if new == "bull" else r["push_dn"]
                    send(f"🟡 {name} is GOLDEN {arrow} on 1H\nAll 7 conditions aligned. Pushes: {pushes}/{MAX_PUSH}\n"
                         f"Next: check the 15m structure and wait for a fib pullback, reversal, then continuation.")
                elif NOTIFY_LOST and old in ("bull", "bear"):
                    send(f"⚪ {name} is no longer Golden on 1H (was {old}).")
                state[name] = new
        except Exception as e:  # keep going if one symbol fails
            print(f"{name}: error {e}")
    save_state(state)


def sleep_until_next_hour():
    now = datetime.now(timezone.utc)
    nxt = (now + timedelta(hours=1)).replace(minute=1, second=30, microsecond=0)
    time.sleep(max(30, (nxt - now).total_seconds()))


if __name__ == "__main__":
    if "--test" in sys.argv:
        send("✅ TGA bot test message. Telegram is working.")
    elif "--once" in sys.argv:
        check_all()
    else:
        while True:
            check_all()
            sleep_until_next_hour()
