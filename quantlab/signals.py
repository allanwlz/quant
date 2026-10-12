"""Trend signals: each algorithm maps a Panel to a bool hold matrix [T, N].

``hold[t]`` uses data up to and including the close of day t; the backtester
trades on it at the open of t+1. Prices are back-adjusted and forward-filled.
"""
import numpy as np
import pandas as pd


def _roll(arr, n, how):
    return getattr(pd.DataFrame(arr).rolling(n, min_periods=n), how)().to_numpy(dtype=np.float32)


def _prev(arr):
    out = np.full_like(arr, np.nan)
    out[1:] = arr[:-1]
    return out


def latch(entry, exit):
    """Enter on ``entry``, stay in until ``exit`` (exit wins on the same day)."""
    out = np.zeros(entry.shape, dtype=bool)
    state = np.zeros(entry.shape[1], dtype=bool)
    for t in range(len(entry)):
        state = np.where(state, ~exit[t], entry[t] & ~exit[t])
        out[t] = state
    return out


def atr(panel, n=20):
    high, low, close = panel['adj_high'], panel['adj_low'], panel['adj_close']
    prev = _prev(close)
    tr = np.fmax(high - low, np.fmax(np.abs(high - prev), np.abs(low - prev)))
    return _roll(tr, n, 'mean')


def trailing_stop(panel, hold, k, n=20):
    """Chandelier exit overlay: leave when close < highest close since entry - k*ATR;
    re-entry needs the base signal to switch off and on again."""
    close, band = panel['adj_close'], k * atr(panel, n)
    out = np.zeros(hold.shape, dtype=bool)
    state = np.zeros(hold.shape[1], dtype=bool)
    stopped = np.zeros_like(state)
    peak = np.zeros(hold.shape[1], dtype=np.float32)
    for t in range(len(hold)):
        stopped &= hold[t]                       # base signal off → may re-enter later
        enter = hold[t] & ~state & ~stopped
        peak = np.where(enter, close[t], np.where(state, np.fmax(peak, close[t]), peak))
        state = (state | enter) & hold[t]
        hit = state & (close[t] < peak - band[t])
        stopped |= hit
        state &= ~hit
        out[t] = state
    return out


def rolling_slope_r2(logp, n, chunk=800):
    """Least-squares slope (per bar) and R² of ``logp`` over trailing windows of n bars."""
    T, N = logp.shape
    slope = np.full((T, N), np.nan, dtype=np.float32)
    r2 = np.full((T, N), np.nan, dtype=np.float32)
    x = np.arange(T, dtype=np.float64)[:, None]
    sx, sxx = n * (n - 1) / 2, (n - 1) * n * (2 * n - 1) / 6
    varx = n * sxx - sx ** 2

    def rsum(a):
        c = np.cumsum(a, axis=0)
        c[n:] = c[n:] - c[:-n]
        return c

    for lo in range(0, N, chunk):  # chunked to stay within memory
        y = logp[:, lo:lo + chunk].astype(np.float64)
        ok = ~np.isnan(y)
        y0 = np.where(ok, y, 0.0)
        cnt, sy, sxy, syy = rsum(ok.astype(np.float64)), rsum(y0), rsum(x * y0), rsum(y0 * y0)
        cov = n * (sxy - (x - n + 1) * sy) - sx * sy
        vary = n * syy - sy ** 2
        full = cnt >= n
        with np.errstate(divide='ignore', invalid='ignore'):
            slope[:, lo:lo + chunk] = np.where(full, cov / varx, np.nan)
            r2[:, lo:lo + chunk] = np.where(full & (vary > 0), cov ** 2 / (varx * vary), np.nan)
    return slope, r2


# --- algorithms -------------------------------------------------------------

def buy_hold(panel):
    return np.ones(panel.shape, dtype=bool)


def above_ma(panel, n=200):
    close = panel['adj_close']
    return close > _roll(close, n, 'mean')


def ma_cross(panel, fast=50, slow=200):
    close = panel['adj_close']
    return _roll(close, fast, 'mean') > _roll(close, slow, 'mean')


def donchian(panel, entry=55, exit=20):
    close = panel['adj_close']
    upper = _prev(_roll(panel['adj_high'], entry, 'max'))
    lower = _prev(_roll(panel['adj_low'], exit, 'min'))
    return latch(close > upper, close < lower)


def tsmom(panel, lookbacks=(20, 60, 120, 250)):
    close = panel['adj_close']
    votes = np.zeros(panel.shape, dtype=np.int16)
    for n in lookbacks:
        past = np.full_like(close, np.nan)
        past[n:] = close[:-n]
        votes += close > past
    return votes * 2 > len(lookbacks)


def slope_r2(panel, n=90, min_slope=0.2, min_r2=0.5):
    """Smooth uptrend: annualised log-slope above ``min_slope`` with fit R² above ``min_r2``."""
    slope, r2 = rolling_slope_r2(np.log(panel['adj_close']), n)
    return (slope * 252 > min_slope) & (r2 > min_r2)


def volume_breakout(panel, n=55, vol_mult=1.5, exit_ma=20):
    """New n-day high on volume above ``vol_mult`` x its n-day average; exit below the exit MA."""
    close, volume = panel['adj_close'], np.nan_to_num(panel['volume'], nan=0.0)
    upper = _prev(_roll(panel['adj_high'], n, 'max'))
    loud = volume > vol_mult * _prev(_roll(volume, n, 'mean'))
    return latch((close > upper) & loud, close < _roll(close, exit_ma, 'mean'))


ALGOS = {f.__name__: f for f in (buy_hold, above_ma, ma_cross, donchian, tsmom, slope_r2, volume_breakout)}


def compute(panel, algo, atr_stop=None, **params):
    hold = ALGOS[algo](panel, **params)
    if atr_stop:
        hold = trailing_stop(panel, hold, atr_stop)
    return hold
