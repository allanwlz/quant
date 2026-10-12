"""Daily portfolio simulator for one market.

Sizing (docs/discussion.md): every security with a hold signal that may be entered
gets ``base_weight``; when their sum exceeds the room left under ``gross`` all are
scaled down pro rata, so cash is held when few signals are on. Holdings whose
signal is on but which left the entry set are frozen (no adds, no trims) and wait
for the signal to exit. Existing positions are only resized when the gap exceeds
both ``band_rel`` of the target and ``band_abs`` of NAV.

Market timing: an optional ``scale`` series in [0, 1] multiplies every target. When it
changes, all candidates are resized regardless of the band, and when it falls, frozen
holdings are trimmed by the same ratio (they are never added to).

Timing: decide on day t's close, trade at day t+1's open with slippage. CN buys
are blocked when t+1 opens at the up limit, sells when it opens at the down limit;
suspended securities cannot trade. Positions are tracked in back-adjusted units,
so splits are neutral and dividends are reinvested.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import markets
from .panel import ffill


@dataclass(frozen=True)
class Config:
    base_weight: float = 0.05
    gross: float = 1.0
    band_rel: float = 0.30
    band_abs: float = 0.015
    initial: float = 1_000_000.0


@dataclass
class Result:
    nav: pd.Series
    exposure: pd.Series
    positions: pd.Series
    trades: pd.DataFrame


def _costs(value, rate, minimum, extra):
    return np.where(value > 0, np.maximum(value * rate, minimum), 0.0) + value * extra


def run(panel, hold, entry_ok, cfg=Config(), scale=None):
    T, N = panel.shape
    rules = markets.RULES[panel.market]
    days = panel.dates.date
    adj, price = panel['adj'].astype(np.float64), np.nan_to_num(panel['adj_close'].astype(np.float64))
    open_raw, ref_raw = panel['open'].astype(np.float64), ffill(panel['close']).astype(np.float64)
    valid = panel['valid']
    lot = np.where(np.isnan(panel.lot), rules.lot, panel.lot)
    if panel.market == 'CN':
        up, down = panel['up_limit'], panel['down_limit']
    scale = np.ones(T) if scale is None else np.asarray(scale, dtype=np.float64)
    applied = scale[0]
    q = np.zeros(N)                    # holdings in back-adjusted units
    cash = cfg.initial
    nav, exposure, npos, trades = np.empty(T), np.empty(T), np.empty(T, dtype=np.int32), []

    for t in range(T):
        value = q * price[t]
        total = cash + value.sum()
        nav[t], exposure[t], npos[t] = total, value.sum() / total, (q > 0).sum()
        if t == T - 1:
            break

        held, h = q > 0, hold[t]
        exits = held & ~h
        frozen = held & h & ~entry_ok[t]
        cand = h & entry_ok[t]
        target = np.zeros(N)
        n = cand.sum()
        if n:
            room = max(0.0, cfg.gross - value[frozen].sum() / total)
            target[cand] = cfg.base_weight * min(1.0, room / (cfg.base_weight * n)) * scale[t]
        gap = target - value / total
        changed = scale[t] != applied
        resize = cand & (~held | changed | (np.abs(gap) > np.maximum(cfg.band_rel * target, cfg.band_abs)))
        if changed and scale[t] < applied:
            gap[frozen] = value[frozen] / total * (scale[t] / applied - 1)
            resize |= frozen
        applied = scale[t]
        if not (resize.any() or exits.any()):
            continue

        k = t + 1
        o, a = open_raw[k], adj[k]
        tradable = valid[k] & (o > 0)
        can_buy, can_sell = tradable.copy(), tradable.copy()
        if panel.market == 'CN':
            can_buy &= ~(o >= up[k] - 1e-6)
            can_sell &= ~(o <= down[k] + 1e-6)
        with np.errstate(invalid='ignore', divide='ignore'):
            shares = np.where(resize, np.trunc(gap * total / ref_raw[t] / lot) * lot, 0.0)
        shares[exits] = -q[exits] * a[exits]
        shares = np.nan_to_num(shares)

        sell = (shares < 0) & can_sell
        if sell.any():
            px = o[sell] * (1 - rules.slippage)
            gross_value = -shares[sell] * px
            cost = _costs(gross_value, rules.commission, rules.commission_min,
                          markets.stamp_sell(panel.market, days[k]) + rules.fees)
            cash += (gross_value - cost).sum()
            q[sell] = np.where(exits[sell], 0.0, q[sell] + shares[sell] / a[sell])
            trades.append((k, np.flatnonzero(sell), shares[sell], px, cost))

        buy = (shares > 0) & can_buy
        if buy.any():
            px = o[buy] * (1 + rules.slippage)
            extra = markets.stamp_buy(panel.market, days[k]) + rules.fees
            qty = shares[buy]
            need = qty * px + _costs(qty * px, rules.commission, rules.commission_min, extra)
            if need.sum() > cash:  # not enough cash: scale buys down, whole lots only
                qty = np.floor(qty * max(0.0, cash) / need.sum() / lot[buy]) * lot[buy]
                need = qty * px + _costs(qty * px, rules.commission, rules.commission_min, extra)
            ok = qty > 0
            cash -= need[ok].sum()
            idx = np.flatnonzero(buy)[ok]
            q[idx] += qty[ok] / a[idx]
            trades.append((k, idx, qty[ok], px[ok], need[ok] - qty[ok] * px[ok]))

    rows = [(panel.dates[k], panel.codes[j], s, p, s * p, c)
            for k, idx, sh, px, cost in trades for j, s, p, c in zip(idx, sh, px, cost)]
    trade_frame = pd.DataFrame(rows, columns=['date', 'code', 'shares', 'price', 'value', 'cost'])
    index = panel.dates
    return Result(pd.Series(nav, index), pd.Series(exposure, index), pd.Series(npos, index), trade_frame)


def metrics(result, start=None, end=None):
    lo = pd.Timestamp(start) if start is not None else result.nav.index[0]
    hi = pd.Timestamp(end) if end is not None else result.nav.index[-1]
    nav = result.nav[(result.nav.index >= lo) & (result.nav.index <= hi)]
    trades = result.trades[(result.trades['date'] >= lo) & (result.trades['date'] <= hi)]
    years = max(len(nav) - 1, 1) / 252
    returns = nav.pct_change().dropna()
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    vol = returns.std() * np.sqrt(252)
    drawdown = (nav / nav.cummax() - 1).min()
    positions = result.positions.reindex(nav.index)
    entries = int((trades['shares'] > 0).sum())  # buys incl. adds; upper bound on new positions
    return {
        'cagr': cagr, 'vol': vol, 'sharpe': cagr / vol if vol else np.nan, 'max_dd': drawdown,
        'calmar': cagr / -drawdown if drawdown else np.nan,
        'exposure': result.exposure.reindex(nav.index).mean(),
        'positions': positions.mean(),
        'turnover': trades['value'].abs().sum() / 2 / nav.mean() / years,   # one-way, per year
        'trades_per_month': len(trades) / (years * 12),
        'cost_drag': trades['cost'].sum() / nav.mean() / years,
        'hold_days': positions.sum() / entries if entries else np.nan,
    }
