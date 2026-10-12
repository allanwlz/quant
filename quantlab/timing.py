"""Market-level timing: a ``scale`` series in [0, 1] for the whole book, from one index.

Each rule has one or two parameters and reads only data up to the decision day's close
(the engine trades on the next open).

- ``ma``: hold when the index closes above its n-day moving average, checked at month end
  (Faber's monthly rule; daily checks whipsaw around the line).
- ``mom``: hold when the index is up over the last n days, checked at month end.
- ``vol``: scale = target / realized vol over n days, capped at 1 and rounded to quarters;
  target = k x the expanding median of that realized vol, so no level is fitted in hindsight.
- ``ma+vol``: product of the two.
"""
import itertools

import numpy as np
import pandas as pd

from . import backtest, research, signals

BENCHMARK = {'CN': '000300.SH', 'US': 'SPY.US', 'HK': '02800.HK'}
HOLDOUT = '2024-01-01'          # sealed out-of-sample start; never used to pick parameters
STEP = 0.25                     # vol scale moves in quarters to keep manual orders sane


def index_close(con, code, dates):
    rows = con.execute('SELECT date, close * coalesce(adj_factor, 1) AS px FROM bars WHERE code=? ORDER BY date',
                       [code]).df()
    px = rows.set_index(pd.to_datetime(rows['date']))['px']
    return px.reindex(px.index.union(dates)).ffill().reindex(dates)


def _month_end(raw):
    """Hold each month-end reading until the next month end."""
    idx = raw.index
    last = idx.to_series().groupby(idx.to_period('M')).transform('max') == idx
    return raw.where(last).ffill().fillna(1.0)


def rule(px, kind, n=200, k=1.0, vol_n=20):
    if kind == 'none':
        return pd.Series(1.0, px.index)
    if kind == 'ma':
        ma = px.rolling(n, min_periods=n).mean()
        return _month_end((px > ma).astype(float).where(ma.notna()))
    if kind == 'mom':
        past = px.shift(n)
        return _month_end((px > past).astype(float).where(past.notna()))
    if kind == 'vol':
        sigma = np.log(px).diff().rolling(n, min_periods=n).std() * np.sqrt(252)
        target = k * sigma.expanding(min_periods=250).median()
        raw = (target / sigma).clip(upper=1.0)
        return (np.floor(raw / STEP) * STEP).fillna(1.0)
    if kind == 'ma+vol':
        return rule(px, 'ma', n=n) * rule(px, 'vol', n=vol_n, k=k)
    raise ValueError(kind)


RULES = ([('none', {})]
         + [('ma', {'n': n}) for n in (100, 150, 200, 250)]
         + [('mom', {'n': n}) for n in (120, 250)]
         + [('vol', {'n': n, 'k': k}) for n, k in itertools.product((20, 60), (0.8, 1.0, 1.2))]
         + [('ma+vol', {'n': 200, 'vol_n': 20, 'k': 1.0})])


def _periods(start, end):
    return {'all': (start, end), 'is': (start, pd.Timestamp(HOLDOUT) - pd.Timedelta(days=1)), 'oos': (HOLDOUT, end)}


def evaluate(p, universe, algo, params, scale, start, mode, pools=20, pool_size=40, seed=0, cfg=backtest.Config(initial=1e8)):
    """Metrics per period for every pool (or the one universe run), same pools across rules."""
    hold = signals.compute(p, algo, **params)
    first = np.searchsorted(p.dates, pd.Timestamp(start))
    hold[:first] = False
    runs = []
    if mode == 'universe':
        runs.append(backtest.run(p, hold & universe if algo == 'buy_hold' else hold, universe, cfg, scale))
    else:
        rng = np.random.default_rng(seed)
        for _ in range(pools):
            pool = research.random_pools(universe, pool_size, rng) & universe
            runs.append(backtest.run(p, (pool & hold) if algo == 'buy_hold' else hold, pool, cfg, scale))
    rows = []
    for i, result in enumerate(runs):
        for period, (lo, hi) in _periods(pd.Timestamp(start), p.dates[-1]).items():
            rows.append({'pool': i, 'period': period, **backtest.metrics(result, lo, hi)})
    return pd.DataFrame(rows)
