"""Market-level timing: a ``scale`` series in [0, 1] for the whole book, from one index.

Each rule has one or two parameters and reads only data up to the decision day's close
(the engine trades on the next open).

- ``ma``: hold when the index closes above its n-day moving average, checked at month end
  (Faber's monthly rule; daily checks whipsaw around the line).
- ``mom``: hold when the index is up over the last n days, checked at month end.
- ``vol``: scale = target / realized vol over n days, capped at 1 and rounded to quarters;
  target = k x the expanding median of that realized vol, so no level is fitted in hindsight.
- ``ma+vol``: product of the two.
- ``value`` (CN): market earnings yield minus the cash rate, ranked against its own history;
  scale 0.5 / 0.75 / 1 in its bottom / middle / top third, checked at month end.
- ``crowd`` (CN): 20-day free-float turnover of the whole market; scale 0.5 while it sits
  above the ``q`` percentile of its own history, else 1.
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


def cash_rate(con, market, dates):
    """Annual cash rate as a decimal on each date (forward-filled; 0 before the first quote)."""
    rows = con.execute('SELECT date, rate FROM rates WHERE market=? ORDER BY date', [market]).df()
    r = rows.set_index(pd.to_datetime(rows['date']))['rate'] / 100
    return r.reindex(r.index.union(dates)).ffill().reindex(dates).fillna(0.0).to_numpy()


def cn_market(con, dates, code='000300.SH'):
    """Index earnings yield (1 / PE TTM) and 20-day free-float turnover, plus the cash rate,
    with history back to 2005 so percentiles are not dominated by one episode."""
    rows = con.execute('SELECT date, 1 / pe_ttm AS ep, turnover_rate_f AS turnover FROM index_basic WHERE code=? ORDER BY date',
                       [code]).df()
    frame = rows.set_index(pd.to_datetime(rows['date']))[['ep', 'turnover']]
    frame['turnover'] = frame['turnover'].rolling(20, min_periods=20).mean()
    rates = con.execute("SELECT date, rate / 100 AS cash FROM rates WHERE market='CN' ORDER BY date").df()
    frame = frame.join(rates.set_index(pd.to_datetime(rates['date']))['cash'], how='left').ffill()
    frame['ep_gap'] = frame['ep'] - frame['cash']
    frame['value_pct'] = _pct(frame['ep_gap'].dropna()).reindex(frame.index)   # point-in-time rank over all history since 2005
    frame['heat_pct'] = _pct(frame['turnover'].dropna()).reindex(frame.index)
    return frame.reindex(frame.index.union(dates)).ffill().reindex(dates)


def _pct(series, min_periods=500):
    """Expanding percentile of today's value within its own history (no look-ahead)."""
    return series.expanding(min_periods=min_periods).rank(pct=True)


def _month_end(raw):
    """Hold each month-end reading until the next month end."""
    idx = raw.index
    last = idx.to_series().groupby(idx.to_period('M')).transform('max') == idx
    return raw.where(last).ffill().fillna(1.0)


def rule(px, kind, n=200, k=1.0, vol_n=20, q=0.9, market=None):
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
    if kind == 'value':
        pct = market['value_pct']
        return _month_end(pd.Series(np.select([pct < 1 / 3, pct < 2 / 3], [0.5, 0.75], 1.0), px.index).where(pct.notna()))
    if kind == 'crowd':
        return pd.Series(np.where(market['heat_pct'] > q, 0.5, 1.0), px.index)
    if kind == 'ma+vol':
        return rule(px, 'ma', n=n) * rule(px, 'vol', n=vol_n, k=k)
    raise ValueError(kind)


RULES = ([('none', {})]
         + [('ma', {'n': n}) for n in (100, 150, 200, 250)]
         + [('mom', {'n': n}) for n in (120, 250)]
         + [('vol', {'n': n, 'k': k}) for n, k in itertools.product((20, 60), (0.8, 1.0, 1.2))]
         + [('ma+vol', {'n': 200, 'vol_n': 20, 'k': 1.0})])


CN_RULES = [('value', {})] + [('crowd', {'q': q}) for q in (0.8, 0.9, 0.95)]


def _periods(start, end):
    return {'all': (start, end), 'is': (start, pd.Timestamp(HOLDOUT) - pd.Timedelta(days=1)), 'oos': (HOLDOUT, end)}


def evaluate(p, universe, algo, params, scale, start, mode, pools=20, pool_size=40, seed=0, cfg=backtest.Config(initial=1e8),
             cash=None):
    """Metrics per period for every pool (or the one universe run), same pools across rules."""
    hold = signals.compute(p, algo, **params)
    first = np.searchsorted(p.dates, pd.Timestamp(start))
    hold[:first] = False
    runs = []
    if mode == 'universe':
        runs.append(backtest.run(p, hold & universe if algo == 'buy_hold' else hold, universe, cfg, scale, cash))
    else:
        rng = np.random.default_rng(seed)
        for _ in range(pools):
            pool = research.random_pools(universe, pool_size, rng) & universe
            runs.append(backtest.run(p, (pool & hold) if algo == 'buy_hold' else hold, pool, cfg, scale, cash))
    rows = []
    for i, result in enumerate(runs):
        for period, (lo, hi) in _periods(pd.Timestamp(start), p.dates[-1]).items():
            m = backtest.metrics(result, lo, hi)
            if cash is not None:   # Sharpe on return over cash, so sitting out is not flattered by low vol
                inside = (p.dates >= pd.Timestamp(lo)) & (p.dates <= pd.Timestamp(hi))
                m['xsharpe'] = (m['cagr'] - cash[inside].mean()) / m['vol'] if m['vol'] else np.nan
            rows.append({'pool': i, 'period': period, **m})
    return pd.DataFrame(rows)


def index_backtest(tr, scale, cash, cost):
    """Hold ``scale`` of a total-return index and the rest in cash. The scale decided at
    t's close earns t+1's return; each unit of change in scale pays ``cost``."""
    r = tr.pct_change().fillna(0).to_numpy()
    s = np.concatenate([[0.0], scale[:-1]])                 # yesterday's decision
    days = np.diff(tr.index.values).astype('timedelta64[D]').astype(float)
    c = np.concatenate([[0.0], cash[:-1] * days / 365])
    ret = s * r + (1 - s) * c - np.abs(np.diff(s, prepend=0.0)) * cost
    return pd.Series(np.cumprod(1 + ret), tr.index), pd.Series(c, tr.index)


def nav_metrics(nav, cash, lo, hi):
    inside = (nav.index >= pd.Timestamp(lo)) & (nav.index <= pd.Timestamp(hi))
    nav, cash = nav[inside], cash[inside]
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    vol = nav.pct_change().std() * np.sqrt(252)
    cash_cagr = np.prod(1 + cash.iloc[1:]) ** (1 / years) - 1
    return {'cagr': cagr, 'max_dd': (nav / nav.cummax() - 1).min(), 'xsharpe': (cagr - cash_cagr) / vol}
