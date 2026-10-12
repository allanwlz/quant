"""Evaluate trend algorithms on a market universe or on random stand-in stock pools.

``universe`` mode holds every in-universe security with a signal (pure timing value,
hundreds of names). ``pools`` mode mimics the real workflow: a 20-50 name pool that
is refreshed every two weeks, here drawn at random so it carries no selection skill;
repeated draws give a distribution instead of one lucky path.
"""
import itertools
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import backtest, panel as panel_mod, signals, universe as universe_mod

WARMUP_DAYS = 400


def prepare(con, market, start, end):
    """Load with a warm-up margin so long lookbacks are ready on ``start``."""
    p = panel_mod.load(con, market, start - timedelta(days=WARMUP_DAYS), end)
    return p, universe_mod.build(p, con)


def random_pools(universe, size, rng, refresh=10, turnover=0.2):
    """Bool [T, N]: a pool of ``size`` names, re-checked every ``refresh`` days; members
    that left the universe plus a ``turnover`` share of the rest are replaced at random."""
    T, N = universe.shape
    pool = np.zeros(N, dtype=bool)
    out = np.zeros((T, N), dtype=bool)
    for t in range(T):
        if t % refresh == 0:
            pool &= universe[t]
            members = np.flatnonzero(pool)
            drop = rng.choice(members, int(len(members) * turnover), replace=False) if len(members) else members
            pool[drop] = False
            free = np.flatnonzero(universe[t] & ~pool)
            add = min(size - pool.sum(), len(free))
            if add > 0:
                pool[rng.choice(free, add, replace=False)] = True
        out[t] = pool
    return out


def evaluate(p, universe, algo, params, start, mode='universe', pools=20, pool_size=40, seed=0,
             atr_stop=None, cfg=backtest.Config()):
    hold = signals.compute(p, algo, atr_stop=atr_stop, **params)
    first = np.searchsorted(p.dates, pd.Timestamp(start))
    hold[:first] = False                        # start in cash on ``start``
    if mode == 'universe':
        result = backtest.run(p, hold, universe, cfg)
        return backtest.metrics(result, start), result
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(pools):
        pool = random_pools(universe, pool_size, rng) & universe
        rows.append(backtest.metrics(backtest.run(p, hold, pool, cfg), start))
    frame = pd.DataFrame(rows)
    summary = {f'{k}_p50': v for k, v in frame.median().items()}
    summary.update({f'{k}_p10': frame[k].quantile(0.1) for k in ('cagr', 'max_dd', 'calmar')})
    summary.update({f'{k}_p90': frame[k].quantile(0.9) for k in ('cagr', 'max_dd', 'calmar')})
    return summary, frame


def grid(spec):
    """{'entry': [20, 55], 'exit': [10, 20]} -> list of param dicts."""
    keys = list(spec)
    return [dict(zip(keys, values)) for values in itertools.product(*(spec[k] for k in keys))]


def sweep(p, universe, algo, spec, start, **kwargs):
    rows = []
    for params in grid(spec):
        summary, _ = evaluate(p, universe, algo, params, start, **kwargs)
        rows.append({**params, **summary})
    return pd.DataFrame(rows)
