"""Index-level timing rules on long histories (CSI 300 from 2006, SPY from 1994).

Usage: python scripts/index_timing_long.py CN|US <duckdb path> <out.csv>
"""
import sys, duckdb, numpy as np, pandas as pd
from quantlab import timing
market = sys.argv[1]
db_path = sys.argv[2]
out = sys.argv[3]
con = duckdb.connect(db_path, read_only=True)
if market == 'CN':
    sig_code, tr_code, start, cost = '000300.SH', 'H00300.CSI', '2006-01-01', 0.001
    periods = {'2006-2015': ('2006-01-01', '2015-12-31'), '2016-2023': ('2016-01-01', '2023-12-31'), '2024-': ('2024-01-01', '2026-12-31'), 'all': (start, '2026-12-31')}
else:
    sig_code, tr_code, start, cost = 'SPY.US', 'SPY.US', '1994-02-01', 0.0005
    periods = {'1994-2015': ('1994-02-01', '2015-12-31'), '2016-2023': ('2016-01-01', '2023-12-31'), '2024-': ('2024-01-01', '2026-12-31'), 'all': (start, '2026-12-31')}
dates = pd.DatetimeIndex([r[0] for r in con.execute('SELECT date FROM bars WHERE code=? ORDER BY date', [tr_code]).fetchall()])
px = timing.index_close(con, sig_code, dates)
tr = timing.index_close(con, tr_code, dates)
cash = timing.cash_rate(con, market, dates).copy()
first = np.flatnonzero(cash > 0)[0]
cash[:first] = cash[first]                                    # back-fill before the first quote
rules = list(timing.RULES)
market_frame = None
if market == 'CN':
    market_frame = timing.cn_market(con, dates)
    rules += timing.CN_RULES
rows = []
for kind, rp in rules:
    scale = timing.rule(px, kind, market=market_frame, **rp)
    scale[scale.index < pd.Timestamp(start)] = 0.0             # warm-up in cash for every rule
    nav, c = timing.index_backtest(tr, scale.to_numpy(), cash, cost)
    live = scale[scale.index >= pd.Timestamp(start)]
    for name, (lo, hi) in periods.items():
        rows.append({'rule': kind + (str(rp).replace("'", '') if rp else ''), 'period': name, **timing.nav_metrics(nav, c, lo, hi),
                     'avg_scale': live[(live.index >= lo) & (live.index <= hi)].mean(),
                     'switches_per_year': (live.diff().abs() > 0).sum() / ((live.index[-1] - live.index[0]).days / 365.25)})
d = pd.DataFrame(rows)
d.to_csv(out, index=False)
w = d.pivot_table(index='rule', columns='period', values=['cagr', 'max_dd', 'xsharpe'], sort=False)
w = w.reindex(columns=[(m, p) for m in ('cagr', 'max_dd', 'xsharpe') for p in periods])
w['scale'] = d[d.period == 'all'].set_index('rule')['avg_scale']
w['sw/yr'] = d[d.period == 'all'].set_index('rule')['switches_per_year']
pd.set_option('display.width', 250)
print(w.round(3).to_string())
