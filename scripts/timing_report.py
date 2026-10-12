"""Summarise a quant timing result: medians per base x rule x period, win rate vs no timing.

Usage: python scripts/timing_report.py test/results/<dir>/timing.csv
"""
import sys, pandas as pd
d = pd.read_csv(sys.argv[1])
key = ['base', 'pool', 'period']
none = d[d.rule == 'none'].set_index(key)[['cagr', 'max_dd', 'calmar', 'sharpe', 'xsharpe']]
d = d.join(none, on=key, rsuffix='_0')
d['win_sharpe'] = d.xsharpe > d.xsharpe_0
d['d_cagr'] = d.cagr - d.cagr_0
d['d_dd'] = d.max_dd - d.max_dd_0
g = d.groupby(['base', 'rule', 'period'], sort=False)
s = g.agg(cagr=('cagr', 'median'), dd=('max_dd', 'median'), sharpe=('xsharpe', 'median'), calmar=('calmar', 'median'),
          d_cagr=('d_cagr', 'median'), d_dd=('d_dd', 'median'), win=('win_sharpe', 'mean'),
          tpm=('trades_per_month', 'median'), scale=('avg_scale', 'first')).reset_index()
s['rule'] = s.rule.str.replace('"', '')
w = s.pivot_table(index=['base', 'rule'], columns='period', values=['cagr', 'dd', 'sharpe', 'd_cagr', 'd_dd', 'win'], sort=False)
w.columns = [f'{a}_{b}' for a, b in w.columns]
w['tpm'] = g['trades_per_month'].median().xs('all', level=2).values if False else s[s.period == 'all'].set_index(['base', 'rule'])['tpm']
w['scale'] = s[s.period == 'all'].set_index(['base', 'rule'])['scale']
cols = ['cagr_all', 'dd_all', 'sharpe_all', 'd_cagr_all', 'd_dd_all', 'win_is', 'win_oos', 'sharpe_is', 'sharpe_oos', 'dd_oos', 'tpm', 'scale']
pd.set_option('display.width', 250)
for base, t in w[cols].groupby(level=0, sort=False):
    print('\n==', base)
    print(t.droplevel(0).round(3).to_string())
