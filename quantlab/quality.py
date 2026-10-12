"""Point-in-time A-share quality screen: industry not shrinking + some moat.

Each month, using only reports announced before that month's first trading day:

1. Industry not declining: the Shenwan L2 industry's combined trailing-12-month
   revenue is not lower than a year earlier (same member set).
2. Moat, at least ``min_moat`` of:
   a. average deducted ROE of the last 3 annual reports >= ``min_roe``;
   b. gross margin above the L2 industry median in each of the last 3 annual reports;
   c. TTM revenue ranked top ``top_n`` or top ``top_pct`` within the L2 industry.
3. Listed for at least ``min_years``.

Shenwan 2021 membership only starts in 2021 for most names, so earlier dates use each
stock's first recorded industry (classification is stable; a mild look-ahead, noted).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Rules:
    min_roe: float = 10.0
    top_n: int = 5
    top_pct: float = 0.2
    min_moat: int = 2
    min_years: float = 3.0
    stale_days: int = 400        # ignore a stock whose latest report is older than this


def load(con):
    fina = con.execute('SELECT code, end_date, ann_date, roe, roe_dt, gross_margin FROM cn_fina').df()
    income = con.execute('SELECT code, end_date, ann_date, revenue, total_revenue FROM cn_income').df()
    industry = con.execute('SELECT code, l1, l2, in_date, out_date FROM cn_industry').df()
    listed = con.execute("SELECT code, list_date FROM securities WHERE market='CN'").df()
    for frame in (fina, income):
        frame['end_date'] = pd.to_datetime(frame['end_date'])
        frame['ann_date'] = pd.to_datetime(frame['ann_date'])
    income['revenue'] = income['revenue'].fillna(income['total_revenue'])
    industry['in_date'] = pd.to_datetime(industry['in_date'])
    industry['out_date'] = pd.to_datetime(industry['out_date'])
    listed['list_date'] = pd.to_datetime(listed['list_date'])
    return fina, income, industry, listed


def _known(frame, asof):
    """Latest version of each (code, period) announced strictly before ``asof``."""
    seen = frame[frame['ann_date'] < asof]
    return seen.sort_values('ann_date').drop_duplicates(['code', 'end_date'], keep='last')


def _industry_at(industry, asof):
    live = industry[(industry['in_date'] <= asof) & (industry['out_date'].isna() | (industry['out_date'] > asof))]
    first = industry.sort_values('in_date').drop_duplicates('code')          # back-fill before 2021
    out = pd.concat([live, first[~first['code'].isin(live['code'])]])
    return out.drop_duplicates('code').set_index('code')[['l1', 'l2']]


def _ttm(income):
    """Trailing-12-month revenue per (code, end_date) from cumulative year-to-date figures."""
    rev = income.set_index(['code', 'end_date'])['revenue']
    prev_year = income['end_date'] - pd.DateOffset(years=1)
    prev_annual = pd.to_datetime(dict(year=income['end_date'].dt.year - 1, month=12, day=31))
    same_q = rev.reindex(pd.MultiIndex.from_arrays([income['code'], prev_year])).to_numpy()
    annual = rev.reindex(pd.MultiIndex.from_arrays([income['code'], prev_annual])).to_numpy()
    q4 = income['end_date'].dt.month.to_numpy() == 12
    ttm = np.where(q4, income['revenue'].to_numpy(), income['revenue'].to_numpy() + annual - same_q)
    return pd.Series(ttm, index=rev.index)


def screen(data, asof, rules=Rules()):
    """DataFrame indexed by code with each criterion and the final ``eligible`` flag."""
    fina, income, industry, listed = data
    asof = pd.Timestamp(asof)
    ind = _industry_at(industry, asof)

    ttm = _ttm(_known(income, asof)).dropna()
    ttm = ttm[ttm > 0].reset_index(name='ttm')
    year_ago = ttm.set_index(['code', 'end_date'])['ttm']
    latest = ttm.sort_values('end_date').drop_duplicates('code', keep='last')
    latest = latest[latest['end_date'] >= asof - pd.Timedelta(days=rules.stale_days)].copy()
    latest['ttm_prev'] = year_ago.reindex(pd.MultiIndex.from_arrays(
        [latest['code'], latest['end_date'] - pd.DateOffset(years=1)])).to_numpy()
    out = latest.set_index('code')[['end_date', 'ttm', 'ttm_prev']].join(ind, how='inner')

    both = out.dropna(subset=['ttm_prev'])
    growth = both.groupby('l2')['ttm'].sum() / both.groupby('l2')['ttm_prev'].sum() - 1
    out['industry_growth'] = out['l2'].map(growth)
    out['industry_ok'] = out['industry_growth'].fillna(0) >= 0          # too little data: give benefit of doubt

    known = _known(fina, asof)
    annual = known[known['end_date'].dt.month == 12].copy()
    annual['roe_x'] = annual['roe_dt'].fillna(annual['roe'])
    annual['l2'] = annual['code'].map(ind['l2'])
    annual['gm_median'] = annual.groupby(['l2', 'end_date'])['gross_margin'].transform('median')
    annual['gm_above'] = annual['gross_margin'] > annual['gm_median']
    last3 = annual.sort_values('end_date').groupby('code').tail(3)
    last3 = last3[last3['end_date'] >= asof - pd.DateOffset(years=4)]
    g = last3.groupby('code')
    out['roe3'] = g['roe_x'].mean()
    out['years'] = g.size()
    out['gm_wins'] = g['gm_above'].sum()
    out['moat_roe'] = (out['years'] >= 3) & (out['roe3'] >= rules.min_roe)
    out['moat_margin'] = (out['years'] >= 3) & (out['gm_wins'] >= 3)

    rank = out.groupby('l2')['ttm'].rank(ascending=False)
    size = out.groupby('l2')['ttm'].transform('size')
    out['rev_rank'] = rank
    out['moat_scale'] = (rank <= rules.top_n) | (rank <= np.ceil(size * rules.top_pct))
    out['moat'] = out[['moat_roe', 'moat_margin', 'moat_scale']].sum(axis=1)

    age = (asof - out.index.map(listed.set_index('code')['list_date'])).days / 365.25
    out['listed_ok'] = np.asarray(age) >= rules.min_years
    out['eligible'] = out['industry_ok'] & (out['moat'] >= rules.min_moat) & out['listed_ok']
    return out


def mask(panel, con, rules=Rules()):
    """Bool [T, N]: eligible on each day per the screen as of that month's first trading day."""
    data = load(con)
    dates = pd.DatetimeIndex(panel.dates)
    firsts = dates.to_series().groupby(dates.to_period('M')).first()
    col = pd.Index(panel.codes)
    out = np.zeros(panel.shape, dtype=bool)
    for k, asof in enumerate(firsts):
        lo = dates.get_loc(asof)
        hi = dates.get_loc(firsts.iloc[k + 1]) if k + 1 < len(firsts) else len(dates)
        picked = col.get_indexer(screen(data, asof, rules).query('eligible').index)
        out[lo:hi, picked[picked >= 0]] = True
    return out
