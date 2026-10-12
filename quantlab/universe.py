"""Point-in-time tradable universe: which securities may be *entered* on each day."""
import numpy as np
import pandas as pd

from .markets import RULES


def rolling_mean(arr, window):
    return pd.DataFrame(arr).rolling(window, min_periods=window).mean().to_numpy(dtype=np.float32)


def build(panel, con=None, min_listed_days=365, min_amount=None, window=20, exclude_st=True, hk_connect=True):
    """Bool matrix [T, N]: has a bar today, listed long enough, liquid enough,
    not ST (CN), and southbound-eligible (HK)."""
    T, N = panel.shape
    mask = panel['valid'].copy()
    age = panel.dates.to_numpy(dtype='datetime64[D]')[:, None] - panel.first_date[None, :]
    mask &= age >= np.timedelta64(min_listed_days, 'D')
    floor = RULES[panel.market].min_amount if min_amount is None else min_amount
    turnover = np.nan_to_num(panel['turnover'], nan=0.0)
    mask &= np.nan_to_num(rolling_mean(turnover, window), nan=0.0) >= floor
    if con is not None and panel.market == 'CN' and exclude_st:
        mask &= ~_st_mask(panel, con)
    if con is not None and panel.market == 'HK' and hk_connect:
        mask &= _connect_mask(panel, con)
    return mask


def _st_mask(panel, con):
    spans = con.execute("SELECT code, start_date, end_date FROM cn_names WHERE name LIKE '%ST%'").df()
    out = np.zeros(panel.shape, dtype=bool)
    col = {c: j for j, c in enumerate(panel.codes)}
    dates = panel.dates.to_numpy(dtype='datetime64[D]')
    for code, start, end in spans.itertuples(index=False):
        j = col.get(code)
        if j is None:
            continue
        lo = np.searchsorted(dates, np.datetime64(start, 'D'))
        hi = len(dates) if pd.isna(end) else np.searchsorted(dates, np.datetime64(end, 'D'), side='right')
        out[lo:hi, j] = True
    return out


def _connect_mask(panel, con):
    """Eligible on an HK day if held via southbound on the latest CN day on or before it."""
    rows = con.execute('SELECT date, code FROM hk_connect WHERE date BETWEEN ? AND ?',
                       [panel.dates[0].date() - pd.Timedelta(days=10), panel.dates[-1].date()]).df()
    if rows.empty:
        return np.zeros(panel.shape, dtype=bool)
    held = pd.crosstab(pd.to_datetime(rows['date']), rows['code']).astype(bool)
    held = held.reindex(columns=panel.codes, fill_value=False)
    held = held.reindex(held.index.union(panel.dates)).ffill().fillna(False).reindex(panel.dates)
    return held.to_numpy(dtype=bool)
