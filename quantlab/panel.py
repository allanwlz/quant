"""Dense date x security matrices loaded from the store, one market at a time.

Arrays are float32 with NaN where a security has no bar (not listed or suspended).
``adj_close`` and friends are back-adjusted and forward-filled so signals see a
continuous series through suspensions.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FIELDS = ('open', 'high', 'low', 'close', 'volume', 'amount', 'adj_factor', 'up_limit', 'down_limit')


@dataclass
class Panel:
    market: str
    dates: pd.DatetimeIndex
    codes: np.ndarray
    data: dict
    lot: np.ndarray
    first_date: np.ndarray            # first bar ever stored (or listing date), datetime64[D]
    _cache: dict = field(default_factory=dict, repr=False)

    def __getitem__(self, name):
        if name in self.data:
            return self.data[name]
        if name not in self._cache:
            self._cache[name] = self._derive(name)
        return self._cache[name]

    @property
    def shape(self):
        return len(self.dates), len(self.codes)

    def _derive(self, name):
        if name == 'valid':
            return ~np.isnan(self.data['close'])
        if name == 'adj':
            return ffill(self.data['adj_factor'])
        if name.startswith('adj_'):   # adj_open / adj_high / adj_low / adj_close, forward-filled
            raw = self.data[name[4:]]
            return ffill(raw * self['adj']) if name == 'adj_close' else raw * self['adj']
        if name == 'turnover':        # traded value; HK sources omit amount
            amount = self.data['amount']
            return np.where(np.isnan(amount), self.data['volume'] * self.data['close'], amount).astype(np.float32)
        raise KeyError(name)

    def frame(self, name):
        return pd.DataFrame(self[name], index=self.dates, columns=self.codes)


def ffill(arr):
    """Forward-fill NaNs down the time axis."""
    return pd.DataFrame(arr).ffill().to_numpy(dtype=np.float32)


def load(con, market, start, end, codes=None, fields=FIELDS):
    dates = pd.DatetimeIndex([r[0] for r in con.execute(
        'SELECT date FROM calendar WHERE market=? AND date BETWEEN ? AND ? ORDER BY date', [market, start, end]).fetchall()])
    if codes is None:
        codes = [r[0] for r in con.execute(
            """SELECT DISTINCT b.code FROM bars b JOIN securities s USING(code)
               WHERE s.market=? AND s.kind='stock' AND b.date BETWEEN ? AND ? ORDER BY 1""", [market, start, end]).fetchall()]
    codes = np.asarray(codes, dtype=object)
    meta = con.execute(
        """SELECT m.code, s.lot, coalesce(s.list_date, f.first) FROM (SELECT unnest(?::TEXT[]) AS code) m
           LEFT JOIN securities s USING(code)
           LEFT JOIN (SELECT code, min(date) AS first FROM bars GROUP BY code) f USING(code)""", [list(codes)]).df()
    meta = meta.set_index('code').reindex(codes)
    T, N = len(dates), len(codes)
    data = {f: np.full((T, N), np.nan, dtype=np.float32) for f in fields}
    con.register('_codes', pd.DataFrame({'code': codes, 'j': np.arange(N, dtype=np.int32)}))
    con.register('_dates', pd.DataFrame({'date': dates.date, 'i': np.arange(T, dtype=np.int32)}))
    try:
        for f in fields:  # one column at a time keeps peak memory low on the 3 GB server
            got = con.execute(f'SELECT d.i, c.j, b.{f} FROM bars b JOIN _codes c USING(code) JOIN _dates d USING(date) '
                              f'WHERE b.date BETWEEN ? AND ? AND b.{f} IS NOT NULL', [start, end]).fetchnumpy()
            data[f][got['i'], got['j']] = got[f]
    finally:
        con.unregister('_codes')
        con.unregister('_dates')
    lot = meta.iloc[:, 0].astype(float).to_numpy()
    first = pd.to_datetime(meta.iloc[:, 1]).to_numpy(dtype='datetime64[D]')
    return Panel(market, dates, codes, data, lot, first)
