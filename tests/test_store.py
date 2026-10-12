from datetime import date

import pandas as pd

from quantlab import db, sync


def test_upsert_replaces_and_keeps_keys(tmp_path):
    con = db.connect(tmp_path / 't.duckdb')
    row = {'code': '600519.SH', 'date': date(2024, 1, 2), 'close': 1.0}
    db.upsert(con, 'bars', pd.DataFrame([row]), ['code', 'date'])
    db.upsert(con, 'bars', pd.DataFrame([{**row, 'close': 2.0}]), ['code', 'date'])
    assert con.execute('SELECT count(*), max(close) FROM bars').fetchone() == (1, 2.0)


def test_sync_state_roundtrip(tmp_path):
    con = db.connect(tmp_path / 't.duckdb')
    db.mark_done(con, 'cn_daily', '2024-01-02', 10)
    db.mark_done(con, 'cn_daily', '2024-01-02', 12)
    assert db.done_keys(con, 'cn_daily') == {'2024-01-02'}


class FakeTushare:
    def query(self, api, fields='', **params):
        tables = {
            'daily': [{'ts_code': '300750.SZ', 'open': 1, 'high': 2, 'low': 1, 'close': 2, 'pre_close': 1, 'vol': 3, 'amount': 4}],
            'adj_factor': [{'ts_code': '300750.SZ', 'adj_factor': 1.5}],
            'stk_limit': [{'ts_code': '300750.SZ', 'up_limit': 1.2, 'down_limit': 0.8}],
            'daily_basic': [{'ts_code': '300750.SZ', 'turnover_rate': 1, 'turnover_rate_f': 1, 'pe_ttm': 1, 'pb': 1,
                             'total_share': 2, 'float_share': 2, 'total_mv': 5, 'circ_mv': 5}],
        }
        return pd.DataFrame(tables[api])


def test_cn_day_normalises_units():
    bars, basic = sync.cn_day(FakeTushare(), date(2024, 1, 2))
    row = bars.iloc[0]
    assert (row['volume'], row['amount'], row['adj_factor'], row['up_limit']) == (300, 4000, 1.5, 1.2)
    assert basic.iloc[0]['total_mv'] == 50000
