"""Fill the local store from Tushare (CN, calendars, HK connect) and TickFlow (HK/US bars).

Every task is resumable: per-date CN pulls are recorded in ``sync_state`` and skipped
on the next run, per-symbol HK/US pulls only request bars after the last stored date.
"""
import logging
from datetime import date, datetime, timedelta

import pandas as pd

from . import db

log = logging.getLogger(__name__)

CN_INDEXES = {'000300.SH': '沪深300', '000905.SH': '中证500', '000852.SH': '中证1000', '000906.SH': '中证800',
              '399006.SZ': '创业板指', '000688.SH': '科创50', '000001.SH': '上证指数'}
CALENDAR_APIS = {'CN': ('trade_cal', {'exchange': 'SSE'}), 'HK': ('hk_tradecal', {}), 'US': ('us_tradecal', {})}


def _ymd(d):
    return d.strftime('%Y%m%d')


def _dates(series):
    return pd.to_datetime(series, format='%Y%m%d', errors='coerce').dt.date


def _chunks(start, end, days):
    while start <= end:
        stop = min(end, start + timedelta(days=days))
        yield start, stop
        start = stop + timedelta(days=1)


def sync_calendar(con, ts, market, start, end):
    api, extra = CALENDAR_APIS[market]
    frames = [ts.query(api, 'cal_date,is_open', start_date=_ymd(a), end_date=_ymd(b), **extra)
              for a, b in _chunks(start, end, 1800)]
    cal = pd.concat(frames, ignore_index=True)
    cal = cal[cal['is_open'].astype(int) == 1]
    rows = pd.DataFrame({'market': market, 'date': _dates(cal['cal_date'])}).drop_duplicates()
    with con.cursor() as cur:
        cur.execute('DELETE FROM calendar WHERE market=? AND date BETWEEN ? AND ?', [market, start, end])
        db.upsert(cur, 'calendar', rows, ['market', 'date'])
    log.info('%s 交易日历 %d 天', market, len(rows))
    return len(rows)


def sync_cn_securities(con, ts):
    fields = 'ts_code,name,industry,market,list_date,delist_date,list_status'
    frame = pd.concat([ts.query('stock_basic', fields, list_status=s) for s in ('L', 'D', 'P')], ignore_index=True)
    rows = pd.DataFrame({'code': frame['ts_code'], 'market': 'CN', 'kind': 'stock', 'name': frame['name'],
                         'board': frame['market'], 'industry': frame['industry'],
                         'list_date': _dates(frame['list_date']), 'delist_date': _dates(frame['delist_date']),
                         'status': frame['list_status'], 'currency': 'CNY', 'updated': datetime.now()})
    indexes = pd.DataFrame({'code': list(CN_INDEXES), 'market': 'CN', 'kind': 'index', 'name': list(CN_INDEXES.values()),
                            'status': 'L', 'currency': 'CNY', 'updated': datetime.now()})
    count = db.upsert(con, 'securities', rows, ['code']) + db.upsert(con, 'securities', indexes, ['code'])
    log.info('A 股证券 %d 条（含退市）', count)
    return count


def sync_hk_securities(con, ts):
    fields = 'ts_code,name,list_date,delist_date,list_status,market'
    frame = pd.concat([ts.query('hk_basic', fields, list_status=s) for s in ('L', 'D')], ignore_index=True)
    rows = pd.DataFrame({'code': frame['ts_code'].str.zfill(8), 'market': 'HK', 'kind': 'stock', 'name': frame['name'],
                         'board': frame['market'], 'list_date': _dates(frame['list_date']),
                         'delist_date': _dates(frame['delist_date']), 'status': frame['list_status'],
                         'currency': 'HKD', 'updated': datetime.now()})
    return db.upsert(con, 'securities', rows, ['code'])


def _open_days(con, market, start, end):
    return [r[0] for r in con.execute('SELECT date FROM calendar WHERE market=? AND date BETWEEN ? AND ? ORDER BY date',
                                      [market, start, min(end, date.today())]).fetchall()]


def cn_day(ts, day):
    """All A-share bars for one trading day, merged from daily/adj_factor/stk_limit."""
    key = _ymd(day)
    bars = ts.query('daily', 'ts_code,open,high,low,close,pre_close,vol,amount', trade_date=key)
    if bars.empty:
        return bars, bars
    adj = ts.query('adj_factor', 'ts_code,adj_factor', trade_date=key)
    limits = ts.query('stk_limit', 'ts_code,up_limit,down_limit', trade_date=key)
    basic = ts.query('daily_basic', 'ts_code,turnover_rate,turnover_rate_f,pe_ttm,pb,total_share,float_share,total_mv,circ_mv',
                     trade_date=key)
    bars = bars.merge(adj, on='ts_code', how='left').merge(limits, on='ts_code', how='left')
    bars = bars.rename(columns={'ts_code': 'code', 'vol': 'volume'})
    bars['volume'] = bars['volume'] * 100      # 手 → 股
    bars['amount'] = bars['amount'] * 1000     # 千元 → 元
    bars['date'], bars['source'] = day, 'tushare'
    basic = basic.rename(columns={'ts_code': 'code'})
    for col in ('total_share', 'float_share', 'total_mv', 'circ_mv'):
        basic[col] = basic[col] * 10000        # 万股 / 万元
    basic['date'] = day
    return bars, basic


def sync_cn_daily(con, ts, start, end):
    days = [d for d in _open_days(con, 'CN', start, end) if d.isoformat() not in db.done_keys(con, 'cn_daily')]
    log.info('A 股日线待拉取 %d 个交易日', len(days))
    for i, day in enumerate(days, 1):
        bars, basic = cn_day(ts, day)
        if bars.empty:
            log.warning('%s 尚无日线，跳过', day)
            continue
        with con.cursor() as cur:
            cur.execute('BEGIN')
            db.upsert(cur, 'bars', bars, ['code', 'date'])
            db.upsert(cur, 'cn_daily_basic', basic, ['code', 'date'])
            db.mark_done(cur, 'cn_daily', day.isoformat(), len(bars))
            cur.execute('COMMIT')
        if i % 20 == 0 or i == len(days):
            log.info('A 股 %d/%d %s：%d 只', i, len(days), day, len(bars))
    return len(days)


def sync_cn_indexes(con, ts, start, end):
    for code in CN_INDEXES:
        frames = [ts.query('index_daily', 'trade_date,open,high,low,close,pre_close,vol,amount', ts_code=code,
                           start_date=_ymd(a), end_date=_ymd(b)) for a, b in _chunks(start, end, 3000)]
        frame = pd.concat(frames, ignore_index=True).rename(columns={'vol': 'volume'})
        if frame.empty:
            continue
        frame['date'] = _dates(frame.pop('trade_date'))
        frame['volume'] = frame['volume'] * 100
        frame['amount'] = frame['amount'] * 1000
        frame['code'], frame['adj_factor'], frame['source'] = code, 1.0, 'tushare'
        db.upsert(con, 'bars', frame, ['code', 'date'])
        log.info('指数 %s %d 条', code, len(frame))


def sync_hk_connect(con, ts, start, end):
    """Southbound (港股通) holdings per day; a stock held on a day was eligible that day."""
    days = [d for d in _open_days(con, 'CN', max(start, date(2017, 3, 17)), end)
            if d.isoformat() not in db.done_keys(con, 'hk_connect')]
    log.info('港股通持股待拉取 %d 天', len(days))
    for i, day in enumerate(days, 1):
        frame = ts.query('hk_hold', 'ts_code,vol,ratio', trade_date=_ymd(day), exchange='HK')
        if frame.empty:
            continue
        rows = pd.DataFrame({'date': day, 'code': frame['ts_code'].str.zfill(8), 'volume': frame['vol'], 'ratio': frame['ratio']})
        with con.cursor() as cur:
            cur.execute('BEGIN')
            db.upsert(cur, 'hk_connect', rows, ['date', 'code'])
            db.mark_done(cur, 'hk_connect', day.isoformat(), len(rows))
            cur.execute('COMMIT')
        if i % 50 == 0 or i == len(days):
            log.info('港股通 %d/%d %s：%d 只', i, len(days), day, len(rows))
    return len(days)


def _ensure_security(con, code):
    market = code.rsplit('.', 1)[-1]
    con.execute("INSERT INTO securities(code,market,kind,currency) VALUES (?,?,'stock',?) ON CONFLICT DO NOTHING",
                [code, market, {'HK': 'HKD', 'US': 'USD'}[market]])


def sync_overseas(con, tf, codes, start, end, overlap=10):
    """HK/US bars per symbol via TickFlow; incremental after the last stored date."""
    failed = []
    for i, code in enumerate(codes, 1):
        last = con.execute('SELECT max(date) FROM bars WHERE code=?', [code]).fetchone()[0]
        begin = max(start, last - timedelta(days=overlap)) if last else start
        if last and last >= min(end, date.today()) - timedelta(days=1):
            continue
        try:
            frame = tf.daily(code, begin, end)
        except Exception as exc:  # keep going; report at the end
            log.warning('%s 拉取失败：%s', code, exc)
            failed.append(code)
            continue
        if frame.empty:
            failed.append(code)
            continue
        frame = frame[frame['date'] <= end]  # the UTC window can spill one HK session past ``end``
        frame['amount'] = frame['amount'].where(frame['amount'] > 0)  # HK amount comes back as 0
        frame['source'] = 'tickflow'
        db.upsert(con, 'bars', frame, ['code', 'date'])
        _ensure_security(con, code)
        if i % 25 == 0 or i == len(codes):
            log.info('海外 %d/%d %s：%d 条', i, len(codes), code, len(frame))
    return failed
