"""Fill the local store from Tushare (CN, calendars, HK connect) and TickFlow (HK/US bars).

Every task is resumable: per-date CN pulls are recorded in ``sync_state`` and skipped
on the next run, per-symbol HK/US pulls only request bars after the last stored date.
"""
import logging
from concurrent.futures import ThreadPoolExecutor
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
    rows['lot'] = 100.0
    count = db.upsert(con, 'securities', rows, ['code']) + db.upsert(con, 'securities', indexes, ['code'])
    log.info('A 股证券 %d 条（含退市）', count)
    sync_cn_names(con, ts)
    return count


def sync_cn_names(con, ts, page=10000):
    frames, offset = [], 0
    while True:
        frame = ts.query('namechange', 'ts_code,name,start_date,end_date', offset=offset, limit=page)
        frames.append(frame)
        if len(frame) < page:
            break
        offset += page
    names = pd.concat(frames, ignore_index=True).dropna(subset=['start_date'])
    rows = pd.DataFrame({'code': names['ts_code'], 'name': names['name'], 'start_date': _dates(names['start_date']),
                         'end_date': _dates(names['end_date'])}).drop_duplicates(['code', 'start_date', 'name'])
    with con.cursor() as cur:
        cur.execute('DELETE FROM cn_names')
        db.upsert(cur, 'cn_names', rows, ['code', 'start_date', 'name'])
    log.info('A 股更名记录 %d 条', len(rows))


def sync_hk_securities(con, ts):
    fields = 'ts_code,name,list_date,delist_date,list_status,market,trade_unit'
    frame = pd.concat([ts.query('hk_basic', fields, list_status=s) for s in ('L', 'D')], ignore_index=True)
    rows = pd.DataFrame({'code': frame['ts_code'].str.zfill(8), 'market': 'HK', 'kind': 'stock', 'name': frame['name'],
                         'board': frame['market'], 'list_date': _dates(frame['list_date']),
                         'delist_date': _dates(frame['delist_date']), 'status': frame['list_status'],
                         'currency': 'HKD', 'lot': frame['trade_unit'], 'updated': datetime.now()})
    return db.upsert(con, 'securities', rows, ['code'])


def _open_days(con, market, start, end):
    return [r[0] for r in con.execute('SELECT date FROM calendar WHERE market=? AND date BETWEEN ? AND ? ORDER BY date',
                                      [market, start, min(end, date.today())]).fetchall()]


def cn_day(ts, day):
    """All A-share bars for one trading day, merged from daily/adj_factor/stk_limit."""
    key = _ymd(day)
    requests = [('daily', 'ts_code,open,high,low,close,pre_close,vol,amount'),
                ('adj_factor', 'ts_code,adj_factor'),
                ('stk_limit', 'ts_code,up_limit,down_limit'),
                ('daily_basic', 'ts_code,turnover_rate,turnover_rate_f,pe_ttm,pb,total_share,float_share,total_mv,circ_mv')]
    with ThreadPoolExecutor(len(requests)) as pool:  # latency-bound; Tushare allows 500 calls/min
        bars, adj, limits, basic = pool.map(lambda r: ts.query(r[0], r[1], trade_date=key), requests)
    if bars.empty:
        return bars, bars
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
    con.execute("INSERT INTO securities(code,market,kind,currency,lot) VALUES (?,?,'stock',?,?) ON CONFLICT DO NOTHING",
                [code, market, {'HK': 'HKD', 'US': 'USD'}[market], {'HK': None, 'US': 1.0}[market]])


def sync_overseas(con, tf, codes, start, end, overlap=10):
    """HK/US bars per symbol via TickFlow; incremental after the last stored date."""
    failed = []
    synced_today = {r[0] for r in con.execute(
        "SELECT key FROM sync_state WHERE task='tickflow' AND synced_at::DATE = current_date").fetchall()}
    delisted = dict(con.execute('SELECT code, delist_date FROM securities WHERE delist_date IS NOT NULL').fetchall())
    for i, code in enumerate(codes, 1):
        last = con.execute('SELECT max(date) FROM bars WHERE code=?', [code]).fetchone()[0]
        final = min(end, date.today(), delisted.get(code, date.max) - timedelta(days=7))
        if code in synced_today or (last and last >= final):
            continue
        begin = max(start, last - timedelta(days=overlap)) if last else start
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
        db.mark_done(con, 'tickflow', code, len(frame))
        if i % 25 == 0 or i == len(codes):
            log.info('海外 %d/%d %s：%d 条', i, len(codes), code, len(frame))
    return failed


def _paged(ts, api, fields, page=6000, **params):
    frames, offset = [], 0
    while True:
        frame = ts.query(api, fields, offset=offset, limit=page, **params)
        frames.append(frame)
        if len(frame) < page:
            return pd.concat(frames, ignore_index=True)
        offset += page


def _periods(start_year):
    today = date.today()
    return [date(y, m, d) for y in range(start_year, today.year + 1)
            for m, d in ((3, 31), (6, 30), (9, 30), (12, 31)) if date(y, m, d) < today]


def sync_cn_fundamentals(con, ts, start_year=2011, refresh_recent=4):
    """Quarterly financial indicators and income statements for all A-shares, with
    announcement dates so research can use only what was public at the time.
    The last ``refresh_recent`` periods are re-pulled to pick up late filers."""
    periods = _periods(start_year)
    done = db.done_keys(con, 'cn_fina')
    todo = [p for p in periods if p.isoformat() not in done or p in periods[-refresh_recent:]]
    fina_fields = 'ts_code,ann_date,end_date,roe,roe_dt,roic,grossprofit_margin,netprofit_margin,or_yoy,netprofit_yoy,debt_to_assets'
    for period in todo:
        fina = _paged(ts, 'fina_indicator_vip', fina_fields, period=_ymd(period))
        income = _paged(ts, 'income_vip', 'ts_code,ann_date,end_date,report_type,revenue,total_revenue,n_income_attr_p',
                        period=_ymd(period), report_type='1')
        fina = pd.DataFrame({'code': fina['ts_code'], 'end_date': _dates(fina['end_date']), 'ann_date': _dates(fina['ann_date']),
                             'roe': fina['roe'], 'roe_dt': fina['roe_dt'], 'roic': fina['roic'],
                             'gross_margin': fina['grossprofit_margin'], 'net_margin': fina['netprofit_margin'],
                             'revenue_yoy': fina['or_yoy'], 'profit_yoy': fina['netprofit_yoy'],
                             'debt_to_assets': fina['debt_to_assets']}).dropna(subset=['ann_date'])
        income = pd.DataFrame({'code': income['ts_code'], 'end_date': _dates(income['end_date']),
                               'ann_date': _dates(income['ann_date']), 'revenue': income['revenue'],
                               'total_revenue': income['total_revenue'], 'net_profit': income['n_income_attr_p']}
                              ).dropna(subset=['ann_date'])
        with con.cursor() as cur:
            cur.execute('BEGIN')
            db.upsert(cur, 'cn_fina', fina.drop_duplicates(['code', 'end_date', 'ann_date'], keep='last'),
                      ['code', 'end_date', 'ann_date'])
            db.upsert(cur, 'cn_income', income.drop_duplicates(['code', 'end_date', 'ann_date'], keep='last'),
                      ['code', 'end_date', 'ann_date'])
            db.mark_done(cur, 'cn_fina', period.isoformat(), len(fina))
            cur.execute('COMMIT')
        log.info('财务 %s：指标 %d 条，利润表 %d 条', period, len(fina), len(income))


def sync_cn_industry(con, ts):
    """Shenwan 2021 L1/L2/L3 membership with in/out dates (current and historical members)."""
    l1 = ts.query('index_classify', 'index_code', level='L1', src='SW2021')['index_code']
    frames = [_paged(ts, 'index_member_all', 'l1_name,l2_name,l3_name,ts_code,in_date,out_date', l1_code=code, is_new=flag)
              for code in l1 for flag in ('Y', 'N')]
    m = pd.concat(frames, ignore_index=True)
    rows = pd.DataFrame({'code': m['ts_code'], 'l1': m['l1_name'], 'l2': m['l2_name'], 'l3': m['l3_name'],
                         'in_date': _dates(m['in_date']), 'out_date': _dates(m['out_date'])}
                        ).dropna(subset=['in_date']).drop_duplicates(['code', 'l3', 'in_date'])
    with con.cursor() as cur:
        cur.execute('DELETE FROM cn_industry')
        db.upsert(cur, 'cn_industry', rows, ['code', 'l3', 'in_date'])
    log.info('申万行业成分 %d 条（%d 只）', len(rows), rows['code'].nunique())


RATE_SOURCES = {'CN': ('shibor', 'date,3m', '3m'), 'HK': ('hibor', 'date,3m', '3m'), 'US': ('us_tbr', 'date,w13_ce', 'w13_ce')}


def sync_rates(con, ts, start, end):
    """Short-term cash rates (annual %) used as the return on idle cash in backtests."""
    for market, (api, fields, col) in RATE_SOURCES.items():
        frames = [ts.query(api, fields, start_date=_ymd(a), end_date=_ymd(b)) for a, b in _chunks(start, end, 1500)]
        f = pd.concat(frames, ignore_index=True).dropna(subset=[col])
        rows = pd.DataFrame({'market': market, 'date': _dates(f['date']), 'rate': f[col].astype(float)}
                            ).drop_duplicates(['market', 'date'])
        db.upsert(con, 'rates', rows, ['market', 'date'])
        log.info('%s 现金利率 %d 条', market, len(rows))


def sync_index_basic(con, ts, start, end, codes=('000300.SH', '000905.SH')):
    """Index valuation and turnover (Tushare index_dailybasic; history from 2004)."""
    for code in codes:
        frames = [ts.query('index_dailybasic', 'ts_code,trade_date,pe_ttm,pb,turnover_rate,turnover_rate_f',
                           ts_code=code, start_date=_ymd(a), end_date=_ymd(b)) for a, b in _chunks(start, end, 2500)]
        f = pd.concat(frames, ignore_index=True)
        rows = pd.DataFrame({'code': f['ts_code'], 'date': _dates(f['trade_date']), 'pe_ttm': f['pe_ttm'], 'pb': f['pb'],
                             'turnover_rate': f['turnover_rate'], 'turnover_rate_f': f['turnover_rate_f']}
                            ).drop_duplicates(['code', 'date'])
        db.upsert(con, 'index_basic', rows, ['code', 'date'])
        log.info('%s 指数估值 %d 条（%s 起）', code, len(rows), rows['date'].min())
