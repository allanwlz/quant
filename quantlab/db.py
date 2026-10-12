"""Local DuckDB market store.

Prices are stored unadjusted; ``adj_factor`` is the cumulative backward factor, so
``close * adj_factor`` is the back-adjusted close. Volume is in shares, amount and
market values in the security's own currency, dates are exchange-local.
"""
import duckdb

SCHEMA = """
CREATE TABLE IF NOT EXISTS securities(
    code TEXT PRIMARY KEY,          -- 600519.SH / 00700.HK / AAPL.US / 000300.SH
    market TEXT NOT NULL,           -- CN / HK / US
    kind TEXT NOT NULL,             -- stock / index
    name TEXT,
    board TEXT,                     -- 主板 / 创业板 / 科创板 / 北交所 / CDR ...
    industry TEXT,
    list_date DATE,
    delist_date DATE,
    status TEXT,                    -- L 上市 / D 退市 / P 暂停
    currency TEXT,
    updated TIMESTAMP DEFAULT now()
);
CREATE TABLE IF NOT EXISTS calendar(
    market TEXT, date DATE, PRIMARY KEY(market, date)   -- open days only
);
CREATE TABLE IF NOT EXISTS bars(
    code TEXT, date DATE,
    open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, pre_close DOUBLE,
    volume DOUBLE, amount DOUBLE,
    adj_factor DOUBLE,
    up_limit DOUBLE, down_limit DOUBLE,  -- CN only, exact exchange limit prices
    source TEXT,
    PRIMARY KEY(code, date)
);
CREATE TABLE IF NOT EXISTS cn_daily_basic(
    code TEXT, date DATE,
    turnover_rate DOUBLE, turnover_rate_f DOUBLE, pe_ttm DOUBLE, pb DOUBLE,
    total_share DOUBLE, float_share DOUBLE, total_mv DOUBLE, circ_mv DOUBLE,
    PRIMARY KEY(code, date)
);
CREATE TABLE IF NOT EXISTS hk_connect(
    date DATE, code TEXT, volume DOUBLE, ratio DOUBLE,  -- southbound holdings, point-in-time eligibility proxy
    PRIMARY KEY(date, code)
);
ALTER TABLE securities ADD COLUMN IF NOT EXISTS lot DOUBLE;  -- board lot in shares (HK per stock)
CREATE TABLE IF NOT EXISTS cn_names(
    code TEXT, name TEXT, start_date DATE, end_date DATE,     -- name history, used for point-in-time ST status
    PRIMARY KEY(code, start_date, name)
);
CREATE TABLE IF NOT EXISTS cn_fina(
    code TEXT, end_date DATE, ann_date DATE,             -- one row per announcement (restatements kept)
    roe DOUBLE, roe_dt DOUBLE, roic DOUBLE, gross_margin DOUBLE, net_margin DOUBLE,
    revenue_yoy DOUBLE, profit_yoy DOUBLE, debt_to_assets DOUBLE,
    PRIMARY KEY(code, end_date, ann_date)
);
CREATE TABLE IF NOT EXISTS cn_income(
    code TEXT, end_date DATE, ann_date DATE,              -- cumulative year-to-date figures, consolidated
    revenue DOUBLE, total_revenue DOUBLE, net_profit DOUBLE,
    PRIMARY KEY(code, end_date, ann_date)
);
CREATE TABLE IF NOT EXISTS cn_industry(
    code TEXT, l1 TEXT, l2 TEXT, l3 TEXT, in_date DATE, out_date DATE,   -- Shenwan 2021 membership spans
    PRIMARY KEY(code, l3, in_date)
);
CREATE TABLE IF NOT EXISTS sync_state(
    task TEXT, key TEXT, rows INTEGER, synced_at TIMESTAMP DEFAULT now(),
    PRIMARY KEY(task, key)
);
"""


def connect(path, read_only=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    if not read_only:
        con.execute(SCHEMA)
    return con


def upsert(con, table, frame, key):
    """Insert or replace rows of ``frame`` (columns must exist in ``table``)."""
    if frame is None or frame.empty:
        return 0
    cols = list(frame.columns)
    updates = ', '.join(f'{c}=excluded.{c}' for c in cols if c not in key) or 'NOTHING'
    action = f'DO UPDATE SET {updates}' if updates != 'NOTHING' else 'DO NOTHING'
    con.register('_incoming', frame)
    try:
        con.execute(f'INSERT INTO {table} ({", ".join(cols)}) SELECT {", ".join(cols)} FROM _incoming '
                    f'ON CONFLICT ({", ".join(key)}) {action}')
    finally:
        con.unregister('_incoming')
    return len(frame)


def done_keys(con, task):
    return {r[0] for r in con.execute('SELECT key FROM sync_state WHERE task=?', [task]).fetchall()}


def mark_done(con, task, key, rows):
    con.execute('INSERT INTO sync_state(task,key,rows,synced_at) VALUES (?,?,?,now()) '
                'ON CONFLICT (task,key) DO UPDATE SET rows=excluded.rows, synced_at=excluded.synced_at', [task, key, rows])
