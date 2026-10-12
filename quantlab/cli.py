"""Command line: ``quant sync cn|hk|us|calendar`` and ``quant status``."""
import argparse
import logging
from datetime import date
from pathlib import Path

from . import config, db, sync
from .sources.tickflow import TickFlow
from .sources.tushare import Tushare

DEFAULT_START = date(2014, 1, 1)


def _parse_date(text):
    return date.fromisoformat(text)


def status(con):
    print('证券:', con.execute('SELECT market, kind, count(*) FROM securities GROUP BY ALL ORDER BY ALL').fetchall())
    print('日历:', con.execute('SELECT market, min(date), max(date), count(*) FROM calendar GROUP BY ALL ORDER BY ALL').fetchall())
    print('日线:', con.execute("""SELECT s.market, count(DISTINCT b.code), min(b.date), max(b.date), count(*)
                                  FROM bars b LEFT JOIN securities s USING(code) GROUP BY ALL ORDER BY ALL""").fetchall())
    print('港股通:', con.execute('SELECT count(DISTINCT code), min(date), max(date) FROM hk_connect').fetchall())


def main(argv=None):
    parser = argparse.ArgumentParser(prog='quant')
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('sync', help='拉取数据到本地库（可断点续传）')
    s.add_argument('target', choices=['calendar', 'cn', 'hk', 'us'])
    s.add_argument('--start', type=_parse_date, default=DEFAULT_START)
    s.add_argument('--end', type=_parse_date, default=date.today())
    s.add_argument('--codes-file', type=Path, help='美股代码清单，每行一个，如 AAPL.US')
    sub.add_parser('status', help='本地库概况')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    settings = config.load()
    con = db.connect(settings.db_path, read_only=args.command == 'status')
    if args.command == 'status':
        return status(con)

    if args.target == 'calendar':
        ts = Tushare(settings.tushare_token)
        for market in ('CN', 'HK', 'US'):
            sync.sync_calendar(con, ts, market, args.start, date(date.today().year, 12, 31))
    elif args.target == 'cn':
        ts = Tushare(settings.tushare_token)
        sync.sync_cn_securities(con, ts)
        sync.sync_cn_indexes(con, ts, args.start, args.end)
        sync.sync_cn_daily(con, ts, args.start, args.end)
    elif args.target == 'hk':
        ts = Tushare(settings.tushare_token)
        sync.sync_hk_securities(con, ts)
        sync.sync_hk_connect(con, ts, args.start, args.end)
        codes = [r[0] for r in con.execute('SELECT DISTINCT code FROM hk_connect ORDER BY code').fetchall()]
        failed = sync.sync_overseas(con, TickFlow(settings.tickflow_token), codes, args.start, args.end)
        logging.info('港股完成，失败 %d 只：%s', len(failed), failed[:20])
    elif args.target == 'us':
        if not args.codes_file:
            parser.error('us 需要 --codes-file')
        codes = [line.strip() for line in args.codes_file.read_text().splitlines() if line.strip() and not line.startswith('#')]
        failed = sync.sync_overseas(con, TickFlow(settings.tickflow_token), codes, args.start, args.end)
        logging.info('美股完成，失败 %d 只：%s', len(failed), failed[:20])


if __name__ == '__main__':
    main()
