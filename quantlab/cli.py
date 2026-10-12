"""Command line: ``quant sync cn|hk|us|calendar`` and ``quant status``."""
import argparse
import json
import logging
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, db, sync
from .sources.tickflow import TickFlow
from .sources.tushare import Tushare

DEFAULT_START = date(2014, 1, 1)
RESULTS = config.ROOT.parent / 'test' / 'results'


def _parse_date(text):
    return date.fromisoformat(text)


def _value(text):
    """'55' → 55, '0.3' → 0.3, '20/60/120' → (20, 60, 120)."""
    if '/' in text:
        return tuple(_value(t) for t in text.split('/'))
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def _params(items, many=False):
    out = {}
    for item in items or []:
        key, _, value = item.partition('=')
        out[key] = [_value(v) for v in value.split(',')] if many else _value(value)
    return out


def research(args, con):
    from . import backtest, research as lab
    p, universe = lab.prepare(con, args.market, args.start, args.end)
    logging.info('%s 面板 %d 天 x %d 只，起始日可交易范围 %d 只', args.market, *p.shape,
                 int(universe[p.dates.searchsorted(str(args.start))].sum()))
    initial = args.initial or 1e8  # study the algorithm, not lot rounding or minimum fees
    cfg = backtest.Config(base_weight=args.base_weight, band_rel=args.band_rel, band_abs=args.band_abs, initial=initial)
    common = dict(start=args.start, mode=args.mode, pools=args.pools, pool_size=args.pool_size,
                  atr_stop=args.atr_stop, cfg=cfg)
    out = RESULTS / f"{datetime.now():%Y%m%d-%H%M%S}-{args.market}-{args.algo}-{args.command}"
    out.mkdir(parents=True, exist_ok=True)
    if args.command == 'backtest':
        summary, detail = lab.evaluate(p, universe, args.algo, _params(args.param), **common)
        if args.mode == 'universe':
            detail.nav.to_csv(out / 'nav.csv')
            detail.trades.to_csv(out / 'trades.csv', index=False)
        else:
            detail.to_csv(out / 'pools.csv', index=False)
        frame = pd.DataFrame([summary])
    else:
        frame = lab.sweep(p, universe, args.algo, _params(args.grid, many=True), **common)
    frame.to_csv(out / 'metrics.csv', index=False)
    (out / 'args.json').write_text(json.dumps(vars(args), default=str, ensure_ascii=False, indent=1))
    with pd.option_context('display.width', 200, 'display.max_columns', 30, 'display.float_format', '{:.3f}'.format):
        print(frame.to_string(index=False))
    print('结果目录:', out)


def study(args, con):
    """Run every algorithm x parameter x ATR stop in ``studies.GRIDS`` on one market."""
    from . import backtest, research as lab, studies
    p, universe = lab.prepare(con, args.market, args.start, args.end)
    if args.pool_filter == 'quality':
        from . import quality
        universe &= quality.mask(p, con)
    cfg = backtest.Config(initial=1e8)
    tag = f"{args.mode}{'-' + args.pool_filter if args.pool_filter else ''}{'-short' if args.shortlist else ''}"
    out = RESULTS / f"{datetime.now():%Y%m%d-%H%M%S}-{args.market}-study-{tag}"
    out.mkdir(parents=True, exist_ok=True)
    rows, todo = [], list(studies.SHORTLIST if args.shortlist else studies.runs())
    logging.info('%s 面板 %d 天 x %d 只，共 %d 组', args.market, *p.shape, len(todo))
    for i, (algo, params, stop) in enumerate(todo, 1):
        summary, _ = lab.evaluate(p, universe, algo, params, args.start, mode=args.mode, pools=args.pools,
                                  atr_stop=stop, cfg=cfg)
        rows.append({'algo': algo, 'params': json.dumps(params), 'atr_stop': stop, **summary})
        if i % 10 == 0 or i == len(todo):
            pd.DataFrame(rows).to_csv(out / 'study.csv', index=False)   # partial results survive a crash
            logging.info('%d/%d', i, len(todo))
    frame = pd.DataFrame(rows)
    key = 'calmar' if args.mode == 'universe' else 'calmar_p50'
    with pd.option_context('display.width', 220, 'display.max_columns', 20, 'display.float_format', '{:.3f}'.format):
        print(frame.sort_values(key, ascending=False).head(25).to_string(index=False))
    print('结果目录:', out)


def timing_study(args, con):
    """Market timing rules (``timing.RULES``) overlaid on a few base strategies."""
    from . import research as lab, timing
    p, universe = lab.prepare(con, args.market, args.start, args.end)
    quality = None
    if args.market == 'CN':
        from . import quality as q
        quality = universe & q.mask(p, con)
    bases = [('pools', 'buy_hold', {}), ('universe', 'buy_hold', {})]
    bases += ([('pools', 'ma_cross', {'fast': 100, 'slow': 300})] if args.market == 'CN'
              else [('pools', 'donchian', {'entry': 200, 'exit': 100})])
    if quality is not None:
        bases = [(m, a, prm, 'quality') for m, a, prm in bases] + [('pools', 'buy_hold', {}, 'all')]
    else:
        bases = [(m, a, prm, 'all') for m, a, prm in bases]
    px = timing.index_close(con, timing.BENCHMARK[args.market], p.dates)
    cash = timing.cash_rate(con, args.market, p.dates)
    logging.info('平均现金利率 %.2f%%', cash[np.searchsorted(p.dates, pd.Timestamp(args.start)):].mean() * 100)
    out = RESULTS / f"{datetime.now():%Y%m%d-%H%M%S}-{args.market}-timing{'' if args.rules == 'all' else '-' + args.rules}"
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    rules = [] if args.rules == 'cd' else list(timing.RULES)
    market = None
    if args.market == 'CN' and args.rules != 'ab':
        market = timing.cn_market(con, p.dates)
        rules += ([('none', {})] if args.rules == 'cd' else []) + timing.CN_RULES
    for mode, algo, prm, uni in bases:
        for kind, rp in rules:
            scale = timing.rule(px, kind, market=market, **rp).to_numpy()
            frame = timing.evaluate(p, quality if uni == 'quality' else universe, algo, prm, scale, args.start, mode,
                                    pools=args.pools, cash=cash)
            frame.insert(0, 'base', f"{mode}-{uni}-{algo}{json.dumps(prm) if prm else ''}")
            frame.insert(1, 'rule', f"{kind}{json.dumps(rp) if rp else ''}")
            frame['avg_scale'] = scale[np.searchsorted(p.dates, pd.Timestamp(args.start)):].mean()
            frames.append(frame)
            pd.concat(frames).to_csv(out / 'timing.csv', index=False)
            logging.info('%s %s %s 完成', mode, algo, kind)
    print('结果目录:', out)


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
    s.add_argument('target', choices=['calendar', 'cn', 'hk', 'us', 'fundamentals', 'rates'])
    s.add_argument('--start', type=_parse_date, default=DEFAULT_START)
    s.add_argument('--end', type=_parse_date, default=date.today())
    s.add_argument('--codes-file', type=Path, help='美股代码清单，每行一个，如 AAPL.US')
    sub.add_parser('status', help='本地库概况')
    for name, text in (('backtest', '单组参数回测'), ('sweep', '参数网格扫描')):
        r = sub.add_parser(name, help=text)
        r.add_argument('--market', choices=['CN', 'HK', 'US'], required=True)
        r.add_argument('--algo', required=True)
        r.add_argument('--start', type=_parse_date, default=date(2016, 1, 1))
        r.add_argument('--end', type=_parse_date, default=date.today())
        r.add_argument('--mode', choices=['universe', 'pools'], default='universe')
        r.add_argument('--pools', type=int, default=20, help='pools 模式随机股票池数量')
        r.add_argument('--pool-size', type=int, default=40)
        r.add_argument('--atr-stop', type=float, help='ATR 跟踪止损倍数')
        r.add_argument('--base-weight', type=float, default=0.05)
        r.add_argument('--band-rel', type=float, default=0.30)
        r.add_argument('--band-abs', type=float, default=0.015)
        r.add_argument('--initial', type=float, help='初始资金，默认 1 亿')
        if name == 'backtest':
            r.add_argument('-p', '--param', action='append', help='算法参数，如 -p entry=55 -p lookbacks=20/60/120')
        else:
            r.add_argument('-g', '--grid', action='append', help='参数网格，如 -g entry=20,55,100')
    r = sub.add_parser('study', help='标准网格：所有算法 x 参数 x ATR 止损')
    r.add_argument('--market', choices=['CN', 'HK', 'US'], required=True)
    r.add_argument('--start', type=_parse_date, default=date(2016, 1, 1))
    r.add_argument('--end', type=_parse_date, default=date.today())
    r.add_argument('--mode', choices=['universe', 'pools'], default='universe')
    r.add_argument('--pools', type=int, default=20)
    r.add_argument('--shortlist', action='store_true', help='只跑 studies.SHORTLIST（pools 模式较慢时用）')
    r.add_argument('--pool-filter', choices=['quality'], help='先用时点选股规则缩小可选范围（仅 CN）')
    r = sub.add_parser('timing', help='大盘择时规则叠加在基础策略上（总仓位缩放）')
    r.add_argument('--market', choices=['CN', 'HK', 'US'], required=True)
    r.add_argument('--start', type=_parse_date, default=date(2016, 1, 1))
    r.add_argument('--end', type=_parse_date, default=date.today())
    r.add_argument('--pools', type=int, default=20)
    r.add_argument('--rules', choices=['all', 'ab', 'cd'], default='all', help='ab：趋势与波动率；cd：估值与拥挤度（仅 CN）')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    settings = config.load()
    con = db.connect(settings.db_path, read_only=args.command != 'sync')
    if args.command == 'status':
        return status(con)
    if args.command in ('backtest', 'sweep'):
        return research(args, con)
    if args.command == 'study':
        return study(args, con)
    if args.command == 'timing':
        return timing_study(args, con)

    if args.target == 'calendar':
        ts = Tushare(settings.tushare_token)
        for market in ('CN', 'HK', 'US'):
            sync.sync_calendar(con, ts, market, args.start, date(date.today().year, 12, 31))
    elif args.target == 'cn':
        ts = Tushare(settings.tushare_token)
        sync.sync_cn_securities(con, ts)
        sync.sync_cn_indexes(con, ts, args.start, args.end)
        sync.sync_cn_daily(con, ts, args.start, args.end)
    elif args.target == 'rates':
        ts = Tushare(settings.tushare_token)
        sync.sync_rates(con, ts, args.start, args.end)
        sync.sync_index_basic(con, ts, args.start, args.end)
    elif args.target == 'fundamentals':
        ts = Tushare(settings.tushare_token)
        sync.sync_cn_industry(con, ts)
        sync.sync_cn_fundamentals(con, ts)
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
