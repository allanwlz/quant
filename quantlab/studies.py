"""Standard parameter grids for comparing trend algorithms across markets."""

GRIDS = {
    'buy_hold': {},
    'above_ma': {'n': [50, 100, 150, 200, 250]},
    'ma_cross': {'fast': [20, 50, 100], 'slow': [100, 200, 300]},
    'donchian': {'entry': [20, 55, 100, 200, 250], 'exit': [10, 20, 50, 100]},
    'tsmom': {'lookbacks': [(20, 60, 120, 250), (60, 120, 250), (120, 250), (250,), (60,)]},
    'slope_r2': {'n': [60, 90, 120, 250], 'min_slope': [0.1, 0.2, 0.4], 'min_r2': [0.3, 0.5]},
    'volume_breakout': {'n': [55, 100, 250], 'vol_mult': [1.5, 2.0], 'exit_ma': [20, 50, 100]},
}
ATR_STOPS = [None, 3.0, 5.0]   # applied to every grid point


def runs():
    from .research import grid
    for algo, spec in GRIDS.items():
        for params in grid(spec):
            for stop in ATR_STOPS:
                if algo == 'buy_hold' and stop:
                    continue
                if algo == 'ma_cross' and params['fast'] >= params['slow']:
                    continue
                if algo == 'donchian' and params['exit'] >= params['entry']:
                    continue
                yield algo, params, stop

# Plateau picks from the universe-mode studies (docs/discussion.md), for the slower pools mode.
SHORTLIST = [
    ('buy_hold', {}, None),
    ('above_ma', {'n': 200}, None),
    ('above_ma', {'n': 200}, 5.0),
    ('ma_cross', {'fast': 50, 'slow': 200}, None),
    ('ma_cross', {'fast': 100, 'slow': 300}, None),
    ('donchian', {'entry': 55, 'exit': 20}, None),
    ('donchian', {'entry': 200, 'exit': 100}, None),
    ('tsmom', {'lookbacks': (60, 120, 250)}, None),
    ('slope_r2', {'n': 250, 'min_slope': 0.2, 'min_r2': 0.3}, None),
    ('slope_r2', {'n': 120, 'min_slope': 0.4, 'min_r2': 0.3}, None),
    ('volume_breakout', {'n': 55, 'vol_mult': 2.0, 'exit_ma': 50}, None),
    ('volume_breakout', {'n': 100, 'vol_mult': 2.0, 'exit_ma': 50}, None),
    ('volume_breakout', {'n': 55, 'vol_mult': 2.0, 'exit_ma': 100}, 3.0),
]
