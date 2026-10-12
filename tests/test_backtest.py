import numpy as np
import pandas as pd
import pytest

from quantlab import backtest, signals
from quantlab.panel import Panel


def make_panel(close, market='US', open_=None, adj=None, up=None, down=None, lot=1.0):
    close = np.asarray(close, dtype=np.float32)
    if close.ndim == 1:
        close = close[:, None]
    T, N = close.shape
    o = close if open_ is None else np.asarray(open_, dtype=np.float32).reshape(T, N)
    data = {'open': o, 'high': np.maximum(o, close), 'low': np.minimum(o, close), 'close': close,
            'volume': np.full((T, N), 1e6, np.float32), 'amount': close * 1e6,
            'adj_factor': np.ones((T, N), np.float32) if adj is None else np.asarray(adj, np.float32).reshape(T, N),
            'up_limit': np.full((T, N), np.nan, np.float32) if up is None else np.asarray(up, np.float32).reshape(T, N),
            'down_limit': np.full((T, N), np.nan, np.float32) if down is None else np.asarray(down, np.float32).reshape(T, N)}
    dates = pd.bdate_range('2024-01-01', periods=T)
    return Panel(market, dates, np.array([f'S{j}' for j in range(N)], dtype=object), data,
                 np.full(N, lot), np.full(N, np.datetime64('2000-01-01', 'D')))


def free(panel, hold, **cfg):
    """Run with every security enterable."""
    return backtest.run(panel, hold, np.ones(panel.shape, bool), backtest.Config(**cfg))


def test_decide_at_close_fill_next_open():
    panel = make_panel([10, 10, 10, 10], open_=[10, 11, 12, 12])
    hold = np.array([[False], [True], [True], [True]])
    result = free(panel, hold, base_weight=0.5)
    trade = result.trades.iloc[0]
    assert trade['date'] == panel.dates[2]                 # signal on day 1 → trade on day 2
    assert trade['price'] == pytest.approx(12 * 1.0005)    # day 2 open plus US slippage
    assert result.positions.iloc[1] == 0 and result.positions.iloc[2] == 1


def test_cn_limit_up_open_blocks_buy():
    panel = make_panel([10, 11, 12.1, 12.1], market='CN', open_=[10, 11, 12.1, 12], up=[11, 11, 12.1, 13.3], lot=100)
    hold = np.array([[False], [True], [True], [True]])
    result = backtest.run(panel, hold, np.ones(panel.shape, bool), backtest.Config(base_weight=0.5))
    assert list(result.trades['date']) == [panel.dates[3]]   # day 2 opened limit-up, filled on day 3


def test_weights_scale_when_signals_exceed_gross():
    panel = make_panel(np.full((3, 40), 10.0))
    result = free(panel, np.ones((3, 40), bool))
    assert result.positions.iloc[1] == 40
    assert result.exposure.iloc[1] == pytest.approx(1.0, abs=0.01)   # 40 x 5% scaled to 100%


def test_band_suppresses_small_resizes():
    prices = np.array([[10, 10], [10, 10], [10.5, 10], [10.5, 10]], dtype=np.float32)
    panel = make_panel(prices)
    result = free(panel, np.ones((4, 2), bool), base_weight=0.3)
    assert len(result.trades) == 2                          # initial buys only


def test_split_is_value_neutral():
    close = [10, 10, 10, 5, 5]
    adj = [1, 1, 1, 2, 2]
    panel = make_panel(close, open_=close, adj=adj)
    result = free(panel, np.ones((5, 1), bool), base_weight=1.0, band_abs=0.5)
    assert result.nav.iloc[-1] == pytest.approx(result.nav.iloc[2], rel=1e-9)


def test_exit_sells_everything():
    panel = make_panel([10] * 5)
    hold = np.array([[True], [True], [False], [False], [False]])
    result = free(panel, hold, base_weight=0.5)
    assert result.positions.iloc[-1] == 0
    assert result.trades['shares'].sum() == pytest.approx(0)


def test_latch_and_trailing_stop():
    entry = np.array([[1], [0], [0], [0]], bool)
    exit = np.array([[0], [0], [1], [0]], bool)
    assert signals.latch(entry, exit)[:, 0].tolist() == [True, True, False, False]
    panel = make_panel([10, 12, 14, 9, 9, 15])
    held = signals.trailing_stop(panel, np.ones((6, 1), bool), k=0.1, n=1)
    assert held[:, 0].tolist() == [True, True, True, False, False, False]   # no re-entry while base stays on


def test_rolling_slope_matches_polyfit():
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(size=(60, 3)), axis=0).astype(np.float32)
    slope, r2 = signals.rolling_slope_r2(y, 20, chunk=2)
    window = y[40:60, 1].astype(float)
    b, a = np.polyfit(np.arange(20), window, 1)
    fit = a + b * np.arange(20)
    expected_r2 = 1 - ((window - fit) ** 2).sum() / ((window - window.mean()) ** 2).sum()
    assert slope[59, 1] == pytest.approx(b, rel=1e-4)
    assert r2[59, 1] == pytest.approx(expected_r2, rel=1e-4)
    assert np.isnan(slope[18, 0])


def test_random_pools_respect_universe_and_size():
    from quantlab import research
    rng = np.random.default_rng(1)
    universe = np.ones((40, 100), bool)
    universe[:, :10] = False
    pools = research.random_pools(universe, 30, rng)
    assert (pools.sum(axis=1) == 30).all()
    assert not pools[:, :10].any()
    assert (pools[9] != pools[10]).any() and (pools[10] == pools[19]).all()   # changes only on refresh days
