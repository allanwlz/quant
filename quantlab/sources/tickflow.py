"""TickFlow daily klines for HK/US (same source the /invest site uses)."""
import logging
import time
from datetime import UTC, date, datetime, timedelta

import httpx
import pandas as pd

log = logging.getLogger(__name__)


class TickFlow:
    def __init__(self, token='', client=None, interval=2.1, retries=8, max_interval=10.0):
        self.token, self.interval, self.retries, self.max_interval = token, interval, retries, max_interval
        self.base = 'https://api.tickflow.org' if token else 'https://free-api.tickflow.org'
        self.client = client or httpx.Client(timeout=60)
        self._last = 0.0

    def _klines(self, symbol, start, end, adjust):
        stamp = lambda d: int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp() * 1000)
        params = {'symbol': symbol, 'period': '1d', 'adjust': adjust, 'count': 10000,
                  'start_time': stamp(start), 'end_time': stamp(end) + 86399000}
        headers = {'x-api-key': self.token} if self.token else {}
        for attempt in range(self.retries):
            wait = self._last + self.interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                response = self.client.get(self.base + '/v1/klines', params=params, headers=headers)
            except httpx.HTTPError as exc:
                log.warning('TickFlow %s 网络错误：%s', symbol, exc)
                time.sleep(min(600, 30 * 2 ** attempt))
                continue
            if response.status_code == 200:
                data = response.json().get('data') or {}
                break
            # Throttled or failing: log what the server says, slow down for good, back off 30s → 10min.
            log.warning('TickFlow %s %s：HTTP %s retry-after=%s %s', symbol, adjust, response.status_code,
                        response.headers.get('retry-after'), response.text[:200])
            if response.status_code not in (429,) and response.status_code < 500:
                raise RuntimeError(f'TickFlow {symbol} {adjust}: HTTP {response.status_code}')
            self.interval = min(self.max_interval, self.interval * 1.5)
            retry_after = response.headers.get('retry-after', '')
            time.sleep(float(retry_after) if retry_after.isdigit() else min(600, 30 * 2 ** attempt))
        else:
            raise RuntimeError(f'TickFlow {symbol} {adjust}: 重试 {self.retries} 次仍失败')
        frame = pd.DataFrame({k: data.get(k, []) for k in ('timestamp', 'open', 'high', 'low', 'close', 'volume', 'amount')})
        if frame.empty:
            return frame
        frame['date'] = [date(1970, 1, 1) + timedelta(days=round(int(t) / 86400000)) for t in frame.pop('timestamp')]
        return frame

    def daily(self, symbol, start, end):
        """Unadjusted bars with a cumulative backward factor (adj close / raw close)."""
        raw = self._klines(symbol, start, end, 'none')
        if raw.empty:
            return raw
        back = self._klines(symbol, start, end, 'backward')[['date', 'close']].rename(columns={'close': 'adj_close'})
        raw = raw.merge(back, on='date', how='left')
        raw['adj_factor'] = raw.pop('adj_close') / raw['close']
        raw.insert(0, 'code', symbol)
        return raw
