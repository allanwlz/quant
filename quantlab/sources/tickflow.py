"""TickFlow daily klines for HK/US (same source the /invest site uses)."""
import time
from datetime import UTC, date, datetime, timedelta

import httpx
import pandas as pd


class TickFlow:
    def __init__(self, token='', client=None, interval=2.1, retries=4):
        self.token, self.interval, self.retries = token, interval, retries
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
                if response.status_code == 429 or response.status_code >= 500:
                    raise httpx.HTTPStatusError('retry', request=response.request, response=response)
                response.raise_for_status()
                data = response.json().get('data') or {}
                break
            except (httpx.HTTPError, ValueError):
                time.sleep(5 * 2 ** attempt)
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
