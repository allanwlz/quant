"""Minimal Tushare Pro HTTP client with pacing and retry."""
import threading
import time

import httpx
import pandas as pd

URL = 'https://api.tushare.pro'


class TushareError(RuntimeError):
    pass


class Tushare:
    def __init__(self, token, client=None, interval=0.15, retries=5):
        if not token:
            raise TushareError('QUANT_TUSHARE_TOKEN 未配置')
        self.token, self.interval, self.retries = token, interval, retries
        self.client = client or httpx.Client(timeout=60)
        self._last, self._lock = 0.0, threading.Lock()

    def query(self, api, fields='', **params):
        """Return a DataFrame; retries network errors and per-minute rate limits."""
        for attempt in range(self.retries):
            with self._lock:  # pacing is shared across threads
                wait = self._last + self.interval - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                self._last = time.monotonic()
            try:
                response = self.client.post(URL, json={'api_name': api, 'token': self.token,
                                                       'params': params, 'fields': fields})
                response.raise_for_status()
                payload = response.json()
            except (httpx.HTTPError, ValueError):
                time.sleep(2 ** attempt)
                continue
            if payload.get('code') == 0:
                data = payload.get('data') or {}
                return pd.DataFrame(data.get('items') or [], columns=data.get('fields') or [])
            message = str(payload.get('msg', ''))
            if '每分钟' in message or '频率' in message:  # rate limited: back off and retry
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            raise TushareError(f'{api}: {message}')
        raise TushareError(f'{api}: 重试 {self.retries} 次仍失败')
