"""Per-market trading rules and costs used by the backtester."""
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Rules:
    currency: str
    lot: float                  # default board lot when the security has none
    commission: float           # rate per side
    commission_min: float       # per order, in currency
    stamp_buy: float
    stamp_sell: float
    fees: float                 # exchange / clearing / SEC, both sides
    slippage: float             # fraction of price, adverse, both sides
    min_amount: float           # default 20-day average turnover floor for the universe


RULES = {
    'CN': Rules('CNY', 100, 0.00025, 5.0, 0.0, 0.0005, 0.00003, 0.001, 5e7),
    'HK': Rules('HKD', 100, 0.0005, 15.0, 0.001, 0.001, 0.0001, 0.002, 2e7),
    'US': Rules('USD', 1, 0.0002, 1.0, 0.0, 0.0, 0.00003, 0.0005, 5e7),
}


def stamp_sell(market, day):
    """A-share stamp duty was 0.1% until 2023-08-27; HK was 0.13% from 2021-08-01 to 2023-11-16."""
    if market == 'CN' and day < date(2023, 8, 28):
        return 0.001
    if market == 'HK' and date(2021, 8, 1) <= day < date(2023, 11, 17):
        return 0.0013
    return RULES[market].stamp_sell


def stamp_buy(market, day):
    if market == 'HK' and date(2021, 8, 1) <= day < date(2023, 11, 17):
        return 0.0013
    return RULES[market].stamp_buy
