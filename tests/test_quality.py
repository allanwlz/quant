import pandas as pd

from quantlab import quality


def test_ttm_from_year_to_date():
    rows = [('A', '2023-06-30', 40.0), ('A', '2023-12-31', 100.0), ('A', '2024-06-30', 60.0), ('A', '2024-12-31', 130.0)]
    income = pd.DataFrame(rows, columns=['code', 'end_date', 'revenue'])
    income['end_date'] = pd.to_datetime(income['end_date'])
    ttm = quality._ttm(income)
    assert ttm[('A', pd.Timestamp('2024-06-30'))] == 120.0       # 60 + 100 - 40
    assert ttm[('A', pd.Timestamp('2024-12-31'))] == 130.0
    assert pd.isna(ttm[('A', pd.Timestamp('2023-06-30'))])        # no prior year


def test_known_ignores_future_announcements():
    frame = pd.DataFrame({'code': ['A', 'A'], 'end_date': pd.to_datetime(['2024-12-31'] * 2),
                          'ann_date': pd.to_datetime(['2025-03-20', '2025-08-01']), 'revenue': [1.0, 2.0]})
    assert quality._known(frame, pd.Timestamp('2025-05-01'))['revenue'].tolist() == [1.0]
