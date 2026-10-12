# quant

个人量化研究：本地行情库 → 回测 → 每日信号（人工下单）。讨论记录与已定结论见 `../docs/discussion.md`。

## 目录（东京工作区约定）

| 路径 | 内容 |
| --- | --- |
| `quant/src` | 本仓库（`quantlab` 包、测试） |
| `quant/prd/quant.env` | 凭据与数据目录（0600，不进 Git），格式见 `.env.example` |
| `quant/prd/data/market.duckdb` | 本地行情库 |
| `quant/prd/logs/` | 同步日志 |

## 使用

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/quant sync calendar          # A/港/美交易日历
.venv/bin/quant sync cn                # A 股全市场（含创业板、科创板、北交所、退市股）+ 主要指数
.venv/bin/quant sync hk                # 港股通持股（时点资格）+ 历史上进过港股通的港股日线
.venv/bin/quant sync us --codes-file us.txt
.venv/bin/quant status
.venv/bin/pytest -q
```

默认从 2014-01-01 开始；所有同步可断点续传，重复运行只补缺失部分。

## 数据口径

- 价格不复权；`adj_factor` 为累计后复权因子，`close * adj_factor` 即后复权收盘价。
- 成交量单位为股，成交额与市值为本币元；港股成交额来源缺失，记为空。
- A 股 `up_limit` / `down_limit` 是交易所当日涨跌停价，回测据此判断能否成交。
- 来源：A 股与日历、港股通持股来自 Tushare；港美股日线来自 TickFlow（与 `/invest` 相同）。
