# model_template

**Starter template** for a bar-strategy project wired onto
[wfolab](../wfolab) (walk-forward harness) and
[datafeed](../datafeed) (data sources).

Copy this directory, rename `mymodel/` to your project name, implement
your signal in `mymodel/strategies/stub.py`, and you inherit:

- Rolling walk-forward optimisation with the Meyers selection filter
- Fast backtest engine (~8× vs reference loop)
- Per-fold IS + OOS P&L charts with buy-&-hold control
- IS-vs-OOS overfitting gauge in the text report
- Monte Carlo robustness on OOS trade P&Ls
- Parallel grid search (`--jobs N`)
- Six data sources via a single `datafeed.load_data()` call

## Quick start

```powershell
# Install dependencies (editable — source changes take effect immediately)
pip install -e E:\Data\QTPI\Models\wfolab
pip install -e E:\Data\QTPI\Models\datafeed
pip install -e .

# Smoke test — no network, no credentials
python run.py single --source synthetic --n-bars 3000

# Walk-forward on Yahoo SPY daily
python run.py wfo --source yahoo --symbol SPY --train-bars 756 --oos-bars 126
python run.py wfo --source ctrader --symbol SPY --train-bars 756 --oos-bars 126
```

## Data sources

| `--source`  | Market                   | Credentials needed?     |
|-------------|--------------------------|-------------------------|
| `synthetic` | GBM random-walk          | None                    |
| `csv`       | Any (`--csv FILE`)       | None                    |
| `yahoo`     | Equities/ETFs/FX/crypto  | None                    |
| `alpaca`    | US equities (intraday)   | `ALPACA_*` in `.env`    |
| `etoro`     | eToro instruments        | `ETORO_*` in `.env`     |
| `ctrader`   | FX/CFDs (Pepperstone)    | `CTRADER_*` in `.env`   |

Timeframe
['1Min', '5Min', '15Min', '30Min', '1H', '4H', '1D', '1W']

E:\Data\QTPI\Models\ModA-D-WF0>python run.py wfo --source cTrader --symbol SPY --train-bars 756 --oos-bars 126
usage: run.py wfo [-h] [--source {synthetic,csv,yahoo,alpaca,etoro,ctrader}] [--symbol SYMBOL] [--timeframe TIMEFRAME]
                  [--n-bars N_BARS] [--csv FILE] [--bars-per-year BARS_PER_YEAR] [--equity EQUITY]
                  [--commission COMMISSION] [--slippage SLIPPAGE] [--out-dir OUT_DIR] [--label LABEL]
                  [--train-bars TRAIN_BARS] [--oos-bars OOS_BARS] [--min-trades MIN_TRADES] [--max-trades MAX_TRADES]
                  [--bottom-mru-p BOTTOM_MRU_P] [--jobs JOBS] [--benchmark BENCHMARK] [--no-benchmark]

See [datafeed/docs/SOURCES.md](../datafeed/docs/SOURCES.md) for depth, bar
counts, and credential setup.

## Implementing your strategy

```python
# mymodel/strategies/stub.py

from dataclasses import dataclass
from wfolab import Strategy, SignalFrame

@dataclass(frozen=True)
class Params:
    fast: int; slow: int; threshold: float
    def as_dict(self): ...

class MyStrategy(Strategy):
    def grid(self):          # list of Params
    def feature_key(self, p): return (p.fast, p.slow)
    def compute_features(self, data, key): ...  # expensive; returns DataFrame + 'atr'
    def apply_params(self, features, p): ...    # cheap; returns SignalFrame
```

The `feature_key / compute_features / apply_params` split is optional but
important for speed: params that only threshold precomputed curves share one
feature computation per WFO fold. See `stub.py` for a worked example.

## Outputs

Everything lands in `results/` (gitignored):

| File | Contents |
|------|----------|
| `*_performance.png` | Equity curve, drawdown, trade P&L histogram |
| `*_wfo_pnl.png` | IS + OOS cumulative P&L per fold with benchmark overlay |
| `*_report.txt` | Metrics + MC + IS-vs-OOS overfitting gauge |
| `*_param_evolution.csv` | Per-fold chosen params + IS + OOS metrics |
