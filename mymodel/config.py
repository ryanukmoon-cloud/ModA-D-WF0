"""
Project-level configuration containers.

DataConfig duck-types datafeed.FeedConfig so run.py can pass it directly to
datafeed.load_data(). Execution/WFO/risk settings re-export from wfolab so
you only import from one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Re-export the generic configs from wfolab so consuming code only needs one import.
from wfolab.config import (   # noqa: F401  (re-export)
    BacktestConfig,
    WFOConfig,
    CostConfig,
    SizingConfig,
    ExitConfig,
    RiskLimits,
)


@dataclass
class DataConfig:
    """Duck-types datafeed.FeedConfig.  Add source-specific fields as needed."""
    source: str = "synthetic"        # synthetic | csv | yahoo | alpaca | etoro | ctrader
    symbol: str = "SPY"
    timeframe: str = "1D"
    n_bars: int = 3000
    csv_path: Optional[str] = None
    bars_per_year: int = 252         # 252 for daily; 252*390 for 1-min; etc.

    # Synthetic GBM extras (ignored for other sources)
    seed: int = 42
    annual_drift: float = 0.05
    annual_vol: float = 0.20
    start_price: float = 100.0
    regime_switch: bool = False

    # cTrader / Alpaca / eToro extras
    account_type: str = "demo"       # demo | live
    env: str = "demo"
    datetime_col: str = "datetime"   # CSV column name for the timestamp


@dataclass
class RunConfig:
    """Top-level run settings (not model params — those live in Params)."""
    out_dir: str = "results"
    label: str = "mymodel"
    n_jobs: int = 1                  # >1 → parallel WFO grid search
    initial_equity: float = 100_000.0
    benchmark: Optional[str] = "^GSPC"  # Yahoo ticker for buy-&-hold control; None to skip

    # WFO defaults (can also be passed directly to WalkForward)
    train_bars: int = 756            # ~3 years of daily bars
    oos_bars: int = 126              # ~6 months of daily bars
