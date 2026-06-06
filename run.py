"""
run.py — CLI orchestrator for mymodel.

Subcommands
-----------
  single   Run one fixed parameter set over the whole series.
  wfo      Rolling walk-forward optimisation (per-fold IS + OOS report).

Data sources (--source)
-----------------------
  synthetic   GBM random-walk (no credentials needed; good for smoke tests)
  csv         Local CSV file (--csv path/to/file.csv)
  yahoo       Yahoo Finance (--symbol AAPL --timeframe 1D)
  alpaca      Alpaca Markets (ALPACA_API_KEY / ALPACA_SECRET_KEY in .env)
  etoro       eToro Open API  (ETORO_API_KEY / ETORO_USER_KEY in .env)
  ctrader     cTrader / Pepperstone  (CTRADER_CLIENT_ID / ... in .env)

Examples
--------
  python run.py single --source synthetic --n-bars 3000
  python run.py wfo    --source yahoo --symbol SPY --train-bars 756 --oos-bars 126
  python run.py wfo    --source csv   --csv data/EURUSD_4H.csv --jobs 4
"""
from __future__ import annotations

import argparse
import os
import sys

# ---- make sibling packages importable without installing ------------------
_here = os.path.dirname(os.path.abspath(__file__))
for _pkg_dir in [
    r"E:\Data\QTPI\Models\datafeed",
    r"E:\Data\QTPI\Models\wfolab",
]:
    if os.path.isdir(_pkg_dir) and _pkg_dir not in sys.path:
        sys.path.insert(0, _pkg_dir)
# ---------------------------------------------------------------------------

from dotenv import load_dotenv          # pip install python-dotenv
load_dotenv()

import numpy as np
import pandas as pd

from wfolab import (
    WalkForward, WFOConfig, BacktestConfig, Backtester,
    monte_carlo,
)
from wfolab.metrics import compute_metrics, equity_metrics, trade_metrics

from mymodel.config import DataConfig, RunConfig
from mymodel.strategies import DualEMAStrategy
from mymodel.reporting import build_report


# --------------------------------------------------------------------------- #
# bars_per_year auto-detection
# --------------------------------------------------------------------------- #

# Approximate trading bars per calendar year, keyed by source-type and timeframe.
# FX/CFD (cTrader, eToro): 260 trading days, ~24 h/day sessions.
# Equity/ETF  (Yahoo, Alpaca, CSV/synthetic default): 252 days, 6.5 h/day.
_BARS_PER_YEAR: dict[str, dict[str, int]] = {
    "equity": {
        "1D": 252, "4H": 403, "1H": 1638, "30Min": 3276,
        "15Min": 6552, "10Min": 9828, "5Min": 19656, "1Min": 98280,
    },
    "fx": {
        "1D": 260, "4H": 1560, "1H": 6240, "30Min": 12480,
        "15Min": 24960, "10Min": 37440, "5Min": 74880, "1Min": 374400,
    },
}
_FX_SOURCES = {"ctrader", "etoro"}


def _auto_bars_per_year(timeframe: str, source: str) -> int:
    """Return a sensible bars-per-year estimate from timeframe + data source."""
    market = "fx" if source in _FX_SOURCES else "equity"
    return _BARS_PER_YEAR.get(market, {}).get(timeframe, 252)


def _resolve_bars_per_year(args) -> int:
    """Return explicit --bars-per-year if given, else auto-detect from timeframe."""
    explicit = getattr(args, "bars_per_year", None)
    if explicit is not None:
        return int(explicit)
    tf  = getattr(args, "timeframe", "1D")
    src = getattr(args, "source", "synthetic")
    bpy = _auto_bars_per_year(tf, src)
    return bpy


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _load_data(args) -> pd.DataFrame:
    """Load OHLCV data via the datafeed package."""
    from datafeed import load_data     # type: ignore[import]

    cfg = DataConfig(
        source    = args.source,
        symbol    = getattr(args, "symbol", "SPY"),
        timeframe = getattr(args, "timeframe", "1D"),
        n_bars    = getattr(args, "n_bars", 3000),
        csv_path  = getattr(args, "csv", None),
    )
    df = load_data(cfg)
    bpy = _resolve_bars_per_year(args)
    print(f"Loaded {len(df):,} bars  ({df.index[0]}  ->  {df.index[-1]})  "
          f"[timeframe={cfg.timeframe}  bars/year={bpy}]")
    return df


def _load_benchmark(symbol: str, start, end) -> pd.Series | None:
    """Fetch a daily close series for the benchmark via Yahoo Finance."""
    if not symbol:
        return None
    try:
        from datafeed import load_data     # type: ignore[import]

        import types
        _cfg = types.SimpleNamespace(
            source="yahoo", symbol=symbol, timeframe="1D",
            n_bars=10_000, csv_path=None,
        )
        df = load_data(_cfg)
        s = df["close"].loc[start:end]
        print(f"Benchmark {symbol}: {len(s)} bars")
        return s
    except Exception as exc:
        print(f"[warn] benchmark fetch failed ({exc}); skipping")
        return None


def _bt_config(args) -> BacktestConfig:
    from wfolab.config import CostConfig
    return BacktestConfig(
        initial_equity = getattr(args, "equity", 100_000.0),
        costs          = CostConfig(
            commission_pct  = getattr(args, "commission", 0.0),
            slippage_points = getattr(args, "slippage", 0.0),
        ),
    )


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #

def run_single(args) -> None:
    """Backtest one fixed parameter set; useful for debugging a signal."""
    data = _load_data(args)
    rc   = RunConfig(out_dir=args.out_dir, label=args.label,
                     initial_equity=getattr(args, "equity", 100_000.0))

    strategy = DualEMAStrategy()
    # Use the first combo in the grid as the default, or hard-code one here:
    params = strategy.grid()[0]
    print(f"Params: {params}")

    sig  = strategy.signals(data, params)
    res  = Backtester(_bt_config(args)).run(sig)
    m    = compute_metrics(res, bars_per_year=_resolve_bars_per_year(args))
    mc   = monte_carlo(res.trade_pnl)

    print(f"Net P&L: {m['net_profit']:,.0f}  |  Trades: {int(m['n_trades'])}  "
          f"|  Sharpe: {m['sharpe']:.2f}  |  PF: {m['profit_factor']:.2f}")

    result = build_report(
        out_dir       = rc.out_dir,
        label         = rc.label,
        equity        = res.equity,
        trade_pnl     = res.trade_pnl,
        extra_metrics = m,
        montecarlo    = mc,
        bars_per_year = _resolve_bars_per_year(args),
        initial_equity= rc.initial_equity,
    )
    print(result["text"])
    print(f"\nReport : {result['report_path']}")
    print(f"Chart  : {result['chart_path']}")


def run_wfo(args) -> None:
    """Rolling walk-forward optimisation with per-fold IS + OOS reporting."""
    data = _load_data(args)
    rc   = RunConfig(
        out_dir        = args.out_dir,
        label          = args.label,
        n_jobs         = getattr(args, "jobs", 1),
        initial_equity = getattr(args, "equity", 100_000.0),
    )

    wfo_cfg = WFOConfig(
        train_bars   = args.train_bars,
        oos_bars     = args.oos_bars,
        min_trades   = getattr(args, "min_trades", 10),
        max_trades   = getattr(args, "max_trades", 200),
        bottom_mru_p = getattr(args, "bottom_mru_p", 20),
    )

    strategy = DualEMAStrategy()
    wf = WalkForward(wfo_cfg, _bt_config(args),
                     bars_per_year=_resolve_bars_per_year(args),
                     n_jobs=rc.n_jobs,
                     strategy=strategy)

    print(f"Grid: {len(strategy.grid())} combos  |  "
          f"train={args.train_bars} bars  oos={args.oos_bars} bars  "
          f"jobs={rc.n_jobs}")
    wfo_result = wf.run(data, progress=True)

    if wfo_result.oos_equity.empty:
        print("[warn] No WFO folds completed — series too short.")
        return

    pe = wfo_result.param_evolution
    bm_sym = None if getattr(args, "no_benchmark", False) else getattr(args, "benchmark", "^GSPC")
    benchmark = _load_benchmark(bm_sym, data.index[0], data.index[-1]) if bm_sym else None

    bpy = _resolve_bars_per_year(args)
    m = equity_metrics(wfo_result.oos_equity, bpy)
    # Augment with trade-level metrics from the stitched OOS P&Ls.
    pnl = wfo_result.oos_trades_pnl
    if len(pnl):
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        gross_profit = float(wins.sum())
        gross_loss   = abs(float(losses.sum()))
        m["n_trades"]      = len(pnl)
        m["profit_factor"] = gross_profit / gross_loss if gross_loss else float("inf")
        m["win_rate"]      = float(len(wins) / len(pnl))
        m["expectancy"]    = float(pnl.mean())
    mc = monte_carlo(pnl)

    result = build_report(
        out_dir        = rc.out_dir,
        label          = rc.label,
        equity         = wfo_result.oos_equity,
        trade_pnl      = wfo_result.oos_trades_pnl,
        extra_metrics  = m,
        montecarlo     = mc,
        param_evolution= pe,
        bars_per_year  = _resolve_bars_per_year(args),
        benchmark      = benchmark,
        initial_equity = rc.initial_equity,
    )
    print(result["text"])
    print(f"\nReport : {result['report_path']}")
    print(f"Chart  : {result['chart_path']}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _common(p: argparse.ArgumentParser) -> None:
    """Shared flags for all subcommands."""
    p.add_argument("--source",       default="synthetic",
                   choices=["synthetic","csv","yahoo","alpaca","etoro","ctrader"])
    p.add_argument("--symbol",       default="SPY")
    p.add_argument("--timeframe",    default="1D")
    p.add_argument("--n-bars",       type=int, default=3000, dest="n_bars")
    p.add_argument("--csv",          default=None, metavar="FILE")
    p.add_argument("--bars-per-year",type=int, default=None, dest="bars_per_year",
                   metavar="N",
                   help="Bars per calendar year for annualising metrics. "
                        "Auto-detected from --timeframe + --source if omitted "
                        "(e.g. 1D equity=252, 4H FX=1560).")
    p.add_argument("--equity",       type=float, default=100_000.0)
    p.add_argument("--commission",   type=float, default=0.0,
                   help="Round-trip commission as a fraction (e.g. 0.001 = 0.1%%)")
    p.add_argument("--slippage",     type=int, default=0,
                   help="Slippage in ticks per fill")
    p.add_argument("--out-dir",      default="results", dest="out_dir")
    p.add_argument("--label",        default="mymodel")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="mymodel — wfolab-powered strategy runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ---- single ----
    ps = sub.add_parser("single", help="One fixed param set over the whole series")
    _common(ps)

    # ---- wfo ----
    pw = sub.add_parser("wfo", help="Rolling walk-forward optimisation")
    _common(pw)
    pw.add_argument("--train-bars",  type=int, default=756, dest="train_bars")
    pw.add_argument("--oos-bars",    type=int, default=126, dest="oos_bars")
    pw.add_argument("--min-trades",  type=int, default=10,  dest="min_trades")
    pw.add_argument("--max-trades",  type=int, default=200, dest="max_trades")
    pw.add_argument("--bottom-mru-p",type=int, default=20,  dest="bottom_mru_p")
    pw.add_argument("--jobs",        type=int, default=1)
    pw.add_argument("--benchmark",   default="^GSPC",
                    help="Yahoo ticker for buy-&-hold control (default: ^GSPC)")
    pw.add_argument("--no-benchmark",action="store_true", dest="no_benchmark")

    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "single":
        run_single(args)
    elif args.cmd == "wfo":
        run_wfo(args)


if __name__ == "__main__":
    main()
