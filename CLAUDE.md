# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A **starter template** for a new model project wired onto the two methodology
libraries:
- **`datafeed`** (`E:\Data\QTPI\Models\datafeed`) — all data sources (synthetic,
  CSV, Yahoo, Alpaca, eToro, cTrader).
- **`wfolab`** (`E:\Data\QTPI\Models\wfolab`) — walk-forward harness, Backtester,
  Meyers selection filter, metrics, Monte Carlo.

The project supplies a `Strategy` (the signal). The libraries supply the rest.

## Commands (run from this directory)

```powershell
$env:PYTHONUTF8="1"
# Quick smoke test (no credentials, no network)
python run.py single --source synthetic --n-bars 3000
# Walk-forward on synthetic data
python run.py wfo    --source synthetic --n-bars 5000 --train-bars 756 --oos-bars 126
# Walk-forward on Yahoo daily SPY
python run.py wfo    --source yahoo --symbol SPY --train-bars 756 --oos-bars 126
# Parallel WFO on a local CSV (4H bars → adjust --bars-per-year)
python run.py wfo    --source csv --csv data/EURUSD_4H.csv --bars-per-year 1560 --jobs 4
```

Outputs (`results/`): `*_performance.png`, `*_wfo_pnl.png`, `*_report.txt`,
`*_param_evolution.csv`.

## Architecture

```
run.py          ← CLI; calls datafeed.load_data(), WalkForward, build_report()
mymodel/
  config.py     ← DataConfig (duck-types FeedConfig) + re-exports wfolab configs
  strategies/
    stub.py     ← DualEMAStrategy (THE SIGNAL — replace / extend this)
  reporting.py  ← thin wrapper: equity chart, WFO IS/OOS chart, text report
```

The `Strategy` ABC contract (from `wfolab.Strategy`):
- `grid()` → list of frozen-dataclass Params (must be hashable)
- `feature_key(p)` → group key for the expensive-vs-cheap split
- `compute_features(data, key)` → DataFrame with precomputed curves + `atr` column
- `apply_params(features, p)` → `SignalFrame` (adds `long_entry`/`short_entry`)

## How to implement your own signal

1. Edit `mymodel/strategies/stub.py` (or add a new file alongside it).
2. Define a frozen `Params` dataclass with an `as_dict()` method.
3. Implement the four `Strategy` methods above.
4. Import your class in `mymodel/strategies/__init__.py`.
5. Swap `DualEMAStrategy` for your class in `run.py`.

**Feature / threshold split** — if some params only threshold precomputed curves
(cheap) and others define the curves (expensive), use `feature_key()` to group
them. WalkForward will call `compute_features` once per feature-group instead of
once per combo.  No split needed?  Return `params` itself from `feature_key`.

## Dependencies

`wfolab` and `datafeed` are expected as editable installs or on `PYTHONPATH`.
`run.py` prepends their hard-coded paths automatically for local dev:

```powershell
pip install -e E:\Data\QTPI\Models\wfolab
pip install -e E:\Data\QTPI\Models\datafeed
pip install -e .   # this project
```

## Environment

Windows 10, PowerShell default. Python 3.13. Credentials via `.env`
(copy `.env.example`, fill in, never commit). Results land in `results/`
(gitignored).
