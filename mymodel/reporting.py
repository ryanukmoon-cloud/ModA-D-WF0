"""
Minimal reporting: equity chart + text summary.

Uses only wfolab output objects (WFOResult / BacktestResult) and matplotlib —
no dependency on any consumer package.  Extend as needed.
"""
from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wfolab.metrics import equity_metrics, max_drawdown


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return (equity - peak) / peak


def _rebase_benchmark(benchmark: Optional[pd.Series],
                      dates: pd.DatetimeIndex,
                      initial_equity: float) -> Optional[np.ndarray]:
    """Rebase benchmark to initial_equity, sampled at fold dates."""
    if benchmark is None or benchmark.empty:
        return None
    b = benchmark.astype(float).copy()
    if getattr(b.index, "tz", None) is not None:
        b.index = b.index.tz_localize(None)
    b = b.sort_index()
    grid = pd.Index(sorted(set(list(b.index) + list(dates))))
    aligned = b.reindex(grid).ffill().reindex(dates).dropna()
    if aligned.empty:
        return None
    return aligned.to_numpy() / aligned.iloc[0] * initial_equity


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #

def plot_equity(equity: pd.Series, trade_pnl: np.ndarray,
                path: str, title: str = "Model") -> str:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    axes[0].plot(equity.index, equity.values, lw=1.2)
    axes[0].set_title(f"{title} — Equity Curve")
    axes[0].grid(alpha=0.3)

    dd = _drawdown_series(equity) * 100
    axes[1].fill_between(dd.index, dd.values, 0, color="firebrick", alpha=0.5)
    axes[1].set_title("Drawdown (%)")
    axes[1].grid(alpha=0.3)

    if trade_pnl is not None and len(trade_pnl):
        axes[2].hist(trade_pnl, bins=40, color="steelblue", alpha=0.8)
        axes[2].axvline(0, color="k", lw=0.8)
    axes[2].set_title("Trade P&L Distribution")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_wfo_pnl(param_evolution: pd.DataFrame, path: str,
                 initial_equity: float = 100_000.0,
                 benchmark: Optional[pd.Series] = None,
                 title: str = "Walk-Forward") -> str:
    """Two-panel IS / OOS cumulative P&L with optional buy-&-hold control."""
    pe = param_evolution.sort_values("fold")
    dates = pd.DatetimeIndex(pd.to_datetime(pe["oos_start"]))
    is_cum  = initial_equity + pe["is_net_profit"].fillna(0).cumsum().to_numpy()
    oos_cum = initial_equity + pe["oos_net_profit"].fillna(0).cumsum().to_numpy()
    bench   = _rebase_benchmark(benchmark, dates, initial_equity)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    a1.plot(dates, is_cum, color="steelblue", lw=1.5, label="In-Sample cum. net P&L")
    if bench is not None:
        a1.plot(dates, bench, color="grey", lw=1.1, ls="--", label="Benchmark (buy & hold)")
    a1.axhline(initial_equity, color="k", lw=0.6)
    a1.set_title(f"{title} — In-Sample cumulative net P&L [optimistic; overlapping windows]")
    a1.legend(fontsize=8); a1.grid(alpha=0.3)

    a2.plot(dates, oos_cum, color="seagreen", lw=1.5, label="Out-of-Sample cum. net P&L")
    if bench is not None:
        a2.plot(dates, bench, color="grey", lw=1.1, ls="--", label="Benchmark (buy & hold)")
    a2.axhline(initial_equity, color="k", lw=0.6)
    a2.set_title(f"{title} — Out-of-Sample cumulative net P&L [the honest result]")
    a2.legend(fontsize=8); a2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Text report
# --------------------------------------------------------------------------- #

def _is_oos_summary(pe: pd.DataFrame) -> list[str]:
    if pe is None or pe.empty or "oos_net_profit" not in pe.columns:
        return []
    n = len(pe)
    is_np  = pe["is_net_profit"].fillna(0)
    oos_np = pe["oos_net_profit"].fillna(0)
    is_pf  = pe["is_profit_factor"].replace([np.inf, -np.inf], np.nan).dropna()
    oos_pf = pe["oos_profit_factor"].replace([np.inf, -np.inf], np.nan).dropna()
    degraded = float((oos_np.to_numpy() < is_np.to_numpy()).mean()) if n else 0.0
    oos_pos  = float((oos_np > 0).mean()) if n else 0.0
    lines = [
        "",
        " In-Sample vs Out-of-Sample (overfitting gauge)",
        f"   Folds                : {n}",
        f"   Mean IS  net / fold  : {is_np.mean():,.0f}",
        f"   Mean OOS net / fold  : {oos_np.mean():,.0f}",
    ]
    if len(is_pf):
        lines.append(f"   Mean IS  profit factr: {is_pf.mean():.2f}")
    if len(oos_pf):
        lines.append(f"   Mean OOS profit factr: {oos_pf.mean():.2f}")
    lines += [
        f"   OOS worse than IS    : {degraded:.0%} of folds",
        f"   OOS profitable       : {oos_pos:.0%} of folds",
    ]
    return lines


def build_report(
    out_dir: str,
    label: str,
    equity: pd.Series,
    trade_pnl: np.ndarray,
    extra_metrics: Optional[dict] = None,
    montecarlo=None,
    param_evolution: Optional[pd.DataFrame] = None,
    bars_per_year: int = 252,
    benchmark: Optional[pd.Series] = None,
    initial_equity: float = 100_000.0,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    # Charts
    png = os.path.join(out_dir, f"{label}_performance.png")
    plot_equity(equity, trade_pnl, png, title=label)

    has_wfo = (param_evolution is not None and not param_evolution.empty
               and "oos_net_profit" in param_evolution.columns)
    if has_wfo:
        plot_wfo_pnl(param_evolution,
                     os.path.join(out_dir, f"{label}_wfo_pnl.png"),
                     initial_equity=initial_equity,
                     benchmark=benchmark, title=label)

    # Metrics
    m = equity_metrics(equity, bars_per_year)
    if extra_metrics:
        m.update(extra_metrics)

    lines = [
        "=" * 64,
        f" Report: {label}",
        "=" * 64,
        f" Bars                 : {len(equity)}",
        f" Period               : {equity.index[0]}  ->  {equity.index[-1]}",
        f" Start / End equity   : {equity.iloc[0]:,.0f}  ->  {equity.iloc[-1]:,.0f}",
        f" Net P&L              : {equity.iloc[-1] - equity.iloc[0]:,.0f}",
        "",
        " Performance",
        f"   Ann. return        : {m.get('ann_return', 0):.2%}",
        f"   Ann. vol           : {m.get('ann_vol', 0):.2%}",
        f"   Sharpe             : {m.get('sharpe', 0):.2f}",
        f"   Sortino            : {m.get('sortino', 0):.2f}",
        f"   Calmar / MAR       : {m.get('calmar', 0):.2f}",
        f"   Max drawdown       : {m.get('max_drawdown', 0):.2%}",
        f"   Equity R^2         : {m.get('equity_r2', 0):.3f}",
        f"   Modified K-ratio   : {m.get('mod_k_ratio', 0):.3f}",
    ]
    if extra_metrics:
        lines += [
            "",
            " Trade quality",
            f"   Profit factor      : {extra_metrics.get('profit_factor', float('nan')):.2f}",
            f"   Win rate           : {extra_metrics.get('win_rate', 0):.2%}",
            f"   Expectancy/trade   : {extra_metrics.get('expectancy', 0):,.1f}",
        ]
    if montecarlo is not None:
        lines += [
            "",
            f" Monte Carlo ({montecarlo.method}, {montecarlo.n_runs:,} runs)",
            f"   Median final P&L   : {montecarlo.median_final:,.0f}",
            f"   5% / 95% final P&L : {montecarlo.p05_final:,.0f} / {montecarlo.p95_final:,.0f}",
            f"   Prob(loss)         : {montecarlo.prob_loss:.2%}",
            f"   Median max DD      : {montecarlo.median_max_dd:,.0f}",
            f"   95% max DD         : {montecarlo.p95_max_dd:,.0f}",
        ]
    if has_wfo:
        lines += _is_oos_summary(param_evolution)
    lines.append("=" * 64)

    text = "\n".join(lines)
    txt_path = os.path.join(out_dir, f"{label}_report.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    if has_wfo:
        param_evolution.to_csv(
            os.path.join(out_dir, f"{label}_param_evolution.csv"), index=False)

    return {"text": text, "report_path": txt_path, "chart_path": png, "metrics": m}
