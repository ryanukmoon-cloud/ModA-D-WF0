"""
Dual-EMA + ATR-normalised spread strategy  (starter stub).

Replace this with your own signal logic. The structure here — frozen Params
dataclass, feature_key, compute_features / apply_params split — is what the
wfolab WalkForward harness expects.  The only constraint: apply_params must be
fast (just threshold comparisons on precomputed columns).

Feature / threshold split recap
--------------------------------
compute_features(data, key)  — expensive; key = (fast, slow, atr_len)
apply_params(features, p)    — cheap; just vectorised boolean ops on columns
                               that already exist in features

This lets WalkForward run compute_features ONCE per feature-group and reuse the
result across the 4 threshold combos, instead of recomputing it 4 × per group.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from wfolab import Strategy, SignalFrame


# --------------------------------------------------------------------------- #
# Parameter container
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Params:
    fast: int          # fast EMA period
    slow: int          # slow EMA period
    atr_len: int       # ATR lookback for normalisation
    threshold: float   # entry when |spread_atr| exceeds this

    def as_dict(self) -> dict:
        return {"fast": self.fast, "slow": self.slow,
                "atr_len": self.atr_len, "thresh": self.threshold}


# --------------------------------------------------------------------------- #
# Strategy
# --------------------------------------------------------------------------- #

class DualEMAStrategy(Strategy):
    """Long when fast EMA crosses above slow EMA by > threshold × ATR.
    Short when fast EMA crosses below slow EMA by > threshold × ATR.
    Flat otherwise.
    """

    # ------------------------------------------------------------------ grid

    def grid(self) -> list[Params]:
        """Cartesian product of all param combinations to grid-search."""
        combos = []
        for fast in [10, 20, 50]:
            for slow in [50, 100, 200]:
                if fast >= slow:          # skip degenerate combos
                    continue
                for atr_len in [14, 21]:
                    for threshold in [0.1, 0.5, 1.0, 2.0]:
                        combos.append(Params(fast, slow, atr_len, threshold))
        return combos

    # ---------------------------------------------------------------- key / features

    def feature_key(self, p: Params) -> Any:
        """Features depend only on (fast, slow, atr_len); threshold is cheap."""
        return (p.fast, p.slow, p.atr_len)

    def compute_features(self, data: pd.DataFrame, key: Any) -> pd.DataFrame:
        """Compute the expensive, threshold-independent curves.

        Must return a DataFrame that includes at minimum the columns required
        by apply_params, plus 'atr' (used by the engine for position sizing
        and stop calculations).
        """
        fast, slow, atr_len = key
        df = data[["open", "high", "low", "close"]].copy()

        df["ema_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
        df["spread"]   = df["ema_fast"] - df["ema_slow"]

        # ATR (Wilder smoothing)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"]  - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.ewm(span=atr_len, adjust=False).mean()

        # Normalised spread: how many ATRs is the EMA separation?
        df["spread_atr"] = df["spread"] / df["atr"].replace(0, np.nan)

        return df.dropna()

    def apply_params(self, features: pd.DataFrame, p: Params) -> SignalFrame:
        """Apply thresholds to precomputed curves.  Must be fast — no rolling
        windows, no ewm, just vectorised comparisons."""
        df = features.copy()
        df["long_entry"]  = df["spread_atr"] >  p.threshold
        df["short_entry"] = df["spread_atr"] < -p.threshold
        return SignalFrame(df, params=p)
