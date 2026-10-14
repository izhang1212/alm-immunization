# liability bucket: a fixed schedule of cash flows the insurer owes
    # liabilities have no market YTM, so PV comes from discounting on same curve as assets
    # Macaulay/modified duration use the internal yield: the single yield that reproduces that PV (the liability analogue of YTM)

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import LIABILITIES_FILE
from instruments.cashflows import pv_from_curve, pv_from_yield, solve_yield


@dataclass
class LiabilityBucket:
    times: np.ndarray
    amounts: np.ndarray
    name: str = "Liabilities"

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float)
        self.amounts = np.asarray(self.amounts, dtype=float)
        if self.times.shape != self.amounts.shape:
            raise ValueError("times and amounts must have the same length.")
        if np.any(self.times <= 0):
            raise ValueError("Liability times must be positive.")
        if np.any(np.diff(self.times) <= 0):
            raise ValueError("Liability times must be strictly increasing.")
        if np.any(self.amounts < 0):
            raise ValueError("Liability amounts must be non-negative.")

    @classmethod
    def from_csv(cls, path=LIABILITIES_FILE):
        """Load from CSV with columns: time, amount."""
        df = pd.read_csv(path)
        return cls(times=df["time"].to_numpy(), amounts=df["amount"].to_numpy())

    def pv(self, curve):
        """PV off a curve."""
        return pv_from_curve(self.times, self.amounts, curve)

    def pv_at_yield(self, y):
        """PV at a single yield y."""
        return pv_from_yield(self.times, self.amounts, y)

    def internal_yield(self, pv):
        """Single yield that reproduces a given PV."""
        return solve_yield(self.times, self.amounts, pv)