"""
Asset portfolio: a set of bond holdings treated as one cash flow stream.

A holding is (bond, units), where units = number of bonds held. Bonds are
quoted per 100 face, so units = face amount / 100, and portfolio values
come out in dollars.

The portfolio merges every holding's cash flows onto one timeline and
exposes `.times`, `.amounts` and `.name`, the same interface as a Bond or
LiabilityBucket. That means every function in duration/metrics.py works
on a portfolio unchanged.

Aggregation:
  - PV and DV01 add exactly across holdings (curve-based, same shift).
  - Effective duration of the portfolio = DV01-weighted, i.e.
    sum(units * DV01_i) / (PV * 1bp), which equals the PV-weighted average
    of holding effective durations.
  - Macaulay/modified are computed on the merged stream at the portfolio's
    own yield. On a flat curve this equals the PV-weighted average of the
    holdings' durations; on a sloped curve it is only approximately so.
"""

from dataclasses import dataclass, field

import numpy as np

from instruments.cashflows import pv_from_curve


# Cash flow times from different bonds are merged if they agree to this
# many decimal places (e.g. 0.5 from a 2Y bond and 0.5 from a 30Y bond).
TIME_DECIMALS = 10


@dataclass
class Holding:
    bond: object
    units: float

    def __post_init__(self):
        if self.units < 0:
            raise ValueError(
                f"{self.bond.name}: negative units ({self.units}). "
                "Part 1 portfolios are long-only."
            )

    @property
    def times(self):
        return self.bond.times

    @property
    def amounts(self):
        return self.units * self.bond.amounts

    def value(self, curve):
        """Dollar value of the holding."""
        return pv_from_curve(self.times, self.amounts, curve)


@dataclass
class Portfolio:
    holdings: list = field(default_factory=list)
    name: str = "Assets"

    @classmethod
    def from_units(cls, bonds, units, name="Assets"):
        """Build from parallel lists of bonds and unit counts."""
        if len(bonds) != len(units):
            raise ValueError("bonds and units must have the same length.")
        return cls([Holding(b, float(u)) for b, u in zip(bonds, units)], name=name)

    @property
    def times(self):
        """Sorted unique cash flow times across all holdings."""
        return self._merged()[0]

    @property
    def amounts(self):
        """Total cash flow at each time in `times`."""
        return self._merged()[1]

    def _merged(self):
        if not self.holdings:
            return np.array([], dtype=float), np.array([], dtype=float)
        all_times = np.round(np.concatenate([h.times for h in self.holdings]), TIME_DECIMALS)
        all_amounts = np.concatenate([h.amounts for h in self.holdings])
        times, index = np.unique(all_times, return_inverse=True)
        amounts = np.zeros(len(times))
        np.add.at(amounts, index, all_amounts)
        return times, amounts

    def holding_values(self, curve):
        """Dollar value of each holding, keyed by bond name."""
        return {h.bond.name: h.value(curve) for h in self.holdings}

    def weights(self, curve):
        """Each holding's share of portfolio value."""
        values = self.holding_values(curve)
        total = sum(values.values())
        return {name: v / total for name, v in values.items()}