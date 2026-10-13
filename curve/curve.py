"""
Curve interface used by every other module in the project.

A curve is defined entirely by its zero rates z(t), quoted with the
project-wide compounding frequency k (config.COMPOUNDING_FREQ):

    DF(t) = (1 + z(t) / k) ** (-k * t)

Subclasses (flat, Nelson-Siegel, bootstrapped) only implement zero_rate(t).
Discount factors and parallel shifts are handled here, so pricing and risk
code never needs to know which kind of curve it was given.
"""

from abc import ABC, abstractmethod

import numpy as np

from config import BP, COMPOUNDING_FREQ


class Curve(ABC):
    """Base zero curve. Subclasses implement zero_rate(t)."""

    compounding = COMPOUNDING_FREQ

    @abstractmethod
    def zero_rate(self, t):
        """
        Zero rate(s) for maturity t in years, as decimals (0.04 = 4%),
        compounded `compounding` times per year.

        t may be a float or a numpy array; the return matches its shape.
        """

    def df(self, t):
        """Discount factor(s) for maturity t in years."""
        t = np.asarray(t, dtype=float)
        if np.any(t < 0):
            raise ValueError("Discount factors are only defined for t >= 0.")
        k = self.compounding
        z = self.zero_rate(t)
        return (1.0 + z / k) ** (-k * t)

    def shifted(self, shift_bp):
        """
        New curve with every zero rate moved by shift_bp basis points.
        The original curve is left unchanged.
        """
        return ShiftedCurve(self, shift_bp * BP)


class ShiftedCurve(Curve):
    """A base curve with a constant (parallel) shift added to its zero rates."""

    def __init__(self, base, shift):
        self.base = base
        self.shift = shift  # decimal, e.g. 0.0001 = 1bp

    def zero_rate(self, t):
        return self.base.zero_rate(t) + self.shift

    def __repr__(self):
        return f"ShiftedCurve({self.base!r}, shift={self.shift / BP:+.1f}bp)"