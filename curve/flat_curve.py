# Build synthetic flat curve for testing and learning
    # FlatCurve: the same zero rate at every maturity. Under a flat curve, every
    # bond's YTM equals the curve rate, so Macaulay/modified/effective duration
    # can be checked against each other with no curve-shape noise.  
    
import numpy as np

from curve.curve import Curve


DEFAULT_FLAT_RATE = 0.04  # 4%, compounded per config.COMPOUNDING_FREQ


class FlatCurve(Curve):
    """Constant zero rate at all maturities."""

    def __init__(self, rate=DEFAULT_FLAT_RATE):
        self.rate = rate

    def zero_rate(self, t):
        return np.full(np.shape(t), self.rate, dtype=float)

    def __repr__(self):
        return f"FlatCurve(rate={self.rate:.4%})"


