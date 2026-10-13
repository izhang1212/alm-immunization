# shared cf math for any stream of (time, amount) pairs
    # bonds and liabilites are both just cf streams, so discounting and yield-solving live here once and are reused by both

import numpy as np
from scipy.optimize import brentq

from config import COMPOUNDING_FREQ, YTM_MAX_ITER, YTM_TOL


# Search range for yield solving. Valid for streams of positive cash flows,
# whose PV falls monotonically as yield rises, so there is exactly one root.
YIELD_BRACKET = (-0.5, 1.0)

# PV of cash flows discounted on a given curve
def pv_from_curve(times, amounts, curve):
    return float(np.sum(np.asarray(amounts, dtype=float) * curve.df(times)))

# discount factors at a single yield y, using project compounding
def yield_discount_factors(times, y):
    k = COMPOUNDING_FREQ
    return (1.0 + y / k) ** (-k * np.asarray(times, dtype=float))

# PV of cash flows discounted at a single yield y
def pv_from_yield(times, amounts, y):
    return float(np.sum(np.asarray(amounts, dtype=float) * yield_discount_factors(times, y)))

# solve for single yield y s.t. pv_from_yield(times, amounts, y) == target_pv
    # for bond, this is YTM; for liabilites, it is their internal yield
def solve_yield(times, amounts, target_pv):
    lo, hi = YIELD_BRACKET
    return brentq(
        lambda y: pv_from_yield(times, amounts, y) - target_pv,
        lo,
        hi,
        xtol=YTM_TOL,
        maxiter=YTM_MAX_ITER,
    )