"""
First-order interest rate risk metrics for any cash flow stream.

Every function takes an `instrument` exposing `.times`, `.amounts` and
`.name` (a Bond, a LiabilityBucket, or anything shaped like them) and a
curve. PV always comes from the curve, so assets and liabilities are
valued on the same basis.

Two families of measures:

  Yield-based (Macaulay, modified):
      Computed at the instrument's own yield: its YTM for a bond, its
      internal yield for liabilities. Each instrument gets its own yield,
      so these do NOT aggregate exactly across a portfolio on a sloped curve.

  Curve-based (effective duration, DV01):
      Computed by shifting the whole curve up and down and repricing.
      Every instrument sees the same shift, so these DO aggregate exactly.
      This is what the matching step uses.

Under a flat curve with matching compounding (config.COMPOUNDING_FREQ),
effective duration equals modified duration up to a tiny central-difference
error. That equality is the main sanity check for this module.
"""

import numpy as np

from config import BP, BUMP_BP, COMPOUNDING_FREQ
from instruments.cashflows import pv_from_curve, solve_yield, yield_discount_factors


# ---------------------------------------------------------------------------
# Value and yield
# ---------------------------------------------------------------------------

# PV off the curve. Per 100 face for a bond, in dollars for liabilities 
def present_value(instrument, curve):
    return pv_from_curve(instrument.times, instrument.amounts, curve)

# Single yield that reproduces the curve PV (YTM for a bond).
def instrument_yield(instrument, curve):
    return solve_yield(instrument.times, instrument.amounts, present_value(instrument, curve))


# ---------------------------------------------------------------------------
# Yield-based measures
# ---------------------------------------------------------------------------

# PV-weighted average time to cash flows, in years, at instrument's yield
    # D_mac = sum( t_i * CF_i * DF_y(t_i) ) / P
def macaulay_duration(instrument, curve):
    y = instrument_yield(instrument, curve)
    times = np.asarray(instrument.times, dtype=float)
    pv_flows = np.asarray(instrument.amounts, dtype=float) * yield_discount_factors(times, y)
    return float(np.sum(times * pv_flows) / np.sum(pv_flows))

# Percent price change per unit change in yield (first derivative of price with respect to yield)
    # D_mod = -(1/P) dP/dy = D_mac / (1 + y/k)
def modified_duration(instrument, curve):
    y = instrument_yield(instrument, curve)
    return macaulay_duration(instrument, curve) / (1.0 + y / COMPOUNDING_FREQ)


# ---------------------------------------------------------------------------
# Curve-based measures (bump and reprice)
# ---------------------------------------------------------------------------

# PV with the curve shifted down and up by bump_bp.
def bumped_values(instrument, curve, bump_bp):
    pv_down = present_value(instrument, curve.shifted(-bump_bp))
    pv_up = present_value(instrument, curve.shifted(bump_bp))
    return pv_down, pv_up

# Effective duration under a parallel curve shift.
    # D_eff = (P(-h) - P(+h)) / (2 * P * h)
def effective_duration(instrument, curve, bump_bp=BUMP_BP):
    pv = present_value(instrument, curve)
    pv_down, pv_up = bumped_values(instrument, curve, bump_bp)
    h = bump_bp * BP
    return (pv_down - pv_up) / (2.0 * pv * h)

# DV01: Dollar value of a 1bp parallel rise in rates
    # DV01 = -(dP/dy) * 1bp = (P(-h) - P(+h)) / (2 * h) * 1bp
    # Per 100 face for a bond, in dollars for liabilities.
def dv01(instrument, curve, bump_bp=BUMP_BP):
    pv_down, pv_up = bumped_values(instrument, curve, bump_bp)
    return (pv_down - pv_up) / (2.0 * bump_bp)

# Analytical DV01 from modified duration: D_mod * P * 1bp
def dv01_from_duration(instrument, curve):
    return modified_duration(instrument, curve) * present_value(instrument, curve) * BP


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

# All Part 1 metrics for one instrument, as a dict (one table row).
def compute_metrics(instrument, curve):
    return {
        "name": instrument.name,
        "pv": present_value(instrument, curve),
        "yield": instrument_yield(instrument, curve),
        "macaulay": macaulay_duration(instrument, curve),
        "modified": modified_duration(instrument, curve),
        "effective": effective_duration(instrument, curve),
        "dv01": dv01(instrument, curve),
    }