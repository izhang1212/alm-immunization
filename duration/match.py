"""
Duration matching: build asset portfolios whose PV and DV01 equal the
liabilities'.

Three constructions:

  solve_barbell:
      Two bonds, one shorter and one longer than the liabilities.
      Solve the 2x2 linear system for units:

          n_s * P_s    + n_l * P_l    = PV_L
          n_s * DV01_s + n_l * DV01_l = DV01_L

      PV and DV01 are matched exactly. A long-only solution exists only if
      the liability duration lies strictly between the two bonds' durations.

  solve_bullet:
      One bond, sized to match PV only. DV01 matches only if that bond's
      duration happens to equal the liabilities'. Shows what you get
      without a duration constraint.

  duration_matched_zero:
      A synthetic zero-coupon bond whose maturity is solved so its effective
      duration equals the liabilities', sized to match PV. This is the exact
      bullet: PV and DV01 both matched with a single cash flow. It's the
      comparison case for the barbell in Part 2 (same duration, different
      convexity).

All matching is on effective duration / DV01, which are curve-based and
aggregate exactly across holdings.
"""

import numpy as np
from scipy.optimize import brentq

from config import FACE_VALUE
from duration.metrics import dv01, effective_duration, present_value
from duration.portfolio import Portfolio
from instruments.bonds import Bond


# Search range (years) for the duration-matched zero's maturity.
ZERO_MATURITY_BRACKET = (0.01, 100.0)


def solve_barbell(short_bond, long_bond, liabilities, curve, name="Barbell"):
    """Units of two bonds that match liability PV and DV01 exactly."""
    a = np.array([
        [present_value(short_bond, curve), present_value(long_bond, curve)],
        [dv01(short_bond, curve), dv01(long_bond, curve)],
    ])
    b = np.array([present_value(liabilities, curve), dv01(liabilities, curve)])

    if np.isclose(np.linalg.det(a), 0.0):
        raise ValueError(
            f"{short_bond.name} and {long_bond.name} have the same duration; "
            "the system has no unique solution."
        )

    units = np.linalg.solve(a, b)

    if np.any(units < 0):
        d_liab = effective_duration(liabilities, curve)
        d_short = effective_duration(short_bond, curve)
        d_long = effective_duration(long_bond, curve)
        raise ValueError(
            f"No long-only barbell: liability duration {d_liab:.3f} must lie "
            f"strictly between {short_bond.name} ({d_short:.3f}) and "
            f"{long_bond.name} ({d_long:.3f})."
        )

    return Portfolio.from_units([short_bond, long_bond], units, name=name)


def solve_bullet(bond, liabilities, curve, name=None):
    """Units of a single bond that match liability PV (DV01 not targeted)."""
    units = present_value(liabilities, curve) / present_value(bond, curve)
    return Portfolio.from_units([bond], [units], name=name or f"Bullet ({bond.name})")


def duration_matched_zero(liabilities, curve, name="Zero (matched)"):
    """
    Zero-coupon bond whose effective duration equals the liabilities',
    sized to match PV. Matches both PV and DV01 with one cash flow.
    """
    target = effective_duration(liabilities, curve)

    def gap(maturity):
        zero = Bond(name="ZERO_MATCH", coupon=0.0, maturity=maturity, face=FACE_VALUE)
        return effective_duration(zero, curve) - target

    lo, hi = ZERO_MATURITY_BRACKET
    maturity = brentq(gap, lo, hi)
    zero = Bond(name=f"ZERO_{maturity:.2f}Y", coupon=0.0, maturity=maturity, face=FACE_VALUE)
    return solve_bullet(zero, liabilities, curve, name=name)


def match_gaps(assets, liabilities, curve):
    """
    How well assets match liabilities. Every gap is assets minus liabilities.

    duration_gap uses D_A - (PV_L / PV_A) * D_L, which is zero exactly when
    DV01s match, even if PVs differ. Plain D_A - D_L is only the right
    measure when PV_A == PV_L.
    """
    pv_a, pv_l = present_value(assets, curve), present_value(liabilities, curve)
    d_a, d_l = effective_duration(assets, curve), effective_duration(liabilities, curve)
    dv01_a, dv01_l = dv01(assets, curve), dv01(liabilities, curve)
    return {
        "name": assets.name,
        "pv_gap": pv_a - pv_l,
        "dv01_gap": dv01_a - dv01_l,
        "duration_gap": d_a - (pv_l / pv_a) * d_l,
    }