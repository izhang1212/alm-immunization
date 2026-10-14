"""
Parallel shock testing: how surplus (assets - liabilities) moves when the
whole curve shifts.

For each shock the table records:
  - actual surplus change, from fully repricing both sides on the shifted curve
  - first-order prediction, from DV01 alone:  -(DV01_A - DV01_L) * shift_bp

For a DV01-matched portfolio the prediction is ~0 at every shock, so the
actual change IS the error of the duration approximation. That residual is
convexity, which Part 2 measures directly.
"""

import pandas as pd

from config import PARALLEL_SHOCKS_BP
from duration.metrics import dv01, present_value


def surplus_change(assets, liabilities, curve, shift_bp):
    """Repriced and predicted surplus change for one parallel shift."""
    shocked = curve.shifted(shift_bp)

    assets_base, liab_base = present_value(assets, curve), present_value(liabilities, curve)
    assets_shocked = present_value(assets, shocked)
    liab_shocked = present_value(liabilities, shocked)

    actual = (assets_shocked - liab_shocked) - (assets_base - liab_base)
    predicted = -(dv01(assets, curve) - dv01(liabilities, curve)) * shift_bp

    return {
        "portfolio": assets.name,
        "shock_bp": shift_bp,
        "assets_pv": assets_shocked,
        "liabilities_pv": liab_shocked,
        "surplus": assets_shocked - liab_shocked,
        "surplus_change": actual,
        "predicted_change": predicted,
        "approx_error": actual - predicted,
    }


def shock_table(portfolios, liabilities, curve, shocks_bp=PARALLEL_SHOCKS_BP):
    """Long-format DataFrame: one row per (portfolio, shock)."""
    rows = [
        surplus_change(p, liabilities, curve, s)
        for p in portfolios
        for s in shocks_bp
    ]
    return pd.DataFrame(rows)