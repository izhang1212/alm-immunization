"""
Part 1: duration matching on a flat curve.

Run from the project root:  python main.py

Change the settings below (curve rate, barbell pairs, bullet bond) or the
CSVs in data/ and every table and chart updates to match.
"""

import numpy as np
import pandas as pd

import plots
from config import FACE_VALUE, OUTPUT_DIR
from curve.flat_curve import FlatCurve
from duration.match import duration_matched_zero, match_gaps, solve_barbell, solve_bullet
from duration.metrics import compute_metrics, dv01, dv01_from_duration
from duration.shocks import shock_table
from instruments.bonds import load_bonds
from instruments.liabilities import LiabilityBucket


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

FLAT_RATE = 0.04

# (short bond, long bond) pairs to solve as barbells. Names from data/bonds.csv.
BARBELL_PAIRS = [
    ("BOND_02Y", "BOND_30Y"),
    ("ZERO_05Y", "ZERO_20Y"),
]

# Single bond sized to match PV only (DV01 not targeted).
BULLET_BOND = "BOND_10Y"

# Include a synthetic zero whose duration exactly matches the liabilities.
INCLUDE_MATCHED_ZERO = True

# Tolerance for the sanity checks.
TOL = 1e-6


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def check(label, passed, detail=""):
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}{f'  ({detail})' if detail else ''}")


def format_metrics(df):
    out = df.copy()
    out["yield"] = out["yield"].map("{:.4%}".format)
    for col in ["macaulay", "modified", "effective"]:
        out[col] = out[col].map("{:.4f}".format)
    out["pv"] = out["pv"].map("{:,.4f}".format)
    out["dv01"] = out["dv01"].map("{:,.6f}".format)
    return out.set_index("name")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def main():
    curve = FlatCurve(FLAT_RATE)
    bonds = load_bonds()
    bonds_by_name = {b.name: b for b in bonds}
    liabilities = LiabilityBucket.from_csv()

    # 1. Instrument metrics ---------------------------------------------------
    section(f"1. Instrument metrics  (flat curve {FLAT_RATE:.2%}; bonds per {FACE_VALUE:.0f} face, liabilities in $)")
    instrument_metrics = pd.DataFrame(
        [compute_metrics(b, curve) for b in bonds] + [compute_metrics(liabilities, curve)]
    )
    print(format_metrics(instrument_metrics))

    # 2. Sanity checks --------------------------------------------------------
    section("2. Sanity checks")

    ytm_err = max(abs(r["yield"] - FLAT_RATE) for _, r in instrument_metrics.iterrows())
    check("Every YTM equals the flat curve rate", ytm_err < TOL, f"max error {ytm_err:.2e}")

    par = [b for b in bonds if np.isclose(b.coupon, FLAT_RATE)]
    if par:
        par_err = max(abs(b.price(curve) - FACE_VALUE) for b in par)
        check("Par bonds (coupon = curve rate) price at face", par_err < TOL,
              f"{', '.join(b.name for b in par)}; max error {par_err:.2e}")

    zeros = [b for b in bonds if b.is_zero_coupon]
    if zeros:
        mac = instrument_metrics.set_index("name")["macaulay"]
        zero_err = max(abs(mac[b.name] - b.maturity) for b in zeros)
        check("Zero-coupon Macaulay duration equals maturity", zero_err < TOL,
              f"max error {zero_err:.2e}")

    eff_mod_err = (instrument_metrics["effective"] - instrument_metrics["modified"]).abs().max()
    check("Effective duration ≈ modified duration (flat curve)", eff_mod_err < 1e-4,
          f"max gap {eff_mod_err:.2e}")

    all_instruments = bonds + [liabilities]
    dv01_rel_err = max(
        abs(dv01(i, curve) - dv01_from_duration(i, curve)) / dv01(i, curve) for i in all_instruments
    )
    check("Bump DV01 ≈ D_mod × P × 1bp", dv01_rel_err < 1e-5, f"max relative gap {dv01_rel_err:.2e}")

    # 3. Build matched portfolios ---------------------------------------------
    section("3. Matched portfolios")
    portfolios = []
    for short_name, long_name in BARBELL_PAIRS:
        try:
            p = solve_barbell(bonds_by_name[short_name], bonds_by_name[long_name], liabilities,
                              curve, name=f"Barbell {short_name}/{long_name}")
            portfolios.append(p)
        except ValueError as e:
            print(f"  Skipped barbell {short_name}/{long_name}: {e}")

    portfolios.append(solve_bullet(bonds_by_name[BULLET_BOND], liabilities, curve))
    if INCLUDE_MATCHED_ZERO:
        portfolios.append(duration_matched_zero(liabilities, curve))

    for p in portfolios:
        print(f"\n  {p.name}")
        for h in p.holdings:
            print(f"    {h.bond.name:<12} {h.units:>14,.2f} units  "
                  f"(face ${h.units * h.bond.face:>16,.2f}, value ${h.value(curve):>16,.2f})")

    # 4. Portfolio metrics and match quality ----------------------------------
    section("4. Portfolio metrics vs. liabilities")
    portfolio_metrics = pd.DataFrame(
        [compute_metrics(p, curve) for p in portfolios] + [compute_metrics(liabilities, curve)]
    )
    print(format_metrics(portfolio_metrics))

    print()
    gaps = pd.DataFrame([match_gaps(p, liabilities, curve) for p in portfolios]).set_index("name")
    print(gaps.map(lambda v: f"{v:,.6f}"))

    # 5. DV01 aggregation -----------------------------------------------------
    section("5. DV01 adds across holdings")
    for p in portfolios:
        summed = sum(h.units * dv01(h.bond, curve) for h in p.holdings)
        direct = dv01(p, curve)
        check(p.name, abs(summed - direct) < 1e-6 * direct,
              f"sum of holdings ${summed:,.4f} vs portfolio ${direct:,.4f}")

    # 6. Parallel shocks ------------------------------------------------------
    section("6. Surplus change under parallel shocks ($)")
    shocks = shock_table(portfolios, liabilities, curve)
    pivot = shocks.pivot(index="shock_bp", columns="portfolio", values="surplus_change")
    print(pivot[[p.name for p in portfolios]].map(lambda v: f"{v:,.0f}"))

    # 7. Charts ---------------------------------------------------------------
    section(f"7. Charts written to {OUTPUT_DIR}")
    liability_dv01 = dv01(liabilities, curve)
    paths = [
        plots.plot_durations(instrument_metrics, liabilities.name),
        plots.plot_cash_flows(portfolios, liabilities),
        plots.plot_dv01_match(portfolio_metrics[portfolio_metrics["name"] != liabilities.name],
                              liability_dv01),
        plots.plot_surplus_vs_shock(shocks),
        plots.plot_approximation_error(shocks),
    ]
    for path in paths:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()