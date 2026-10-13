# Project-wide conventions for the ALM immunization project.

from pathlib import Path

# Paths

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
CURVE_DIR = PROJECT_ROOT / "curve"
OUTPUT_DIR = PROJECT_ROOT / "output"

BONDS_FILE = DATA_DIR / "bonds.csv"
LIABILITIES_FILE = DATA_DIR / "liabilities.csv"


# Rate conventions

# One compounding convention for the whole project: zero rates on the curve AND bond yields are both quoted with this frequency.
#   2 -> semiannual (US Treasury convention)
#
# Using the same convention on both sides means:
#   - under a flat curve at rate y, every bond's YTM is exactly y
#   - bumping the curve by 1bp is the same as bumping each bond's yield by 1bp
#   - effective duration (bump-and-reprice) matches modified duration
# Mixing conventions (e.g. continuous zero rates, semiannual yields) breaks those equalities and is the most common source of duration bugs.
COMPOUNDING_FREQ = 2

# Default coupon frequency for bonds in the synthetic universe.
COUPON_FREQ = 2

# Default face (par) value per bond.
FACE_VALUE = 100.0


# Units

# Time is measured in years as floats (e.g. 2.5 = two and a half years).
# No day-count conventions or calendar dates: synthetic data doesn't need them.

# One basis point in decimal terms.
BP = 1e-4


# Risk measurement

# Bump size for effective duration / DV01 via central difference:
#   eff_dur = (P(y - h) - P(y + h)) / (2 * P * h)
# 1bp is small enough that the central-difference error is negligible.
BUMP_BP = 1.0

# Sign convention: DV01 is POSITIVE for a long bond.
#   DV01 = -(dP/dy) * BP
# i.e. the dollar amount the position LOSES when rates rise by 1bp.


# Scenarios

# Parallel shocks applied to the curve when testing the hedge (in bp).
PARALLEL_SHOCKS_BP = [-200, -100, -50, -10, 0, 10, 50, 100, 200]


# Numerics

# YTM root-finder settings.
YTM_TOL = 1e-12
YTM_MAX_ITER = 100

# Seed for synthetic data generation, so results are reproducible.
RANDOM_SEED = 42