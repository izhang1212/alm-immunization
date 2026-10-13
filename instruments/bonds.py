# fixed-coupon bullet bonds
    # valuation date is t = 0, and is assumed to fall on a coupon date, so there is no accrued interest (clean = dirty)
    # coupons are annual rates in decimal (0.05 = 5%), paid coupon / freq per period

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import BONDS_FILE, COUPON_FREQ, FACE_VALUE
from instruments.cashflows import pv_from_curve, pv_from_yield, solve_yield


@dataclass
class Bond:
    name: str
    coupon: float
    maturity: float
    freq: int = COUPON_FREQ
    face: float = FACE_VALUE

    def __post_init__(self):
        if self.maturity <= 0:
            raise ValueError(f"{self.name}: maturity must be positive.")
        if self.coupon < 0:
            raise ValueError(f"{self.name}: coupon must be non-negative.")
        periods = self.maturity * self.freq
        if self.coupon > 0 and not np.isclose(periods, round(periods)):
            raise ValueError(
                f"{self.name}: maturity {self.maturity} is not a whole number "
                f"of coupon periods at freq {self.freq}."
            )

    @property
    def is_zero_coupon(self):
        return self.coupon == 0

    @property
    def times(self):
        """Cash flow times in years."""
        if self.is_zero_coupon:
            return np.array([self.maturity], dtype=float)
        n = int(round(self.maturity * self.freq))
        return np.arange(1, n + 1, dtype=float) / self.freq

    @property
    def amounts(self):
        """Cash flow amounts: coupons, with face added to the final payment."""
        if self.is_zero_coupon:
            return np.array([self.face], dtype=float)
        amounts = np.full(len(self.times), self.coupon / self.freq * self.face)
        amounts[-1] += self.face
        return amounts

    # Price off a curve: each cf is discounted at its own zero rate
    def price(self, curve):
        return pv_from_curve(self.times, self.amounts, curve)

    # Price bond at a single yield y
    def price_at_yield(self, y):
        return pv_from_yield(self.times, self.amounts, y)

    # yield to maturity implied by a given price
    def ytm(self, price):
        return solve_yield(self.times, self.amounts, price)

# load a bond universe from CSV with cols: name, coupon, maturity, freq, face
def load_bonds(path=BONDS_FILE):
    df = pd.read_csv(path)
    return [
        Bond(
            name=str(row["name"]),
            coupon=float(row["coupon"]),
            maturity=float(row["maturity"]),
            freq=int(row["freq"]),
            face=float(row["face"]),
        )
        for _, row in df.iterrows()
    ]