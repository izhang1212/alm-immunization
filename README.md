# alm-immunization

## overview

**Purpose:** To show understanding of basic Asset-Liability Management strategies when it comes to managing a particular portfolio.

**Goal:** Given a liability cash flow bucket and a small set of bonds, build an asset portfolio with the same PV and DV01 as the liabilities, then show it holds up under small parallel rate moves.


## process
1. Set up rate environment 
    - Discount factores, compounding conventions, build/bootstrap a yield curve
2. Define liability bucket
    - Build table of (time, cashflow) and compute its PV
3. Define asset universe
    - Build bond cash flow generator (coupon, maturity, freq, face), pricer, and YTM solver
4. Compute metrics
    - Macaulay duration, modified duration, DV01, effective duration (via +- 10 bps shock)
5. Apply metrics at portfolio level
    - Given bond weights, compute portfolio PV, duration, DV01
6. Solve the match
    - Build two-bond barbell (one short and one long bond, bracketing the liability duration). Solve 2x2 linear system for face amounts:
        1. PV_A = PV_L
        2. DV01_A = DV01_L

## outputs