"""Validation against Petroleum Production Systems, Examples 7-8 and 7-9.

Same checks as BUENROSTRO-HB-BB-Methods-HK5-P7-8-VER1.ipynb, as pytest tests.
"""
import numpy as np
import pytest

from pete662_nodal.hagedorn_brown import (hb_dimensionless_groups, hb_holdup_from_groups,
                                          mixture_gradient, bubble_flow_limit)
from pete662_nodal.beggs_brill import bb_flow_regime, bb_holdup

D = 2.259 / 12.0           # ft, 2-7/8 in tubing
USL, USG, UM = 4.67, 8.72, 13.39
RHO_L, RHO_G, MU_L, MU_G, SIGMA = 49.9, 2.6, 2.0, 0.0131, 30.0
P = 800.0


# ---------------- Example 7-8: modified Hagedorn-Brown ----------------
def test_ex78_not_bubble_flow():
    lam_g = USG / UM
    assert bubble_flow_limit(UM, D) == pytest.approx(0.13)
    assert lam_g > 0.13


def test_ex78_dimensionless_groups():
    NvL, Nvg, ND, NL = hb_dimensionless_groups(USL, USG, RHO_L, MU_L, SIGMA, D)
    assert NvL == pytest.approx(10.28, abs=0.01)
    assert Nvg == pytest.approx(19.20, abs=0.02)
    assert ND == pytest.approx(29.35, abs=0.01)
    assert NL == pytest.approx(0.00926, abs=5e-5)


def test_ex78_holdup_from_equations():
    """Direct evaluation of Eqs. 7-103..7-107 gives HL ~ 0.508; the book's
    worked example uses chart readings (HL ~ 0.46).  Documented difference."""
    groups = hb_dimensionless_groups(USL, USG, RHO_L, MU_L, SIGMA, D)
    HL, det = hb_holdup_from_groups(*groups, P)
    assert det["CNL"] == pytest.approx(0.00235, abs=2e-5)
    assert det["psi"] == pytest.approx(1.062, abs=0.002)
    assert HL == pytest.approx(0.508, abs=0.002)


def test_ex78_gradient_with_book_holdup():
    """With the book's HL = 0.46 the gradient arithmetic gives 0.197 psi/ft
    (book reports 0.190; difference documented in the VER1 notebook)."""
    HL = 0.46
    rho_bar = HL * RHO_L + (1 - HL) * RHO_G
    assert rho_bar == pytest.approx(24.36, abs=0.01)
    mdot = 614000.0
    mu_mix = MU_L ** HL * MU_G ** (1 - HL)
    Re = 2.2e-2 * mdot / (D * mu_mix)
    assert Re == pytest.approx(5.42e5, rel=0.005)
    # use the book's f = 0.0046 directly: reproduce the 0.1974 psi/ft check
    grad = (rho_bar + 0.0046 * mdot ** 2 / (7.413e10 * D ** 5 * rho_bar)) / 144.0
    assert grad == pytest.approx(0.1974, abs=5e-4)
    # and the library function with a computed friction factor lands close
    assert mixture_gradient(rho_bar, mdot, mu_mix, D, 0.0006) == pytest.approx(0.197, abs=0.003)


# ---------------- Example 7-9: Beggs-Brill + Payne ----------------
LAM = 0.35
NFR = UM ** 2 / (32.174 * D)


def test_ex79_regime_boundaries():
    reg, L = bb_flow_regime(LAM, NFR)
    assert NFR == pytest.approx(29.6, abs=0.05)
    assert L["L1"] == pytest.approx(230, abs=1)
    assert L["L2"] == pytest.approx(0.0124, abs=2e-4)
    assert L["L3"] == pytest.approx(0.462, abs=0.005)
    assert L["L4"] == pytest.approx(606, rel=0.03)   # 590 from the equation; book rounds
    assert reg == "intermittent"


def test_ex79_holdup():
    HL, det = bb_holdup(LAM, NFR, NvL=10.28, theta_deg=90.0)
    assert det["HL0"] == pytest.approx(0.454, abs=0.001)
    assert det["C"] == pytest.approx(0.0351, abs=1e-4)
    assert det["psi"] == pytest.approx(1.01, abs=0.002)
    assert det["HL_original"] == pytest.approx(0.459, abs=0.001)
    assert HL == pytest.approx(0.424, abs=0.001)
