"""Regression against the Homework 5 / Exam 1 notebook results, plus IPR and
nodal-solver behaviour checks."""
import numpy as np
import pytest

from pete662_nodal import (WellboreCase, IPRInputs, ReservoirInputs, oil_rate, pwf_from_rate,
                           pressure_traverse, operating_point, run_sensitivity)
from pete662_nodal.hagedorn_brown import griffith_holdup, griffith_gradient

HW5 = WellboreCase(depth=8000, ptf=200, T_surface=150, T_bottom=180, D_in=2.259, GOR=400)
EX1 = WellboreCase(depth=9000, ptf=100, T_surface=150, T_bottom=180, D_in=2.44, GOR=750)
EX1_RES = ReservoirInputs(3000, 4336, k_md=13, h_ft=115, mu_o_cp=0.5, Bo=1.46,
                          re_ft=1490, rw_ft=0.25, kro=0.5)


# ---- notebook regression ----
def test_hw5_hb_matches_notebook_outside_bubble_flow():
    """At 300 STB/d the traverse never enters Griffith flow -> identical to notebook."""
    assert pressure_traverse(300, HW5, "HB") == pytest.approx(1621.96, abs=0.01)


@pytest.mark.parametrize("q, notebook", [(550, 1734.48), (800, 1830.92), (1200, 1971.69), (2000, 2254.31)])
def test_hw5_hb_griffith_fix_small_increase(q, notebook):
    """These rates reach Griffith bubble flow near bottom; the corrected
    liquid-velocity friction raises pwf by only a few psi."""
    p = pressure_traverse(q, HW5, "HB")
    assert notebook <= p < notebook + 5


@pytest.mark.parametrize("q, expected", [(50, 2597.45), (300, 1925.73), (1200, 2036.3), (2000, 2337.02)])
def test_hw5_bb_matches_notebook(q, expected):
    assert pressure_traverse(q, HW5, "BB") == pytest.approx(expected, abs=0.05)


@pytest.mark.parametrize("q, expected", [(50, 1135.79), (300, 998.82), (1050, 1381.53), (3000, 2229.01)])
def test_exam1_vlp_matches_notebook(q, expected):
    assert pressure_traverse(q, EX1, "HB") == pytest.approx(expected, abs=0.01)


def test_griffith_gradient_hand_calc():
    um, usg, usL = 3.0, 0.3, 2.7
    HL = griffith_holdup(um, usg)
    assert usL / um <= HL <= 1
    rhoL, rhoG, muL, D = 50.0, 5.0, 1.0, 0.2
    mL = usL * np.pi * D ** 2 / 4 * 86400 * rhoL
    uL = usL / HL
    Re = 1488 * rhoL * uL * D / muL
    from pete662_nodal.flow import fanning_factor
    f = fanning_factor(Re, 0.0006)
    rho_bar = HL * rhoL + (1 - HL) * rhoG
    expected = (rho_bar + 2 * f * rhoL * uL ** 2 / (32.174 * D)) / 144
    assert griffith_gradient(rho_bar, rhoL, muL, HL, mL, D, 0.0006) == pytest.approx(expected, rel=0.01)


# ---- IPR ----
def test_exam1_productivity_index():
    assert EX1_RES.J == pytest.approx(0.913, abs=0.001)
    assert oil_rate(0, EX1_RES.to_ipr()) == pytest.approx(1521.2, abs=0.2)


def test_ipr_inverse_roundtrip():
    ipr = IPRInputs(3500, 2200, 1.5)
    for q in (100, 1000, 2500):
        assert oil_rate(pwf_from_rate(q, ipr), ipr) == pytest.approx(q)


# ---- nodal solver ----
def test_exam1_operating_point():
    """Notebook secant method: ~1110 STB/d, ~1408 psia."""
    r = operating_point(EX1_RES, EX1, "HB")
    assert r.status == "flowing"
    assert r.q == pytest.approx(1110, abs=2)
    assert r.pwf == pytest.approx(1408, abs=2)
    # IPR and VLP agree at the answer
    assert pressure_traverse(r.q, EX1, "HB") == pytest.approx(r.pwf, abs=1.0)


def test_no_natural_flow_detected():
    weak = IPRInputs(1000, 4336, 0.5)            # depleted reservoir, low PI
    r = operating_point(weak, EX1, "HB", n_points=12)
    assert r.status == "no natural flow"
    assert r.q is None


def test_sensitivity_tubing_size_runs():
    df, _ = run_sensitivity(EX1_RES, EX1, "D_in", [1.995, 2.44, 2.992], n_points=15)
    assert list(df["status"]) == ["flowing"] * 3
    assert df["q (STB/d)"].is_monotonic_increasing


def test_sensitivity_skin_reduces_rate():
    df, _ = run_sensitivity(EX1_RES, EX1, "skin", [0, 5, 10], n_points=15)
    assert df["q (STB/d)"].is_monotonic_decreasing
