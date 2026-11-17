"""Validation of CorrelationPVT against Petroleum Production Systems
Example 3-3 (oil properties) and Chapter 4 (gas properties)."""
import math
from dataclasses import replace

import pytest

from pete662_nodal import (WellboreCase, ReservoirInputs, CorrelationPVT, AppendixBPVT,
                           make_pvt, pressure_traverse, operating_point)
from pete662_nodal.pvt import dak_z, lee_gas_viscosity, standing_pseudocritical

# Example 3-3: Appendix-B oil (32 API, gas gravity 0.71), separator at 100 psig,
# 500 STB/d oil, WOR = 1.5, Rp = 500 scf/STB, at 2000 psia and 150 deg F.
EX33 = WellboreCase(depth=8000, ptf=200, T_surface=150, T_bottom=180, D_in=2.259,
                    GOR=500, WOR=1.5, API=32, gamma_g=0.71, mu_w=1.0,
                    pvt_model="correlations")
P, T = 2000.0, 150.0


@pytest.fixture
def pvt33():
    return CorrelationPVT(EX33, gamma_gd=0.85)   # book reads gamma_gd = 0.85 from Fig. 3-4


# ---------------- Example 3-3 (Vasquez-Beggs, Beggs-Robinson) ----------------
def test_ex33_solution_gor(pvt33):
    assert pvt33.Rs(P, T) == pytest.approx(369, abs=2)


def test_ex33_oil_fvf(pvt33):
    assert pvt33.Bo(P, T) == pytest.approx(1.22, abs=0.005)


def test_ex33_gas_fvf(pvt33):
    """Book: Bg = 6.97e-3 ft3/scf (implies Z = 0.805)."""
    assert pvt33.Bg(P, T) == pytest.approx(6.97e-3, rel=0.01)


def test_ex33_oil_density(pvt33):
    assert pvt33.rho_o(P, T) == pytest.approx(47.8, abs=0.1)


def test_ex33_oil_viscosity(pvt33):
    assert pvt33.mu_od(T) == pytest.approx(4.19, abs=0.01)
    assert pvt33.mu_o(P, T) == pytest.approx(1.16, abs=0.01)


def test_ex33_liquid_mixture(pvt33):
    rhoL, muL = pvt33.liquid_props(P, T)
    assert rhoL == pytest.approx(55.9, abs=0.1)     # Eq. 3-29 -> 55.9 lbm/ft3
    assert muL == pytest.approx(1.06, abs=0.01)     # Eq. 3-31 -> 1.06 cp


def test_ex33_volumetric_rates(pvt33):
    qo = 500.0
    qL_bbl = qo * (EX33.WOR + pvt33.Bo(P, T))
    assert qL_bbl == pytest.approx(1360, abs=5)
    qg_ft3 = pvt33.Bg(P, T) * (EX33.GOR - pvt33.Rs(P, T)) * qo
    assert qg_ft3 == pytest.approx(457, rel=0.02)


# ---------------- Chapter 4 gas properties ----------------
def test_standing_pseudocritical_eq_4_8_4_9():
    Tpc, ppc = standing_pseudocritical(0.709)
    assert Tpc == pytest.approx(392.1, abs=0.5)
    assert ppc == pytest.approx(669.7, abs=0.5)


def test_dak_against_independent_hall_yarborough():
    """DAK and Hall-Yarborough both fit the Standing-Katz chart; they agree
    within 0.5 % over the traverse range.  (Example 4-2 prints Z = 0.89 for
    'Eq. 4-5' at ppr = 5.96, Tpr = 1.69; both correlations give ~0.914.)"""
    from scipy.optimize import brentq

    def hy(ppr, tpr):
        t = 1 / tpr
        a = 0.06125 * t * math.exp(-1.2 * (1 - t) ** 2)
        f = lambda y: (-a * ppr + (y + y * y + y ** 3 - y ** 4) / (1 - y) ** 3
                       - (14.76 * t - 9.76 * t * t + 4.58 * t ** 3) * y * y
                       + (90.7 * t - 242.2 * t * t + 42.4 * t ** 3) * y ** (2.18 + 2.82 * t))
        return a * ppr / brentq(f, 1e-9, 0.99)

    for ppr, tpr in [(0.5, 1.4), (1, 1.2), (2, 1.5), (3, 2.0), (4, 1.3), (5.96, 1.69)]:
        assert dak_z(ppr, tpr) == pytest.approx(hy(ppr, tpr), rel=0.005)


def test_dak_ideal_gas_limit():
    assert dak_z(1e-4, 1.5) == pytest.approx(1.0, abs=1e-3)


def test_lee_viscosity_vs_chart_example_4_4():
    """Natural gas of Ex. 4-2/4-4 (gamma 0.65, 180 F, 4000 psi): the charts
    (Figs. 4-4, 4-5) give 0.0226 cp.  Lee et al. lands within ~6 %."""
    M = 0.65 * 28.97
    z = dak_z(4000 / 671, 640 / 378)
    rho = M * 4000 / (z * 10.73 * 640)
    assert lee_gas_viscosity(rho, 180, M) == pytest.approx(0.0226, rel=0.07)


def test_gas_properties_smooth():
    """No jumps like the Appendix-B step Z-factor."""
    pvt = make_pvt(EX33)
    for p in (999, 2499):
        assert pvt.rho_gas(p + 2, T) / pvt.rho_gas(p, T) == pytest.approx((p + 2) / p, rel=0.002)


# ---------------- bubble point / undersaturated behaviour ----------------
def test_bubble_point_of_produced_fluid(pvt33):
    pb = pvt33.bubble_point(T)
    assert pvt33.rs_vb(pb, T) == pytest.approx(EX33.GOR, rel=1e-9)
    assert pvt33.Rs(pb + 500, T) == EX33.GOR                 # no extra gas above pb
    assert pvt33.Bo(pb + 500, T) < pvt33.Bo(pb, T)            # compressed oil
    assert pvt33.Bo(pb + 1e-6, T) == pytest.approx(pvt33.Bo(pb - 1e-6, T), rel=1e-6)
    assert pvt33.mu_o(pb + 500, T) > pvt33.mu_o(pb, T)
    assert pvt33.rho_o(pb + 1e-6, T) == pytest.approx(pvt33.rho_o(pb - 1e-6, T), rel=1e-6)


def test_api_branch_heavy_oil():
    heavy = CorrelationPVT(replace(EX33, API=22.0))
    assert heavy._rs[1] == pytest.approx(1.0937)
    assert 0 < heavy.Rs(1500, 150) < heavy.Rs(2500, 150)


# ---------------- integration ----------------
def test_pvt_model_selection():
    assert isinstance(make_pvt(replace(EX33, pvt_model="appendix_b")), AppendixBPVT)
    assert isinstance(make_pvt(EX33), CorrelationPVT)
    with pytest.raises(ValueError):
        make_pvt(replace(EX33, pvt_model="nonsense"))


def test_exam1_with_correlations_close_to_appendix_b():
    ex1 = WellboreCase(depth=9000, ptf=100, T_surface=150, T_bottom=180, D_in=2.44, GOR=750)
    res = ReservoirInputs(3000, 4336, k_md=13, h_ft=115, mu_o_cp=0.5, Bo=1.46,
                          re_ft=1490, rw_ft=0.25, kro=0.5)
    a = operating_point(res, ex1, "HB", n_points=15)
    c = operating_point(res, replace(ex1, pvt_model="correlations"), "HB", n_points=15)
    assert c.status == "flowing"
    assert c.q == pytest.approx(a.q, rel=0.06)       # ~3 % lower: denser correlation oil
    assert pressure_traverse(c.q, replace(ex1, pvt_model="correlations")) == pytest.approx(c.pwf, abs=1)
