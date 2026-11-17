"""Fluid properties for the pressure traverse.

Two interchangeable PVT models (same method names, all take p in psia and
T in deg F):

``AppendixBPVT``  - Appendix-B curve approximation from the HW5 / Exam 1
                    notebooks (one specific oil; ignores temperature).  Kept
                    so results can be compared with the homework.
``CorrelationPVT`` - general black-oil correlations from the textbook:
    * Vasquez & Beggs Rs, Bo (PPS Sec. 3.2.2, Eqs. 3-8 to 3-15)
    * oil density, Eqs. 3-17 / 3-18
    * Beggs & Robinson dead/live oil viscosity, Vasquez & Beggs above pb
      (Eqs. 3-19 to 3-27)
    * Standing pseudocritical properties from gas gravity (Eqs. 4-8, 4-9)
    * Dranchuk & Abou-Kassem Z-factor (Eqs. 4-5, 4-6)
    * Lee, Gonzalez & Eakin gas viscosity (Eqs. 4-23 to 4-26)
    * liquid mixture density / viscosity, Eqs. 3-29 and 3-31

Select with ``WellboreCase(pvt_model="appendix_b" | "correlations")`` or
build directly with ``make_pvt(case)``.

Bubble point in CorrelationPVT: the produced oil cannot hold more gas than the
producing GOR, so Rs = min(Rs_VB(p, T), GOR).  The pressure where
Rs_VB = GOR is the bubble point of the produced fluid at that temperature
(book: "At the bubble point ... Rs = Rp", Sec. 3.2.2).  ``case.pb`` and
``case.Rsb`` are NOT used by CorrelationPVT (they still feed the IPR).
"""
import math
import numpy as np

from .inputs import WellboreCase

R_GAS = 10.73      # psia ft3 / (lb-mol R)
M_AIR = 28.97


# =====================================================================
#  Model 1 - Appendix-B curves (HW5 / Exam 1), unchanged numbers
# =====================================================================
class AppendixBPVT:
    """KNOWN LIMITATIONS: one fluid only, no temperature dependence, linear
    Rs, step-function Z (gas density jumps ~3-6 % at 1000 and 2500 psia)."""
    name = "Appendix-B curves"

    def __init__(self, case: WellboreCase):
        self.case = case
        pb = case.pb
        self._bo_p = np.array([0, 1000, 2000, 3000, pb, 6000.0])
        self._bo_v = np.array([1.08, 1.18, 1.26, 1.33, 1.46, 1.43])
        self._mu_p = np.array([0, 1000, 2000, 3000, 4000, 6000.0])
        self._mu_v = np.array([1.90, 1.10, 0.75, 0.60, 0.50, 0.55])
        self._rho_p = np.array([0, 1000, 2000, 3000, 4000, pb, 6000.0])
        self._rho_v = np.array([49.0, 47.0, 44.0, 42.0, 40.0, 38.9, 39.5])

    def Rs(self, p, T_F=None):
        c = self.case
        return min(c.Rsb, c.Rsb * max(p, 0.0) / c.pb)

    def Bo(self, p, T_F=None):
        return float(np.interp(p, self._bo_p, self._bo_v))

    def mu_o(self, p, T_F=None):
        return float(np.interp(p, self._mu_p, self._mu_v))

    def rho_o(self, p, T_F=None):
        return float(np.interp(p, self._rho_p, self._rho_v))

    def z_factor(self, p, T_F):
        if p < 1000:
            return 0.98
        if p < 2500:
            return 0.93
        return 0.90

    def Bg(self, p, T_F):
        """Gas FVF, ft3/scf."""
        return 0.02827 * self.z_factor(p, T_F) * (T_F + 460.0) / max(p, 1.0)

    def rho_gas(self, p, T_F):
        return M_AIR * self.case.gamma_g * p / (self.z_factor(p, T_F) * R_GAS * (T_F + 460.0))

    def mu_gas(self, p, T_F):
        return self.case.mu_g

    def liquid_props(self, p, T_F=None):
        """Oil + water liquid density (lbm/ft3) and viscosity (cp) - notebook form."""
        c = self.case
        ro, mo = self.rho_o(p), self.mu_o(p)
        if c.WOR <= 0:
            return ro, mo
        fw = c.WOR / (1.0 + c.WOR)
        return (1 - fw) * ro + fw * 62.4 * c.gamma_w, (1 - fw) * mo + fw * c.mu_w


# =====================================================================
#  Model 2 - textbook correlations
# =====================================================================
# Dranchuk & Abou-Kassem constants (Eq. 4-5)
_DAK = (0.3265, -1.0700, -0.5339, 0.01569, -0.05165, 0.5475,
        -0.7361, 0.1844, 0.1056, 0.6134, 0.7210)


def standing_pseudocritical(gamma_g):
    """Natural-gas pseudocritical T (deg R) and p (psia), PPS Eqs. 4-8 and 4-9
    (coefficients as printed in the textbook)."""
    Tpc = 168.0 + 325.0 * gamma_g - 12.5 * gamma_g ** 2
    ppc = 677.0 + 15.0 * gamma_g - 35.7 * gamma_g ** 2
    return Tpc, ppc


def dak_z(ppr, Tpr, tol=1e-10, max_iter=100):
    """Dranchuk & Abou-Kassem Z-factor (Eqs. 4-5, 4-6), solved by Newton
    iteration on the reduced density rho_r = 0.27 ppr / (Z Tpr)."""
    if ppr <= 0:
        return 1.0
    A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11 = _DAK
    c1 = A1 + A2 / Tpr + A3 / Tpr ** 3 + A4 / Tpr ** 4 + A5 / Tpr ** 5
    c2 = A6 + A7 / Tpr + A8 / Tpr ** 2
    c3 = A9 * (A7 / Tpr + A8 / Tpr ** 2)

    def z_of_rho(r):
        return (1 + c1 * r + c2 * r ** 2 - c3 * r ** 5
                + A10 * (1 + A11 * r ** 2) * r ** 2 / Tpr ** 3 * math.exp(-A11 * r ** 2))

    z = 1.0
    for _ in range(max_iter):
        r = 0.27 * ppr / (z * Tpr)
        f = z - z_of_rho(r)
        # derivative of f w.r.t. z by finite difference (robust, cheap)
        h = 1e-6 * z
        r2 = 0.27 * ppr / ((z + h) * Tpr)
        df = (h - (z_of_rho(r2) - z_of_rho(r))) / h
        z_new = z - f / df
        if z_new <= 0.05:
            z_new = 0.5 * z
        if abs(z_new - z) < tol:
            return z_new
        z = z_new
    raise RuntimeError(f"DAK Z-factor did not converge (ppr={ppr:.3f}, Tpr={Tpr:.3f}).")


def lee_gas_viscosity(rho_g_lbft3, T_F, M):
    """Lee, Gonzalez & Eakin gas viscosity (cp), Eqs. 4-23 to 4-26."""
    T = T_F + 460.0
    rho = rho_g_lbft3 / 62.428          # g/cc
    A = (9.379 + 0.01607 * M) * T ** 1.5 / (209.2 + 19.26 * M + T)
    B = 3.448 + 986.4 / T + 0.01009 * M
    C = 2.447 - 0.2224 * B
    return 1e-4 * A * math.exp(B * rho ** C)


class CorrelationPVT:
    """Black-oil properties from the textbook correlations (see module docstring).

    Optional arguments
    ------------------
    sep_p_psia, sep_T_F : separator conditions for the Vasquez-Beggs gas-gravity
        correction (Eq. 3-8).  None = separator at the 100 psig reference.
    gamma_gd : dissolved-gas gravity for the oil density (Eq. 3-17).  The book
        reads it from Fig. 3-4; default uses the separator gas gravity.
    """
    name = "Textbook correlations (Vasquez-Beggs / Beggs-Robinson / DAK / Lee)"

    def __init__(self, case: WellboreCase, sep_p_psia=None, sep_T_F=None, gamma_gd=None):
        self.case = case
        api, gg = case.API, case.gamma_g
        if sep_p_psia is not None and sep_T_F is not None:
            gg = gg * (1 + 5.912e-5 * api * sep_T_F * math.log10(sep_p_psia / 114.7))
        self.gamma_gs = gg
        self.gamma_gd = gamma_gd if gamma_gd is not None else gg
        self.gamma_o = 141.5 / (131.5 + api)
        if api <= 30:
            self._rs = (1 / 27.64, 1.0937, 11.172)
            self._bo = (4.677e-4, 0.1751e-4, -1.8106e-8)
        else:
            self._rs = (1 / 56.06, 1.187, 10.393)
            self._bo = (4.670e-4, 0.11e-4, 0.1337e-8)
        self.Tpc, self.ppc = standing_pseudocritical(case.gamma_g)
        self.M_gas = M_AIR * case.gamma_g
        self._cache = {}

    # ---------------- oil: saturated (Vasquez-Beggs) ----------------
    def rs_vb(self, p, T_F):
        """Vasquez-Beggs solution GOR (scf/STB) for a saturated oil, Eqs. 3-9/3-10."""
        c1, c2, c3 = self._rs
        A = self.case.API / (T_F + 460.0)
        return c1 * self.gamma_gs * max(p, 0.0) ** c2 * 10 ** (c3 * A)

    def bubble_point(self, T_F):
        """Bubble point of the PRODUCED fluid: Rs_VB(pb, T) = GOR (psia)."""
        c1, c2, c3 = self._rs
        A = self.case.API / (T_F + 460.0)
        gor = max(self.case.GOR, 1e-6)
        return (gor / (c1 * self.gamma_gs * 10 ** (c3 * A))) ** (1.0 / c2)

    def _bo_sat(self, Rs, T_F):
        C1, C2, C3 = self._bo
        F = (T_F - 60.0) * (self.case.API / self.gamma_gs)
        return 1.0 + C1 * Rs + C2 * F + C3 * Rs * F

    def Rs(self, p, T_F):
        return min(self.rs_vb(p, T_F), self.case.GOR)

    def Bo(self, p, T_F):
        pb = self.bubble_point(T_F)
        if p <= pb:
            return self._bo_sat(self.rs_vb(p, T_F), T_F)
        Bob = self._bo_sat(self.case.GOR, T_F)
        a = 1e-5 * (-1433 + 5 * self.case.GOR + 17.2 * T_F - 1180 * self.gamma_gs
                    + 12.61 * self.case.API)                      # Eq. 3-15
        return Bob * math.exp(-a * math.log(p / pb))

    def rho_o(self, p, T_F):
        pb = self.bubble_point(T_F)
        if p <= pb:                                              # Eq. 3-17
            Rs = self.rs_vb(p, T_F)
            return (62.4 * self.gamma_o + 0.01361 * self.gamma_gd * Rs) / self._bo_sat(Rs, T_F)
        Bob = self._bo_sat(self.case.GOR, T_F)                   # Eq. 3-18
        rho_ob = (62.4 * self.gamma_o + 0.01361 * self.gamma_gd * self.case.GOR) / Bob
        return rho_ob * Bob / self.Bo(p, T_F)

    def mu_od(self, T_F):
        """Beggs-Robinson dead-oil viscosity (cp), Eqs. 3-19 to 3-22."""
        x = 10 ** (3.0324 - 0.02023 * self.case.API) * T_F ** -1.163
        return 10 ** x - 1.0

    def _mu_live(self, Rs, T_F):
        a = 10.715 * (Rs + 100) ** -0.515
        b = 5.44 * (Rs + 150) ** -0.338
        return a * self.mu_od(T_F) ** b                          # Eqs. 3-23 to 3-25

    def mu_o(self, p, T_F):
        pb = self.bubble_point(T_F)
        if p <= pb:
            return self._mu_live(self.rs_vb(p, T_F), T_F)
        mu_ob = self._mu_live(self.case.GOR, T_F)                 # Eqs. 3-26, 3-27
        m = 2.6 * p ** 1.187 * math.exp(-11.513 - 8.98e-5 * p)
        return mu_ob * (p / pb) ** m

    # ---------------- gas ----------------
    def z_factor(self, p, T_F):
        key = (round(p, 6), round(T_F, 6))
        z = self._cache.get(key)
        if z is None:
            z = dak_z(p / self.ppc, (T_F + 460.0) / self.Tpc)
            if len(self._cache) > 20000:
                self._cache.clear()
            self._cache[key] = z
        return z

    def Bg(self, p, T_F):
        """Gas FVF, ft3/scf (Eq. 4-28 with psc = 14.7 psia, Tsc = 520 R)."""
        return 0.02827 * self.z_factor(p, T_F) * (T_F + 460.0) / max(p, 1.0)

    def rho_gas(self, p, T_F):
        return self.M_gas * p / (self.z_factor(p, T_F) * R_GAS * (T_F + 460.0))

    def mu_gas(self, p, T_F):
        return lee_gas_viscosity(self.rho_gas(p, T_F), T_F, self.M_gas)

    # ---------------- liquid mixture (Eqs. 3-29, 3-31) ----------------
    def liquid_props(self, p, T_F):
        c = self.case
        Bo, ro, mo = self.Bo(p, T_F), self.rho_o(p, T_F), self.mu_o(p, T_F)
        if c.WOR <= 0:
            return ro, mo
        rw = 62.4 * c.gamma_w
        rhoL = (c.WOR * rw + Bo * ro) / (c.WOR + Bo)                       # volume-weighted
        muL = (c.WOR * rw * c.mu_w + Bo * ro * mo) / (c.WOR * rw + Bo * ro)  # mass-weighted
        return rhoL, muL


PVT_MODELS = {"appendix_b": AppendixBPVT, "correlations": CorrelationPVT}


def make_pvt(case: WellboreCase):
    """Build the PVT object selected by ``case.pvt_model``."""
    try:
        return PVT_MODELS[case.pvt_model](case)
    except KeyError:
        raise ValueError(f"pvt_model must be one of {list(PVT_MODELS)}") from None
