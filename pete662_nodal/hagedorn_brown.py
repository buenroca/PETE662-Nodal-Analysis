"""Modified Hagedorn-Brown pressure gradient (PPS Sec. 7.4.3, Eqs. 7-91 to 7-116).

Non-bubble holdup equations are the ones validated against Example 7-8 in
BUENROSTRO-HB-BB-Methods-HK5-P7-8-VER1.ipynb (moved here unchanged).

CHANGE FROM THE NOTEBOOKS - Griffith bubble-flow friction:
  The notebooks switched to the Griffith HOLDUP when lambda_g < L_B but kept
  the mixture-mass-rate friction term.  The book (Eqs. 7-111 to 7-116) bases
  Griffith friction on the in-situ average LIQUID velocity u_L = u_sL / H_L:

      dp/dz = rho_bar/144 + f * m_L^2 / (7.413e10 * D^5 * rho_L * H_L^2) / 144
      N_Re  = 2.2e-2 * m_L / (D * H_L * mu_L)

  with m_L = liquid-only mass rate (lbm/d).  Kinetic energy is neglected.
"""
import numpy as np

from .flow import velocities, fanning_factor, Gradient

PA = 14.7  # psia, atmospheric pressure in Eq. 7-104
GRIFFITH_SLIP_VELOCITY = 0.8  # ft/s


def hb_dimensionless_groups(usL, usg, rhoL, muL, sigma_L, D):
    """N_vL, N_vg, N_D, N_L (Eqs. 7-95 to 7-98). D in ft."""
    NvL = 1.938 * usL * (rhoL / sigma_L) ** 0.25
    Nvg = 1.938 * usg * (rhoL / sigma_L) ** 0.25
    ND = 120.872 * D * (rhoL / sigma_L) ** 0.5
    NL = 0.15726 * muL * (1.0 / (rhoL * sigma_L ** 3)) ** 0.25
    return NvL, Nvg, ND, NL


def hb_holdup_from_groups(NvL, Nvg, ND, NL, p):
    """Original Hagedorn-Brown holdup from the curve fits, Eqs. 7-103 to 7-107.

    Returns (HL, details) - HL is NOT yet limited to >= lambda_L.
    """
    CNL = (0.0019 + 0.0322 * NL - 0.6642 * NL ** 2 + 4.9951 * NL ** 3) / \
          (1.0 - 10.0147 * NL + 33.8696 * NL ** 2 + 277.2817 * NL ** 3)
    CNL = max(CNL, 1e-8)
    H = (NvL / Nvg ** 0.575) * (p / PA) ** 0.1 * (CNL / ND)
    ratio = np.sqrt((0.0047 + 1123.32 * H + 729489.64 * H ** 2) /
                    (1.0 + 1097.1566 * H + 722153.97 * H ** 2))
    B = Nvg * NL ** 0.38 / ND ** 2.14
    psi = (1.0886 - 69.9473 * B + 2334.3497 * B ** 2 - 12896.6838 * B ** 3) / \
          (1.0 - 53.4401 * B + 1517.9369 * B ** 2 - 8419.8115 * B ** 3)
    return ratio * psi, dict(CNL=CNL, H=H, HL_over_psi=ratio, B=B, psi=psi)


def bubble_flow_limit(um, D):
    """L_B, Eq. 7-91 (floored at 0.13). D in ft."""
    return max(0.13, 1.071 - 0.2218 * um ** 2 / D)


def griffith_holdup(um, usg, vs=GRIFFITH_SLIP_VELOCITY):
    """Griffith bubble-flow liquid holdup (PPS Sec. 7.4.3, Griffith correlation)."""
    disc = max((1.0 + um / vs) ** 2 - 4.0 * usg / vs, 0.0)
    return 1.0 - 0.5 * (1.0 + um / vs - np.sqrt(disc))


def mixture_gradient(rho_bar, mdot, mu_mix, D, rel_roughness, friction_method="haaland"):
    """Eq. 7-93 field-unit form with Eq. 7-110 Reynolds number (KE neglected)."""
    Re = 2.2e-2 * mdot / (D * max(mu_mix, 1e-12))
    f = fanning_factor(Re, rel_roughness, friction_method)
    return (rho_bar + f * mdot ** 2 / (7.413e10 * D ** 5 * max(rho_bar, 1e-9))) / 144.0


def griffith_gradient(rho_bar, rhoL, muL, HL, mL, D, rel_roughness, friction_method="haaland"):
    """Griffith bubble-flow gradient, Eqs. 7-111 to 7-116."""
    Re = 2.2e-2 * mL / (D * HL * max(muL, 1e-12))
    f = fanning_factor(Re, rel_roughness, friction_method)
    return (rho_bar + f * mL ** 2 / (7.413e10 * D ** 5 * rhoL * HL ** 2)) / 144.0


def hb_gradient(qo, p, T_F, case, pvt) -> Gradient:
    """Modified Hagedorn-Brown dp/dz (psi/ft) for vertical upward flow."""
    v = velocities(qo, p, T_F, case, pvt)
    rhoL, muL = pvt.liquid_props(p, T_F)
    rhoG = pvt.rho_gas(p, T_F)
    D = case.D

    if v.usg <= 1e-12:                       # no free gas
        HL, regime = 1.0, "single-liquid"
    elif v.lamG < bubble_flow_limit(v.um, D):
        HL = min(max(griffith_holdup(v.um, v.usg), v.lamL), 1.0)
        regime = "Griffith bubble"
    else:
        NvL, Nvg, ND, NL = hb_dimensionless_groups(v.usL, v.usg, rhoL, muL, case.sigma_L, D)
        HL, _ = hb_holdup_from_groups(NvL, Nvg, ND, NL, p)
        HL = min(max(HL, v.lamL), 1.0)       # modification: HL >= no-slip holdup
        regime = "Hagedorn-Brown"

    rho_bar = HL * rhoL + (1 - HL) * rhoG
    mL = v.qL_ft3d * rhoL                    # lbm/d, liquid only

    if regime == "Griffith bubble":
        grad = griffith_gradient(rho_bar, rhoL, muL, HL, mL, D,
                                 case.rel_roughness, case.friction_method)
    else:
        mdot = mL + v.qg_ft3d * rhoG          # lbm/d, total
        mu_mix = muL ** HL * pvt.mu_gas(p, T_F) ** (1 - HL)   # H-B mixture viscosity for Eq. 7-110
        grad = mixture_gradient(rho_bar, mdot, mu_mix, D,
                                case.rel_roughness, case.friction_method)
    return Gradient(grad, HL, regime)
