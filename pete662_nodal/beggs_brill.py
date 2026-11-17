"""Beggs-Brill pressure gradient with the Payne et al. upward-flow holdup correction
(PPS Sec. 7.4.3, Table 7-2).

Moved unchanged from the Homework 5 notebooks; the holdup steps reproduce
textbook Example 7-9 (HL0 = 0.454, C = 0.0351, psi = 1.01, HL = 0.424).

TODO (minor): in the TRANSITION regime the textbook method computes the
segregated and intermittent holdups each with its own inclination
correction and then interpolates.  This code interpolates the horizontal
holdups first and applies the intermittent correction to the blend.
"""
import numpy as np

from .flow import velocities, fanning_factor, Gradient

G = 32.174
GC = 32.174

BB_H = {  # a, b, c for horizontal holdup HL0 = a*lam^b / NFR^c
    "segregated": (0.98, 0.4846, 0.0868),
    "intermittent": (0.845, 0.5351, 0.0173),
    "distributed": (1.065, 0.5824, 0.0609),
}
BB_UPHILL = {  # d, e, f, g for C = (1-lam) ln(d lam^e NvL^f NFR^g)
    "segregated": (0.011, -3.768, 3.539, -1.614),
    "intermittent": (2.96, 0.305, -0.4473, 0.0978),
}


def bb_flow_regime(lam, NFR):
    L1 = 316 * lam ** 0.302
    L2 = 0.0009252 * lam ** (-2.4684)
    L3 = 0.1 * lam ** (-1.4516)
    L4 = 0.5 * lam ** (-6.738)
    if (lam < 0.01 and NFR < L1) or (lam >= 0.01 and NFR < L2):
        reg = "segregated"
    elif 0.01 <= lam < 0.4 and L2 <= NFR <= L3:
        reg = "transition"
    elif (lam < 0.4 and L3 < NFR <= L1) or (lam >= 0.4 and L3 < NFR <= L4):
        reg = "intermittent"
    else:
        reg = "distributed"
    return reg, dict(L1=L1, L2=L2, L3=L3, L4=L4)


def bb_base_holdup(reg, lam, NFR):
    a, b, c = BB_H[reg]
    return max(lam, a * lam ** b / max(NFR, 1e-12) ** c)


def bb_holdup(lam, NFR, NvL, theta_deg=90.0, payne=True):
    """Beggs-Brill holdup for upward flow.  Returns (HL, details)."""
    reg, L = bb_flow_regime(lam, NFR)
    if reg == "transition":
        A = (L["L3"] - NFR) / (L["L3"] - L["L2"])
        HL0 = A * bb_base_holdup("segregated", lam, NFR) + (1 - A) * bb_base_holdup("intermittent", lam, NFR)
        corr_reg = "intermittent"
    else:
        HL0 = bb_base_holdup(reg, lam, NFR)
        corr_reg = reg

    if theta_deg > 0 and corr_reg in BB_UPHILL:
        d, e, f, g = BB_UPHILL[corr_reg]
        arg = d * lam ** e * max(NvL, 1e-12) ** f * max(NFR, 1e-12) ** g
        C = max(0.0, (1 - lam) * np.log(max(arg, 1e-12)))
    else:
        C = 0.0          # distributed flow uphill: C = 0

    th = np.radians(theta_deg)
    psi = 1 + C * (np.sin(1.8 * th) - np.sin(1.8 * th) ** 3 / 3.0)
    HL_original = max(lam, HL0 * psi)
    HL = 0.924 * HL_original if (payne and theta_deg > 0) else HL_original
    HL = float(np.clip(HL, lam, 1.0))
    return HL, dict(regime=reg, HL0=HL0, C=C, psi=psi, HL_original=HL_original, **L)


def bb_gradient(qo, p, T_F, case, pvt) -> Gradient:
    """Beggs-Brill + Payne dp/dz (psi/ft) along the flow direction."""
    v = velocities(qo, p, T_F, case, pvt)
    rhoL, muL = pvt.liquid_props(p, T_F)
    rhoG = pvt.rho_gas(p, T_F)
    D = case.D
    lam = float(np.clip(v.lamL, 1e-6, 0.999999))
    um = v.um
    NFR = um ** 2 / (G * D)
    NvL = 1.938 * v.usL * (rhoL / case.sigma_L) ** 0.25

    HL, det = bb_holdup(lam, NFR, NvL, case.theta_deg)

    rho_bar = HL * rhoL + (1 - HL) * rhoG
    rho_ns = lam * rhoL + (1 - lam) * rhoG
    mu_ns = lam * muL + (1 - lam) * pvt.mu_gas(p, T_F)
    Re = 1488 * rho_ns * um * D / max(mu_ns, 1e-12)
    fn = fanning_factor(Re, case.rel_roughness, case.friction_method)

    x = lam / max(HL ** 2, 1e-12)
    if 1 < x < 1.2:
        S = np.log(2.2 * x - 1.2)
    else:
        lx = np.log(max(x, 1e-12))
        den = -0.0523 + 3.182 * lx - 0.8725 * lx ** 2 + 0.01853 * lx ** 4
        S = lx / den if abs(den) > 1e-12 else 0.0
    ftp = fn * np.exp(S)

    th = np.radians(case.theta_deg)
    grad_pe = rho_bar * np.sin(th) / 144.0
    grad_f = 2 * ftp * rho_ns * um ** 2 / (GC * D * 144.0)
    Ek = um * v.usg * rho_ns / (GC * max(p * 144.0, 1e-12))
    grad = (grad_pe + grad_f) / max(1 - Ek, 0.2)
    return Gradient(grad, HL, det["regime"])
