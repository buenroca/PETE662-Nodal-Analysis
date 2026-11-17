"""Fixed-length iterative pressure traverse, wellhead -> bottom (top-down).

Algorithm moved unchanged from the notebooks: for each dz step, guess the
step pressure drop, evaluate the gradient at the mid-step pressure, and
damp the update (average old and new) until |dP_new - dP_old| < 0.01 psi.
"""
import numpy as np
import pandas as pd

from .inputs import WellboreCase
from .pvt import make_pvt
from .hagedorn_brown import hb_gradient
from .beggs_brill import bb_gradient
from .flow import velocities

METHODS = {"HB": hb_gradient, "BB": bb_gradient}
METHOD_NAMES = {"HB": "Modified Hagedorn-Brown", "BB": "Beggs-Brill (Payne)"}


def step_details(qo, p, T_F, HL, regime, method, case, pvt):
    """Display quantities for one traverse step (same equations the gradient uses):
    superficial velocities, liquid/gas/mixture density and the friction Reynolds number."""
    v = velocities(qo, p, T_F, case, pvt)
    rhoL, muL = pvt.liquid_props(p, T_F)
    rhoG, muG = pvt.rho_gas(p, T_F), pvt.mu_gas(p, T_F)
    rho_mix = HL * rhoL + (1 - HL) * rhoG
    D = case.D
    if method == "BB":
        lam = min(max(v.lamL, 1e-6), 0.999999)
        rho_ns = lam * rhoL + (1 - lam) * rhoG
        mu_ns = lam * muL + (1 - lam) * muG
        Re = 1488 * rho_ns * v.um * D / mu_ns
    elif regime == "Griffith bubble":                  # liquid-velocity Reynolds number
        Re = 2.2e-2 * v.qL_ft3d * rhoL / (D * HL * muL)
    else:                                              # H-B mixture Reynolds number, Eq. 7-110
        mdot = v.qL_ft3d * rhoL + v.qg_ft3d * rhoG
        Re = 2.2e-2 * mdot / (D * muL ** HL * muG ** (1 - HL))
    return (v.usL, v.usg, v.um, v.lamL, rhoL, rhoG, rho_mix, Re)


def pressure_traverse(qo, case: WellboreCase, method="HB", pvt=None,
                      return_profile=False, tol=0.01, max_iter=50):
    """Bottom (node) pressure for oil rate qo (STB/d).

    Returns pwf, or (pwf, profile DataFrame) if return_profile=True.
    """
    key = method.upper()
    if key not in METHODS:
        raise ValueError("method must be 'HB' or 'BB'")
    if qo <= 0:
        raise ValueError("qo must be > 0 (multiphase correlations are undefined at zero rate).")
    gradient = METHODS[key]
    pvt = pvt or make_pvt(case)

    p, z, rows = float(case.ptf), 0.0, []
    while z < case.depth - 1e-9:
        step = min(case.dz, case.depth - z)
        Tmid = case.T_at_depth(z + step / 2)
        dp_old = pvt.liquid_props(p, Tmid)[0] / 144.0 * step     # hydrostatic first guess
        for it in range(max_iter):
            pavg = p + dp_old / 2
            grad, HL, reg = gradient(qo, pavg, Tmid, case, pvt)
            dp_new = grad * step
            if abs(dp_new - dp_old) < tol:
                break
            dp_old = 0.5 * (dp_old + dp_new)
        row = (z + step, p, pavg, Tmid, grad, dp_new, p + dp_new, HL, reg, it + 1)
        if return_profile:
            row += step_details(qo, pavg, Tmid, HL, reg, key, case, pvt)
        rows.append(row)
        p += dp_new
        z += step

    if not return_profile:
        return p
    prof = pd.DataFrame(rows, columns=[
        "Depth ft", "P_in psia", "P_avg psia", "T_avg F", "dp/dz psi/ft",
        "dP psi", "P_out psia", "HL", "Regime", "Iterations",
        "u_sL ft/s", "u_sg ft/s", "u_m ft/s", "lambda_L", "rho_L lbm/ft3", "rho_g lbm/ft3",
        "rho_mix lbm/ft3", "N_Re"])
    return p, prof


def vlp_curve(rates, case: WellboreCase, method="HB", pvt=None) -> pd.DataFrame:
    """VLP table: node pressure vs. oil rate."""
    pvt = pvt or make_pvt(case)
    rates = np.asarray(rates, dtype=float)
    return pd.DataFrame({"qo (STB/d)": rates,
                         "VLP pwf (psia)": [pressure_traverse(q, case, method, pvt) for q in rates]})
