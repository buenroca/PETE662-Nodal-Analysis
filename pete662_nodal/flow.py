"""Shared two-phase flow helpers: superficial velocities and friction factor."""
from typing import NamedTuple
import numpy as np

from .inputs import WellboreCase


class Velocities(NamedTuple):
    usL: float    # ft/s
    usg: float    # ft/s
    um: float     # ft/s
    lamL: float   # no-slip liquid fraction
    lamG: float   # no-slip gas fraction
    qL_ft3d: float  # in-situ liquid rate, ft3/d
    qg_ft3d: float  # in-situ free-gas rate, ft3/d


class Gradient(NamedTuple):
    """Unpacks like the notebooks' (grad, HL, regime) tuple."""
    grad: float    # psi/ft
    HL: float
    regime: str


def velocities(qo, p, T_F, case: WellboreCase, pvt) -> Velocities:
    qL_ft3d = (qo * pvt.Bo(p, T_F) + qo * case.WOR) * 5.615          # Bw ~ 1
    qg_ft3d = qo * max(case.GOR - pvt.Rs(p, T_F), 0.0) * pvt.Bg(p, T_F)
    usL = qL_ft3d / 86400.0 / case.A
    usg = qg_ft3d / 86400.0 / case.A
    um = usL + usg
    lamL = usL / um if um > 0 else 1.0
    return Velocities(usL, usg, um, lamL, 1.0 - lamL, qL_ft3d, qg_ft3d)


def fanning_factor(Re, rel_roughness, method="haaland"):
    """Fanning friction factor.

    method="haaland": explicit Haaland equation (what the notebooks used).
    method="chen":    Chen (1979) equation, PPS Eq. 7-35 (the book's choice).
    The two agree within ~1-2 % in the turbulent range.
    """
    Re = max(Re, 1.0)
    if Re < 2100:
        return 16.0 / Re
    e = rel_roughness
    if method == "haaland":
        fD = 1.0 / (-1.8 * np.log10((e / 3.7) ** 1.11 + 6.9 / Re)) ** 2
        return fD / 4.0
    if method == "chen":
        inner = e ** 1.1098 / 2.8257 + (7.149 / Re) ** 0.8981
        inv_sqrt_f = -4.0 * np.log10(e / 3.7065 - 5.0452 / Re * np.log10(inner))
        return 1.0 / inv_sqrt_f ** 2
    raise ValueError("method must be 'haaland' or 'chen'")
