"""Oil-well inflow performance (IPR).

* PI above the bubble point - PPS Ch. 2 (undersaturated oil).
* Vogel and generalized Vogel below the bubble point - PPS Ch. 3, Secs. 3.4-3.5.
* Skin enters through the radial-flow PI - PPS Ch. 2 and Ch. 6.

``IPRInputs`` / ``oil_rate`` / ``ipr_table`` / ``pwf_from_rate`` come from
Project2/PETE662_01_IPR_Jupyter.ipynb unchanged.  ``ReservoirInputs`` adds
the radial-flow PI from the Exam 1 notebook so permeability and skin can be
used as sensitivity variables.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import brentq


@dataclass(frozen=True)
class IPRInputs:
    reservoir_pressure: float     # psia
    bubblepoint_pressure: float   # psia
    productivity_index: float     # STB/d/psi (tangent PI at p_r if saturated)

    def __post_init__(self):
        vals = (self.reservoir_pressure, self.bubblepoint_pressure, self.productivity_index)
        if not all(np.isfinite(v) for v in vals):
            raise ValueError("Inputs must be finite.")
        if min(vals) <= 0:
            raise ValueError("Pressures and productivity index must be positive.")

    def to_ipr(self) -> "IPRInputs":
        return self


def oil_rate(pwf: float, inputs: IPRInputs) -> float:
    """Oil rate (STB/d) at flowing bottomhole pressure pwf (psia)."""
    pr, pb, j = inputs.reservoir_pressure, inputs.bubblepoint_pressure, inputs.productivity_index
    if not np.isfinite(pwf) or not 0 <= pwf <= pr:
        raise ValueError("pwf must be finite and between 0 and reservoir pressure.")
    if pr <= pb:                                   # saturated: Vogel
        x = pwf / pr
        return (j * pr / 1.8) * (1 - 0.2 * x - 0.8 * x * x)
    if pwf >= pb:                                  # single-phase PI
        return j * (pr - pwf)
    x = pwf / pb                                   # generalized Vogel
    return j * (pr - pb) + (j * pb / 1.8) * (1 - 0.2 * x - 0.8 * x * x)


def ipr_table(inputs: IPRInputs, n: int = 101) -> pd.DataFrame:
    if n < 2:
        raise ValueError("n must be at least 2")
    pressures = np.linspace(0, inputs.reservoir_pressure, n)
    return pd.DataFrame({"pwf_psia": pressures,
                         "qo_stb_day": [oil_rate(float(p), inputs) for p in pressures]})


def pwf_from_rate(q_target: float, inputs: IPRInputs) -> float:
    """Inverse IPR: pwf (psia) that delivers q_target (STB/d)."""
    qmax = oil_rate(0.0, inputs)
    if not np.isfinite(q_target) or not 0 <= q_target <= qmax:
        raise ValueError(f"Rate must be in [0, {qmax:.2f}] STB/day")
    if q_target == 0:
        return inputs.reservoir_pressure
    if q_target == qmax:
        return 0.0
    return brentq(lambda p: oil_rate(p, inputs) - q_target,
                  0.0, inputs.reservoir_pressure, xtol=1e-9)


def productivity_index(k_md, h_ft, mu_cp, Bo, re_ft, rw_ft, skin=0.0, kro=1.0,
                       pseudo_steady=True):
    """Radial-flow PI, STB/d/psi.

    J = 0.00708 k kro h / [mu Bo (ln(re/rw) - 0.75 + s)]   (pseudo-steady state)
    J = 0.00708 k kro h / [mu Bo (ln(re/rw) + s)]          (steady state)
    """
    den = np.log(re_ft / rw_ft) + skin - (0.75 if pseudo_steady else 0.0)
    if den <= 0:
        raise ValueError("ln(re/rw) - 0.75 + s must be positive; check skin / radii.")
    return 0.00708 * k_md * kro * h_ft / (mu_cp * Bo * den)


@dataclass(frozen=True)
class ReservoirInputs:
    """Reservoir description that builds an IPR (lets k and skin be varied)."""
    reservoir_pressure: float   # psia (average)
    bubblepoint_pressure: float # psia
    k_md: float
    h_ft: float
    mu_o_cp: float
    Bo: float
    re_ft: float
    rw_ft: float
    skin: float = 0.0
    kro: float = 1.0
    pseudo_steady: bool = True

    @property
    def J(self) -> float:
        return productivity_index(self.k_md, self.h_ft, self.mu_o_cp, self.Bo, self.re_ft,
                                  self.rw_ft, self.skin, self.kro, self.pseudo_steady)

    def to_ipr(self) -> IPRInputs:
        return IPRInputs(self.reservoir_pressure, self.bubblepoint_pressure, self.J)
